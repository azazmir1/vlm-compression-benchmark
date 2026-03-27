"""
TegraStats Monitor — Jetson power, thermal, and memory monitoring.

Parses tegrastats output in a background thread to capture:
- Total board power (VDD_IN or POM_5V_IN)
- GPU/CPU temperature
- RAM usage (shared unified memory)
- GPU frequency

Works on Jetson Orin Nano 8GB. On non-Jetson systems, silently returns zeros.

Usage:
    monitor = TegraStatsMonitor()
    monitor.start()
    # ... run inference ...
    monitor.stop()
    stats = monitor.stats()
    print(f"Avg power: {stats['avg_power_w']:.1f} W")
"""

import subprocess
import threading
import re
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class TegraStats:
    """Aggregated tegrastats measurements."""
    # Power (watts)
    avg_power_w: float = 0.0
    peak_power_w: float = 0.0
    power_samples: List[float] = field(default_factory=list)

    # Temperature (Celsius)
    avg_gpu_temp_c: float = 0.0
    peak_gpu_temp_c: float = 0.0
    avg_cpu_temp_c: float = 0.0
    peak_cpu_temp_c: float = 0.0

    # RAM (MB) — unified memory
    avg_ram_used_mb: float = 0.0
    peak_ram_used_mb: float = 0.0

    # GPU utilization (%)
    avg_gpu_util_pct: float = 0.0
    peak_gpu_util_pct: float = 0.0

    num_samples: int = 0

    def to_dict(self) -> Dict:
        return {
            'avg_power_w': round(self.avg_power_w, 2),
            'peak_power_w': round(self.peak_power_w, 2),
            'avg_gpu_temp_c': round(self.avg_gpu_temp_c, 1),
            'peak_gpu_temp_c': round(self.peak_gpu_temp_c, 1),
            'avg_cpu_temp_c': round(self.avg_cpu_temp_c, 1),
            'peak_cpu_temp_c': round(self.peak_cpu_temp_c, 1),
            'avg_ram_used_mb': round(self.avg_ram_used_mb, 1),
            'peak_ram_used_mb': round(self.peak_ram_used_mb, 1),
            'avg_gpu_util_pct': round(self.avg_gpu_util_pct, 1),
            'peak_gpu_util_pct': round(self.peak_gpu_util_pct, 1),
            'num_samples': self.num_samples,
        }


class TegraStatsMonitor:
    """
    Background monitor that parses tegrastats output.

    Runs `tegrastats --interval <ms>` in a subprocess and parses each line
    for power, temperature, RAM, and GPU frequency data.
    """

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._available = self._check_available()

        # Raw samples
        self._power_samples: List[float] = []
        self._gpu_temp_samples: List[float] = []
        self._cpu_temp_samples: List[float] = []
        self._ram_samples: List[float] = []
        self._gpu_util_samples: List[float] = []

    def _check_available(self) -> bool:
        """Check if tegrastats is available on this system."""
        try:
            result = subprocess.run(
                ['tegrastats', '--help'],
                capture_output=True, timeout=5
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    def start(self):
        """Start monitoring in background thread."""
        if not self._available:
            return

        self._clear()
        self._stop_event.clear()

        try:
            self._process = subprocess.Popen(
                ['tegrastats', '--interval', str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
        except (FileNotFoundError, OSError):
            self._available = False
            return

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring and wait for thread to finish."""
        self._stop_event.set()
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def stats(self) -> TegraStats:
        """Compute aggregated statistics from samples."""
        s = TegraStats()
        s.num_samples = len(self._power_samples)

        if self._power_samples:
            s.avg_power_w = sum(self._power_samples) / len(self._power_samples)
            s.peak_power_w = max(self._power_samples)
            s.power_samples = self._power_samples.copy()

        if self._gpu_temp_samples:
            s.avg_gpu_temp_c = sum(self._gpu_temp_samples) / len(self._gpu_temp_samples)
            s.peak_gpu_temp_c = max(self._gpu_temp_samples)

        if self._cpu_temp_samples:
            s.avg_cpu_temp_c = sum(self._cpu_temp_samples) / len(self._cpu_temp_samples)
            s.peak_cpu_temp_c = max(self._cpu_temp_samples)

        if self._ram_samples:
            s.avg_ram_used_mb = sum(self._ram_samples) / len(self._ram_samples)
            s.peak_ram_used_mb = max(self._ram_samples)

        if self._gpu_util_samples:
            s.avg_gpu_util_pct = sum(self._gpu_util_samples) / len(self._gpu_util_samples)
            s.peak_gpu_util_pct = max(self._gpu_util_samples)

        return s

    def _read_loop(self):
        """Read tegrastats output line by line."""
        while not self._stop_event.is_set() and self._process:
            line = self._process.stdout.readline()
            if not line:
                break
            self._parse_line(line.strip())

    def _parse_line(self, line: str):
        """
        Parse a tegrastats output line.

        Example Orin Nano line:
        03-27-2026 10:00:00 RAM 3456/7620MB ... VDD_IN 5432mW/5432mW
        GR3D_FREQ 76% ... gpu@45C cpu@42C ...
        """
        # Power: VDD_IN or POM_5V_IN (milliwatts)
        power_match = re.search(r'(?:VDD_IN|POM_5V_IN|VDD_CPU_GPU_CV)\s+(\d+)mW', line)
        if power_match:
            self._power_samples.append(int(power_match.group(1)) / 1000.0)

        # RAM usage: RAM XXXX/YYYYmb
        ram_match = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
        if ram_match:
            self._ram_samples.append(int(ram_match.group(1)))

        # GPU temperature: gpu@XXC or GPU@XXC
        gpu_temp = re.search(r'[Gg][Pp][Uu]@(\d+(?:\.\d+)?)C', line)
        if gpu_temp:
            self._gpu_temp_samples.append(float(gpu_temp.group(1)))

        # CPU temperature: cpu@XXC or CPU@XXC
        cpu_temp = re.search(r'[Cc][Pp][Uu]@(\d+(?:\.\d+)?)C', line)
        if cpu_temp:
            self._cpu_temp_samples.append(float(cpu_temp.group(1)))

        # GPU utilization: GR3D_FREQ XX% (this is utilization, not frequency)
        gpu_util = re.search(r'GR3D_FREQ\s+(\d+)%', line)
        if gpu_util:
            self._gpu_util_samples.append(float(gpu_util.group(1)))

    def _clear(self):
        self._power_samples.clear()
        self._gpu_temp_samples.clear()
        self._cpu_temp_samples.clear()
        self._ram_samples.clear()
        self._gpu_util_samples.clear()

    @property
    def available(self) -> bool:
        return self._available

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        return False
