"""
Hook Manager for Component-Level Profiling

Manages PyTorch forward hooks for timing model components. Registers hooks
on all leaf modules, categorizes them, and collects timing data.
"""

import torch.nn as nn
from typing import Dict, List
from .timing_tracker import TimingTracker
from .module_categorizer import ModuleCategorizer


class HookManager:
    """
    Manages forward hooks for component-level profiling.

    Registers pre and post-hooks on all leaf modules to time their execution.
    Categorizes modules and aggregates timing data by category.
    """

    def __init__(self, categorizer: ModuleCategorizer, tracker: TimingTracker):
        self.categorizer = categorizer
        self.tracker = tracker
        self.hooks = []
        self.module_categories = {}
        self.current_token_idx = None
        self._enabled = True

    def register_hooks(self, model: nn.Module):
        """Register forward hooks on all leaf modules."""
        num_registered = 0
        for name, module in model.named_modules():
            if len(list(module.children())) > 0:
                continue
            category = self.categorizer.categorize(name, type(module))
            if category == "ignore":
                continue
            self.module_categories[name] = category

            pre_hook = module.register_forward_pre_hook(self._make_pre_hook(name))
            post_hook = module.register_forward_hook(self._make_post_hook(name))
            self.hooks.extend([pre_hook, post_hook])
            num_registered += 1

        print(f"Registered {len(self.hooks)} hooks on {num_registered} modules "
              f"({len(set(self.module_categories.values()))} categories)")

    def _make_pre_hook(self, module_name: str):
        def hook(module, input):
            if self._enabled:
                self.tracker.record_start(module_name, self.current_token_idx)
        return hook

    def _make_post_hook(self, module_name: str):
        def hook(module, input, output):
            if self._enabled:
                self.tracker.record_end(module_name, self.current_token_idx)
        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def get_timings(self) -> Dict[str, List[Dict]]:
        """Get computed timings grouped by category."""
        if not self.tracker.timings:
            self.tracker.compute_timings()

        category_timings = {}
        for key, timing_data in self.tracker.timings.items():
            if '_t' in key and key.split('_t')[-1].isdigit():
                base_name = key.rsplit('_t', 1)[0]
            else:
                base_name = key

            category = self.module_categories.get(base_name, "other")
            if category not in category_timings:
                category_timings[category] = []

            category_timings[category].append({
                'module': base_name,
                'elapsed_ms': timing_data['elapsed_ms'],
                'token_idx': timing_data.get('token_idx')
            })

        return category_timings

    def get_category_summary(self) -> Dict[str, Dict]:
        """Get summary statistics for each category."""
        category_timings = self.get_timings()
        summary = {}
        total_time = 0.0

        for category, timings in category_timings.items():
            total_ms = sum(t['elapsed_ms'] for t in timings)
            summary[category] = {
                'total_ms': total_ms,
                'count': len(timings),
                'avg_ms': total_ms / len(timings) if timings else 0.0
            }
            total_time += total_ms

        for category in summary:
            if total_time > 0:
                summary[category]['percentage'] = (summary[category]['total_ms'] / total_time) * 100
            else:
                summary[category]['percentage'] = 0.0

        return summary

    def print_summary(self):
        summary = self.get_category_summary()
        print("\n" + "=" * 80)
        print("Component Timing Summary")
        print("=" * 80)
        print(f"{'Category':<25} {'Time (ms)':<12} {'Percentage':<12} {'Calls':<8}")
        print("-" * 80)

        sorted_cats = sorted(summary.items(), key=lambda x: x[1]['total_ms'], reverse=True)
        for category, stats in sorted_cats:
            print(f"{category:<25} {stats['total_ms']:<12.2f} {stats['percentage']:<12.1f}% {stats['count']:<8}")
        print("=" * 80)

    def reset(self):
        self.tracker.clear()
        self.current_token_idx = None

    def get_module_count(self) -> int:
        return len(self.module_categories)

    def get_categories(self) -> List[str]:
        return list(set(self.module_categories.values()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        return False

    def __repr__(self) -> str:
        return (f"HookManager(hooks={len(self.hooks)}, "
                f"modules={len(self.module_categories)}, enabled={self._enabled})")
