# VLM Benchmark Results

Cross-device profiling results for Vision-Language Model compression benchmarking.

## Directory Structure

```
├── jetson/profiling/       # NVIDIA Jetson Orin Nano 8GB
├── rpi/profiling/          # Raspberry Pi (to be added)
└── ...                     # Other devices
```

## Profiling Attributes (per JSON)

- **Timing**: prefill_ms, decode_ms, per_token_ms[], preprocessing_ms
- **Components** (11 categories): vision_embeddings, vision_attention, vision_mlp, vision_normalization, vision_encoder, projection, text_embeddings, attention, feedforward, normalization, output
- **Memory**: gpu_peak_mb, gpu_mem_load_mb, num_params_M
- **Power** (device-specific): avg_power_w, peak_power_w, avg_gpu_temp_c
- **Accuracy**: exact_match, contains, token_f1, bleu, rouge_l
- **Throughput**: avg_tok_s, avg_decode_tok_s, samples_per_s
- **Metadata**: model_id, method, family, device, load_time_s

## Models Profiled (Jetson Orin Nano 8GB)

| Model | Method | Prefill | Decode | Total | Decode t/s | Memory | Power |
|-------|--------|---------|--------|-------|------------|--------|-------|
| LFM2-VL-450M | fp16 | 149ms | 102ms | 268ms | 14.1 | 902MB | 11.8W |
| SmolVLM-256M | fp16 | 1555ms | 357ms | 1624ms | 7.5 | 887MB | 12.5W |
| SmolVLM-500M | fp16 | 870ms | 367ms | 1630ms | 7.0 | 1473MB | 13.7W |
| FastVLM-0.5B | fp16 | 509ms | 3414ms | 3987ms | 8.5 | 1346MB | 8.6W |
| InternVL2.5-1B | fp16 | 371ms | 222ms | 593ms | 7.7 | 1886MB | 13.7W |
| InternVL2.5-2B | fp16 | 461ms | 200ms | 661ms | 7.8 | 4798MB | 18.1W |
| Ovis2-1B | fp16 | 1421ms | 199ms | 1620ms | 6.7 | 2674MB | 17.9W |
| LFM2-VL-1.6B | hqq_int4 | 775ms | 676ms | 1476ms | 1.9 | 2600MB | 16.5W |
| LFM2-VL-3B | hqq_int4 | 1524ms | 1586ms | 3136ms | 0.9 | 3381MB | 16.5W |
| SmolVLM-2.2B | hqq_int4 | 3133ms | 1819ms | 5151ms | 1.4 | 2957MB | 18.7W |
| Qwen2.5-VL-3B | pt_int4 | 2282ms | 1645ms | 3960ms | 0.8 | 4969MB | 17.2W |
| FastVLM-1.5B | pt_int4 | 1068ms | — | 19546ms | — | — | — |
| InternVL2.5-4B | pt_int8 | 1186ms | 915ms | 2100ms | 1.5 | 5408MB | 18.1W |
| Ovis2-2B | pt_int4 | 2585ms | 883ms | 3469ms | 1.5 | 3741MB | 18.3W |
