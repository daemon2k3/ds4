# Measurement tooling for the M3 Ultra study

Each tool below was used in this study; the companion paper explains the results.

## ds4-bench + variant benches (in-tree)

Canonical context-frontier curves (prefill + decode at every 2k ctx), ABBA/BAAB
bit-identical variant A/B harnesses:

```sh
./ds4-bench -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128 | tee m3_ultra.csv
python3 speed-bench/plot_speed.py m3_ultra.csv --title "M3 Ultra t/s"

./speed-bench/metal_prefill_variant_bench  -m ds4flash.gguf --candidate-env DS4_METAL_DISABLE_INDEXER_LLT
./speed-bench/metal_decode_schedule_bench  -m ds4flash.gguf --candidate-env DS4_METAL_DISABLE_INDEXER_LLT --include-selection
```

Note: `metal_getenv` lock file — bench tools and a running server refuse to
share a process lock; set `DS4_LOCK_FILE=/tmp/dev.lock` to coexist
(concurrent GPU work contaminates timings, use for correctness runs only).

## ds4_probe.py — HTTP SSE probes (this folder)

Client-side TTFT / decode-t/s / prompt-cache accounting
(`cached_tokens`, `cache_write_tokens`) for cold vs warm repeats at arbitrary
prompt sizes. Edit `BASE`, run. Used for the disk-KV restore-vs-recompute
measurements (e.g. 12.7s -> 5.8s TTFT on repeated 6.2k prompts) and for the
long-context cold/warm decode probes at 117.6k ctx.

## Stage profilers (built into ds4, env-gated)

| env | prints |
|---|---|
| `DS4_METAL_GRAPH_PREFILL_SPLIT_PROFILE=1` | per-layer attn/ffn encode+execute ms per prefill chunk |
| `DS4_METAL_GRAPH_PREFILL_PROFILE=1` | per-chunk totals |
| `DS4_METAL_INDEXER_STAGE_PROFILE=1` | indexer score/topk/attention (prefill + decode) |
| `DS4_METAL_LAYER_STAGE_PROFILE=1` (+ `_LAYER=N`) | attn part & ffn sub-stages (hc, router, shared, ...) |

Absolute values carry profiler synchronization overhead (~1.4x on decode);
ratios are the information. Profiler-tainted runs must never be compared to
clean runs.

## gguf_analyze.py — GGUF census (this folder)

`python3 gguf_analyze.py file.gguf` — metadata (arch, indexer dims, top_k,
XV kv/runtime formats), per-class tensor census with params/GiB/bits-per-param.
Handles the DS4 custom GGUF variant (u64 dims) and sizes tensors from offset
deltas. Used to characterize the active quant (hybrid q2-hybrid vs q4 file).

## mpsbench.mm — MPS GEMM envelope (this folder)

`c++ -framework Foundation -framework Metal -framework MetalPerformanceShaders -o /tmp/mpsbench mpsbench.mm && /tmp/mpsbench`
Measures MPS half-precision GEMM TFLOP/s on the study's exact dense chain
shapes (the practical prefill GEMM ceiling of M3 Ultra).

## Indexer score dump for the approximate-scoring study

Env-gated instrumentation (branch `metal-indexer-score-speedups`):

```
DS4_INDEXER_DUMP_DIR=/path DS4_INDEXER_DUMP_LAYERS=2,22,42 \
DS4_INDEXER_DUMP_TOKENS=32000:32037,117000:117037,496000:496037 ./ds4-server ...
```

Writes per sampled token: s_l<layer>_t<token>.bin (header + scores + q + w)
and per (layer,chunk): k_l<layer>_p<pos0>.bin (header + K rows).
Binarу format: 9x u32 header (magic 'DSQI', ver, layer, token, pos0, n_comp,
ratio, n_head, head_dim) + f32 scale + f32 payload.

## analyze_indexer.py — feasibility matrix (this folder)

`python3 analyze_indexer.py <dumpdir>` — per token/layer: exact-prune rate
(norm-bound, L1/L2), pooled-pilot selection agreement, fp8 agreement;
aggregates by layer/depth. (numpy required.)
