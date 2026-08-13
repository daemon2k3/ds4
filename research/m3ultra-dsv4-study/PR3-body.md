# PR: Metal code-layer performance wins (rb16 prefill dispatch, vectorized K/Q staging in the LLT scorer, double-buffered head tiles, simd_max reducers, host fixes, zero-copy payload spans)

## Summary

A set of code-layer (non-algorithmic) optimizations over the DS4 Metal backend, measured end-to-end on an Apple **M3 Ultra 512GB + DeepSeek V4 Flash (q4, 153GiB, resident)** against the current `main` + LLT scorer. Every change preserves existing semantics; the two sampling-path items are strictly opt-in via env flags and change behavior only when explicitly requested. **Deterministic per build per lineage; the deep-context dump-diff verification protocol below is included.**

## What is in this PR

| area | change | effect (measured, same-quant isolated A/B pair, alternating trials) |
|---|---|---|
| prefill attention dispatch | dispatch `heads8_rb16` on prefill (was `heads8`) — 16 staged rows per stage kernel with identical per-row online-softmax sequence | part of **+5.2%** cold prefill @117k ctx; warm TTFT −8.0% |
| LLT indexer K/Q staging | float4→half4 vectorized staging (mirrors the NAX kernel) | part of decode +3.9% @117k |
| LLT head-loop double buffer | stage tile t+1 while `simdgroup_multiply_accumulate`-ing tile t | the deep-context decode/TTFT gains above |
| 5 × fp8/hadamard finalize kernels | replace 7-barrier tree max-reduce with `simd_max` + one cross-group exchange (`max` is IEEE-exact — bit-identical) | decode −1–2% |
| Metal decode dispatch env reads | process-lifetime env cache (was ~200–400 linear `environ` scans per token across 43 layers) | host-path hygiene |
| speculative verify scratch | session-owned 2×vocab `spec_row_logits` (was 1.3–2.6 MB `xmalloc`+free per 1–2 committed tokens with DSpark/MTP) | page-fault churn removal |
| tokenizer signature matching | precomputed literal lengths (`sizeof(lit)-1`) instead of `strlen` per scanned byte | 10–50 ms on 100k-token prompt ingest |
| KV disk cache store lookup | mtime gate **off by default** (second-resolution `st_mtime` can skip fresh entries in the same second — deterministic lineage integrity beats ms-scale refresh savings) | correctness-first default |
| checkpoint payload spans | `fwrite`/`fread` directly into shared (unified-memory) Metal tensor backing stores + GPU fence (fwd both ways), removing the 8MB staging loop | warm restore TTFT 30k ctx: 11.4s vs 12.7–19.2s |
| sampling top-p<1 expf loop | **opt-in only** `DS4_SAMPLE_ACCELERATE_EXP=1`: Accelerate `vvexpf` (measured 0.35 ns/op vs scalar `expf` 1.29 ns/op ≈ 2.6–4×) — **not bit-identical to scalar expf, changes the sampling stream by construction — off by default** | micro-benchmarked only |
| Makefile | Darwin `-framework Accelerate` link (for the vvexpf opt-in); `DS4_LTO=thin` opt-in (Apple Silicon release builds) | build hygiene |

## End-to-end measured outcomes

Isolated identical-instance pair on the same q4 model, `--dspark` + `--mtp` production-config, 5+ alternating trials each metric:

```
decode @117k ctx:   32.47 → 35.76 t/s       +3.9%   (medians; ranges non-overlapping)
warm TTFT @117k:    24.62 → 22.65 s         −8.0%   (medians)
cold 117k prefill:  244.5 → 232.5 s         −5.2%
short ctx (8k):     prefill 514.3 → 515.9 t/s, decode 49.79 → 49.69 t/s (parity)
```

## Verification protocol used for this PR's bit-stability claim

The in-tree `speed-bench/*_variant_bench` harnesses show a slot-order bias on this configuration (full-vocab mismatch on control-vs-control) — I filed a separate issue report for it, attached. So the bit-stability was verified via the engine's own debug dump-diff protocol instead:

```bash
# for each variant: dump the layer-43 result_output tensor at pos0, binary-compare
DS4_METAL_GRAPH_DUMP_PREFIX=/tmp/up_a DS4_METAL_GRAPH_DUMP_NAME=result_output \
DS4_METAL_GRAPH_DUMP_POS=0 ./ds4 -m ds4flash.gguf --prompt-file /tmp/prompt.txt
```

Compared across factory `main`, kernel-subset, host-subset, and full branch tips at multiple prompt sizes (96 / 512 / 2048 / 5000-word): `nan=0 inf=0` everywhere, deterministic per build, no all-scan anomalies. Several negative candidates were found on the way (not in this PR): a Metal runtime-arg toggle family that produced broken Metal shader states (archived at `toggle-attempts-ref` in `daemon2k3/ds4` for the record, plus bisect notes), and an f16 indexer-compressed-K cache (`indexer-comp-f16` — nondeterministic at depth, parked).

Honest gaps: precision interplay with the existing NAX/quality paths (M5-gated; untouched by this PR); stream-changing vvexpf is opt-in only; Apple M3 Ultra is the only hardware this was measured on (Kernel-level invariant claims rest on identical math/order, marked as such in each commit message).

## How to review

Branch: `daemon2k3/ds4:code-layer-wins` (tip `594757c`). Recommended reading order: the big host-side commit, then the kernel commit, then the zero-copy/C1 commits. Rollback flags follow the engine's one-env-per-change convention where applicable (`DS4_DISABLE_ZEROCOPY_PAYLOAD_SPANS`; `DS4_SAMPLE_ACCELERATE_EXP` is itself an opt-in).

Open upstream-prep question kept from the session: the slot-order bias in the bench ABBA harness (separate issue with exact repro lines included).

## Evidence

A/B ledger, dump-diff protocol table, C1 (vvexpf) micro-data, the toggle-series bisect kill-chain, and ops notes: `research/m3ultra-dsv4-study/evidence/README.md` (same fork, `research` branch).

