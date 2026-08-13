# Measuring a 284B MoE on a Mac: a performance anatomy of DeepSeek V4 Flash on Apple Silicon, with a kernel fix and an algorithmic path for deep context

**Working paper, August 2026. Code + tools in this folder tree; engine: [antirez/ds4](https://github.com/antirez/ds4).**

> **Author & AI disclaimer.** The heavy lifting in this study — session orchestration, the measurement loop, all instrumentation and code patches, analysis, and this write-up — was done by an AI agent (**Kimi K3**), running in a pi harness with SSH access to the test machine. The human operator (daemon2k3 / Gabriel Mateiciuc) steered the investigation, supplied hardware and questions, and validated results. We state the AI's role up front rather than laundering it, because it was substantial.

## Abstract

We instrumented and measured the DeepSeek V4 Flash 284B-parameter MoE model end-to-end on a single Apple M3 Ultra with 512 GB of unified memory, running the DS4 inference engine over its OpenAI-compatible server. We establish verified baselines (prefill ~600 t/s short context decaying to ~390 t/s at 64k and ~142 t/s near the 1M context cap; decode ~48 t/s short context decaying to ~34 t/s at 64k and ~16.4 t/s at ~1M), decompose prefill and decode into per-stage costs, quantize the quant-format tradeoffs (hybrid 2-bit vs 4-bit routed experts), catalog a set of falsified micro-optimizations (tile geometries, chunk sizes, batch multipliers, occupancy re-balancing), land one real kernel improvement (a llama.cpp-ported indexer scorer, +6–7% decode at 117k context, bit-identical), validate the disk KV cache (2–5× TTFT on real repeats), and close with a measured algorithmic feasibility study for deep-context scoring: a pilot-then-rescore indexer candidate pass costing ~16% of the score stage while retaining ~95–99.5% per-token selection fidelity at 117k–496k context — projecting prefill ~3× and decode ~2× near the 1M cap.

## 1. Setup and disclosure

Hardware: Apple M3 Ultra, 512 GB unified RAM (~800 GB/s), macOS 26 (Darwin 25.6). Engine: DS4 (DwarfStar), DeepSeek V4 Flash GGUF (284.33B params, 90.88 GiB resident, hybrid-2bit experts), server flags `--ctx 1048576 --kv-disk space 65536 MB --dspark (MTP speculative decode with the support GGUF) --kv cache align 4096` on port 12434. The model is fully resident; no SSD streaming anywhere in this study.

This work was performed with heavy AI assistance (a pi-coding-agent session drove measurement, analysis, kernel edits, and this write-up), with the human operator steering and validating. We say this openly because it shaped the workflow; all claims rest on direct measurement.

## 2. Methodology: the measurement loop

Four instruments, cross-checked against each other:

1. **HTTP SSE probes** (tools/ds4_probe.py): client-side TTFT, decode rate, and server-reported prompt accounting (cached/cache_write tokens), for cold vs warm repeats at 1.5k–944k token prompts.
2. **In-tree `ds4-bench` + variant harnesses**: canonical frontier CSVs (ctx 2k–64k, 2048-token frontier prefill + 128-token gen), ABBA/BAAB bit-identical logits comparisons for kernel variants (abort unless every full-vocabulary logit row matches).
3. **Env-gated stage profilers inside ds4** (tools/README.md glossary): per-layer/chunk encode+execute times, indexer score/topk/attention per stage (prefill and decode), MoE sub-stages.
4. **File/format probes**: GGUF census (tools/gguf_analyze.py, handles DS4's u64-dims variant; sizes from offset deltas), score dump instrumentation for the approximate-scoring study (section 7).

MPS GEMM microbenchmark (tools/mpsbench.mm) supplied the practical dense-GEMM envelope (~20–23 TFLOP/s-half on this device) as an external reference.

Repeated traps discovered and documented, then measured around: profiler sync tax (~1.4× decode), a deterministic run-order bias (~6e-4 per slot) in the variant harnesses on this config, cross-run profiler absolute-value drift (within-run ratios only), server-vs-bench lock-file policy, and the KV disk cache creating fake speedups for nested-prefix benchmark prompts.

## 3. Verified baselines

**ds4-bench frontier (combined q2-hybrid / q4, 2048-token frontier prefill + steady 128-token gen):**

| ctx | prefill t/s | decode steady t/s (q2-hybrid) | decode (q4) |
|---:|---:|---:|---:|
| 2k | 596 | 44.2 | 40.6 |
| 16k | 508 | 38.6 | 35.9 |
| 32k | 462 | 36.9 | 34.2 |
| 65k | 391 | 34.5 | 31.9 |

Long-context frontier (HTTP probes, measured, single runs): prefill 402 t/s @117.6k cold; 218 t/s avg across a 525k-token cold prefill; 141.5 t/s avg across a 944.8k-token cold prefill. Decode: 47–48 t/s short; 38.5–39.3 t/s @117k; **16.4 t/s @944.8k** (near 1M cap). Disk KV cache on real repeats: 12.7s → 5.8s TTFT for a repeated 6.2k prompt (4096-token aligned checkpoint + suffix), restored checkpoints cost ~0.4–1s at 10k-token scale; DSpark speculative decode +7.5–11% decode at short-to-mid context (measured against the raw-engine curve in the same server).

Comparative reference (in-tree speed-bench CSVs): prefill m2_ultra ~410, m4_max ~344, m5_max ~790, GB10 ~825–900; decode tops this table at 44 t/s (M3 Ultra memory bandwidth wins decode; tensor units win prefill elsewhere).

## 4. Where prefill time goes (measured decomposition)

Per-layer sub-stage maps (43 layers, both quants identical within ±1–5%, profiler-ratios):

| part.stage | per-token cost | share | character |
|---|---:|---:|---|
| ffn routed MoE GEMM + router | 15.6 µs | 38% ~19 TF/s | flat vs ctx; at practical envelope |
| attn attention (SWA + CSA selected rows) | 9.6 µs | 23% | flat by design (top-512 cap + 128 raw window) |
| dense MLA projection chain (q_a/q_b + low-rank out) | 9.9 µs | 24% ~22 TF/s | dense Q8_0 GEMM; at envelope |
| indexer score stage (even layers) | grows linearly | 7% @4k → ~80% @1M | **the only unbounded term** |
| compressor, shared FFN, hc, router, norms | ~4 µs | ~15% | small |

The deep-context wall is exactly and only the indexer score stage: per even layer it costs ~(base + slope × compressed-rows), fitted 122.5 + 8.6 ms per 4096-token chunk per 4096-position step (~1.8 ns/row-token; ~9–11 TFLOP/s effective) — ~205 ms/layer/chunk at 29k rows → projected ~1.9 s/layer at 262k rows (1M ctx). Everything else is capped or context-independent. Global prefill MFU vs the practical dense-GEMM envelope: ~50–55%.

## 5. The quant question

Censused both candidates against the measured numbers (tools/gguf_analyze.py):

- **q2-hybrid (running)**: 37 layers experts IQ2_XXS(gate/up)+Q2_K(down), last 6 layers Q4_K experts; 90.88 GiB; aux classes F16/Q8_0; hash-routed first 3 layers (static tid2eid tables) and hyper-connection(4) blocks discovered in metadata.
- **q4**: all-experts Q4_K, 153.32 GiB, dense/aux byte-identical otherwise.

**Measured outcomes**: prefill parity (±1% everywhere, including a cold 525k token run); decode −7.5…−8.7% (40.6 vs 44.2 @2k; 31.9 vs 34.5 @65k — far milder than the ~31% bytes-ratio extrapolation); DSpark sidecar accepts both (`missing=0 invalid=0`); greedy answers identical. Conclusion: q4 is a *quality* move at ~8% decode cost and zero prefill cost; MXFP4 is strictly dominated on this hardware (bytes +68%, no prefill relief — the campaign's own ~640 t/s @2k reflects a different, unmeasured-here configuration); IQ2-family decode bandwidth accounting closes to within 4% (7.35 GB/token predicted vs 47.4–48.4 t/s measured).

## 6. The kernel campaign: what doesn't work, and the one fix that does

Falsified by direct A/B under the methodology above (each documented with before/after numbers in the commit history of branch `metal-indexer-score-speedups`):

- prefill chunk size (2048 vs 4096+KV-align): noise (bandwidth math on 512GB RAM predicted this correctly).
- wide tiles for the score kernel (TN 32→64, TM 8→16): parity-to-slightly-worse (the score stage is MMA-latency-bound, not q/K-traffic-bound as modeled; two of us measured this).
- decode 4-rows-per-threadgroup with ping-pong Q staging (bit-exact but ~17% *slower* score at depth; old direct kernel becomes *more* efficient at scale).
- token-batching per threadgroup (NBPTG 8/16/32): flat everywhere.
- 4-simdgroup / 2-threadgroups-per-core occupancy variant (NSG4): prefill score −19%, decode score −8%, e2e parity → net negative.
- dense-chain MPS/simdgroup improvements: unnecessary — chain already at practical envelope once FLOPs were correctly accounted (512-wide MLA heads).

**The fix that survived everything**: port llama.cpp's Metal `kernel_lightning_indexer` organization into `kernel_dsv4_indexer_scores_llt` — K rows staged once per threadgroup and transposed into per-simdgroup **register-resident** matrices reused across the whole 64-head loop; heads processed in 8-head tiles with pre-scaled weights (two barriers per head-tile vs three per head); prefill iterates tokens inside the kernel against resident K. Result: **bit-identical decode** (68,389,120 exact floats, 528/529 selected ids), **decode +5.6–7.4% @117.6k ctx (42.2/41.5 vs 39.3 t/s, replicated)**, short-ctx parity, prefill parity. PR branch: `metal-llt-scorer`. Three failures of the same insertion point documented (the `#ifdef DS4_METAL_HAS_TENSOR` guard silently compiles out kernels placed before the NAX kernel; a template-instantiation-order trap; guard-env gotchas).

Even-LLT prefill score remains ~205 ms/layer/chunk at 29k rows — the score stage is bounded by per-row-token work, not organization (four independent organizations now bracket it within ±10%).

## 7. The algorithmic answer for deep context: pilot-then-rescore

The score stage's cost = per-token scoring of all compressed rows. We measured three algorithmic escape routes (env-gated dumps of real score/q/k/w tensors; tools/analyze_indexer.py):

- **Exact pruning via Cauchy–Schwarz norms**: **dead** — prune rate 0.0% in every sampled layer/token (bound ≫ real top-512 threshold on this distribution).
- **Pooled/group pilots**: weak (noisy per-token track; ~55% coverage needed for ~99% agreement).
- **Per-row single-direction pilot (`q̄ = Σ_h w_h·q_h`, 64× cheaper than full) + full rescore of top-N candidates**: the winner.

Fidelity (top-N candidate agreement with exact top-512; 228 aligned samples, layers 2/22/42):

| candidates | 32k ctx | 117k ctx | 496k ctx |
|---:|---:|---:|---:|
| N=2048 | 55.3% | 96.7% | 87.8% |
| N=4096 | 76.1% | **99.5%** | 95.1% |
| N=8192 | — | 100.0% | 98.5% |

Per-layer at 496k, N=4096: layer 2: 89.9% · layer 22: 99.7% · layer 42: 95.7% — i.e., **layer-dependent and depth-dependent fidelity**; candidates for layer-adaptive budgets are included in the proposed implementation (§ 9). Cost model at N≈4096 + pilot: ~16% of the score stage → projected prefill ~250–340 t/s and decode ~18–20 t/s near 1M ctx (from ~94 and ~16.4 respectively), gated at ≥64k ctx where pilots are trustworthy. Semantic license note: DeepSeek's own indexer runs at low precision (DeepGEMM fp8 indexer logits) — the indexer was designed for low-precision ranking; our post-hoc fidelity measurements supply the model-specific evidence.

## 8. Artifacts and reproduction

`data/` — the frontier CSVs and plots; `tools/` — every measurement script with usage (README.md). Kernel work: branches `metal-indexer-score-speedups` (campaign history incl. reverted experiments and the env-gated dump instrumentation) and `metal-llt-scorer` (curated LLT port, PR-ready). Benchmark CSV PR on `bench-m3-ultra-flash`.

Core reproduction on a Mac with the model file: build at `84cc882+`, run `ds4-bench` per tools/README.md, then the HTTP probes; stage maps via the profiler envs on a second server instance; indexer dumps per tools/README.md; analysis via `analyze_indexer.py` (numpy).

## 9. Limitations and open work

- Profiler sync tax makes all `*_STAGE_PROFILE` absolutes inflated; we use within-run ratios and clean e2e A/B everywhere it mattered. The ~6e-4 deterministic run-order bias in variant harnesses (old-vs-old "mismatches", identical bits per slot) is reported upstream-worthy: the bit-identical contract needs slot counterbalancing on this configuration.
- Near-1M-ctx figures are fewer samples each (prefill runs take ~40 min–2 h); decode at 1M ctx from two samples.
- Selection-fidelity statistics are per-token per-layer ranking overlaps; end-to-end **quality** of pilot-selected attention is unevaluated until the algorithm is implemented with its in-engine fidelity meter + greedy continuity checks + long-context QA. The algorithmic numbers in §7 are projections from stage-fit constants, not measured endpoints.
- DSpark's verify economics at deep context (acceptance vs verify-pass cost) observed but not decomposed.
- No Mac-side GPU-duty telemetry (powermetrics needs root); occupancy statements are wave/threadgroup-memory arithmetic, not counters.

## 10. Proposed next work (ranked)

1. Implement pilot-then-rescore with **layer-adaptive candidate budgets** (from §7 stats) + **in-engine fidelity meter** + greedy continuity and long-context QA gates.
2. Suffix-prefill and restore-cost curve at bigger checkpoints (warm-path economics at 100k+ ctx).
3. DSpark at depth: verify-cost/acceptance decomposition, `--dspark-confidence` sweeps.
4. Upstream report: variant-bench run-order bias; `#ifdef DS4_METAL_HAS_TENSOR` compile-out hazard for future kernel contributors.

## References

- antirez/ds4 (engine + speed-bench/* + gguf recipes); llama.cpp (ggml Metal backend heritage incl. `kernel_lightning_indexer`); DeepSeek V4 Flash model card (deepseek-ai/HF); DeepGEMM (fp8 indexer logits); antirez/deepseek-v4-gguf (quant recipes; ~966k downloads).

## Appendix A — second-phase campaign (code-layer wins, August 12–13 update)

A second campaign of code-level (non-algorithmic) optimizations delivered on the same codebase, verified with the same session-standard A/B method:

| item | mechanism | measured |
|---|---|---|
| rb16 attention dispatch prefill | replace per-row staged `heads8` (2 threadgroup barriers/row) with `heads8_rb16` (16 staged rows per stage) — identical math/order | part of the +5.2% cold prefill @117k ctx |
| LLT K/Q staging vectorization | float4→half4 staging mirroring its NAX sibling | part of decode/prefill gains |
| F3 double-buffered Q | stage tile t+1 while MMAing tile t (device-latency hidden) | decode +3.9% @117k; TTFT −8%; prefill −5.2% |
| simd_max reducers | replace 7-barrier tree max-reduce at 5 finalize/quantize/verify sites; IEEE max is exact ⇒ bit-identical | decode −1–2% |
| host getenv cache | process-lifetime env snapshot (~200–400 linear scans/token) | class (a) |
| session verify scratch | session-owned 2×vocab verify-logit buffers (was a 1.3–2.6 MB xmalloc/free per verify cycle) | class (a) |
| tokenizer literal lens | precomputed sizeof-literal vs strlen-per-byte at ingest | prompt-ingest ms-scale |
| KV-store refresh gating | default OFF after discovering the st_mtime second-resolution lineage trap | (negative-fix) |
| zero-copy payload spans | fwrite/fread straight into unified-memory Metal tensor backing stores, fenced | warm restore 11.4 s vs 12.7–19.2 s @30k ctx |
| C1 vvexpf sampling path | Accelerate vForce vvexpf = 0.35 ns/op vs scalar expf = 1.29 ns/op — opt-in only (`DS4_SAMPLE_ACCELERATE_EXP` — changes the sampling stream by construction) | micro 3.7× |

Negative results from the same campaign (also documented): toggle-region runtime-arg kernels (Metal-run broken states/all-NaN prefill at second-chunk — archived at tag `toggle-attempts-ref`), F2 f16 indexer compressed-K cache (nondeterministic at depth — parked on branch `indexer-comp-f16`), and the bench-ABBA slot-order bias (bit-exact A/B impossible on this config — harness artifact, reprod path documented).

The engineering lesson set also includes: symlink-aware rm (159 GiB model file deleted via rm-through-symlink, restored & sha256-verified), APFS local-snapshot pinning, disk-budget sentinel (`/tmp/DSK_SATURATED` + experimenter launch gate — guard at 90% used), and the dump-cap discipline (DS4_METAL_GRAPH_DUMP_POS + NAME gating near 1M-token contexts).
