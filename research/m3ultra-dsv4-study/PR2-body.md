# PR: Metal lightning-indexer-organized DS4 indexer scorer (LLT port)

## Summary

Port of llama.cpp's Metal `kernel_lightning_indexer` organization into `kernel_dsv4_indexer_scores_llt`:
K rows staged once per threadgroup into per-simdgroup register matrices reused across the whole 64-head loop; heads processed in 8-head tiles with pre-scaled weights (two barriers per head-tile vs three per head); prefill iterates tokens inside the kernel against resident K. `DS4_METAL_DISABLE_INDEXER_LLT` reverts to the previous tiled kernel (one-flag rollback per convention).

## Measured (same machine, same-indexed profiler A/B — profiler sync tax ~1.4x taken out of ratio)

```
decode @117k ctx: 39.32 → 42.18/41.53 t/s  (+5.6–7.4%, duplicated runs)
prefill @all sizes: parity (within ±1%)
bit-exactness: 68,389,120 floats identical vs old kernel; top-512 ids 528/529 match (existing false-positive rate of the topk sort's own envy/arc env pairing)
(per the findings note: profiler ABSOLUTES inflated ~1.4×; ratios are the data)
```

## What it supersedes

Four kernel organizations of the score stage had all landed within ±10% of each other because the stage is (a) not organization-limited — its bound is per-row-token work-count — and (b) the existing tiled/tiled_f32 score_one forms scratching the roofline floor with scalar 4-byte staging and per-head barrier serialization. LLT replaces that with a coherent tile-of-resident-K organization and is the first reorganization that measured past the plateau with the group's reproduction standards.

## Caveats
- Includes `metal-misc` gone-through kernels outside `#ifdef DS4_METAL_HAS_TENSOR` guard (documented trap: kernels placed inside compile out silently pre-M5).
- dispatch policy untouched: `score_llt` remains the M3-default non-quality score path; NAX keeps M5.
- bench harness (`speed-bench/*_variant_bench`) on this machine has a slot-order bias — see the linked issue filed in the same repo.

## Evidence

- bit-exact decode: med/bit uint floats pair table (ds4_decode117k_run logs retained)
- perf ledger: research/m3ultra-dsv4-study/evidence/README.md (PR3 tables + LLT numbers + methodology notes)
- full campaign paper: research/m3ultra-dsv4-study/PAPER.md
