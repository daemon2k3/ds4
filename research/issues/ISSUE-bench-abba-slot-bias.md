# Issue: speed-bench variant harnesses show slot-order bias — control-vs-control logits mismatch (bit-exact A/B impossible on this config)

**Component**: `speed-bench/metal_prefill_variant_bench.c` (and `metal_decode_schedule_bench.c`)
**Hardware/config**: Apple M3 Ultra, 512 GB, macOS 26 (Darwin 25.6), DS4 @ 84cc882+, DeepSeek V4 Flash (284B, GGUF q4 imatrix 0731, `--ctx 1048576`, `--kv-cache-boundary-align-tokens 4096`, `--dspark` + MTP support sidecar)
**Reporter**: daemon2k3 (work done with heavy AI assistance, disclosed)

## Summary

On the above config, `metal_prefill_variant_bench` — with the documented ABBA machinery against deterministic run-order — reports *full-vocabulary raw logit mismatches* between counterbalanced slots for **control-vs-control reproductions**, i.e. even when the compared engine paths are identical binaries and identical env knobs. Bit-exact A/B verification of kernel variants is therefore impossible on this config; the only route to classifying numeric changes is the engine-debug dump-diff protocol (below). The observed pattern:

```
metal-prefill-variant-bench: model=ds4flash.gguf prompt=ds4.c prefix=8192
    warmup=32 ctx=8193 repeats=2 candidate_env=DS4_METAL_DISABLE_INDEXER_LLT
run=1 repeat=1 pattern=ABBA slot=1 variant=control tokens=8192
    seconds=13.926444 tokens_per_second=588.2334 exact=yes
metal-prefill-variant-bench: raw logit mismatch at run 2: differing=129280/129280
metal-prefill-variant-bench: first mismatch id=0
    reference=-0x1.54284p+4 (0xc1aa1420) observed=-0x1.58b41cp+4 (0xc1ac5a0e)
```

Separately (with `metal_decode_schedule_bench --include-selection`):

```
metal-decode-schedule-bench: model=ds4flash.gguf prompt=ds4.c prefix=2048
    ctx=4096 warmup=16 measured=512 control=2/32 candidate=2/32
    candidate_env=DS4_METAL_DISABLE_INDEXER_LLT_F3 include_selection=yes
metal-decode-schedule-bench: raw logit mismatch at frontier 0:
    differing=129280/129280 top0=24483 top1=24483
```

Note the decode case explicitly shows **argmax equality (top0 == top1) with every other vocabulary entry differing** — consistent with a slot-position-dependent global numeric offset, not with data-dependent changes.

## Repro

```
git checkout <ds4 @ 84cc882>    # the matched release baseline
make -j$(sysctl -n hw.ncpu)
./speed-bench/metal_prefill_variant_bench -m ds4flash.gguf \
    --candidate-env DS4_METAL_DISABLE_INDEXER_LLT
./speed-bench/metal_decode_schedule_bench -m ds4flash.gguf \
    --candidate-env DS4_METAL_DISABLE_INDEXER_LLT_F3 --include-selection
```

Run-once reproducible here with the release binary alone (no local source
changes needed). The failure signature is `run 2` 129280/129280 with the
`reference vs observed` delta at last-ULP-scale per element.

## Observations to help diagnosis

1. The mismatch is **always full vocab** (`differing=129280/129280`) appearing at `id=0` / `frontier 0` in every "candidate" run after a clean `run 1` (`exact=yes`).
2. The per-element deltas are last-ULP-scale (±1–4 ULPs; max ±2.4e-6 absolute on "second-slot" cases), never a random scatter; first-mismatch deltas show the same scale on every pair.
3. At `frontier 0` decode: **top0==top1**, i.e. argmax is *always* preserved.
4. Earlier campaign notes already flagged "deterministic run-order bias ~6e-4" on this machine — the larger-sample implementation of this issue simply exposes its size.
5. Bit-stability verified independently via the dump-diff protocol (suggested below for relief of similar work), where kernels/programs produce byte-identical or deterministic outputs per build.

## Suggested direction for a fix

Any of (a) counterbalance at slot positions with explicit per-slot session *and pointer-region* isolation instead of shared engine scratch state across session alternates; (b) a slot counter in the seed/position bookkeeping so logits cannot see slot-position-dependence; (c) abstract the engine's session-scratch cleanup between variants (warm-up state might leak); and (d) **dump-diff protocol** as the engine-grade solution for numeric regression verification (works): dump `result_output` per `(layer, pos)` at deterministic prompts and byte-compare dumps outside the engine loop.

## Workarounds used in my sessions (for reference in case your next change touches this area)

- `DS4_METAL_GRAPH_DUMP_PREFIX=... DS4_METAL_GRAPH_DUMP_NAME=result_output DS4_METAL_GRAPH_DUMP_POS=0` per ds4-cli `--prompt-file` run + byte-level dump compare.
- Treat the slot-position-run always the same across compared variants when used in CI-ish bench loops (`--candidate-env` with the same session parameter order).

## Disclosure

This report was drafted in a session with AI assistance (Kimi K3 / ds4-engineering flow). All claims rest on direct measurement reproduced twice on this machine; happy to produce any further traces you want.
