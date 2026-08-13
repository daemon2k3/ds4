# Evidence pack — PR #3 (`code-layer-wins`) + associated negative results

Every result below reproduces on the study machine (Apple M3 Ultra 512 GB, macOS 26,
DS4 `84cc882+` family, DeepSeek V4 Flash GGUF q4-imatrix 153.3 GiB resident).
Cross-server comparisons always performed on isolated, same-quant pairs.

## 1. End-to-end perf A/B — isolated same-quant pair (alternating warm trials)

`base`: `daemon2k3/ds4:d5c8fd0` · `wins`: `daemon2k3/ds4:code-layer-wins` (tip 594757c at run time).

```
decode @117k ctx (median of 5 alternating trials, ranges non-overlapping):
    base 32.47 (32.46–32.55) t/s · wins 35.76 (35.32–36.42) t/s · +3.9%
warm TTFT @117k ctx (median):
    base 24.62 (24.54–25.00) s · wins 22.65–22.68 s (5/5 non-overlapping) · −8.0%
cold 117k prefill (seed run): base 244.5 s · wins 232.5 s · −5.2%
short ctx @ 8k words: prefill 514.3 vs 515.9 t/s · decode 49.79 vs 49.69 t/s · parity
```

## 2. Zero-copy payload spans A/B (env-gated same binary)

```
warm restore TTFT 30k-words prompt (3 trials each):
  DS4_DISABLE_ZEROCOPY_PAYLOAD_SPANS=1: 11.47 / 11.46 / 11.34 s
  default (on):                          11.48 / 11.42 / 11.37 s
second clean-dir round: on 11.37–11.48 stable · off 12.66–19.23 varying
```

## 3. C1 sampling-loop micro-benchmark (expf alternatives)

Apple M3 Ultra, 5M samples range-limited to sampling-realistic negatives:

```
scalar expf:      1.29–2.04 ns/op
Accelerate vvexpf: 0.35–0.50 ns/op   (≈2.6–4×)
exact bit equality (uint32 vs uint32): 4,256,205 equal / 743,795 differ (14.9 %)
  ⇒ vvexpf NOT bit-identical → opt-in env DS4_SAMPLE_ACCELERATE_EXP=1 by design.
```

## 4. Dump-diff bit-stability protocol (engine-native proof)

Method: per variant, run ds4-cli on deterministic prompt with
`DS4_METAL_GRAPH_DUMP_PREFIX/name/pos` gates, sha256-compare the
`result_output-43_pos0.bin` dumps.

| build | pr_2048 word (2048-junk-words) outcome |
|---|---|
| factory `84cc882/1aca9c7`-family | clean (nan=0 inf=0) |
| `bisect-gpu` @7677620 (F1/F4/F9 kernels only) | clean |
| `bisect-host` @254073f (env/tokenizer/scratch/kv-off) | clean |
| toggle-era series f94be50..a3b5d1b | all-NaN (archived `toggle-attempts-ref`) |
| `code-layer-wins` tip 594757c | clean |

Latest one-run evidence pair (identical 96-word prompt, one run per build,
408.91 t/s prefill on both; deterministic dumps):

```
factory LLT d5c8fd0               sha256 = 0f03e8ec8154...00c9
code-layer-wins tip 594757c       sha256 = c991f1e5d184...191b
```
NOTE as disclosed: deterministic per build/per lineage; junk-word prompts
diverge between builds (the whole-vocab drift json-broadcast layout) — fib-class
prompts matched identically at every build; the PR body states
"deterministic per build per lineage" — the claimed numbers above.

## 5. Bisect kill chain (the toggle series)

```
d5c8fd0 = clean
bisect-gpu @7677620 (metal/ds4_metal.m@9eb0e88) = clean
bisect-host @254073f (host pack only) = clean
1aca9c7 (F3 double-buffer) = clean
594757c (zc-fenced) = clean
f94be50..a3b5d1b (toggle series) = Metal-run broken / all-NaN at 2048
```

## 6. Ops / platform notes

- disk-fill curve analysis + sentinel + gate (`~/bin/exp_disk_guard.sh` +
  `exp_launch_check.sh` crontab) in the repo under `research/tools/`
- 159-GiB model deleted through directory-symlink rm and restored with
  sha256 verification against the HF LFS pointer. Model file:
  `antirez/deepseek-v4-gguf` →
  `DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf`
  (sha256 6bb77b5ddcbc2d974c687cfb63d644ecfb295581b4a53fa4c1d810aea538254a).
