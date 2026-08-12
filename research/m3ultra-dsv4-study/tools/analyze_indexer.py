#!/usr/bin/env python3
"""Approximate indexer-scoring feasibility analysis from ds4 dump files.

Doors evaluated per (layer, token) sample and aggregated by (layer, depth):
  Door1: exact Cauchy-Schwarz early-out prune rate at the true top-512 threshold
         (bound l1 = ||k|| * sum(w||q||); bound l2 = ||k|| * sqrt(sum((w||q||)^2)))
  Door2: 64-row group-pooled pilot -> expand top-M groups -> top-512 set agreement
  +bonus: fp8-quantized-score selection agreement
"""
import glob, os, struct, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "/tmp/indexer-dump"

def read_hdr(fp):
    magic, ver, layer, token, pos0, n_comp, ratio, n_head, head_dim = struct.unpack("<9I", fp.read(36))
    assert magic == 0x49535144, hex(magic)
    (scale,) = struct.unpack("<f", fp.read(4))
    return dict(layer=layer, token=token, pos0=pos0, n_comp=n_comp, ratio=ratio,
                n_head=n_head, head_dim=head_dim, scale=scale)

samples = {}
for sp in sorted(glob.glob(os.path.join(D, "s_l*_t*.bin"))):
    with open(sp, "rb") as fp:
        h = read_hdr(fp)
        scores = np.frombuffer(fp.read(h["n_comp"]*4), dtype="<f4")
        q = np.frombuffer(fp.read(h["n_head"]*h["head_dim"]*4), dtype="<f4").reshape(h["n_head"], h["head_dim"])
        w = np.frombuffer(fp.read(h["n_head"]*4), dtype="<f4")
    kp = os.path.join(D, f"k_l{h['layer']}_p{h['pos0']}.bin")
    if not os.path.exists(kp) or h["n_comp"] == 0:
        continue
    with open(kp, "rb") as fp:
        hk = read_hdr(fp)
        k = np.frombuffer(fp.read(hk["n_comp"]*128*4), dtype="<f4").reshape(hk["n_comp"], 128)
    samples[(h["layer"], h["token"], h["pos0"])] = (h, scores, q, w, k)
print(f"{len(samples)} samples with k")

G = 64
rows = []
for (layer, token, pos0), (h, scores, q, w, k) in sorted(samples.items()):
    visible = min((pos0 + token + 1)//h["ratio"], h["n_comp"])
    if visible < 516: continue
    s = scores[:visible]
    kv = k[:visible]

    tau = np.partition(s, -512)[-512]                     # 512th-threshold
    top_full = set(np.argpartition(s, -512)[-512:].tolist())

    # ---- Door 1: exact pruning bounds
    kn = np.linalg.norm(kv, axis=1)
    wq_norm = w * np.linalg.norm(q, axis=1)
    N1 = wq_norm.sum()
    N2 = np.sqrt((wq_norm**2).sum())
    pr1 = float((kn*N1 < tau).mean())
    pr2 = float((kn*N2 < tau).mean())

    # ---- Door 2: pooled pilot (relu*pilot weights applied identically)
    nfull = (visible // G) * G
    kg = kv[:nfull].reshape(-1, G, 128).mean(axis=1)
    pilot = (np.maximum(kg @ q.T, 0.0) * w * h["scale"]).sum(axis=1)
    agree = {}
    expanded_total = {}
    for M in (8, 16, 32, 64, 128, 256):
        if M > len(pilot): break
        topg = np.argpartition(pilot, -M)[-M:]
        cand = set()
        for g in topg:
            cand.update(range(g*G, min((g+1)*G, visible)))
        agree[M] = len(top_full & cand) / 512.0
        expanded_total[M] = len(cand)/visible

    # ---- fp8 selection agreement (e4m3-ish: quantize to 4-bit exponent/3-bit mantissa of MAGNITUDE-ranked scores)
    sf = (s / np.abs(s).max()).astype(np.float32)
    q8 = np.round(sf * 448) / 448                       # 448 = 7*64 levels
    top_fp8 = set(np.argpartition(q8, -512)[-512:].tolist())
    agr_fp8 = len(top_full & top_fp8)/512.0

    rows.append((layer, token, visible, tau, pr1, pr2, agree, expanded_total, agr_fp8))

print(f"\n{'layer':>5} {'token':>7} {'visible':>8} | {'pruneL1':>7} {'pruneL2':>7} | pilot top-512 agreement @M groups (rows covered) | fp8 agr")
agg1, agg2 = {}, {}
for layer, token, visible, tau, pr1, pr2, agree, cov, agr in rows:
    agg1.setdefault(layer, []).append(pr1); agg2.setdefault(layer, []).append(pr2)
    astr = " ".join(f"{m}:{agree[m]*100:5.1f}%({cov[m]*100:4.1f}cov)" for m in sorted(agree))
    print(f"{layer:>5} {token:>7} {visible:>8} | {pr1*100:6.1f}% {pr2*100:6.1f}% | {astr} | {agr*100:5.1f}%")
print("\n== prune-rate means by layer (Door1) ==")
for layer in sorted(agg1):
    print(f"  layer {layer:2d}: L1 {np.mean(agg1[layer])*100:5.1f}%  L2 {np.mean(agg2[layer])*100:5.1f}%")
