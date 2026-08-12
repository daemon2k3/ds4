#!/usr/bin/env python3
"""ds4 HTTP bench probes: decode rate, cold/warm prefill, disk-KV cache behavior.

Client-side measurements over the ds4-server OpenAI-compatible SSE API.
Usage: edit BASE/PROMPT knobs below, then run. Prints TTFT, decode t/s,
prompt-token accounting (cached vs cache_write tokens), warm/cold ratio.
"""
import json, time, urllib.request, random

BASE = "http://127.0.0.1:12434"
MODEL = "deepseek-v4-flash"

def stream_chat(content, max_tokens, stream_usage=True):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            d = line[6:].strip()
            if d == b"[DONE]":
                break
            c = json.loads(d)
            if c.get("usage"):
                usage = c["usage"]; continue
            ch = c.get("choices") or [{}]
            delta = ch[0].get("delta") or {}
            piece = delta.get("content") or delta.get("reasoning_content")
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return ttft, total, usage

def fire(tag, content, max_tokens):
    ttft, tot, u = stream_chat(content, max_tokens)
    pt, ct = u["prompt_tokens"], u["completion_tokens"]
    det = u.get("prompt_tokens_details", {})
    dec = (ct - 1) / (tot - ttft) if ct > 1 else float("nan")
    print(f"{tag:26s} ptok={pt:7d} ttft={ttft:8.2f}s ({pt/ttft:7.0f} t/s) gen={ct:4d} decode={dec:6.1f} t/s {det}", flush=True)

if __name__ == "__main__":
    rnd = random.Random(0xAB)
    nonce = "".join(rnd.choice("abcdefghijklmnop") for _ in range(8))
    words = "amber basil cinder dune ember fjord garnet harbor ivory jasper krill lagoon mesa nimbus onyx prairie quartz ravine sable tundra umber vortex willow xenon yarrow zephyr".split()
    prompt10k = " ".join(f"{nonce}-{i:05d}-{rnd.choice(words)}" for i in range(1000)) + "\nCount from 1 to 120, one number per line."
    fire("short decode", "Count from 1 to 200, one number per line.", 256)
    fire("cold  ~10k", prompt10k, 16)
    fire("warm  ~10k + gen", prompt10k, 128)
    fire("warm2 ~10k + gen", prompt10k, 128)
