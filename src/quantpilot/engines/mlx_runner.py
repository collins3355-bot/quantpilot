"""Measure an MLX model: perplexity over 512-token chunks, plus speed.

This script runs inside the MLX virtualenv (see scripts/setup-mlx.sh) —
quantpilot invokes it as a subprocess so its own core stays dependency-free.
It prints a single JSON object as the last line of stdout.

Methodology matches llama-perplexity's scored-position set: the corpus is
tokenized, split into fixed 512-token chunks, and token positions
ctx/2+1 .. ctx-1 of each chunk are scored (llama.cpp scores logits offset by
`first`, predicting tokens[i+1]). One remaining difference: llama.cpp forces a
BOS token at each chunk start when the model requests one (add_bos); we do
not, so absolute cross-engine comparability holds exactly only for models
without forced BOS (e.g. Qwen). Across different models, compare relative
degradation vs. each engine's own baseline instead.
"""

from __future__ import annotations

import argparse
import json
import math
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--chunks", type=int, default=32)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument(
        "--kld-vs", default=None,
        help="baseline model dir; also measure KL divergence and top-1 agreement vs. it",
    )
    args = parser.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, stream_generate

    model, tokenizer = load(args.model)
    base_model = None
    if args.kld_vs:
        base_model, _ = load(args.kld_vs)

    with open(args.corpus, encoding="utf-8", errors="replace") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    n_chunks = min(args.chunks, len(tokens) // args.ctx)

    # Score only tokens with >= ctx/2 context, like llama-perplexity does.
    first = args.ctx // 2
    total_nll = 0.0
    total_scored = 0
    kld_sum = 0.0
    top1_same = 0
    model_time = 0.0  # timing covers only this model's forward, not the KLD baseline's
    for i in range(n_chunks):
        chunk = mx.array(tokens[i * args.ctx : (i + 1) * args.ctx])[None]
        t0 = time.perf_counter()
        logits = model(chunk)
        logprobs = nn.log_softmax(logits[0, :-1].astype(mx.float32), axis=-1)
        mx.eval(logprobs)
        model_time += time.perf_counter() - t0
        targets = chunk[0, 1:]
        scored = logprobs[first:]
        nll = -mx.take_along_axis(scored, targets[first:, None], axis=-1).sum()
        mx.eval(nll)
        total_nll += nll.item()
        total_scored += int(targets.size) - first
        if base_model is not None:
            base_lp = nn.log_softmax(
                base_model(chunk)[0, :-1].astype(mx.float32), axis=-1
            )[first:]
            base_p = mx.exp(base_lp)
            kld = (base_p * (base_lp - scored)).sum(axis=-1).sum()
            same = (mx.argmax(scored, axis=-1) == mx.argmax(base_lp, axis=-1)).sum()
            mx.eval(kld, same)
            kld_sum += kld.item()
            top1_same += int(same.item())
    ppl = math.exp(total_nll / total_scored)
    prompt_tps = n_chunks * args.ctx / model_time if model_time > 0 else None

    generate_tps = None
    if args.gen_tokens > 0:  # skipped in the KLD-only invocation
        try:
            generated = 0
            t0 = time.perf_counter()
            for _ in stream_generate(
                model, tokenizer, prompt="The old lighthouse keeper", max_tokens=args.gen_tokens
            ):
                generated += 1
            gen_elapsed = time.perf_counter() - t0
            if generated and gen_elapsed > 0:
                generate_tps = generated / gen_elapsed
        except Exception:
            pass  # speed is best-effort; perplexity is the number that matters

    result = {
        "ppl": ppl,
        "chunks": n_chunks,
        "prompt_tps": prompt_tps,
        "generate_tps": generate_tps,
    }
    if base_model is not None and total_scored:
        result["mean_kld"] = kld_sum / total_scored
        result["same_top_pct"] = top1_same / total_scored * 100.0
    print(json.dumps(result))


if __name__ == "__main__":
    main()
