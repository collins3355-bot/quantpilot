"""Measure an MLX model: perplexity over 512-token chunks, plus speed.

This script runs inside the MLX virtualenv (see scripts/setup-mlx.sh) —
quantpilot invokes it as a subprocess so its own core stays dependency-free.
It prints a single JSON object as the last line of stdout.

Methodology matches llama-perplexity's convention: the corpus is tokenized,
split into fixed 512-token chunks, and only tokens in the second half of each
chunk (i.e. with at least ctx/2 tokens of context) are scored. With the same
tokenizer — e.g. the same model in GGUF and MLX form — absolute perplexities
are then directly comparable across engines; across different models, compare
relative degradation vs. each baseline instead.
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
    args = parser.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, stream_generate

    model, tokenizer = load(args.model)

    with open(args.corpus, encoding="utf-8", errors="replace") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    n_chunks = min(args.chunks, len(tokens) // args.ctx)

    # Score only tokens with >= ctx/2 context, like llama-perplexity does.
    first = args.ctx // 2
    total_nll = 0.0
    total_scored = 0
    start = time.perf_counter()
    for i in range(n_chunks):
        chunk = mx.array(tokens[i * args.ctx : (i + 1) * args.ctx])[None]
        logits = model(chunk)
        logprobs = nn.log_softmax(logits[0, :-1].astype(mx.float32), axis=-1)
        targets = chunk[0, 1:]
        nll = -mx.take_along_axis(
            logprobs[first - 1 :], targets[first - 1 :, None], axis=-1
        ).sum()
        mx.eval(nll)
        total_nll += nll.item()
        total_scored += int(targets.size) - (first - 1)
    elapsed = time.perf_counter() - start
    ppl = math.exp(total_nll / total_scored)
    prompt_tps = n_chunks * args.ctx / elapsed if elapsed > 0 else None

    generate_tps = None
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

    print(
        json.dumps(
            {
                "ppl": ppl,
                "chunks": n_chunks,
                "prompt_tps": prompt_tps,
                "generate_tps": generate_tps,
            }
        )
    )


if __name__ == "__main__":
    main()
