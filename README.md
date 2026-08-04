# quantpilot

**Quantization autotuning for local models.** Give quantpilot a model and a
quality budget; it quantizes the model several ways with battle-tested engines,
measures what each variant actually costs you in quality — on *your* hardware —
and recommends the smallest artifact that stays inside the budget.

Choosing a quantization today is folklore ("Q4_K_M is usually fine").
quantpilot replaces the folklore with a measurement.

```
$ quantpilot bench --source qwen2.5-0.5b-instruct-f16.gguf --corpus wiki.test.raw

| Variant  | Size    | Δ size | Perplexity | Δ PPL   | Prompt t/s | Gen t/s |
|----------|---------|--------|------------|---------|------------|---------|
| baseline | 1.19 GB | +0%    | 14.5670    | +0.00%  | 9,600      | 230     |
| Q4_K_M   | 0.47 GB | -60%   | 14.6743    | +0.74%  | 11,200     | 310     |
| ...      |         |        |            |         |            |         |

Recommendation: Q4_K_M — 60% smaller at +0.74% perplexity, inside the 1% budget.
```

## Requirements

- Python 3.10+ (no Python dependencies)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) binaries on your PATH:
  `brew install llama.cpp` (macOS) or build from source.

Check your setup:

```
quantpilot doctor
```

## Usage

```
quantpilot bench \
  --source path/to/model-f16.gguf \    # an unquantized (F16/F32) GGUF
  --corpus path/to/wiki.test.raw \     # plain text used to measure quality
  --quants Q4_K_M Q5_K_M Q8_0 \        # candidates to try
  --budget 1.0 \                       # max % perplexity increase you'll accept
  --chunks 32                          # how much of the corpus to evaluate
```

Reports land in `reports/` as markdown (for humans) and JSON (for machines).
Quantized artifacts land in `work/` and are reused on later runs.

Or run without installing: `PYTHONPATH=src python3 -m quantpilot bench ...`

## How it measures quality

Perplexity over a held-out text corpus (wikitext-2 by convention), computed by
`llama-perplexity` with full GPU offload. Lower is better; what matters is the
*increase* relative to the unquantized baseline. Speed comes from `llama-bench`
(prompt processing and generation, tokens/second).

## Roadmap

- **KL divergence** against baseline logits (`llama-perplexity --kl-divergence`) —
  a stricter quality signal than perplexity
- **MLX backend** for Apple-silicon-native formats
- **Per-layer mixed precision search** — not just picking a preset, composing one
- **Task evals** (small MMLU/GSM8K slices) alongside perplexity
- **Hardware-aware search**: given "must fit in N GB", search the frontier for you

## License

AGPL-3.0-or-later. For commercial licensing, custom hardware integrations, or
corporate inquiries: collins3355@gmail.com.
