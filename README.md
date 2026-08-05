# quantpilot

**Quantization autotuning for local models.** Give quantpilot a model and a
quality budget; it quantizes the model several ways with battle-tested engines,
measures what each variant actually costs you in quality — on *your* hardware —
and recommends the smallest artifact that stays inside the budget.

Choosing a quantization today is folklore ("Q4_K_M is usually fine").
quantpilot replaces the folklore with a measurement.

Real output from an M1 Max, Qwen3-8B: [GGUF sweep](examples/qwen3-8b-gguf.md) ·
[MLX sweep](examples/qwen3-8b-mlx.md) ·
[GGUF vs. MLX head-to-head](examples/qwen3-8b-gguf-vs-mlx.md) · [per-layer search](examples/qwen3-8b-search.md) — the two
engines' full-precision baselines agree to 0.04%, and the quality frontier is
genuinely mixed (MLX wins at 6-bit, GGUF's K-quants win at 4-bit).

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
- Optional, for the MLX backend (Apple silicon): `scripts/setup-mlx.sh` creates a
  dedicated `.venv-mlx` with [mlx-lm](https://github.com/ml-explore/mlx-lm) —
  quantpilot's core stays dependency-free.

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

Same loop with Apple's MLX stack (converts straight from a Hugging Face repo):

```
quantpilot bench-mlx \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --corpus path/to/wiki.test.raw \
  --quants 4 6 8                       # MLX bit-widths
```

Compose a per-layer mixed-precision recipe instead of picking a preset — it
probes each tensor class's sensitivity on your model, then bumps the most
valuable classes until the composed model meets your quality budget, and
prints the exact `llama-quantize` command to reproduce the recipe:

```
quantpilot search \
  --source path/to/model-bf16.gguf \
  --corpus path/to/wiki.test.raw \
  --base Q4_K_M --bump Q6_K --budget 1.0
```

Merge runs of the *same model* on different engines into one table with a
quality-budget frontier (smallest artifact within each budget, across engines):

```
quantpilot compare reports/Qwen3-8B-BF16.json reports/Qwen3-8B-mlx.json
```

Or run without installing: `PYTHONPATH=src python3 -m quantpilot bench ...`

## How it measures quality

- **Perplexity** over a held-out text corpus (wikitext-2 by convention), computed
  by `llama-perplexity` with full GPU offload. Lower is better; what matters is
  the *increase* relative to the unquantized baseline.
- **KL divergence** of each quant's token distributions against the baseline's
  saved logits — a stricter signal than perplexity, since a quant can luck into
  a good perplexity while disagreeing with the baseline token-by-token. Also
  reported: top-1 agreement (% of positions where the quant picks the same
  token). Skip with `--no-kld` (the baseline logits file runs several GB).
- **Speed** from `llama-bench`: prompt processing and generation tokens/second.

Comparing across engines: both backends score only tokens with ≥ 256 tokens of
context (llama-perplexity's convention), so for the *same model* in GGUF and
MLX form, absolute perplexities are directly comparable. Across *different*
models (different tokenizers), compare each quant's relative degradation
against its own engine's full-precision baseline instead.

## Roadmap

The detailed public roadmap lives in [ROADMAP.md](ROADMAP.md). Short version:
- **Task evals** (small MMLU/GSM8K slices) alongside perplexity
- **Hardware-aware search**: given "must fit in N GB", search the frontier for you

## License

AGPL-3.0-or-later. For commercial licensing, custom hardware integrations, or
corporate inquiries: collins3355@gmail.com.
