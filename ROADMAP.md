# quantpilot roadmap

The mission, unchanged at every step: **replace quantization folklore with
measurements**, and eventually stop *picking* quants and start *compiling*
them.

## v0.3 — compose, don't pick (shipped)

- Per-layer mixed-precision search on the GGUF side: probe each tensor
  class's sensitivity on *your* model, rank by quality-recovered-per-GB,
  greedily compose a recipe that hits your quality budget.
- Every search report ends with a reproducible `llama-quantize` command.

## v0.4 — trust the numbers everywhere

- **Task evals** (small MMLU / GSM8K / HellaSwag slices) alongside perplexity,
  because wikitext PPL is a narrow lens.
- **KL divergence for the MLX backend** (already first-class on GGUF).
- Multiple corpora (code, chat, multilingual) with per-corpus deltas.
- Error bars surfaced everywhere a number is shown.

## v0.5 — hardware-aware targets

- `--fit 12GB`: search for the best model *that actually fits*, counting KV
  cache at your target context length, not just file size.
- Speed-weighted recommendations (quality budget + latency floor).
- Machine profiles so reports from different hardware are comparable.

## v0.6 — calibration-aware compilation

- imatrix (importance matrix) integration and IQ-quant support.
- Bring the mixed-precision search to MLX via quant predicates.
- Layer-level (not just class-level) granularity in the search.

## v1.0 — the boring promises

- Stable CLI and JSON report schema, CI across engines, real docs.
- A public **recipe registry**: community-submitted, reproducible measured
  recipes per model × hardware, so the answer to "which quant?" becomes a
  lookup before it becomes a compute job.

## Sustaining the project

quantpilot is AGPL-3.0. It stays independent and maintained through
commercial licenses for closed-source/enterprise use, GitHub Sponsors, and
paid custom hardware-target work. If your company depends on this, fund it:
collins3355@gmail.com.
