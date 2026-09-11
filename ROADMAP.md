# quantpilot roadmap

The mission, unchanged at every step: **replace quantization folklore with
measurements**, and eventually stop *picking* quants and start *compiling*
them.

## v0.3 — compose, don't pick (shipped)

- Per-layer mixed-precision search on the GGUF side: probe each tensor
  class's sensitivity on *your* model, rank by quality-recovered-per-GB,
  greedily compose a recipe that hits your quality budget.
- Every search report ends with a reproducible `llama-quantize` command.

## v0.4 — trust the numbers everywhere (shipped)

- **HellaSwag task eval** alongside perplexity (`bench --hellaswag`), because
  wikitext PPL is a narrow lens. More tasks (MMLU/GSM8K slices) to follow.
- **KL divergence for the MLX backend**, computed in-process against the
  16-bit baseline — metric parity with GGUF.

## v0.5 — hardware-aware targets (`fit` shipped)

- **Shipped:** `quantpilot fit` — ranks measured artifacts by runtime
  footprint (weights + KV cache at your target context, from a native
  pure-Python GGUF metadata reader) against a memory budget. Validated to
  the MiB against llama.cpp's own KV allocations.
- **Shipped:** paired error bars on every GGUF Δ PPL (per-chunk differences
  against the baseline on identical text), a held-out check for `search`
  (it tunes on one half of the corpus and judges the recipe on the other),
  and speed reported as mean ± spread over 5 llama-bench repetitions.
- Still to come in 0.5.x: speed-weighted recommendations (quality budget +
  latency floor), machine profiles, multiple corpora with per-corpus deltas,
  error bars for MLX reports, fit for MLX reports, and KV math for
  sliding-window/MLA architectures.

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
