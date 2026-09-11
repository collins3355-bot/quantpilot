"""Paired error bars for perplexity comparisons.

llama-perplexity's own "± 0.30" is the uncertainty of an *absolute* PPL, and
it is mostly text-to-text variation: some chunks are just harder to predict.
When two models read the same text that variation cancels, so comparing them
chunk by chunk gives a far tighter interval on the difference, which is the
number quantpilot's quality budgets are about.
"""

from __future__ import annotations

import math
import statistics

Z95 = 1.96  # normal approximation; chunk counts are typically 32 or more


def _pct(nats: float) -> float:
    return (math.exp(nats) - 1.0) * 100.0


def paired_delta_pct(
    variant_nll: list[float] | None, baseline_nll: list[float] | None
) -> tuple[float, float, float] | None:
    """(Δ PPL %, low, high) of a variant vs. a baseline, with a ~95% interval.

    Inputs are per-chunk mean negative log-likelihoods over identical text.
    Returns None when the runs can't be paired (missing data, different chunk
    counts, or fewer than two chunks).
    """
    if not variant_nll or not baseline_nll or len(variant_nll) != len(baseline_nll):
        return None
    if len(variant_nll) < 2:
        return None
    diffs = [v - b for v, b in zip(variant_nll, baseline_nll)]
    mean = statistics.fmean(diffs)
    half = Z95 * statistics.stdev(diffs) / math.sqrt(len(diffs))
    return _pct(mean), _pct(mean - half), _pct(mean + half)
