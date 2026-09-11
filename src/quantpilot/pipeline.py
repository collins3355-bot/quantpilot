"""The core loop: quantize a model several ways, measure each, pick a winner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import stats
from .engines import llamacpp

Progress = Callable[[str], None]


@dataclass
class Variant:
    name: str  # "F16 (baseline)" or a quant type like "Q4_K_M"
    path: Path
    size_bytes: int
    ppl: float
    ppl_err: float | None
    prompt_tps: float | None
    generate_tps: float | None
    mean_kld: float | None = None  # mean KL divergence vs. baseline (0 = identical)
    same_top_pct: float | None = None  # % of tokens with the same top-1 prediction
    hellaswag_acc: float | None = None  # HellaSwag accuracy % (task eval)
    chunk_nll: list[float] | None = None  # per-chunk mean NLL, for paired error bars
    prompt_tps_sd: float | None = None  # spread across llama-bench repetitions
    generate_tps_sd: float | None = None

    def ppl_increase_pct(self, baseline_ppl: float) -> float:
        return (self.ppl - baseline_ppl) / baseline_ppl * 100.0

    def ppl_increase_ci(self, baseline: Variant) -> tuple[float, float] | None:
        """~95% interval on the Δ PPL %, paired chunk by chunk against the baseline."""
        paired = stats.paired_delta_pct(self.chunk_nll, baseline.chunk_nll)
        return (paired[1], paired[2]) if paired else None


@dataclass
class BenchRun:
    source: Path
    baseline: Variant
    variants: list[Variant]  # quantized variants only, in the order they ran
    corpus: Path
    chunks: int
    budget_pct: float  # max acceptable perplexity increase, in percent
    engine: str = "llama.cpp"

    def all_variants(self) -> list[Variant]:
        return [self.baseline, *self.variants]

    def recommendation(self) -> Variant:
        """Smallest artifact whose quality loss stays inside the budget.

        The baseline always qualifies (its loss is zero), so there is
        always a recommendation — worst case, it's "don't quantize".
        """
        within_budget = [
            v
            for v in self.all_variants()
            if v.ppl_increase_pct(self.baseline.ppl) <= self.budget_pct
        ]
        return min(within_budget, key=lambda v: v.size_bytes)


def _measure(
    name: str,
    path: Path,
    corpus: Path,
    chunks: int,
    progress: Progress,
    base_logits: Path | None = None,
    save_logits_to: Path | None = None,
    hellaswag_data: Path | None = None,
    hellaswag_tasks: int = 400,
) -> Variant:
    if save_logits_to is not None:
        progress(f"  measuring perplexity of {name} and saving baseline logits...")
        measured = llamacpp.save_base_logits(path, corpus, chunks, save_logits_to)
    else:
        progress(f"  measuring perplexity of {name} ({chunks} chunks)...")
        measured = llamacpp.perplexity(path, corpus, chunks)
    mean_kld = same_top_pct = None
    if base_logits is not None:
        progress(f"  measuring KL divergence of {name} vs. baseline...")
        stats = llamacpp.kl_divergence(path, base_logits)
        mean_kld, same_top_pct = stats.mean_kld, stats.same_top_pct
    hellaswag_acc = None
    if hellaswag_data is not None:
        progress(f"  scoring HellaSwag ({hellaswag_tasks} tasks) for {name}...")
        hellaswag_acc, _ = llamacpp.hellaswag(path, hellaswag_data, hellaswag_tasks)
    progress(f"  benchmarking speed of {name}...")
    speed = llamacpp.bench(path)
    return Variant(
        name=name,
        path=path,
        size_bytes=path.stat().st_size,
        ppl=measured.ppl,
        ppl_err=measured.err,
        prompt_tps=speed.prompt_tps,
        generate_tps=speed.generate_tps,
        mean_kld=mean_kld,
        same_top_pct=same_top_pct,
        hellaswag_acc=hellaswag_acc,
        chunk_nll=measured.chunk_nll or None,
        prompt_tps_sd=speed.prompt_sd,
        generate_tps_sd=speed.generate_sd,
    )


def run(
    source: Path,
    quants: list[str],
    corpus: Path,
    chunks: int,
    workdir: Path,
    budget_pct: float,
    kld: bool = True,
    hellaswag_data: Path | None = None,
    hellaswag_tasks: int = 400,
    progress: Progress = print,
) -> BenchRun:
    workdir.mkdir(parents=True, exist_ok=True)

    # Baseline logits enable KL divergence; the file is keyed by corpus and
    # chunk count so a changed eval setup never reuses stale logits.
    logits = workdir / f"{source.stem}.{corpus.stem}.{chunks}.kld" if kld else None

    hs = {"hellaswag_data": hellaswag_data, "hellaswag_tasks": hellaswag_tasks}
    progress(f"[1/{len(quants) + 1}] baseline: {source.name}")
    if logits is not None and not logits.exists():
        baseline = _measure(
            "baseline", source, corpus, chunks, progress, save_logits_to=logits, **hs
        )
    else:
        baseline = _measure("baseline", source, corpus, chunks, progress, **hs)
    if kld:
        baseline.mean_kld, baseline.same_top_pct = 0.0, 100.0

    variants = []
    for i, qtype in enumerate(quants, start=2):
        progress(f"[{i}/{len(quants) + 1}] quant: {qtype}")
        dest = workdir / f"{source.stem}-{qtype}.gguf"
        if dest.exists():
            progress(f"  reusing existing {dest.name}")
        else:
            progress(f"  quantizing to {qtype}...")
            llamacpp.quantize(source, dest, qtype)
        variants.append(
            _measure(qtype, dest, corpus, chunks, progress, base_logits=logits, **hs)
        )

    return BenchRun(
        source=source,
        baseline=baseline,
        variants=variants,
        corpus=corpus,
        chunks=chunks,
        budget_pct=budget_pct,
    )
