"""The core loop: quantize a model several ways, measure each, pick a winner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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

    def ppl_increase_pct(self, baseline_ppl: float) -> float:
        return (self.ppl - baseline_ppl) / baseline_ppl * 100.0


@dataclass
class BenchRun:
    source: Path
    baseline: Variant
    variants: list[Variant]  # quantized variants only, in the order they ran
    corpus: Path
    chunks: int
    budget_pct: float  # max acceptable perplexity increase, in percent

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


def _measure(name: str, path: Path, corpus: Path, chunks: int, progress: Progress) -> Variant:
    progress(f"  measuring perplexity of {name} ({chunks} chunks)...")
    ppl, ppl_err = llamacpp.perplexity(path, corpus, chunks)
    progress(f"  benchmarking speed of {name}...")
    speed = llamacpp.bench(path)
    return Variant(
        name=name,
        path=path,
        size_bytes=path.stat().st_size,
        ppl=ppl,
        ppl_err=ppl_err,
        prompt_tps=speed.prompt_tps,
        generate_tps=speed.generate_tps,
    )


def run(
    source: Path,
    quants: list[str],
    corpus: Path,
    chunks: int,
    workdir: Path,
    budget_pct: float,
    progress: Progress = print,
) -> BenchRun:
    workdir.mkdir(parents=True, exist_ok=True)

    progress(f"[1/{len(quants) + 1}] baseline: {source.name}")
    baseline = _measure("baseline", source, corpus, chunks, progress)

    variants = []
    for i, qtype in enumerate(quants, start=2):
        progress(f"[{i}/{len(quants) + 1}] quant: {qtype}")
        dest = workdir / f"{source.stem}-{qtype}.gguf"
        if dest.exists():
            progress(f"  reusing existing {dest.name}")
        else:
            progress(f"  quantizing to {qtype}...")
            llamacpp.quantize(source, dest, qtype)
        variants.append(_measure(qtype, dest, corpus, chunks, progress))

    return BenchRun(
        source=source,
        baseline=baseline,
        variants=variants,
        corpus=corpus,
        chunks=chunks,
        budget_pct=budget_pct,
    )
