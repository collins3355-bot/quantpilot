"""Per-layer mixed-precision search: compose a quant recipe instead of picking a preset.

Not all tensor classes tolerate low precision equally. The search measures
that directly on the target model:

1. PROBE — quantize the whole model at a base type, then once per tensor
   class with just that class bumped to a higher-precision type; the drop in
   perplexity is that class's measured sensitivity.
2. RANK — order classes by perplexity recovered per gigabyte added.
3. COMPOSE — greedily bump the best classes, re-measuring the composed model
   each step, until it meets the quality budget vs. the full-precision
   baseline (or candidates run out).
4. CHECK — re-measure the recipe, the uniform base and the uniform bump on
   held-out text the search never saw. A greedy search partly fits the noise
   of the text it tuned on, so only the held-out numbers are trustworthy. On
   Qwen3-8B, a recipe that tied uniform Q6_K on its tuning text was 0.91%
   worse on held-out text.

The output is a recipe — an exact llama-quantize command anyone can rerun.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import stats
from .engines import llamacpp

Progress = Callable[[str], None]

# Tensor classes worth probing, in architecture order.
TENSOR_CLASSES = [
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_up",
    "ffn_gate",
    "ffn_down",
    "token_embd",
    "output",
]


@dataclass
class Probe:
    cls: str
    size_bytes: int
    size_delta: int  # bytes added vs. the uniform base artifact
    ppl: float
    ppl_recovered: float  # base_ppl - probe_ppl; positive = quality improved

    @property
    def recovery_per_gb(self) -> float:
        if self.size_delta <= 0:
            return 0.0
        return self.ppl_recovered / (self.size_delta / 1024**3)


@dataclass
class Step:
    classes: list[str]  # cumulative overrides at this step
    size_bytes: int
    ppl: float
    dppl_pct: float  # vs. the full-precision baseline


@dataclass
class HoldoutRow:
    size_bytes: int
    ppl: float
    dppl_pct: float  # vs. the full-precision baseline on the held-out text
    ci: tuple[float, float] | None  # ~95% interval on dppl_pct, paired per chunk


@dataclass
class Holdout:
    corpus: Path
    baseline_ppl: float
    base: HoldoutRow  # uniform base type
    bump: HoldoutRow  # uniform bump type — the preset the recipe has to beat
    recipe: HoldoutRow
    recipe_vs_bump: tuple[float, float, float] | None  # (Δ PPL %, low, high)


@dataclass
class SearchResult:
    source: Path
    base: str
    bump: str
    budget_pct: float
    corpus: Path
    chunks: int
    baseline_ppl: float  # full-precision source, on the tuning text
    base_ppl: float  # uniform base quant, on the tuning text
    base_size: int
    probes: list[Probe]  # ranked, best first
    steps: list[Step]
    final_path: Path | None
    met_budget: bool  # judged on the tuning text; see `holdout` for the real test
    failed: list[str] = field(default_factory=list)  # classes whose probe crashed
    tune_corpus: Path | None = None  # the text the search optimized on
    holdout: Holdout | None = None

    def recipe_command(self) -> str:
        if not self.steps:
            return ""
        overrides = {cls: self.bump for cls in self.steps[-1].classes}
        flags = " ".join(llamacpp.override_args(overrides))
        return f"llama-quantize {flags} {self.source.name} <out>.gguf {self.base}"


def rank_probes(probes: list[Probe]) -> list[Probe]:
    """Best quality-recovered-per-byte first; classes that didn't help are dropped."""
    useful = [p for p in probes if p.ppl_recovered > 0 and p.size_delta > 0]
    return sorted(useful, key=lambda p: p.recovery_per_gb, reverse=True)


def split_corpus(corpus: Path, workdir: Path) -> tuple[Path, Path]:
    """Write the first and second halves of `corpus` as separate files.

    The search tunes on the first half and is judged on the second, so its
    reported quality isn't measured on the very text it was optimized for.
    The cut lands on a line boundary.
    """
    text = corpus.read_text(encoding="utf-8", errors="replace")
    mid = len(text) // 2
    cut = text.find("\n", mid)
    cut = mid if cut == -1 else cut + 1
    tune = workdir / f"{corpus.stem}.tune.txt"
    held = workdir / f"{corpus.stem}.holdout.txt"
    tune.write_text(text[:cut], encoding="utf-8")
    held.write_text(text[cut:], encoding="utf-8")
    return tune, held


def _check_holdout(
    source: Path,
    base: str,
    bump: str,
    base_path: Path,
    final_path: Path,
    holdout: Path,
    chunks: int,
    workdir: Path,
    progress: Progress,
    quantize_fn,
    ppl_fn,
) -> Holdout:
    progress(f"held-out check on {holdout.name}: baseline, uniform {base}, uniform {bump}, recipe")
    baseline = ppl_fn(source, holdout, chunks)
    bump_path = workdir / f"{source.stem}-{bump}.gguf"  # same name bench uses, so it's reused
    if not bump_path.exists():
        progress(f"  quantizing uniform {bump} to compare against...")
        quantize_fn(source, bump_path, bump)
    runs = {}
    for role, label, path in (
        ("base", base, base_path),
        ("bump", bump, bump_path),
        ("recipe", "recipe", final_path),
    ):
        progress(f"  measuring {label} on held-out text...")
        runs[role] = (path, ppl_fn(path, holdout, chunks))

    def row(role: str) -> HoldoutRow:
        path, run = runs[role]
        paired = stats.paired_delta_pct(run.chunk_nll, baseline.chunk_nll)
        return HoldoutRow(
            size_bytes=path.stat().st_size,
            ppl=run.ppl,
            dppl_pct=(run.ppl - baseline.ppl) / baseline.ppl * 100.0,
            ci=(paired[1], paired[2]) if paired else None,
        )

    return Holdout(
        corpus=holdout,
        baseline_ppl=baseline.ppl,
        base=row("base"),
        bump=row("bump"),
        recipe=row("recipe"),
        recipe_vs_bump=stats.paired_delta_pct(
            runs["recipe"][1].chunk_nll, runs["bump"][1].chunk_nll
        ),
    )


def run_search(
    source: Path,
    corpus: Path,
    chunks: int,
    workdir: Path,
    base: str = "Q4_K_M",
    bump: str = "Q6_K",
    budget_pct: float = 1.0,
    classes: list[str] | None = None,
    keep_artifacts: bool = False,
    holdout: Path | None = None,
    progress: Progress = print,
    quantize_fn=llamacpp.quantize,
    ppl_fn=llamacpp.perplexity,
) -> SearchResult:
    classes = classes or TENSOR_CLASSES
    workdir.mkdir(parents=True, exist_ok=True)

    if holdout is None:
        tune_corpus, holdout = split_corpus(corpus, workdir)
        progress(f"tuning on the first half of {corpus.name}; the second half is held out")
    else:
        tune_corpus = corpus
        progress(f"tuning on {corpus.name}; held-out check on {holdout.name}")

    progress(f"measuring full-precision baseline ({chunks} chunks)...")
    baseline_ppl = ppl_fn(source, tune_corpus, chunks).ppl

    base_path = workdir / f"{source.stem}-{base}.gguf"
    if not base_path.exists():
        progress(f"quantizing uniform {base}...")
        quantize_fn(source, base_path, base)
    progress(f"measuring uniform {base}...")
    base_ppl = ppl_fn(base_path, tune_corpus, chunks).ppl
    base_size = base_path.stat().st_size

    probes = []
    failed: list[str] = []
    for i, cls in enumerate(classes, start=1):
        progress(f"probe {i}/{len(classes)}: {base} + {cls}={bump}")
        probe_path = workdir / f"{source.stem}-{base}+{cls}-{bump}.gguf"
        try:
            quantize_fn(source, probe_path, base, tensor_overrides={cls: bump})
            ppl = ppl_fn(probe_path, tune_corpus, chunks).ppl
        except llamacpp.EngineError:
            # e.g. tensor shape incompatible with the bump type's block size
            progress(f"  {cls}: probe failed (incompatible with {bump}?), skipping")
            failed.append(cls)
            probe_path.unlink(missing_ok=True)
            continue
        size = probe_path.stat().st_size
        probes.append(
            Probe(
                cls=cls,
                size_bytes=size,
                size_delta=size - base_size,
                ppl=ppl,
                ppl_recovered=base_ppl - ppl,
            )
        )
        if not keep_artifacts:
            probe_path.unlink()

    ranked = rank_probes(probes)
    progress(
        "ranking: " + ", ".join(f"{p.cls} ({p.recovery_per_gb:.3f} ppl/GB)" for p in ranked)
        if ranked
        else "ranking: no class recovered quality — nothing to compose"
    )

    steps: list[Step] = []
    final_path: Path | None = None
    met_budget = False
    active: list[str] = []
    previous_path: Path | None = None
    for p in ranked:
        active = [*active, p.cls]
        step_path = workdir / f"{source.stem}-{base}-mix{len(active)}.gguf"
        progress(f"compose: {base} + {{{', '.join(active)}}}={bump}")
        try:
            quantize_fn(
                source, step_path, base, tensor_overrides={cls: bump for cls in active}
            )
            ppl = ppl_fn(step_path, tune_corpus, chunks).ppl
        except llamacpp.EngineError:
            progress(f"  composing with {p.cls} failed, skipping it")
            active.pop()
            step_path.unlink(missing_ok=True)
            continue
        dppl_pct = (ppl - baseline_ppl) / baseline_ppl * 100.0
        steps.append(
            Step(
                classes=list(active),
                size_bytes=step_path.stat().st_size,
                ppl=ppl,
                dppl_pct=dppl_pct,
            )
        )
        if previous_path is not None and not keep_artifacts:
            previous_path.unlink()
        previous_path = step_path
        final_path = step_path
        if dppl_pct <= budget_pct:
            met_budget = True
            break

    held: Holdout | None = None
    if final_path is not None:
        try:
            held = _check_holdout(
                source, base, bump, base_path, final_path, holdout, chunks,
                workdir, progress, quantize_fn, ppl_fn,
            )
        except llamacpp.EngineError as err:
            progress(f"held-out check failed, reporting tuning-text results only: {err}")

    return SearchResult(
        source=source,
        base=base,
        bump=bump,
        budget_pct=budget_pct,
        corpus=corpus,
        chunks=chunks,
        baseline_ppl=baseline_ppl,
        base_ppl=base_ppl,
        base_size=base_size,
        probes=ranked,
        steps=steps,
        final_path=final_path,
        met_budget=met_budget,
        failed=failed,
        tune_corpus=tune_corpus,
        holdout=held,
    )
