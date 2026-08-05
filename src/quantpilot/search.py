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

The output is a recipe — an exact llama-quantize command anyone can rerun.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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
class SearchResult:
    source: Path
    base: str
    bump: str
    budget_pct: float
    corpus: Path
    chunks: int
    baseline_ppl: float  # full-precision source
    base_ppl: float  # uniform base quant
    base_size: int
    probes: list[Probe]  # ranked, best first
    steps: list[Step]
    final_path: Path | None
    met_budget: bool
    failed: list[str] = field(default_factory=list)  # classes whose probe crashed

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
    progress: Progress = print,
    quantize_fn=llamacpp.quantize,
    ppl_fn=llamacpp.perplexity,
) -> SearchResult:
    classes = classes or TENSOR_CLASSES
    workdir.mkdir(parents=True, exist_ok=True)

    progress(f"measuring full-precision baseline ({chunks} chunks)...")
    baseline_ppl, _ = ppl_fn(source, corpus, chunks)

    base_path = workdir / f"{source.stem}-{base}.gguf"
    if not base_path.exists():
        progress(f"quantizing uniform {base}...")
        quantize_fn(source, base_path, base)
    progress(f"measuring uniform {base}...")
    base_ppl, _ = ppl_fn(base_path, corpus, chunks)
    base_size = base_path.stat().st_size

    probes = []
    failed: list[str] = []
    for i, cls in enumerate(classes, start=1):
        progress(f"probe {i}/{len(classes)}: {base} + {cls}={bump}")
        probe_path = workdir / f"{source.stem}-{base}+{cls}-{bump}.gguf"
        try:
            quantize_fn(source, probe_path, base, tensor_overrides={cls: bump})
            ppl, _ = ppl_fn(probe_path, corpus, chunks)
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
            ppl, _ = ppl_fn(step_path, corpus, chunks)
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
    )
