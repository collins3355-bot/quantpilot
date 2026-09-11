"""Command-line interface.

    quantpilot doctor
    quantpilot bench --source model-f16.gguf --corpus wiki.test.raw
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import __version__, hw
from .engines import llamacpp, mlxlm
from .pipeline import BenchRun, Variant, run as run_pipeline
from .report import render_markdown, to_json

DEFAULT_QUANTS = ["Q4_K_M", "Q5_K_M", "Q8_0"]
DEFAULT_MLX_BITS = [4, 6, 8]
KV_BYTES_CHOICES = ("f32", "f16", "bf16", "q8_0", "q4_0")


def cmd_doctor(_args: argparse.Namespace) -> int:
    info = hw.describe()
    print(f"quantpilot {__version__}")
    print(f"hardware: {info.get('chip') or info.get('machine')}, "
          f"{info.get('memory_gb', '?')} GB RAM, {info['os']}")
    ok = True
    for name in llamacpp.BINARIES:
        try:
            print(f"  {name}: {llamacpp.find_binary(name)}")
        except llamacpp.EngineError as err:
            print(f"  {name}: MISSING — {err}")
            ok = False
    return 0 if ok else 1


def cmd_bench(args: argparse.Namespace) -> int:
    source = Path(args.source)
    corpus = Path(args.corpus)
    for path, flag in ((source, "--source"), (corpus, "--corpus")):
        if not path.exists():
            print(f"error: {flag} {path} does not exist", file=sys.stderr)
            return 1

    bench = run_pipeline(
        source=source,
        quants=args.quants,
        corpus=corpus,
        chunks=args.chunks,
        workdir=Path(args.workdir),
        budget_pct=args.budget,
        kld=not args.no_kld,
        hellaswag_data=Path(args.hellaswag) if args.hellaswag else None,
        hellaswag_tasks=args.hellaswag_tasks,
    )

    hardware = hw.describe()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    md_path = outdir / f"{source.stem}.md"
    md_path.write_text(render_markdown(bench, hardware))
    (outdir / f"{source.stem}.json").write_text(to_json(bench, hardware))

    print()
    print(md_path.read_text())
    print(f"report written to {md_path}")
    return 0


def _mlx_variant(
    name: str, model_dir: Path, corpus: Path, chunks: int, kld_vs: Path | None = None
) -> Variant:
    # Speed and quality come from a solo pass; KLD runs as a second pass so
    # the co-resident baseline can't distort the timings (or the memory
    # footprint of the timed run).
    print(f"  measuring {name}...")
    result = mlxlm.measure(model_dir, corpus, chunks)
    kld_result: dict = {}
    if kld_vs is not None:
        print(f"  measuring KL divergence of {name} vs. baseline...")
        kld_result = mlxlm.measure(model_dir, corpus, chunks, kld_vs=kld_vs, gen_tokens=0)
    return Variant(
        name=name,
        path=model_dir,
        size_bytes=mlxlm.dir_size(model_dir),
        ppl=result["ppl"],
        ppl_err=None,
        prompt_tps=result.get("prompt_tps"),
        generate_tps=result.get("generate_tps"),
        mean_kld=kld_result.get("mean_kld"),
        same_top_pct=kld_result.get("same_top_pct"),
    )


def cmd_bench_mlx(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus)
    if not corpus.exists():
        print(f"error: --corpus {corpus} does not exist", file=sys.stderr)
        return 1
    name = Path(args.model).name
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_src = Path(args.model)
    if not model_src.exists():
        print(f"  downloading {args.model} from Hugging Face...")
        model_src = mlxlm.download_hf(args.model, workdir / f"{name}-hf")

    def ensure_converted(dest: Path, bits: int | None, label: str) -> Path:
        if (dest / "config.json").exists():
            return dest
        if dest.exists():  # leftover from an interrupted conversion
            shutil.rmtree(dest)
        print(f"  converting to MLX {label}...")
        return mlxlm.convert(str(model_src), dest, bits=bits)

    steps = len(args.quants) + 1
    print(f"[1/{steps}] baseline: {name} 16-bit")
    baseline_dir = ensure_converted(workdir / f"{name}-fp16", None, "16-bit")
    baseline = _mlx_variant("16-bit (baseline)", baseline_dir, corpus, args.chunks)
    kld_vs = None if args.no_kld else baseline_dir
    if kld_vs is not None:
        baseline.mean_kld, baseline.same_top_pct = 0.0, 100.0

    variants = []
    for i, bits in enumerate(args.quants, start=2):
        print(f"[{i}/{steps}] quant: {bits}-bit")
        dest = ensure_converted(workdir / f"{name}-{bits}bit", bits, f"{bits}-bit")
        variants.append(_mlx_variant(f"{bits}-bit", dest, corpus, args.chunks, kld_vs=kld_vs))

    bench = BenchRun(
        source=Path(args.model),
        baseline=baseline,
        variants=variants,
        corpus=corpus,
        chunks=args.chunks,
        budget_pct=args.budget,
        engine=f"mlx-lm (group quantization), venv {mlxlm.find_python()}",
    )

    hardware = hw.describe()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    md_path = outdir / f"{name}-mlx.md"
    md_path.write_text(render_markdown(bench, hardware))
    (outdir / f"{name}-mlx.json").write_text(to_json(bench, hardware))

    print()
    print(md_path.read_text())
    print(f"report written to {md_path}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    import json

    from .report import render_search, search_to_json
    from .search import run_search

    source = Path(args.source)
    corpus = Path(args.corpus)
    for path, flag in ((source, "--source"), (corpus, "--corpus")):
        if not path.exists():
            print(f"error: {flag} {path} does not exist", file=sys.stderr)
            return 1

    result = run_search(
        source=source,
        corpus=corpus,
        chunks=args.chunks,
        workdir=Path(args.workdir),
        base=args.base,
        bump=args.bump,
        budget_pct=args.budget,
        classes=args.classes,
        keep_artifacts=args.keep_artifacts,
    )

    ladder = None
    if args.ladder:
        ladder = json.loads(Path(args.ladder).read_text())

    hardware = hw.describe()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    md_path = outdir / f"{source.stem}-search.md"
    md_path.write_text(render_search(result, hardware, ladder))
    (outdir / f"{source.stem}-search.json").write_text(search_to_json(result, hardware))

    print()
    print(md_path.read_text())
    print(f"report written to {md_path}")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    from .fit import FitError, compute_rows, find_shape, load_report, render
    from .memory import KV_BYTES

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"error: {report_path} does not exist", file=sys.stderr)
        return 1
    hardware = hw.describe()
    ram_gb = args.ram if args.ram is not None else hardware.get("memory_gb")
    if not ram_gb:
        print("error: could not detect RAM; pass --ram", file=sys.stderr)
        return 1

    try:
        report = load_report(report_path)
        shape = find_shape(report)
    except FitError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    budget_bytes = ram_gb * args.headroom * 1024**3
    rows = compute_rows(report, shape, args.ctx, args.kv_type, budget_bytes)
    markdown = render(
        report, rows, shape, args.ctx, args.kv_type, ram_gb, args.headroom, hardware
    )

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    md_path = outdir / f"{Path(report['source']).stem}-fit-{args.ctx}.md"
    md_path.write_text(markdown)
    print(markdown)
    print(f"report written to {md_path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .compare import load_report, render_comparison

    reports = []
    for path in args.reports:
        if not Path(path).exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 1
        reports.append(load_report(Path(path)))

    markdown = render_comparison(reports, budgets=tuple(args.budgets))
    names = sorted({Path(r["source"]).name for r in reports})
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    md_path = outdir / f"compare-{'-'.join(names)[:60]}.md"
    md_path.write_text(markdown)
    print(markdown)
    print(f"comparison written to {md_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Long runs are often piped/backgrounded; line-buffer stdout so progress
    # lines appear as they happen instead of all at once on exit.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(
        prog="quantpilot",
        description="Quantization autotuner: find the best quant for your model, "
        "your hardware, and your quality budget.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check that required engines are installed")
    doctor.set_defaults(func=cmd_doctor)

    bench = sub.add_parser("bench", help="quantize, measure, and recommend")
    bench.add_argument("--source", required=True, help="path to an F16/F32 .gguf model")
    bench.add_argument("--corpus", required=True, help="plain-text file for perplexity")
    bench.add_argument(
        "--quants", nargs="+", default=DEFAULT_QUANTS,
        help=f"llama.cpp quant types to try (default: {' '.join(DEFAULT_QUANTS)})",
    )
    bench.add_argument(
        "--chunks", type=int, default=32,
        help="512-token chunks of the corpus to evaluate (default: 32)",
    )
    bench.add_argument(
        "--budget", type=float, default=1.0,
        help="max acceptable perplexity increase in percent (default: 1.0)",
    )
    bench.add_argument(
        "--no-kld", action="store_true",
        help="skip KL-divergence measurement (saves time and a multi-GB logits file)",
    )
    bench.add_argument(
        "--hellaswag", default=None, metavar="DATA",
        help="path to hellaswag_val_full.txt to also score HellaSwag accuracy",
    )
    bench.add_argument(
        "--hellaswag-tasks", type=int, default=400,
        help="number of HellaSwag tasks to score (default: 400)",
    )
    bench.add_argument("--workdir", default="work", help="where quantized files go")
    bench.add_argument("--out", default="reports", help="where reports go")
    bench.set_defaults(func=cmd_bench)

    mlx = sub.add_parser("bench-mlx", help="same loop, but with Apple's MLX stack")
    mlx.add_argument(
        "--model", required=True,
        help="Hugging Face model id (e.g. Qwen/Qwen2.5-0.5B-Instruct) or local HF dir",
    )
    mlx.add_argument("--corpus", required=True, help="plain-text file for perplexity")
    mlx.add_argument(
        "--quants", nargs="+", type=int, default=DEFAULT_MLX_BITS,
        help=f"MLX quantization bit-widths to try (default: {DEFAULT_MLX_BITS})",
    )
    mlx.add_argument("--chunks", type=int, default=32,
                     help="512-token chunks of the corpus to evaluate (default: 32)")
    mlx.add_argument("--budget", type=float, default=1.0,
                     help="max acceptable perplexity increase in percent (default: 1.0)")
    mlx.add_argument(
        "--no-kld", action="store_true",
        help="skip KL-divergence measurement (avoids loading the baseline alongside each quant)",
    )
    mlx.add_argument("--workdir", default="work/mlx", help="where converted models go")
    mlx.add_argument("--out", default="reports", help="where reports go")
    mlx.set_defaults(func=cmd_bench_mlx)

    search = sub.add_parser(
        "search", help="compose a per-layer mixed-precision recipe to hit a quality budget"
    )
    search.add_argument("--source", required=True, help="path to an F16/BF16 .gguf model")
    search.add_argument("--corpus", required=True, help="plain-text file for perplexity")
    search.add_argument("--base", default="Q4_K_M", help="base quant type (default: Q4_K_M)")
    search.add_argument("--bump", default="Q6_K",
                        help="type sensitive classes get bumped to (default: Q6_K)")
    search.add_argument("--budget", type=float, default=1.0,
                        help="max %% PPL increase vs. full-precision baseline (default: 1.0)")
    search.add_argument("--chunks", type=int, default=32,
                        help="512-token chunks of the corpus to evaluate (default: 32)")
    search.add_argument("--classes", nargs="+", default=None,
                        help="tensor classes to probe (default: all)")
    search.add_argument("--keep-artifacts", action="store_true",
                        help="keep probe/intermediate .gguf files instead of deleting them")
    search.add_argument("--ladder", default=None,
                        help="bench report .json for uniform-ladder comparison in the report")
    search.add_argument("--workdir", default="work", help="where quantized files go")
    search.add_argument("--out", default="reports", help="where reports go")
    search.set_defaults(func=cmd_search)

    fit = sub.add_parser(
        "fit", help="rank a bench report's artifacts by what fits your machine at runtime"
    )
    fit.add_argument("report", help="bench report .json (GGUF engine)")
    fit.add_argument("--ram", type=float, default=None,
                     help="memory budget in GB (default: this machine's RAM)")
    fit.add_argument("--ctx", type=int, default=8192,
                     help="target context length in tokens (default: 8192)")
    fit.add_argument("--kv-type", default="f16", choices=sorted(KV_BYTES_CHOICES),
                     help="KV cache storage type (default: f16)")
    fit.add_argument("--headroom", type=float, default=0.75,
                     help="fraction of RAM usable for the model (default: 0.75, "
                     "≈ the Metal working-set limit on Apple silicon)")
    fit.add_argument("--out", default="reports", help="where reports go")
    fit.set_defaults(func=cmd_fit)

    compare = sub.add_parser(
        "compare", help="merge bench report JSONs into one cross-engine table"
    )
    compare.add_argument("reports", nargs="+", help="report .json files from bench/bench-mlx")
    compare.add_argument(
        "--budgets", nargs="+", type=float, default=[0.5, 1.0, 2.0, 5.0],
        help="quality budgets (%% PPL increase) for the frontier section",
    )
    compare.add_argument("--out", default="reports", help="where the comparison goes")
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except llamacpp.EngineError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
