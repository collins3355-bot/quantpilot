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


def _mlx_variant(name: str, model_dir: Path, corpus: Path, chunks: int) -> Variant:
    print(f"  measuring {name}...")
    result = mlxlm.measure(model_dir, corpus, chunks)
    return Variant(
        name=name,
        path=model_dir,
        size_bytes=mlxlm.dir_size(model_dir),
        ppl=result["ppl"],
        ppl_err=None,
        prompt_tps=result.get("prompt_tps"),
        generate_tps=result.get("generate_tps"),
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

    variants = []
    for i, bits in enumerate(args.quants, start=2):
        print(f"[{i}/{steps}] quant: {bits}-bit")
        dest = ensure_converted(workdir / f"{name}-{bits}bit", bits, f"{bits}-bit")
        variants.append(_mlx_variant(f"{bits}-bit", dest, corpus, args.chunks))

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
    mlx.add_argument("--workdir", default="work/mlx", help="where converted models go")
    mlx.add_argument("--out", default="reports", help="where reports go")
    mlx.set_defaults(func=cmd_bench_mlx)

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
