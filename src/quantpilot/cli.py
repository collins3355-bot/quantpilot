"""Command-line interface.

    quantpilot doctor
    quantpilot bench --source model-f16.gguf --corpus wiki.test.raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, hw
from .engines import llamacpp
from .pipeline import run as run_pipeline
from .report import render_markdown, to_json

DEFAULT_QUANTS = ["Q4_K_M", "Q5_K_M", "Q8_0"]


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
    bench.add_argument("--workdir", default="work", help="where quantized files go")
    bench.add_argument("--out", default="reports", help="where reports go")
    bench.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except llamacpp.EngineError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
