"""Wrappers around mlx-lm, Apple's native quantization/inference stack.

mlx-lm lives in its own virtualenv (.venv-mlx, created by scripts/setup-mlx.sh)
so quantpilot's core stays dependency-free; every call shells out to that
interpreter.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .llamacpp import EngineError

RUNNER = Path(__file__).with_name("mlx_runner.py")


def find_python() -> Path:
    override = os.environ.get("QUANTPILOT_MLX_PYTHON")
    candidates = [Path(override)] if override else []
    candidates.append(Path(".venv-mlx/bin/python"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise EngineError(
        "MLX virtualenv not found — run scripts/setup-mlx.sh first "
        "(or point QUANTPILOT_MLX_PYTHON at a python with mlx-lm installed)."
    )


def _run(cmd: list, timeout: int = 7200) -> str:
    proc = subprocess.run(
        [str(part) for part in cmd], capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        pretty = " ".join(str(part) for part in cmd)
        output = proc.stdout + "\n" + proc.stderr
        raise EngineError(f"command failed ({proc.returncode}): {pretty}\n{output[-2000:]}")
    return proc.stdout


def download_hf(repo: str, dest: Path) -> Path:
    """Download a model's full snapshot to a local directory (idempotent).

    Converting from a local copy keeps mlx-lm off the network at save time,
    where newer huggingface_hub versions reject partially-cached snapshots.
    """
    if (dest / "config.json").exists():
        return dest
    script = (
        "import sys; from huggingface_hub import snapshot_download; "
        "snapshot_download(sys.argv[1], local_dir=sys.argv[2])"
    )
    _run([find_python(), "-c", script, repo, dest])
    return dest


def convert(hf_model: str, dest: Path, bits: int | None) -> Path:
    """Convert a Hugging Face model to MLX format, optionally quantized.

    `bits=None` produces the float16 baseline. Tries the modern
    `python -m mlx_lm convert` CLI first, then the legacy module path.
    """
    quant_args = ["-q", "--q-bits", str(bits)] if bits is not None else []
    python = find_python()
    last_error = None
    for entry in (["-m", "mlx_lm", "convert"], ["-m", "mlx_lm.convert"]):
        try:
            _run([python, *entry, "--hf-path", hf_model, "--mlx-path", dest, *quant_args])
            return dest
        except EngineError as err:
            last_error = err
    raise last_error


def parse_runner_output(output: str) -> dict:
    """The runner prints one JSON object as its last line; tolerate noise above."""
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise EngineError("could not find the JSON result line in mlx runner output")


def measure(model_dir: Path, corpus: Path, chunks: int) -> dict:
    """Perplexity + speed for an MLX model directory. Keys: ppl, prompt_tps, generate_tps."""
    output = _run(
        [find_python(), RUNNER, "--model", model_dir, "--corpus", corpus, "--chunks", str(chunks)]
    )
    return parse_runner_output(output)


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
