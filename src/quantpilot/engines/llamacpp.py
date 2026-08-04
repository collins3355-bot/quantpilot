"""Thin wrappers around the llama.cpp command-line tools.

quantpilot does not implement quantization or inference itself — it
orchestrates the battle-tested llama.cpp binaries and measures the results.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Checked when a binary isn't on PATH (Homebrew, local source builds).
EXTRA_BIN_DIRS = [
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
    Path("vendor/llama.cpp/build/bin"),
]

BINARIES = ("llama-quantize", "llama-perplexity", "llama-bench")


class EngineError(RuntimeError):
    pass


def find_binary(name: str) -> Path:
    found = shutil.which(name)
    if found:
        return Path(found)
    for directory in EXTRA_BIN_DIRS:
        candidate = directory / name
        if candidate.exists():
            return candidate
    raise EngineError(
        f"could not find `{name}`. Install llama.cpp (e.g. `brew install llama.cpp`) "
        "or build it from source and put the binaries on your PATH."
    )


def _run(cmd: list, timeout: int = 3600) -> str:
    proc = subprocess.run(
        [str(part) for part in cmd], capture_output=True, text=True, timeout=timeout
    )
    output = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        pretty = " ".join(str(part) for part in cmd)
        raise EngineError(f"command failed ({proc.returncode}): {pretty}\n{output[-2000:]}")
    return output


def quantize(source: Path, dest: Path, qtype: str) -> Path:
    """Produce a quantized copy of `source` using a llama.cpp type such as Q4_K_M."""
    _run([find_binary("llama-quantize"), source, dest, qtype])
    if not dest.exists():
        raise EngineError(f"llama-quantize reported success but {dest} does not exist")
    return dest


_PPL_RE = re.compile(r"Final estimate: PPL = ([0-9.]+)(?: \+/- ([0-9.]+))?")


def parse_perplexity(output: str) -> tuple[float, float | None]:
    match = _PPL_RE.search(output)
    if not match:
        raise EngineError(
            "could not find 'Final estimate: PPL = ...' in llama-perplexity output"
        )
    err = float(match.group(2)) if match.group(2) else None
    return float(match.group(1)), err


def perplexity(model: Path, corpus: Path, chunks: int) -> tuple[float, float | None]:
    """Perplexity (lower = better) over the first `chunks` 512-token chunks of `corpus`.

    -ngl 99 offloads every layer to the GPU (Metal on Apple silicon).
    """
    output = _run(
        [
            find_binary("llama-perplexity"),
            "-m", model,
            "-f", corpus,
            "--chunks", str(chunks),
            "-ngl", "99",
        ]
    )
    return parse_perplexity(output)


@dataclass
class Speed:
    prompt_tps: float | None  # prompt processing, tokens/second
    generate_tps: float | None  # text generation, tokens/second


def parse_bench_json(output: str) -> Speed:
    # llama-bench -o json prints a JSON array; log lines may surround it.
    start, end = output.find("["), output.rfind("]")
    if start == -1 or end == -1:
        raise EngineError("could not find a JSON array in llama-bench output")
    entries = json.loads(output[start : end + 1])
    prompt_tps = generate_tps = None
    for entry in entries:
        if entry.get("n_gen", 0) == 0 and entry.get("n_prompt", 0) > 0:
            prompt_tps = entry.get("avg_ts")
        elif entry.get("n_gen", 0) > 0 and entry.get("n_prompt", 0) == 0:
            generate_tps = entry.get("avg_ts")
    return Speed(prompt_tps=prompt_tps, generate_tps=generate_tps)


def bench(model: Path, repetitions: int = 3) -> Speed:
    """Prompt-processing and generation throughput via llama-bench."""
    output = _run(
        [find_binary("llama-bench"), "-m", model, "-r", str(repetitions), "-o", "json"]
    )
    return parse_bench_json(output)
