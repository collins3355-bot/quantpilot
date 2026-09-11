"""Thin wrappers around the llama.cpp command-line tools.

quantpilot does not implement quantization or inference itself — it
orchestrates the battle-tested llama.cpp binaries and measures the results.
"""

from __future__ import annotations

import json
import math
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


def override_args(tensor_overrides: dict[str, str] | None) -> list[str]:
    """Translate {tensor_class: ggml_type} into llama-quantize CLI flags.

    token_embd and output have dedicated flags; everything else uses the
    repeatable --tensor-type name=type form.
    """
    args: list[str] = []
    for cls, qtype in (tensor_overrides or {}).items():
        qtype = qtype.lower()
        if cls == "token_embd":
            args += ["--token-embedding-type", qtype]
        elif cls == "output":
            args += ["--output-tensor-type", qtype]
        else:
            args += ["--tensor-type", f"{cls}={qtype}"]
    return args


def quantize(
    source: Path, dest: Path, qtype: str, tensor_overrides: dict[str, str] | None = None
) -> Path:
    """Produce a quantized copy of `source` using a llama.cpp type such as Q4_K_M.

    `tensor_overrides` selectively bumps tensor classes to a different type,
    e.g. {"ffn_down": "q6_k"} — the mechanism behind mixed-precision recipes.
    """
    cmd = [find_binary("llama-quantize"), *override_args(tensor_overrides), source, dest, qtype]
    _run(cmd)
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


_RUNNING_RE = re.compile(r"\[(\d+)\](\d+\.\d+)")


def parse_chunk_nll(output: str) -> list[float]:
    """Per-chunk mean negative log-likelihood, recovered from the running estimates.

    llama-perplexity prints the cumulative PPL after each chunk
    ([1]6.41,[2]7.15,...). Every chunk scores the same number of tokens, so
    chunk i's mean NLL is i*ln(P_i) - (i-1)*ln(P_{i-1}). Paired error bars are
    built from these. Returns [] when the running estimates aren't present.
    """
    running = {int(i): float(p) for i, p in _RUNNING_RE.findall(output)}
    n = len(running)
    if n == 0 or set(running) != set(range(1, n + 1)) or min(running.values()) <= 0:
        return []
    totals = [0.0] + [i * math.log(running[i]) for i in range(1, n + 1)]
    return [totals[i] - totals[i - 1] for i in range(1, n + 1)]


@dataclass
class Perplexity:
    ppl: float
    err: float | None  # llama-perplexity's ± on the absolute PPL (mostly text-to-text noise)
    chunk_nll: list[float]  # per-chunk mean NLL, for paired comparisons on identical text


def _perplexity_result(output: str) -> Perplexity:
    ppl, err = parse_perplexity(output)
    return Perplexity(ppl=ppl, err=err, chunk_nll=parse_chunk_nll(output))


def perplexity(model: Path, corpus: Path, chunks: int) -> Perplexity:
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
    return _perplexity_result(output)


_HS_RE = re.compile(r"^\s*(\d+)\t([0-9.]+)%", re.MULTILINE)


def parse_hellaswag(output: str) -> tuple[float, int]:
    """Return (accuracy_pct, tasks_scored) from --hellaswag output.

    The run prints one line per task with the running accuracy; the last
    line is the final score.
    """
    matches = _HS_RE.findall(output)
    if not matches:
        raise EngineError("could not find HellaSwag accuracy lines in output")
    tasks, acc = matches[-1]
    return float(acc), int(tasks)


def hellaswag(model: Path, data: Path, tasks: int) -> tuple[float, int]:
    """HellaSwag accuracy (higher = better) over the first `tasks` tasks."""
    output = _run(
        [
            find_binary("llama-perplexity"),
            "-m", model,
            "--hellaswag",
            "-f", data,
            "--hellaswag-tasks", str(tasks),
            "-ngl", "99",
        ]
    )
    return parse_hellaswag(output)


def save_base_logits(model: Path, corpus: Path, chunks: int, dest: Path) -> Perplexity:
    """Run the baseline over the corpus, saving its full logits to `dest`.

    Returns the baseline perplexity (the run reports it in the same pass).
    Note: the logits file is large — roughly tokens x vocab x 2 bytes,
    i.e. several GB for 32 chunks on a modern vocabulary.
    """
    output = _run(
        [
            find_binary("llama-perplexity"),
            "-m", model,
            "-f", corpus,
            "--chunks", str(chunks),
            "-ngl", "99",
            "--kl-divergence-base", dest,
        ]
    )
    return _perplexity_result(output)


@dataclass
class KLDStats:
    mean_kld: float  # mean KL divergence vs. baseline logits (0 = identical)
    same_top_pct: float | None  # % of tokens where the top-1 prediction matches


_KLD_RE = re.compile(r"Mean\s+KLD:\s+([0-9.eE+-]+)")
_TOP_RE = re.compile(r"Same top p:\s*([0-9.]+)")


def parse_kld(output: str) -> KLDStats:
    kld_match = _KLD_RE.search(output)
    if not kld_match:
        raise EngineError("could not find 'Mean KLD:' in llama-perplexity output")
    top_match = _TOP_RE.search(output)
    return KLDStats(
        mean_kld=float(kld_match.group(1)),
        same_top_pct=float(top_match.group(1)) if top_match else None,
    )


def kl_divergence(model: Path, base_logits: Path) -> KLDStats:
    """Compare a quantized model's token distributions against saved baseline logits."""
    output = _run(
        [
            find_binary("llama-perplexity"),
            "-m", model,
            "--kl-divergence-base", base_logits,
            "--kl-divergence",
            "-ngl", "99",
        ]
    )
    return parse_kld(output)


@dataclass
class Speed:
    prompt_tps: float | None  # prompt processing, tokens/second
    generate_tps: float | None  # text generation, tokens/second
    prompt_sd: float | None = None  # standard deviation across llama-bench repetitions
    generate_sd: float | None = None


def parse_bench_json(output: str) -> Speed:
    # llama-bench -o json prints a JSON array; log lines may surround it.
    start, end = output.find("["), output.rfind("]")
    if start == -1 or end == -1:
        raise EngineError("could not find a JSON array in llama-bench output")
    entries = json.loads(output[start : end + 1])
    speed = Speed(prompt_tps=None, generate_tps=None)
    for entry in entries:
        if entry.get("n_gen", 0) == 0 and entry.get("n_prompt", 0) > 0:
            speed.prompt_tps, speed.prompt_sd = entry.get("avg_ts"), entry.get("stddev_ts")
        elif entry.get("n_gen", 0) > 0 and entry.get("n_prompt", 0) == 0:
            speed.generate_tps, speed.generate_sd = entry.get("avg_ts"), entry.get("stddev_ts")
    return speed


def bench(model: Path, repetitions: int = 5) -> Speed:
    """Prompt-processing and generation throughput via llama-bench.

    Five repetitions (llama-bench's own default) give a usable spread. The
    spread only covers noise within one session: between sessions, thermals
    and background load moved Qwen3-8B generation speed by 20%+ on an M1 Max.
    """
    output = _run(
        [find_binary("llama-bench"), "-m", model, "-r", str(repetitions), "-o", "json"]
    )
    return parse_bench_json(output)
