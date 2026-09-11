"""Runtime memory model: what a GGUF artifact actually costs to run.

File size is only part of the story — the KV cache grows linearly with
context length and can dwarf the quality difference between two quants.
Numbers here are derived from GGUF metadata and validated against
llama.cpp's own reported buffer sizes (see tests and docs).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .engines import gguf

# Bytes per element for KV-cache storage types (block formats amortized).
KV_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,  # 32 elems per 34-byte block
    "q4_0": 18 / 32,
}


@dataclass
class ModelShape:
    arch: str
    block_count: int
    head_count_kv: int
    key_length: int
    value_length: int
    context_length: int | None  # the model's trained maximum, if declared


def shape_from_metadata(meta: dict) -> ModelShape:
    arch = meta.get("general.architecture")
    if not arch:
        raise gguf.GGUFError("GGUF metadata has no general.architecture")

    def need(suffix: str) -> int:
        value = meta.get(f"{arch}.{suffix}")
        if value is None:
            raise gguf.GGUFError(f"GGUF metadata missing {arch}.{suffix}")
        return int(value)

    head_count = need("attention.head_count")
    default_head_dim = need("embedding_length") // head_count
    key_length = int(meta.get(f"{arch}.attention.key_length") or default_head_dim)
    value_length = int(meta.get(f"{arch}.attention.value_length") or default_head_dim)
    ctx = meta.get(f"{arch}.context_length")
    return ModelShape(
        arch=arch,
        block_count=need("block_count"),
        head_count_kv=need("attention.head_count_kv"),
        key_length=key_length,
        value_length=value_length,
        context_length=int(ctx) if ctx is not None else None,
    )


def kv_bytes_per_token(shape: ModelShape, kv_type: str = "f16") -> float:
    """KV-cache bytes per token of context (all layers, K and V)."""
    per_elem = KV_BYTES[kv_type]
    return shape.block_count * shape.head_count_kv * (
        shape.key_length + shape.value_length
    ) * per_elem


@dataclass
class Footprint:
    weights_bytes: int
    kv_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.weights_bytes + self.kv_bytes


def footprint(weights_bytes: int, shape: ModelShape, ctx: int, kv_type: str = "f16") -> Footprint:
    """Estimated steady-state memory to run the model at `ctx` context.

    Covers weights + KV cache. Compute/graph buffers and framework overhead
    are NOT included — treat this as a floor, and leave headroom.
    """
    return Footprint(
        weights_bytes=weights_bytes,
        kv_bytes=round(kv_bytes_per_token(shape, kv_type) * ctx),
    )


def shape_of(model_path: Path) -> ModelShape:
    return shape_from_metadata(gguf.read_metadata(model_path))
