"""Minimal pure-Python GGUF metadata reader.

Reads only the header key-value section — enough to learn a model's
architecture shape (layer count, KV heads, head dims) without loading
weights or adding dependencies. Large arrays (tokenizer vocabularies)
are skipped, not materialized.

Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from pathlib import Path


class GGUFError(RuntimeError):
    pass


# value_type -> (struct format, size in bytes)
_SIMPLE = {
    0: ("B", 1),  # uint8
    1: ("b", 1),  # int8
    2: ("H", 2),  # uint16
    3: ("h", 2),  # int16
    4: ("I", 4),  # uint32
    5: ("i", 4),  # int32
    6: ("f", 4),  # float32
    7: ("?", 1),  # bool
    10: ("Q", 8),  # uint64
    11: ("q", 8),  # int64
    12: ("d", 8),  # float64
}
_STRING, _ARRAY = 8, 9
_SKIP_ARRAYS_OVER = 4096  # tokenizer vocab etc. — walk/seek past, don't keep


def _read(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise GGUFError("truncated GGUF file")
    return data


def _scalar(f, fmt: str, size: int):
    return struct.unpack("<" + fmt, _read(f, size))[0]


def _string(f) -> str:
    n = _scalar(f, "Q", 8)
    if n > 1 << 31:
        raise GGUFError("implausible string length")
    return _read(f, n).decode("utf-8", errors="replace")


def _value(f, vtype: int):
    if vtype in _SIMPLE:
        fmt, size = _SIMPLE[vtype]
        return _scalar(f, fmt, size)
    if vtype == _STRING:
        return _string(f)
    if vtype == _ARRAY:
        etype = _scalar(f, "I", 4)
        count = _scalar(f, "Q", 8)
        if count > 1 << 34:
            raise GGUFError("implausible array length")
        if count > _SKIP_ARRAYS_OVER:
            if etype in _SIMPLE:
                f.seek(_SIMPLE[etype][1] * count, 1)
            else:
                for _ in range(count):
                    _value(f, etype)
            return None
        return [_value(f, etype) for _ in range(count)]
    raise GGUFError(f"unknown GGUF value type {vtype}")


def read_metadata(path: Path) -> dict:
    """Return the GGUF header key-values (large arrays elided as None)."""
    with open(path, "rb") as f:
        if _read(f, 4) != b"GGUF":
            raise GGUFError(f"{path} is not a GGUF file")
        version = _scalar(f, "I", 4)
        if version < 2:
            raise GGUFError(f"GGUF version {version} is not supported")
        _tensor_count = _scalar(f, "Q", 8)
        kv_count = _scalar(f, "Q", 8)
        if kv_count > 1 << 20:
            raise GGUFError("implausible metadata count")
        meta = {}
        for _ in range(kv_count):
            key = _string(f)
            vtype = _scalar(f, "I", 4)
            meta[key] = _value(f, vtype)
        return meta
