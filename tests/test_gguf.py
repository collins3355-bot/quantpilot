import struct
import tempfile
import unittest
from pathlib import Path

from quantpilot.engines import gguf


def s(text):  # gguf string: uint64 length + bytes
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def kv_str(key, value):
    return s(key) + struct.pack("<I", 8) + s(value)


def kv_u32(key, value):
    return s(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def kv_bool(key, value):
    return s(key) + struct.pack("<I", 7) + struct.pack("<?", value)


def kv_big_i32_array(key, count):
    return (
        s(key) + struct.pack("<I", 9) + struct.pack("<I", 5)
        + struct.pack("<Q", count) + b"\x00" * (4 * count)
    )


def kv_small_u32_array(key, values):
    return (
        s(key) + struct.pack("<I", 9) + struct.pack("<I", 4)
        + struct.pack("<Q", len(values))
        + b"".join(struct.pack("<I", v) for v in values)
    )


def synthetic_gguf(kvs):
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
    return header + b"".join(kvs)


class TestGGUFReader(unittest.TestCase):
    def read(self, blob):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(blob)
            path = Path(f.name)
        try:
            return gguf.read_metadata(path)
        finally:
            path.unlink()

    def test_reads_scalars_strings_and_small_arrays(self):
        meta = self.read(
            synthetic_gguf(
                [
                    kv_str("general.architecture", "qwen3"),
                    kv_u32("qwen3.block_count", 36),
                    kv_u32("qwen3.attention.head_count_kv", 8),
                    kv_bool("x.flag", True),
                    kv_small_u32_array("small.arr", [1, 2, 3]),
                ]
            )
        )
        self.assertEqual(meta["general.architecture"], "qwen3")
        self.assertEqual(meta["qwen3.block_count"], 36)
        self.assertEqual(meta["qwen3.attention.head_count_kv"], 8)
        self.assertTrue(meta["x.flag"])
        self.assertEqual(meta["small.arr"], [1, 2, 3])

    def test_elides_large_arrays_but_keeps_reading(self):
        meta = self.read(
            synthetic_gguf(
                [
                    kv_big_i32_array("tokenizer.ggml.token_type", 5000),
                    kv_u32("after.big", 7),
                ]
            )
        )
        self.assertIsNone(meta["tokenizer.ggml.token_type"])
        self.assertEqual(meta["after.big"], 7)

    def test_rejects_bad_magic(self):
        with self.assertRaises(gguf.GGUFError):
            self.read(b"NOPE" + b"\x00" * 20)

    def test_rejects_truncated_file(self):
        blob = synthetic_gguf([kv_u32("a.b", 1)])
        with self.assertRaises(gguf.GGUFError):
            self.read(blob[:-2])


if __name__ == "__main__":
    unittest.main()
