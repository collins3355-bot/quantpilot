import tempfile
import unittest
from pathlib import Path

from quantpilot.engines.llamacpp import override_args
from quantpilot.report import render_search
from quantpilot.search import Probe, rank_probes, run_search

GB = 1024**3


class TestOverrideArgs(unittest.TestCase):
    def test_regular_class_uses_tensor_type(self):
        self.assertEqual(override_args({"ffn_down": "Q6_K"}), ["--tensor-type", "ffn_down=q6_k"])

    def test_embeddings_and_output_use_dedicated_flags(self):
        args = override_args({"token_embd": "Q8_0", "output": "Q6_K"})
        self.assertEqual(
            args, ["--token-embedding-type", "q8_0", "--output-tensor-type", "q6_k"]
        )

    def test_none_is_empty(self):
        self.assertEqual(override_args(None), [])


def probe(cls, delta_gb, recovered):
    return Probe(
        cls=cls,
        size_bytes=int(5 * GB + delta_gb * GB),
        size_delta=int(delta_gb * GB),
        ppl=9.0 - recovered,
        ppl_recovered=recovered,
    )


class TestRanking(unittest.TestCase):
    def test_ranks_by_recovery_per_gb_and_drops_useless(self):
        cheap_win = probe("attn_v", 0.1, 0.05)  # 0.5/GB
        big_win = probe("ffn_down", 0.4, 0.08)  # 0.2/GB
        harmful = probe("attn_q", 0.2, -0.01)
        ranked = rank_probes([big_win, harmful, cheap_win])
        self.assertEqual([p.cls for p in ranked], ["attn_v", "ffn_down"])


class FakeEngine:
    """Deterministic quantize/perplexity standing in for llama.cpp."""

    def __init__(self):
        # sizes in GB and ppls keyed by frozenset of bumped classes
        self.sizes = {frozenset(): 4.7}
        self.ppls = {frozenset(): 9.0}
        self.baseline_ppl = 8.77
        self.crash_on = set()  # classes whose quantize aborts

    def quantize(self, source, dest, qtype, tensor_overrides=None):
        overrides = frozenset((tensor_overrides or {}).keys())
        if overrides & self.crash_on:
            from quantpilot.engines.llamacpp import EngineError

            raise EngineError("simulated abort: incompatible tensor shape")
        dest.write_bytes(b"g" * int(self.sizes[overrides] * 1024))  # KB stand in for GB
        return dest

    def perplexity(self, path, corpus, chunks):
        if "mix" in path.name:
            n = int(path.name.split("mix")[1].split(".")[0])
            key = frozenset(self.order[:n])
        elif "+" in path.name:
            cls = path.name.split("+")[1].rsplit("-", 1)[0]
            key = frozenset([cls])
        elif path.name.endswith("-Q4_K_M.gguf"):
            key = frozenset()
        else:
            return (self.baseline_ppl, 0.1)
        return (self.ppls[key], 0.1)


class TestGreedySearch(unittest.TestCase):
    def test_composes_until_budget_met(self):
        engine = FakeEngine()
        # probe outcomes: ffn_down recovers most per GB, then attn_v; attn_q hurts
        engine.sizes[frozenset(["ffn_down"])] = 5.0
        engine.ppls[frozenset(["ffn_down"])] = 8.92  # recovered 0.08 over 0.3GB
        engine.sizes[frozenset(["attn_v"])] = 4.8
        engine.ppls[frozenset(["attn_v"])] = 8.97  # recovered 0.03 over 0.1GB -> best per GB
        engine.sizes[frozenset(["attn_q"])] = 4.9
        engine.ppls[frozenset(["attn_q"])] = 9.02  # harmful
        # composed: attn_v alone misses 1% budget (8.97 -> +2.28%), +ffn_down meets it
        engine.order = ["attn_v", "ffn_down"]
        engine.sizes[frozenset(["attn_v", "ffn_down"])] = 5.1
        engine.ppls[frozenset(["attn_v", "ffn_down"])] = 8.85  # +0.91% vs 8.77

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "model-bf16.gguf"
            source.write_bytes(b"g" * 16000)
            corpus = tmp / "wiki.test.raw"
            corpus.write_text("hello")
            result = run_search(
                source=source,
                corpus=corpus,
                chunks=4,
                workdir=tmp / "work",
                classes=["attn_q", "attn_v", "ffn_down"],
                budget_pct=1.0,
                progress=lambda _msg: None,
                quantize_fn=engine.quantize,
                ppl_fn=engine.perplexity,
            )

            self.assertTrue(result.met_budget)
            self.assertEqual([p.cls for p in result.probes], ["attn_v", "ffn_down"])
            self.assertEqual(result.steps[-1].classes, ["attn_v", "ffn_down"])
            self.assertAlmostEqual(result.steps[-1].dppl_pct, 0.912, places=2)
            # probe artifacts cleaned up; only the final composed file remains
            remaining = sorted(f.name for f in (tmp / "work").glob("*.gguf"))
            self.assertEqual(remaining, ["model-bf16-Q4_K_M-mix2.gguf", "model-bf16-Q4_K_M.gguf"])
            self.assertIn("--tensor-type ffn_down=q6_k", result.recipe_command())

    def test_crashing_probe_is_skipped_and_reported(self):
        engine = FakeEngine()
        engine.crash_on = {"token_embd"}
        engine.sizes[frozenset(["ffn_down"])] = 5.0
        engine.ppls[frozenset(["ffn_down"])] = 8.80  # meets 1% budget alone
        engine.order = ["ffn_down"]
        engine.sizes[frozenset(["ffn_down"])] = 5.0

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "model-bf16.gguf"
            source.write_bytes(b"g" * 16000)
            corpus = tmp / "wiki.test.raw"
            corpus.write_text("hello")
            result = run_search(
                source=source,
                corpus=corpus,
                chunks=4,
                workdir=tmp / "work",
                classes=["token_embd", "ffn_down"],
                budget_pct=1.0,
                progress=lambda _msg: None,
                quantize_fn=engine.quantize,
                ppl_fn=engine.perplexity,
            )
            self.assertEqual(result.failed, ["token_embd"])
            self.assertEqual([p.cls for p in result.probes], ["ffn_down"])
            self.assertTrue(result.met_budget)

    def test_renders_report(self):
        md = render_search(
            _quick_result(), {"chip": "Apple M1 Max", "memory_gb": 64, "os": "Darwin"}
        )
        self.assertIn("Sensitivity probes", md)
        self.assertIn("| ffn_down | +307 MB", md)
        self.assertIn("Met the 1% budget", md)
        self.assertIn("--tensor-type ffn_down=q6_k", md)


def _quick_result():
    from quantpilot.search import SearchResult, Step

    return SearchResult(
        source=Path("/m/model-bf16.gguf"),
        base="Q4_K_M",
        bump="Q6_K",
        budget_pct=1.0,
        corpus=Path("/d/wiki.test.raw"),
        chunks=32,
        baseline_ppl=8.77,
        base_ppl=9.0,
        base_size=int(4.7 * GB),
        probes=[probe("ffn_down", 0.3, 0.08)],
        steps=[Step(classes=["ffn_down"], size_bytes=int(5 * GB), ppl=8.85, dppl_pct=0.91)],
        final_path=Path("/w/mix1.gguf"),
        met_budget=True,
    )


if __name__ == "__main__":
    unittest.main()
