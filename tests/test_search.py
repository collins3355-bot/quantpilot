import math
import tempfile
import unittest
from pathlib import Path

from quantpilot.engines.llamacpp import EngineError, Perplexity, override_args
from quantpilot.report import render_search
from quantpilot.search import Probe, rank_probes, run_search, split_corpus

GB = 1024**3
HARDWARE = {"chip": "Apple M1 Max", "memory_gb": 64, "os": "Darwin"}
# Per-chunk difficulty shared by every model on the same text (sums to zero,
# so the chunk NLLs average back to ln(ppl)).
WOBBLE = [0.05, -0.05, 0.02, -0.02]


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


class TestSplitCorpus(unittest.TestCase):
    def test_halves_are_complete_and_cut_on_a_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = tmp / "wiki.test.raw"
            text = "".join(f"line {i}\n" for i in range(100))
            corpus.write_text(text)
            tune, held = split_corpus(corpus, tmp)
            self.assertEqual(tune.read_text() + held.read_text(), text)
            self.assertTrue(held.read_text().startswith("line "))
            self.assertGreater(len(tune.read_text()), len(text) // 3)
            self.assertGreater(len(held.read_text()), len(text) // 3)


class FakeEngine:
    """Deterministic quantize/perplexity standing in for llama.cpp."""

    def __init__(self):
        # sizes in GB and ppls keyed by frozenset of bumped classes (Q4_K_M base)
        # or by a uniform type name
        self.sizes = {frozenset(): 4.7, "Q6_K": 6.2}
        self.ppls = {frozenset(): 9.0, "Q6_K": 8.84}
        self.holdout_ppls = {}  # overrides on the held-out text, same keys
        self.baseline_ppl = 8.77
        self.crash_on = set()  # classes whose quantize aborts

    def quantize(self, source, dest, qtype, tensor_overrides=None):
        overrides = frozenset((tensor_overrides or {}).keys())
        if overrides & self.crash_on:
            raise EngineError("simulated abort: incompatible tensor shape")
        key = overrides if qtype == "Q4_K_M" else qtype
        dest.write_bytes(b"g" * int(self.sizes[key] * 1024))  # KB stand in for GB
        return dest

    def _key(self, path):
        if "mix" in path.name:
            n = int(path.name.split("mix")[1].split(".")[0])
            return frozenset(self.order[:n])
        if "+" in path.name:  # probe files also end in -Q6_K.gguf, so check first
            return frozenset([path.name.split("+")[1].rsplit("-", 1)[0]])
        if path.name.endswith("-Q4_K_M.gguf"):
            return frozenset()
        if path.name.endswith("-Q6_K.gguf"):
            return "Q6_K"
        return None  # the full-precision source

    def perplexity(self, path, corpus, chunks):
        key = self._key(path)
        if key is None:
            ppl = self.baseline_ppl
        elif corpus.name.endswith(".holdout.txt") and key in self.holdout_ppls:
            ppl = self.holdout_ppls[key]
        else:
            ppl = self.ppls[key]
        return Perplexity(ppl, 0.1, [math.log(ppl) + w for w in WOBBLE[:chunks]])


def _search(engine, tmp, classes):
    source = tmp / "model-bf16.gguf"
    source.write_bytes(b"g" * 16000)
    corpus = tmp / "wiki.test.raw"
    corpus.write_text("".join(f"line {i}\n" for i in range(40)))
    return run_search(
        source=source,
        corpus=corpus,
        chunks=4,
        workdir=tmp / "work",
        classes=classes,
        budget_pct=1.0,
        progress=lambda _msg: None,
        quantize_fn=engine.quantize,
        ppl_fn=engine.perplexity,
    )


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
            result = _search(engine, tmp, ["attn_q", "attn_v", "ffn_down"])

            self.assertTrue(result.met_budget)
            self.assertEqual([p.cls for p in result.probes], ["attn_v", "ffn_down"])
            self.assertEqual(result.steps[-1].classes, ["attn_v", "ffn_down"])
            self.assertAlmostEqual(result.steps[-1].dppl_pct, 0.912, places=2)
            self.assertEqual(result.tune_corpus.name, "wiki.test.tune.txt")
            # probes cleaned up; the base, the final recipe and the uniform bump remain
            remaining = sorted(f.name for f in (tmp / "work").glob("*.gguf"))
            self.assertEqual(
                remaining,
                ["model-bf16-Q4_K_M-mix2.gguf", "model-bf16-Q4_K_M.gguf", "model-bf16-Q6_K.gguf"],
            )
            self.assertIn("--tensor-type ffn_down=q6_k", result.recipe_command())
            # same behaviour on held-out text, so the recipe holds
            self.assertAlmostEqual(result.holdout.recipe.dppl_pct, 0.912, places=2)
            self.assertIn("Holds on held-out text", render_search(result, HARDWARE))

    def test_crashing_probe_is_skipped_and_reported(self):
        engine = FakeEngine()
        engine.crash_on = {"token_embd"}
        engine.sizes[frozenset(["ffn_down"])] = 5.0
        engine.ppls[frozenset(["ffn_down"])] = 8.80  # meets 1% budget alone
        engine.order = ["ffn_down"]

        with tempfile.TemporaryDirectory() as tmp:
            result = _search(engine, Path(tmp), ["token_embd", "ffn_down"])
            self.assertEqual(result.failed, ["token_embd"])
            self.assertEqual([p.cls for p in result.probes], ["ffn_down"])
            self.assertTrue(result.met_budget)

    def test_holdout_catches_a_recipe_that_only_fits_the_tuning_text(self):
        engine = FakeEngine()
        engine.sizes[frozenset(["ffn_down"])] = 5.0
        engine.ppls[frozenset(["ffn_down"])] = 8.80  # +0.34% on the tuning text
        engine.holdout_ppls[frozenset(["ffn_down"])] = 8.95  # +2.05% on held-out text
        engine.order = ["ffn_down"]

        with tempfile.TemporaryDirectory() as tmp:
            result = _search(engine, Path(tmp), ["ffn_down"])
            self.assertTrue(result.met_budget)  # judged on the tuning text
            self.assertAlmostEqual(result.holdout.recipe.dppl_pct, 2.05, places=1)
            md = render_search(result, HARDWARE)
            self.assertIn("Does not hold on held-out text", md)
            self.assertIn("worse quality", md)  # 8.95 vs. uniform Q6_K at 8.84

    def test_renders_report(self):
        md = render_search(_quick_result(), HARDWARE)
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
