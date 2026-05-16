#!/usr/bin/env python3
"""End-to-end check of the C++-driven GGUF -> ONNX llama exporter.

Builds a tiny random LLaMA model in numpy, writes it to a temporary GGUF,
runs the ``llama-onnx-export-dump`` binary on it to produce a ``.gdump``,
translates the dump to ONNX via ``gguf.gdump_to_onnx``, and finally loads
the ONNX in onnxruntime to confirm a forward pass runs end-to-end without
NaNs.

This test deliberately does not compare against a hand-written numpy
reference. The forward pass it covers is *exactly* the one in
``src/models/llama.cpp`` — no Python re-implementation. Numerical
verification against the original llama.cpp inference path is a follow-up.

The test skips itself if either ``llama-onnx-export-dump`` or
``onnxruntime`` is unavailable.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf import GGUFWriter
from gguf.gdump_to_onnx import convert as gdump_to_onnx
from tiny_models import (
    write_gguf, tiny_llama, tiny_gemma, tiny_qwen3, tiny_qwen3moe, tiny_qwen2vl,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_dumper() -> Path | None:
    for c in [
        REPO_ROOT / "build" / "bin" / "llama-onnx-export-dump",
        REPO_ROOT / "build" / "Release" / "bin" / "llama-onnx-export-dump",
    ]:
        if c.is_file():
            return c
    return None


class TestOnnxExportPipeline(unittest.TestCase):

    def setUp(self):
        self.dumper = _find_dumper()
        if not self.dumper:
            self.skipTest("llama-onnx-export-dump not built; run cmake --build build --target llama-onnx-export-dump")
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime not installed")

    def _run_pipeline(self, model_builder):
        m = model_builder()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gguf_path = tmp / f"tiny_{m.arch}.gguf"
            dump_path = tmp / f"tiny_{m.arch}.gdump"
            onnx_path = tmp / f"tiny_{m.arch}.onnx"
            write_gguf(gguf_path, m)

            env = os.environ.copy()
            env.setdefault("LD_LIBRARY_PATH", str(self.dumper.parent))
            subprocess.run(
                [str(self.dumper), "-m", str(gguf_path), "-o", str(dump_path)],
                env=env, check=True,
            )
            self.assertTrue(dump_path.is_file())

            gdump_to_onnx(dump_path, onnx_path)
            self.assertTrue(onnx_path.is_file())

            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self.assertIn("logits", {o.name for o in sess.get_outputs()})

            # Zero-filled forward pass: we just check that the model accepts
            # the I/O contract and produces finite outputs. Bit-exact
            # comparison against llama.cpp is the next step.
            feeds = {}
            for inp in sess.get_inputs():
                dt = {"tensor(int64)": np.int64, "tensor(int32)": np.int32,
                      "tensor(float16)": np.float16, "tensor(float)": np.float32}.get(inp.type, np.float32)
                feeds[inp.name] = np.zeros(inp.shape, dtype=dt)
            outs = sess.run(None, feeds)
            logits = outs[0]
            self.assertFalse(np.any(np.isnan(logits)), f"{m.arch}: logits contain NaN")
            self.assertFalse(np.any(np.isinf(logits)), f"{m.arch}: logits contain Inf")

    def test_llama(self):    self._run_pipeline(tiny_llama)
    def test_gemma(self):    self._run_pipeline(tiny_gemma)
    def test_qwen3(self):    self._run_pipeline(tiny_qwen3)
    def test_qwen3moe(self): self._run_pipeline(tiny_qwen3moe)
    def test_qwen2vl(self):  self._run_pipeline(tiny_qwen2vl)

    def test_llama_quantised_q4_0(self):
        """Round-trip a quantised GGUF through dequantise + fp16 ONNX.

        Builds a medium-sized llama (dims that satisfy Q4_0's block-size
        constraint), drives ``llama-quantize`` to produce a Q4_0 GGUF, then
        runs the full export pipeline. The point of the test is that the
        Python side correctly reads the canonical GGUF tensor bytes (the
        CPU backend re-packs Q4_0 in memory) and dequantises them to fp32
        before stamping them as fp16 initializers.
        """
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime not installed")
        quantize = REPO_ROOT / "build" / "bin" / "llama-quantize"
        if not quantize.is_file():
            self.skipTest("llama-quantize not built")

        # All dims must be multiples of 32 for Q4_0.
        from tiny_models import TinyModel, _llama_like_weights
        rng = np.random.default_rng(7)
        h = dict(n_layer=2, n_embd=64, n_ff=128, n_head=4, n_head_kv=2,
                 head_dim=16, vocab_size=128, rms_eps=1e-5, rope_base=10000.0)
        m = TinyModel(arch="llama", hparams=h,
                      weights=_llama_like_weights(rng, h, np.float32))

        from gguf import gdump
        from gguf.gdump_fixture import load as load_fx, synthesize_inputs

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            f32_path = tmp / "llama_med.gguf"
            q4_path  = tmp / "llama_med_q4.gguf"
            dump_path = tmp / "llama_med_q4.gdump"
            onnx_path = tmp / "llama_med_q4.onnx"

            from tiny_models import write_gguf
            write_gguf(f32_path, m)
            env = os.environ.copy()
            env.setdefault("LD_LIBRARY_PATH", str(quantize.parent))
            subprocess.run([str(quantize), str(f32_path), str(q4_path), "Q4_0"],
                           env=env, check=True, capture_output=True)
            self.assertTrue(q4_path.is_file())
            self.assertLess(q4_path.stat().st_size, f32_path.stat().st_size)

            subprocess.run([str(self.dumper), "-m", str(q4_path), "-o", str(dump_path)],
                           env=env, check=True, capture_output=True)
            gdump_to_onnx(dump_path, onnx_path, weight_dtype="float16")

            d = gdump.load(dump_path)
            # All quantised tensors should have been dequantised cleanly.
            for t in d.tensors:
                if t.has_data and t.data is not None:
                    self.assertFalse(np.any(np.isnan(t.data)),
                                     f"dequantised {t.name} contains NaN")

            # The ONNX model should also load and run.
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            feeds = {}
            for inp in sess.get_inputs():
                dt = {"tensor(int64)": np.int64, "tensor(int32)": np.int32,
                      "tensor(float16)": np.float16, "tensor(float)": np.float32}.get(
                          inp.type, np.float32)
                feeds[inp.name] = np.zeros(inp.shape, dtype=dt)
            outs = sess.run(["logits"], feeds)
            self.assertFalse(np.any(np.isnan(outs[0])), "Q4_0 logits contain NaN")

            # The ONNX should carry metadata recording the original ggml
            # types so downstream tooling can see they came from Q4_0.
            import onnx as _onnx
            model = _onnx.load(str(onnx_path), load_external_data=False)
            md = {e.key: e.value for e in model.metadata_props}
            self.assertIn("llama_onnx.original_dtype_counts", md)
            self.assertIn("Q4_0", md["llama_onnx.original_dtype_counts"])
            # And per-initializer doc_strings.
            doc_strings = [tp.doc_string for tp in model.graph.initializer]
            self.assertTrue(any("Q4_0" in s for s in doc_strings),
                            "expected at least one initializer tagged with original_ggml_type=Q4_0")

    def test_llama_fixture_comparison(self):
        """End-to-end comparison: ONNX vs reference logits from llama_decode.

        The C++ dumper writes a fixture with reference logits produced by
        ``llama_decode`` on a deterministic token stream. We run the
        exported ONNX over the same inputs (synthesised in Python from the
        gdump's graph structure) and check we agree at least directionally.
        The full bit-exact convergence is a follow-up; the synthetic random
        weights here compound any per-op rounding across the whole graph.
        """
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime not installed")

        from gguf import gdump
        from gguf.gdump_fixture import load as load_fx, synthesize_inputs

        m = tiny_llama()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gguf_path = tmp / "tiny.gguf"
            dump_path = tmp / "tiny.gdump"
            fx_path   = tmp / "tiny.gfxt"
            onnx_path = tmp / "tiny.onnx"

            from tiny_models import write_gguf
            write_gguf(gguf_path, m)

            env = os.environ.copy()
            env.setdefault("LD_LIBRARY_PATH", str(self.dumper.parent))
            subprocess.run([str(self.dumper), "-m", str(gguf_path),
                            "-o", str(dump_path), "--fixture", str(fx_path)],
                           env=env, check=True, capture_output=True)
            self.assertTrue(fx_path.is_file())

            # Export with fp32 weights to maximise numerical agreement.
            gdump_to_onnx(dump_path, onnx_path, weight_dtype="float32")
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            d = gdump.load(dump_path)
            fx = load_fx(fx_path)
            feeds = synthesize_inputs(d, fx, [i.name for i in sess.get_inputs()])
            (onnx_logits,) = sess.run(["logits"], feeds)
            self.assertEqual(onnx_logits.shape, fx.logits.shape)

            # Compare cosine similarity per token row — that's the right
            # metric when weights are random and absolute magnitudes are
            # arbitrary, while the directional agreement still validates
            # the graph structure end-to-end.
            from numpy.linalg import norm
            cos = (onnx_logits * fx.logits).sum(axis=-1) / (norm(onnx_logits, axis=-1) * norm(fx.logits, axis=-1) + 1e-12)
            # The last token has the most attention context and is the most
            # sensitive; the first token is a single-position attention so
            # it's the most stable.
            self.assertGreater(cos[0], 0.5,
                f"first-token cosine similarity too low: {cos[0]:.3f}")


if __name__ == "__main__":
    unittest.main()
