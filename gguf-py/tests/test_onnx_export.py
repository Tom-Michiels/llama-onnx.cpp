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


if __name__ == "__main__":
    unittest.main()
