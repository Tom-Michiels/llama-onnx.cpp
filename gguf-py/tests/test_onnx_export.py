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

from gguf import GGUFWriter
from gguf.gdump_to_onnx import convert as gdump_to_onnx


REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_dumper() -> Path | None:
    for c in [
        REPO_ROOT / "build" / "bin" / "llama-onnx-export-dump",
        REPO_ROOT / "build" / "Release" / "bin" / "llama-onnx-export-dump",
    ]:
        if c.is_file():
            return c
    return None


class Hparams:
    def __init__(self, n_layer, n_embd, n_ff, n_head, n_head_kv, head_dim,
                 vocab_size, rms_eps, rope_base):
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_ff = n_ff
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.rms_eps = rms_eps
        self.rope_base = rope_base


def make_random_weights(h: Hparams, *, seed=0, dtype=np.float32) -> dict:
    rng = np.random.default_rng(seed)
    def rand(shape):
        return (rng.standard_normal(shape).astype(np.float32) * 0.05).astype(dtype)
    weights = {
        "token_embd.weight":   rand((h.vocab_size, h.n_embd)),
        "output_norm.weight":  np.ones((h.n_embd,), dtype=dtype),
    }
    for il in range(h.n_layer):
        weights[f"blk.{il}.attn_norm.weight"]   = np.ones((h.n_embd,), dtype=dtype)
        weights[f"blk.{il}.attn_q.weight"]      = rand((h.n_head * h.head_dim, h.n_embd))
        weights[f"blk.{il}.attn_k.weight"]      = rand((h.n_head_kv * h.head_dim, h.n_embd))
        weights[f"blk.{il}.attn_v.weight"]      = rand((h.n_head_kv * h.head_dim, h.n_embd))
        weights[f"blk.{il}.attn_output.weight"] = rand((h.n_embd, h.n_head * h.head_dim))
        weights[f"blk.{il}.ffn_norm.weight"]    = np.ones((h.n_embd,), dtype=dtype)
        weights[f"blk.{il}.ffn_gate.weight"]    = rand((h.n_ff, h.n_embd))
        weights[f"blk.{il}.ffn_up.weight"]      = rand((h.n_ff, h.n_embd))
        weights[f"blk.{il}.ffn_down.weight"]    = rand((h.n_embd, h.n_ff))
    return weights


def write_tiny_llama_gguf(path: Path, h: Hparams, weights: dict, *, dtype=np.float32):
    w = GGUFWriter(path, "llama")
    w.add_block_count(h.n_layer)
    w.add_embedding_length(h.n_embd)
    w.add_feed_forward_length(h.n_ff)
    w.add_head_count(h.n_head)
    w.add_head_count_kv(h.n_head_kv)
    w.add_key_length(h.head_dim)
    w.add_value_length(h.head_dim)
    w.add_layer_norm_rms_eps(h.rms_eps)
    w.add_rope_dimension_count(h.head_dim)
    w.add_rope_freq_base(h.rope_base)
    w.add_vocab_size(h.vocab_size)
    w.add_context_length(64)
    # llama.cpp's vocab loader insists on tokenizer.ggml.model being present;
    # "none" tells it to skip tokenizer construction entirely.
    w.add_tokenizer_model("none")
    for name, arr in weights.items():
        w.add_tensor(name, arr.astype(dtype))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


class TestOnnxExportPipeline(unittest.TestCase):

    def setUp(self):
        self.dumper = _find_dumper()
        if not self.dumper:
            self.skipTest("llama-onnx-export-dump not built; run cmake --build build --target llama-onnx-export-dump")
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime not installed")

    def _run_pipeline(self, h: Hparams):
        weights = make_random_weights(h, seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gguf_path = tmp / "tiny.gguf"
            dump_path = tmp / "tiny.gdump"
            onnx_path = tmp / "tiny.onnx"
            write_tiny_llama_gguf(gguf_path, h, weights)

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

            # Build zero-filled inputs and run a forward pass; we just check
            # that the model accepts the I/O contract and produces finite
            # outputs. Bitwise-against-llama.cpp comparison is a follow-up.
            feeds = {}
            for inp in sess.get_inputs():
                dt = {"tensor(int64)": np.int64, "tensor(int32)": np.int32,
                      "tensor(float16)": np.float16, "tensor(float)": np.float32}.get(inp.type, np.float32)
                feeds[inp.name] = np.zeros(inp.shape, dtype=dt)
            outs = sess.run(None, feeds)
            logits = outs[0]
            self.assertFalse(np.any(np.isnan(logits)), "logits contain NaN")
            self.assertFalse(np.any(np.isinf(logits)), "logits contain Inf")

    def test_llama_small_gqa(self):
        h = Hparams(
            n_layer=2, n_embd=16, n_ff=32,
            n_head=4, n_head_kv=2, head_dim=4,
            vocab_size=32, rms_eps=1e-5, rope_base=10000.0,
        )
        self._run_pipeline(h)


if __name__ == "__main__":
    unittest.main()
