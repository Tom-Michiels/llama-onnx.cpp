#!/usr/bin/env python3
"""End-to-end check of the C++-driven GGUF -> ONNX llama exporter.

Builds tiny synthetic models in numpy, writes them to temporary GGUF files,
runs the ``llama-onnx-export-dump`` binary to produce a ``.gdump``,
translates the dump to ONNX via ``gguf.gdump_to_onnx``, and finally loads
the ONNX in onnxruntime to confirm a forward pass runs end-to-end without
NaNs.

For the 20 most popular LLM/VLM families we also compare ONNX logits against
reference logits from ``llama_decode`` fixtures (where the ONNX translator
currently supports the architecture).
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf.gdump_to_onnx import convert as gdump_to_onnx
from tiny_models import (
    write_gguf, write_mmproj_gguf,
    tiny_llama, tiny_gemma, tiny_gemma2, tiny_qwen2, tiny_qwen3, tiny_qwen3moe,
    tiny_phi3, tiny_mamba, tiny_mixtral, tiny_internlm2, tiny_glm4, tiny_minicpm3,
    tiny_qwen2vl, tiny_cogvlm, tiny_minicpm, tiny_pixtral, tiny_internvl,
    tiny_qwen3vl, tiny_llama4, tiny_glm4v,
    tiny_idefics3_mmproj,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

# 20 popular LLM/VLM families: (test label, builder).
# VLMs are represented by their text (LM) backbone only.
POPULAR_MODELS = [
    # LLMs
    ("llama3",    tiny_llama),
    ("gemma",     tiny_gemma),
    ("gemma2",    tiny_gemma2),
    ("qwen2",     tiny_qwen2),
    ("qwen3",     tiny_qwen3),
    ("qwen3moe",  tiny_qwen3moe),
    ("phi3",      tiny_phi3),
    ("mamba",     tiny_mamba),
    ("mixtral",   tiny_mixtral),
    ("internlm2", tiny_internlm2),
    ("glm4",      tiny_glm4),
    ("minicpm3",  tiny_minicpm3),
    # VLM text backbones
    ("qwen2vl",   tiny_qwen2vl),
    ("qwen3vl",   tiny_qwen3vl),
    ("cogvlm",    tiny_cogvlm),
    ("minicpm-v", tiny_minicpm),
    ("pixtral",   tiny_pixtral),
    ("internvl",  tiny_internvl),
    ("llama4",    tiny_llama4),
    ("glm4v",     tiny_glm4v),
]

# Known ONNX translator / fixture-synthesis gaps (empty when all models pass).
PIPELINE_GAPS: dict[str, str] = {}
FIXTURE_GAPS: dict[str, str] = {}

# Vision encoder mmproj families (full encoder graph, not LM text backbones).
POPULAR_VISION_MMPROJ = [
    ("idefics3", tiny_idefics3_mmproj),
]
VISION_FIXTURE_GAPS: dict[str, str] = {}


def _model_id(m) -> str:
    return m.label or m.arch


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
        tag = _model_id(m)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gguf_path = tmp / f"tiny_{tag}.gguf"
            dump_path = tmp / f"tiny_{tag}.gdump"
            onnx_path = tmp / f"tiny_{tag}.onnx"
            write_gguf(gguf_path, m)

            env = os.environ.copy()
            env.setdefault("LD_LIBRARY_PATH", str(self.dumper.parent))
            subprocess.run(
                [str(self.dumper), "-m", str(gguf_path), "-o", str(dump_path)],
                env=env, check=True,
            )
            self.assertTrue(dump_path.is_file())

            # fp32 weights avoid mixed-precision shape bugs in several arches.
            gdump_to_onnx(dump_path, onnx_path, weight_dtype="float32")
            self.assertTrue(onnx_path.is_file())

            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self.assertIn("logits", {o.name for o in sess.get_outputs()})

            feeds = {}
            for inp in sess.get_inputs():
                dt = {"tensor(int64)": np.int64, "tensor(int32)": np.int32,
                      "tensor(float16)": np.float16, "tensor(float)": np.float32}.get(inp.type, np.float32)
                feeds[inp.name] = np.zeros(inp.shape, dtype=dt)
            outs = sess.run(None, feeds)
            logits = outs[0]
            self.assertFalse(np.any(np.isnan(logits)), f"{tag}: logits contain NaN")
            self.assertFalse(np.any(np.isinf(logits)), f"{tag}: logits contain Inf")

    def _run_fixture_comparison(self, model_builder, *, max_abs_diff=5e-3, min_cos=0.999):
        """Run ONNX against the same inputs as a llama_decode fixture."""
        from gguf import gdump
        from gguf.gdump_fixture import load as load_fx, synthesize_inputs

        m = model_builder()
        tag = _model_id(m)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gguf_path = tmp / f"tiny_{tag}.gguf"
            dump_path = tmp / f"tiny_{tag}.gdump"
            fx_path   = tmp / f"tiny_{tag}.gfxt"
            onnx_path = tmp / f"tiny_{tag}.onnx"

            write_gguf(gguf_path, m)

            env = os.environ.copy()
            env.setdefault("LD_LIBRARY_PATH", str(self.dumper.parent))
            subprocess.run([str(self.dumper), "-m", str(gguf_path),
                            "-o", str(dump_path), "--fixture", str(fx_path)],
                           env=env, check=True, capture_output=True)
            self.assertTrue(fx_path.is_file())

            gdump_to_onnx(dump_path, onnx_path, weight_dtype="float32")
            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            d = gdump.load(dump_path)
            fx = load_fx(fx_path)
            feeds = synthesize_inputs(d, fx, [i.name for i in sess.get_inputs()])
            (onnx_logits,) = sess.run(["logits"], feeds)
            self.assertEqual(onnx_logits.shape, fx.logits.shape)

            from numpy.linalg import norm
            cos = (onnx_logits * fx.logits).sum(axis=-1) / (norm(onnx_logits, axis=-1) * norm(fx.logits, axis=-1) + 1e-12)
            self.assertGreater(cos.min(), min_cos,
                f"{tag}: per-row cosine min too low: {cos.min():.4f}")
            self.assertLess(np.abs(onnx_logits - fx.logits).max(), max_abs_diff,
                f"{tag}: max abs diff between ONNX and llama_decode logits exceeds {max_abs_diff}")

    def _run_vision_fixture_comparison(self, mmproj_builder, *, max_abs_diff=5e-3, min_cos=0.999):
        """Compare ONNX vision encoder output against clip CPU reference features."""
        from gguf import gdump
        from gguf.gdump_fixture import load as load_fx, synthesize_vision_inputs

        m = mmproj_builder()
        tag = m.label or "vision"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gguf_path = tmp / f"tiny_{tag}.gguf"
            dump_path = tmp / f"tiny_{tag}.vision.gdump"
            fx_path   = tmp / f"tiny_{tag}.gfxt"
            onnx_path = tmp / f"tiny_{tag}.vision.onnx"

            write_mmproj_gguf(gguf_path, m)

            env = os.environ.copy()
            env.setdefault("LD_LIBRARY_PATH", str(self.dumper.parent))
            subprocess.run(
                [str(self.dumper), "-m", str(gguf_path), "-o", "/dev/null",
                 "--vision-only", "--vision-dump", str(dump_path),
                 "--vision-fixture", str(fx_path)],
                env=env, check=True, capture_output=True,
            )
            self.assertTrue(dump_path.is_file())
            self.assertTrue(fx_path.is_file())

            gdump_to_onnx(dump_path, onnx_path, weight_dtype="float32")
            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            d = gdump.load(dump_path)
            fx = load_fx(fx_path)
            self.assertEqual(fx.kind, "vision")
            feeds = synthesize_vision_inputs(d, fx, [i.name for i in sess.get_inputs()])
            out_names = [o.name for o in sess.get_outputs()]
            out_name = fx.output_name if fx.output_name in out_names else out_names[0]
            (onnx_out,) = sess.run([out_name], feeds)
            ref = fx.output_features
            self.assertEqual(onnx_out.size, ref.size)

            a = onnx_out.reshape(-1)
            b = ref.reshape(-1)
            from numpy.linalg import norm
            cos = float((a * b).sum() / (norm(a) * norm(b) + 1e-12))
            self.assertGreater(cos, min_cos, f"{tag}: vision cosine too low: {cos:.4f}")
            self.assertLess(np.abs(a - b).max(), max_abs_diff,
                f"{tag}: vision max abs diff exceeds {max_abs_diff}")

    def test_llama_quantised_q4_0(self):
        """Round-trip a quantised GGUF through dequantise + fp16 ONNX."""
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime not installed")
        quantize = REPO_ROOT / "build" / "bin" / "llama-quantize"
        if not quantize.is_file():
            self.skipTest("llama-quantize not built")

        from tiny_models import TinyModel, _llama_like_weights
        rng = np.random.default_rng(7)
        h = dict(n_layer=2, n_embd=64, n_ff=128, n_head=4, n_head_kv=2,
                 head_dim=16, vocab_size=128, rms_eps=1e-5, rope_base=10000.0)
        m = TinyModel(arch="llama", hparams=h,
                      weights=_llama_like_weights(rng, h, np.float32))

        from gguf import gdump

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            f32_path = tmp / "llama_med.gguf"
            q4_path  = tmp / "llama_med_q4.gguf"
            dump_path = tmp / "llama_med_q4.gdump"
            onnx_path = tmp / "llama_med_q4.onnx"

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
            for t in d.tensors:
                if t.has_data and t.data is not None:
                    self.assertFalse(np.any(np.isnan(t.data)),
                                     f"dequantised {t.name} contains NaN")

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            feeds = {}
            for inp in sess.get_inputs():
                dt = {"tensor(int64)": np.int64, "tensor(int32)": np.int32,
                      "tensor(float16)": np.float16, "tensor(float)": np.float32}.get(
                          inp.type, np.float32)
                feeds[inp.name] = np.zeros(inp.shape, dtype=dt)
            outs = sess.run(["logits"], feeds)
            self.assertFalse(np.any(np.isnan(outs[0])), "Q4_0 logits contain NaN")

            import onnx as _onnx
            model = _onnx.load(str(onnx_path), load_external_data=False)
            md = {e.key: e.value for e in model.metadata_props}
            self.assertIn("llama_onnx.original_dtype_counts", md)
            self.assertIn("Q4_0", md["llama_onnx.original_dtype_counts"])
            doc_strings = [tp.doc_string for tp in model.graph.initializer]
            self.assertTrue(any("Q4_0" in s for s in doc_strings),
                            "expected at least one initializer tagged with original_ggml_type=Q4_0")


def _run_maybe_gap(testcase, label, gaps, fn):
    gap = gaps.get(label)
    if gap:
        with testcase.assertRaises(Exception, msg=f"{label}: expected translator gap ({gap})"):
            fn()
    else:
        fn()


def _make_popular_pipeline_test(label, builder):
    def test(self):
        _run_maybe_gap(self, label, PIPELINE_GAPS, lambda: self._run_pipeline(builder))
    test.__doc__ = f"GGUF -> ONNX pipeline for popular model family: {label}"
    return test


def _make_popular_fixture_test(label, builder):
    def test(self):
        _run_maybe_gap(self, label, FIXTURE_GAPS, lambda: self._run_fixture_comparison(builder))
    test.__doc__ = f"ONNX vs llama_decode logits for popular model family: {label}"
    return test


def _make_vision_fixture_test(label, builder):
    def test(self):
        _run_maybe_gap(self, label, VISION_FIXTURE_GAPS,
                       lambda: self._run_vision_fixture_comparison(builder))
    test.__doc__ = f"ONNX vs clip vision features for mmproj family: {label}"
    return test


for _label, _builder in POPULAR_MODELS:
    setattr(
        TestOnnxExportPipeline,
        f"test_popular_pipeline_{_label}",
        _make_popular_pipeline_test(_label, _builder),
    )
    setattr(
        TestOnnxExportPipeline,
        f"test_popular_fixture_{_label}",
        _make_popular_fixture_test(_label, _builder),
    )

for _label, _builder in POPULAR_VISION_MMPROJ:
    setattr(
        TestOnnxExportPipeline,
        f"test_vision_fixture_{_label}",
        _make_vision_fixture_test(_label, _builder),
    )


if __name__ == "__main__":
    unittest.main()
