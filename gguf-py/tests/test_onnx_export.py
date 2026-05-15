#!/usr/bin/env python3
"""End-to-end check of the GGUF -> ONNX llama exporter.

Builds a tiny random LLaMA model in numpy, writes it to a temporary GGUF file,
runs ``gguf.onnx_export.convert``, then runs both the exported ONNX model (via
onnxruntime) and a numpy reference implementation, and compares the logits and
the KV-cache outputs.

The reference forward pass and the ONNX graph are derived from the same spec
(see ``src/models/llama.cpp``), so this test mostly catches regressions in
shape handling, RoPE sign conventions and KV-cache wiring.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Allow running the test from a source checkout without installing gguf.
if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gguf
from gguf import GGMLQuantizationType, GGUFWriter
from gguf.onnx_export import convert


# ---------------------------------------------------------------------------
# Numpy reference
# ---------------------------------------------------------------------------


def rms_norm_np(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x32 = x.astype(np.float32)
    var = (x32 * x32).mean(axis=-1, keepdims=True)
    out = x32 / np.sqrt(var + eps)
    return (out.astype(x.dtype)) * weight


def silu_np(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(x.dtype)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    e = np.exp((x - m).astype(np.float64))
    return (e / e.sum(axis=axis, keepdims=True)).astype(x.dtype)


def rope_norm_np(
    x: np.ndarray,
    position_ids: np.ndarray,
    rope_base: float,
    rope_scale: float,
    rope_factors: np.ndarray | None,
) -> np.ndarray:
    """NORM-style (consecutive-pair) RoPE; mirrors llama.cpp's NORMAL mode."""
    B, S, H, D = x.shape
    assert D % 2 == 0
    inv_freqs = 1.0 / (rope_base ** (np.arange(0, D, 2, dtype=np.float64) / D))
    inv_freqs = inv_freqs * rope_scale
    if rope_factors is not None:
        inv_freqs = inv_freqs / rope_factors.astype(np.float64)
    angles = position_ids[..., None].astype(np.float64) * inv_freqs[None, None, :]
    cos = np.cos(angles)[:, :, None, :]
    sin = np.sin(angles)[:, :, None, :]
    x_p = x.reshape(B, S, H, D // 2, 2).astype(np.float64)
    x_even = x_p[..., 0]
    x_odd = x_p[..., 1]
    new_even = x_even * cos - x_odd * sin
    new_odd = x_even * sin + x_odd * cos
    out = np.stack([new_even, new_odd], axis=-1).reshape(B, S, H, D)
    return out.astype(x.dtype)


class Hparams:
    def __init__(self, n_layer, n_embd, n_ff, n_head, n_head_kv, head_dim,
                 vocab_size, rms_eps, rope_base, rope_scale=1.0, rope_factors=None):
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_ff = n_ff
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.rms_eps = rms_eps
        self.rope_base = rope_base
        self.rope_scale = rope_scale
        self.rope_factors = rope_factors


def numpy_llama_forward(weights, h: Hparams, input_ids, position_ids, past_kvs):
    B, S = input_ids.shape
    hidden = weights["token_embd.weight"][input_ids]  # [B, S, n_embd]
    new_kvs = []
    for il in range(h.n_layer):
        normed = rms_norm_np(hidden, weights[f"blk.{il}.attn_norm.weight"], h.rms_eps)
        q = normed @ weights[f"blk.{il}.attn_q.weight"].T
        k = normed @ weights[f"blk.{il}.attn_k.weight"].T
        v = normed @ weights[f"blk.{il}.attn_v.weight"].T
        q = q.reshape(B, S, h.n_head, h.head_dim)
        k = k.reshape(B, S, h.n_head_kv, h.head_dim)
        v = v.reshape(B, S, h.n_head_kv, h.head_dim)
        q = rope_norm_np(q, position_ids, h.rope_base, h.rope_scale, h.rope_factors)
        k = rope_norm_np(k, position_ids, h.rope_base, h.rope_scale, h.rope_factors)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        past_k, past_v = past_kvs[il]
        k = np.concatenate([past_k, k], axis=2)
        v = np.concatenate([past_v, v], axis=2)
        new_kvs.append((k.copy(), v.copy()))
        # GQA: repeat along head axis.
        if h.n_head_kv != h.n_head:
            n_rep = h.n_head // h.n_head_kv
            k = np.repeat(k, n_rep, axis=1)
            v = np.repeat(v, n_rep, axis=1)
        # Attention.
        scale = 1.0 / np.sqrt(h.head_dim)
        scores = q @ k.transpose(0, 1, 3, 2) * scale
        past_len = past_k.shape[2]
        total_len = k.shape[2]
        q_pos = np.arange(past_len, total_len)
        k_pos = np.arange(total_len)
        mask = (k_pos[None, :] > q_pos[:, None]).astype(scores.dtype) * np.array(-1e30, dtype=scores.dtype)
        scores = scores + mask[None, None]
        attn = softmax_np(scores, axis=-1)
        ctx = attn @ v  # [B, H, S, D]
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, S, h.n_head * h.head_dim)
        attn_out = ctx @ weights[f"blk.{il}.attn_output.weight"].T
        hidden = hidden + attn_out
        ffn_in = rms_norm_np(hidden, weights[f"blk.{il}.ffn_norm.weight"], h.rms_eps)
        gate = ffn_in @ weights[f"blk.{il}.ffn_gate.weight"].T
        up = ffn_in @ weights[f"blk.{il}.ffn_up.weight"].T
        ffn = silu_np(gate) * up
        ffn = ffn @ weights[f"blk.{il}.ffn_down.weight"].T
        hidden = hidden + ffn
    hidden = rms_norm_np(hidden, weights["output_norm.weight"], h.rms_eps)
    lm_head = weights.get("output.weight", weights["token_embd.weight"])
    logits = hidden @ lm_head.T
    return logits, new_kvs


# ---------------------------------------------------------------------------
# GGUF generation
# ---------------------------------------------------------------------------


def write_tiny_llama_gguf(path: Path, h: Hparams, weights: dict, *, dtype: np.dtype):
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

    for name, arr in weights.items():
        arr_cast = arr.astype(dtype)
        w.add_tensor(name, arr_cast)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def make_random_weights(
    h: Hparams,
    *,
    seed: int = 0,
    dtype=np.float32,
    include_output: bool = False,
    include_rope_freqs: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)
    # Small values to keep softmax / silu numerically tame.
    def rand(shape):
        return (rng.standard_normal(shape).astype(np.float32) * 0.05).astype(dtype)
    weights = {
        "token_embd.weight": rand((h.vocab_size, h.n_embd)),
        "output_norm.weight": np.ones((h.n_embd,), dtype=dtype),
    }
    if include_output:
        weights["output.weight"] = rand((h.vocab_size, h.n_embd))
    if include_rope_freqs:
        assert h.rope_factors is not None
        weights["rope_freqs.weight"] = h.rope_factors.astype(np.float32)
    for il in range(h.n_layer):
        weights[f"blk.{il}.attn_norm.weight"] = np.ones((h.n_embd,), dtype=dtype)
        weights[f"blk.{il}.attn_q.weight"]   = rand((h.n_head * h.head_dim, h.n_embd))
        weights[f"blk.{il}.attn_k.weight"]   = rand((h.n_head_kv * h.head_dim, h.n_embd))
        weights[f"blk.{il}.attn_v.weight"]   = rand((h.n_head_kv * h.head_dim, h.n_embd))
        weights[f"blk.{il}.attn_output.weight"] = rand((h.n_embd, h.n_head * h.head_dim))
        weights[f"blk.{il}.ffn_norm.weight"] = np.ones((h.n_embd,), dtype=dtype)
        weights[f"blk.{il}.ffn_gate.weight"] = rand((h.n_ff, h.n_embd))
        weights[f"blk.{il}.ffn_up.weight"]   = rand((h.n_ff, h.n_embd))
        weights[f"blk.{il}.ffn_down.weight"] = rand((h.n_embd, h.n_ff))
    return weights


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOnnxExport(unittest.TestCase):

    def _run_e2e(
        self,
        h: Hparams,
        *,
        dtype=np.float32,
        include_output: bool = False,
        include_rope_freqs: bool = False,
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime not installed")

        weights = make_random_weights(
            h, seed=42, dtype=dtype,
            include_output=include_output,
            include_rope_freqs=include_rope_freqs,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gguf_path = tmp / "tiny.gguf"
            onnx_path = tmp / "tiny.onnx"
            write_tiny_llama_gguf(gguf_path, h, weights, dtype=dtype)
            convert(gguf_path, onnx_path, dtype="float32")

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

            # Prefill: 5 tokens, batch=2.
            B, S = 2, 5
            rng = np.random.default_rng(123)
            input_ids = rng.integers(0, h.vocab_size, size=(B, S)).astype(np.int64)
            position_ids = np.broadcast_to(np.arange(S, dtype=np.int64), (B, S)).copy()
            empty_kv = np.zeros((B, h.n_head_kv, 0, h.head_dim), dtype=np.float32)
            past_kvs_np = [(empty_kv.copy(), empty_kv.copy()) for _ in range(h.n_layer)]

            feeds = {"input_ids": input_ids, "position_ids": position_ids}
            for il in range(h.n_layer):
                feeds[f"past_key_values.{il}.key"] = empty_kv
                feeds[f"past_key_values.{il}.value"] = empty_kv

            out_names = ["logits"] + [
                f"present.{il}.{which}" for il in range(h.n_layer) for which in ("key", "value")
            ]
            ort_outputs = sess.run(out_names, feeds)
            logits_ort = ort_outputs[0]
            present_ort = []
            for il in range(h.n_layer):
                k = ort_outputs[1 + 2 * il]
                v = ort_outputs[1 + 2 * il + 1]
                present_ort.append((k, v))

            # Reference forward pass.
            weights_ref = {name: arr.astype(np.float32) for name, arr in weights.items()}
            logits_ref, present_ref = numpy_llama_forward(weights_ref, h, input_ids, position_ids, past_kvs_np)

            # The graph runs in fp32 and the reference also runs in fp32, so they
            # should be very close (down to floating-point reduction order).
            np.testing.assert_allclose(logits_ort, logits_ref, atol=2e-4, rtol=2e-4)
            for il in range(h.n_layer):
                np.testing.assert_allclose(present_ort[il][0], present_ref[il][0], atol=2e-4, rtol=2e-4)
                np.testing.assert_allclose(present_ort[il][1], present_ref[il][1], atol=2e-4, rtol=2e-4)

            # Now test a second decode step that consumes the cache from the
            # prefill above. Append one more token at position S for each batch.
            new_id = rng.integers(0, h.vocab_size, size=(B, 1)).astype(np.int64)
            new_pos = np.full((B, 1), S, dtype=np.int64)
            feeds2 = {"input_ids": new_id, "position_ids": new_pos}
            for il in range(h.n_layer):
                feeds2[f"past_key_values.{il}.key"] = present_ort[il][0]
                feeds2[f"past_key_values.{il}.value"] = present_ort[il][1]
            ort_outputs2 = sess.run(out_names, feeds2)
            logits_ort2 = ort_outputs2[0]

            # Reference with the cached state.
            logits_ref2, _ = numpy_llama_forward(weights_ref, h, new_id, new_pos, present_ref)
            np.testing.assert_allclose(logits_ort2, logits_ref2, atol=2e-4, rtol=2e-4)

            # Sanity: prefill the same sequence in one go and compare the last
            # token's logits to the cached-decode result. This validates that
            # the KV cache reproduces the no-cache forward pass.
            full_ids = np.concatenate([input_ids, new_id], axis=1)
            full_pos = np.concatenate([position_ids, new_pos], axis=1)
            feeds_full = {"input_ids": full_ids, "position_ids": full_pos}
            for il in range(h.n_layer):
                feeds_full[f"past_key_values.{il}.key"] = empty_kv
                feeds_full[f"past_key_values.{il}.value"] = empty_kv
            ort_full = sess.run(["logits"], feeds_full)[0]
            np.testing.assert_allclose(ort_full[:, -1:, :], logits_ort2, atol=5e-4, rtol=5e-4)

    def test_small_mha(self):
        # MHA (n_head == n_head_kv) is the simpler case; sanity-check it first.
        h = Hparams(
            n_layer=2, n_embd=16, n_ff=32,
            n_head=4, n_head_kv=4, head_dim=4,
            vocab_size=32, rms_eps=1e-5, rope_base=10000.0,
        )
        self._run_e2e(h)

    def test_small_gqa(self):
        # GQA layout like Llama-3 family (n_head=8, n_head_kv=2 here).
        h = Hparams(
            n_layer=2, n_embd=16, n_ff=32,
            n_head=8, n_head_kv=2, head_dim=4,
            vocab_size=32, rms_eps=1e-5, rope_base=500000.0,
        )
        self._run_e2e(h)

    def test_small_explicit_lm_head(self):
        # An explicit output.weight (not tied to token_embd).
        h = Hparams(
            n_layer=2, n_embd=16, n_ff=32,
            n_head=4, n_head_kv=2, head_dim=4,
            vocab_size=32, rms_eps=1e-5, rope_base=10000.0,
        )
        self._run_e2e(h, include_output=True)

    def test_small_llama3_rope_scaling(self):
        # Llama-3 style frequency rescaling via rope_freqs.weight. The factors
        # below are arbitrary positive values; what matters is that the same
        # vector is fed to both the numpy reference and the ONNX graph.
        head_dim = 8
        rope_factors = np.array([1.0, 1.0, 2.0, 4.0], dtype=np.float32)
        h = Hparams(
            n_layer=2, n_embd=16, n_ff=32,
            n_head=4, n_head_kv=2, head_dim=head_dim,
            vocab_size=32, rms_eps=1e-5, rope_base=500000.0,
            rope_factors=rope_factors,
        )
        self._run_e2e(h, include_rope_freqs=True)


if __name__ == "__main__":
    unittest.main()
