#!/usr/bin/env python3
"""Phase-A handler math tests against numpy reference implementations.

Promoted from /tmp/handler_math_tests.py to live in gguf-py/tests/ so it
runs under CI alongside the other tests.

These verify the formulas inside the handlers are equivalent to ggml's
canonical implementation (translated from ggml/src/ggml-cpu/ops.cpp).
We don't load real GGUFs here — just exercise the math directly, so no
onnxruntime dependency.
"""
import math
import unittest

import numpy as np


# -------------------------------------------------------------------- L2_NORM
def ggml_l2_norm_ref(x: np.ndarray, eps: float) -> np.ndarray:
    """Match ggml_compute_forward_l2_norm_f32."""
    sumsq = (x * x).sum(axis=-1, keepdims=True)
    denom = np.maximum(np.sqrt(sumsq), eps)
    return x / denom


def onnx_l2_norm_emit(x: np.ndarray, eps: float) -> np.ndarray:
    """Mirror of the new _h_l2_norm handler, computed in numpy."""
    x_f = x.astype(np.float32)
    sq = x_f * x_f
    ss = sq.sum(axis=-1, keepdims=True)
    norm = np.sqrt(ss)
    denom = np.maximum(norm, np.float32(eps))
    return (x_f / denom).astype(x.dtype)


# ------------------------------------------------------------------ SWIGLU_OAI
def ggml_swiglu_oai_ref(gate, up, alpha, limit):
    """Match ggml_compute_forward_swiglu_oai_f32."""
    x = np.minimum(gate, limit)
    y = np.clip(up, -limit, limit)
    out_glu = x / (1.0 + np.exp(-alpha * x))
    return out_glu * (y + 1.0)


def onnx_swiglu_oai_emit(gate, up, alpha, limit):
    g = np.minimum(gate, limit)
    u = np.maximum(np.minimum(up, limit), -limit)
    sig = 1.0 / (1.0 + np.exp(-alpha * g))
    return (g * sig) * (u + 1.0)


# ----------------------------------------------------------------------- ADD_ID
def ggml_add_id_ref(src0, src1, ids):
    """Match ggml_compute_forward_add_id_f32: dst[i3,i2,i1,:] = src0[...] + src1[ids[i2,i1],:]."""
    return src0 + src1[ids]


def onnx_add_id_emit(src0, src1, ids):
    return src0 + src1[ids.astype(np.int64)]


# ---------------------------------------------------------------- ROPE YaRN
def ggml_yarn_cos_sin(head_dim, n_ctx_orig, freq_base, freq_scale,
                     ext_factor, attn_factor, beta_fast, beta_slow, positions):
    """Reference: ggml_compute_forward_rope's per-token cache_init + rope_yarn loop."""
    def corr_dim(rot):
        return head_dim * math.log(n_ctx_orig / (rot * 2 * math.pi)) / (2 * math.log(freq_base))
    low  = max(0.0,                math.floor(corr_dim(beta_fast)))
    high = min(float(head_dim - 1), math.ceil (corr_dim(beta_slow)))
    half = head_dim // 2
    cos = np.zeros((len(positions), half), dtype=np.float64)
    sin = np.zeros((len(positions), half), dtype=np.float64)
    theta_scale = freq_base ** (-2.0 / head_dim)
    for ti, pos in enumerate(positions):
        theta = float(pos)
        for i, i0 in enumerate(range(0, head_dim, 2)):
            theta_extrap = theta
            theta_interp = freq_scale * theta_extrap
            t = theta_interp
            mscale = attn_factor
            if ext_factor != 0.0:
                y = (i0 / 2.0 - low) / max(0.001, high - low)
                ramp = 1.0 - min(1.0, max(0.0, y))
                ramp_mix = ramp * ext_factor
                t = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix
                mscale = attn_factor * (1.0 + 0.1 * math.log(1.0 / freq_scale))
            cos[ti, i] = math.cos(t) * mscale
            sin[ti, i] = math.sin(t) * mscale
            theta *= theta_scale
    return cos, sin


def onnx_yarn_cos_sin(head_dim, n_ctx_orig, freq_base, freq_scale, ext_factor,
                     attn_factor, beta_fast, beta_slow, positions):
    """Mirror of the new _h_rope YaRN branch."""
    i0 = np.arange(0, head_dim, 2, dtype=np.float64)
    inv_extrap = 1.0 / (freq_base ** (i0 / head_dim))
    if ext_factor != 0.0:
        def corr(rot):
            return head_dim * math.log(n_ctx_orig / (rot * 2 * math.pi)) / (2 * math.log(freq_base))
        low  = max(0.0,                math.floor(corr(beta_fast)))
        high = min(float(head_dim - 1), math.ceil (corr(beta_slow)))
        denom = max(0.001, high - low)
        y = (i0 / 2.0 - low) / denom
        ramp = 1.0 - np.clip(y, 0.0, 1.0)
        ramp_mix = ramp * ext_factor
        inv_interp = inv_extrap * freq_scale
        inv = inv_interp * (1.0 - ramp_mix) + inv_extrap * ramp_mix
        mscale = attn_factor * (1.0 + 0.1 * math.log(1.0 / freq_scale))
    else:
        inv = inv_extrap * freq_scale
        if attn_factor != 1.0:
            inv = inv * attn_factor
        mscale = 1.0
    angles = positions[:, None].astype(np.float64) * inv[None, :]
    return np.cos(angles) * mscale, np.sin(angles) * mscale


class TestPhaseAMath(unittest.TestCase):
    def test_l2_norm(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((8, 16)).astype(np.float32)
        for eps in (1e-6, 1e-3, 1e-1):
            diff = np.abs(ggml_l2_norm_ref(x, eps) - onnx_l2_norm_emit(x, eps)).max()
            self.assertLess(diff, 1e-7, f"L2_NORM mismatch eps={eps}: {diff}")

    def test_swiglu_oai(self):
        rng = np.random.default_rng(1)
        gate = rng.standard_normal((8, 16)).astype(np.float32) * 3.0
        up   = rng.standard_normal((8, 16)).astype(np.float32) * 3.0
        for alpha, limit in [(1.0, 4.0), (1.702, 6.0), (0.5, 10.0)]:
            a = ggml_swiglu_oai_ref(gate, up, alpha, limit)
            b = onnx_swiglu_oai_emit(gate, up, alpha, limit)
            rel = np.abs(a - b).max() / (np.abs(a).max() + 1e-12)
            self.assertLess(rel, 1e-6, f"SWIGLU_OAI mismatch alpha={alpha},limit={limit}")

    def test_add_id(self):
        rng = np.random.default_rng(2)
        src0 = rng.standard_normal((4, 3, 8)).astype(np.float32)
        src1 = rng.standard_normal((5, 8)).astype(np.float32)
        ids  = rng.integers(0, 5, size=(4, 3)).astype(np.int32)
        diff = np.abs(ggml_add_id_ref(src0, src1, ids) - onnx_add_id_emit(src0, src1, ids)).max()
        self.assertLess(diff, 1e-7)

    def test_rope_yarn(self):
        head_dim = 96
        n_ctx_orig = 4096
        positions = np.arange(0, 32, dtype=np.int64)
        cases = [
            (10000.0, 1.0,  0.0,  1.0,  32.0, 1.0, "no-scaling"),
            (10000.0, 0.25, 0.0,  1.0,  32.0, 1.0, "linear-only"),
            (10000.0, 0.25, 1.0,  1.0,  32.0, 1.0, "yarn-typical"),
            (500000.0, 0.125, 1.0, 1.0, 32.0, 1.0, "yarn-deepseek-style"),
            (10000.0, 0.5,  0.5,  1.0,  32.0, 1.0, "partial-extfactor"),
        ]
        for fb, fs, ef, af, bf, bs, name in cases:
            ref_c, ref_s = ggml_yarn_cos_sin(head_dim, n_ctx_orig, fb, fs, ef, af, bf, bs, positions)
            my_c, my_s   = onnx_yarn_cos_sin(head_dim, n_ctx_orig, fb, fs, ef, af, bf, bs, positions)
            self.assertLess(np.abs(ref_c - my_c).max(), 1e-9, f"YaRN '{name}' cos mismatch")
            self.assertLess(np.abs(ref_s - my_s).max(), 1e-9, f"YaRN '{name}' sin mismatch")


if __name__ == "__main__":
    unittest.main()
