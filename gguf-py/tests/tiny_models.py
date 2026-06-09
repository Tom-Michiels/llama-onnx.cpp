"""Tiny synthetic GGUF model builders for the ONNX export pipeline tests.

Each function returns a path-writable record (arch name, hparams, weight
dict) plus a writer that pours it into a GGUF file. The shapes are kept
small so the resulting graph fits in a few KB of memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gguf import GGUFWriter


@dataclass
class TinyModel:
    arch: str
    hparams: dict
    weights: dict
    # Extra GGUF metadata callbacks. Each entry is (writer-method, args).
    extra_kv: list = field(default_factory=list)
    tokenizer: str = "none"   # "none" -> llama.cpp skips tokenizer init
    # Human-readable id for tests when multiple builders share the same arch.
    label: str = ""


def _rand(rng: np.random.Generator, shape, dtype=np.float32, scale=0.05):
    return (rng.standard_normal(shape).astype(np.float32) * scale).astype(dtype)


def _qkv_norm_tensors(weights, il, n_embd, head_dim, n_head, n_head_kv, dtype):
    weights[f"blk.{il}.attn_q_norm.weight"] = np.ones((head_dim,), dtype=dtype)
    weights[f"blk.{il}.attn_k_norm.weight"] = np.ones((head_dim,), dtype=dtype)


def _qwen2_attn_biases(weights, rng, h, il, dtype):
    weights[f"blk.{il}.attn_q.bias"] = _rand(rng, (h["n_head"]    * h["head_dim"],), dtype)
    weights[f"blk.{il}.attn_k.bias"] = _rand(rng, (h["n_head_kv"] * h["head_dim"],), dtype)
    weights[f"blk.{il}.attn_v.bias"] = _rand(rng, (h["n_head_kv"] * h["head_dim"],), dtype)


def write_gguf(path: Path, m: TinyModel, *, dtype=np.float32) -> None:
    w = GGUFWriter(path, m.arch)
    h = m.hparams
    w.add_block_count(h["n_layer"])
    w.add_embedding_length(h["n_embd"])
    w.add_feed_forward_length(h["n_ff"])
    # Some recurrent / SSM architectures (Mamba, RWKV-7) don't carry attention
    # heads or rope; for those we let the fixture omit those keys.
    if "n_head" in h:
        w.add_head_count(h["n_head"])
    if "n_head_kv" in h:
        w.add_head_count_kv(h["n_head_kv"])
    if "head_dim" in h:
        w.add_key_length(h["head_dim"])
        w.add_value_length(h["head_dim"])
    w.add_layer_norm_rms_eps(h["rms_eps"])
    if "head_dim" in h:
        w.add_rope_dimension_count(h.get("n_rot", h["head_dim"]))
    if "rope_base" in h:
        w.add_rope_freq_base(h["rope_base"])
    w.add_vocab_size(h["vocab_size"])
    w.add_context_length(h.get("context_length", 64))
    if "n_expert" in h:
        w.add_expert_count(h["n_expert"])
    if "n_expert_used" in h:
        w.add_expert_used_count(h["n_expert_used"])
    if "rope_sections" in h:
        # qwen2vl-style mrope sections: 4-vector summing to n_rot/2.
        w.add_rope_dimension_sections(h["rope_sections"])
    for fn_name, args in m.extra_kv:
        getattr(w, fn_name)(*args)
    w.add_tokenizer_model(m.tokenizer)
    for name, arr in m.weights.items():
        w.add_tensor(name, arr.astype(dtype))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


# ---------------------------------------------------------------------------
# Per-arch builders
# ---------------------------------------------------------------------------


def _llama_like_weights(rng, h, dtype, *, with_qk_norm=False, with_moe=False, moe_style="qwen"):
    weights = {
        "token_embd.weight":   _rand(rng, (h["vocab_size"], h["n_embd"]), dtype),
        "output_norm.weight":  np.ones((h["n_embd"],), dtype=dtype),
    }
    for il in range(h["n_layer"]):
        weights[f"blk.{il}.attn_norm.weight"]   = np.ones((h["n_embd"],), dtype=dtype)
        weights[f"blk.{il}.attn_q.weight"]      = _rand(rng, (h["n_head"]    * h["head_dim"], h["n_embd"]), dtype)
        weights[f"blk.{il}.attn_k.weight"]      = _rand(rng, (h["n_head_kv"] * h["head_dim"], h["n_embd"]), dtype)
        weights[f"blk.{il}.attn_v.weight"]      = _rand(rng, (h["n_head_kv"] * h["head_dim"], h["n_embd"]), dtype)
        weights[f"blk.{il}.attn_output.weight"] = _rand(rng, (h["n_embd"], h["n_head"] * h["head_dim"]), dtype)
        weights[f"blk.{il}.ffn_norm.weight"]    = np.ones((h["n_embd"],), dtype=dtype)
        if with_qk_norm:
            _qkv_norm_tensors(weights, il, h["n_embd"], h["head_dim"], h["n_head"], h["n_head_kv"], dtype)
        if with_moe:
            n_exp = h["n_expert"]
            if moe_style == "llama":
                n_ff_exp = h["n_ff"]
            else:
                n_ff_exp = h.get("n_ff_exp", h["n_ff"] // h.get("n_expert_used", 1))
            weights[f"blk.{il}.ffn_gate_inp.weight"]  = _rand(rng, (n_exp, h["n_embd"]), dtype)
            weights[f"blk.{il}.ffn_gate_exps.weight"] = _rand(rng, (n_exp, n_ff_exp, h["n_embd"]), dtype)
            weights[f"blk.{il}.ffn_up_exps.weight"]   = _rand(rng, (n_exp, n_ff_exp, h["n_embd"]), dtype)
            weights[f"blk.{il}.ffn_down_exps.weight"] = _rand(rng, (n_exp, h["n_embd"], n_ff_exp), dtype)
        else:
            weights[f"blk.{il}.ffn_gate.weight"] = _rand(rng, (h["n_ff"], h["n_embd"]), dtype)
            weights[f"blk.{il}.ffn_up.weight"]   = _rand(rng, (h["n_ff"], h["n_embd"]), dtype)
            weights[f"blk.{il}.ffn_down.weight"] = _rand(rng, (h["n_embd"], h["n_ff"]), dtype)
    return weights


def _phi3_like_weights(rng, h, dtype):
    """Phi-3: SWIGLU via merged ffn_up (2*n_ff); no ffn_gate."""
    weights = {
        "token_embd.weight":   _rand(rng, (h["vocab_size"], h["n_embd"]), dtype),
        "output_norm.weight":  np.ones((h["n_embd"],), dtype=dtype),
    }
    for il in range(h["n_layer"]):
        weights[f"blk.{il}.attn_norm.weight"]   = np.ones((h["n_embd"],), dtype=dtype)
        weights[f"blk.{il}.attn_q.weight"]      = _rand(rng, (h["n_head"]    * h["head_dim"], h["n_embd"]), dtype)
        weights[f"blk.{il}.attn_k.weight"]      = _rand(rng, (h["n_head_kv"] * h["head_dim"], h["n_embd"]), dtype)
        weights[f"blk.{il}.attn_v.weight"]      = _rand(rng, (h["n_head_kv"] * h["head_dim"], h["n_embd"]), dtype)
        weights[f"blk.{il}.attn_output.weight"] = _rand(rng, (h["n_embd"], h["n_head"] * h["head_dim"]), dtype)
        weights[f"blk.{il}.ffn_norm.weight"]    = np.ones((h["n_embd"],), dtype=dtype)
        weights[f"blk.{il}.ffn_up.weight"]      = _rand(rng, (2 * h["n_ff"], h["n_embd"]), dtype)
        weights[f"blk.{il}.ffn_down.weight"]    = _rand(rng, (h["n_embd"], h["n_ff"]), dtype)
    return weights


def _gemma2_like_weights(rng, h, dtype):
    weights = _llama_like_weights(rng, h, dtype)
    for il in range(h["n_layer"]):
        weights[f"blk.{il}.post_attention_norm.weight"] = np.ones((h["n_embd"],), dtype=dtype)
        weights[f"blk.{il}.post_ffw_norm.weight"]       = np.ones((h["n_embd"],), dtype=dtype)
    return weights


def tiny_llama(seed=42, *, dtype=np.float32) -> TinyModel:
    rng = np.random.default_rng(seed)
    h = dict(n_layer=2, n_embd=16, n_ff=32, n_head=4, n_head_kv=2, head_dim=4,
             vocab_size=32, rms_eps=1e-5, rope_base=10000.0)
    return TinyModel(arch="llama", hparams=h, weights=_llama_like_weights(rng, h, dtype))


def tiny_gemma(seed=43, *, dtype=np.float32) -> TinyModel:
    rng = np.random.default_rng(seed)
    h = dict(n_layer=2, n_embd=16, n_ff=32, n_head=4, n_head_kv=4, head_dim=4,
             vocab_size=32, rms_eps=1e-6, rope_base=10000.0)
    return TinyModel(arch="gemma", hparams=h, weights=_llama_like_weights(rng, h, dtype))


def tiny_qwen3(seed=44, *, dtype=np.float32) -> TinyModel:
    rng = np.random.default_rng(seed)
    h = dict(n_layer=2, n_embd=16, n_ff=32, n_head=4, n_head_kv=2, head_dim=4,
             vocab_size=32, rms_eps=1e-6, rope_base=1_000_000.0)
    return TinyModel(arch="qwen3", hparams=h,
                     weights=_llama_like_weights(rng, h, dtype, with_qk_norm=True))


def tiny_qwen3moe(seed=45, *, dtype=np.float32) -> TinyModel:
    rng = np.random.default_rng(seed)
    h = dict(n_layer=2, n_embd=16, n_ff=8, n_head=4, n_head_kv=2, head_dim=4,
             vocab_size=32, rms_eps=1e-6, rope_base=1_000_000.0,
             n_expert=4, n_expert_used=2)
    return TinyModel(arch="qwen3moe", hparams=h,
                     weights=_llama_like_weights(rng, h, dtype, with_qk_norm=True, with_moe=True))


def tiny_mamba(seed=47, *, dtype=np.float32) -> TinyModel:
    """Mamba-1 fixture: exercises SSM_CONV and SSM_SCAN (Mamba-1, A is (d_state, d_inner)).

    The expansion factor is fixed at 2 (d_inner = 2 * n_embd), matching the
    assertion in ``llama_model_mamba::load_arch_tensors``.
    """
    rng = np.random.default_rng(seed)
    n_embd = 8
    d_inner = 2 * n_embd
    d_state = 4
    d_conv = 4
    dt_rank = 2
    n_layer = 1
    vocab_size = 32
    h = dict(
        n_layer=n_layer, n_embd=n_embd, n_ff=0, vocab_size=vocab_size,
        rms_eps=1e-5,
        # SSM-specific:
        ssm_d_conv=d_conv, ssm_d_inner=d_inner, ssm_d_state=d_state,
        ssm_dt_rank=dt_rank,
    )
    weights = {
        "token_embd.weight":   _rand(rng, (vocab_size, n_embd), dtype),
        "output_norm.weight":  np.ones((n_embd,), dtype=dtype),
    }
    for il in range(n_layer):
        weights[f"blk.{il}.attn_norm.weight"]      = np.ones((n_embd,), dtype=dtype)
        weights[f"blk.{il}.ssm_in.weight"]         = _rand(rng, (2 * d_inner, n_embd), dtype)
        weights[f"blk.{il}.ssm_conv1d.weight"]     = _rand(rng, (d_inner, d_conv), dtype)
        weights[f"blk.{il}.ssm_conv1d.bias"]       = _rand(rng, (d_inner,), dtype)
        weights[f"blk.{il}.ssm_x.weight"]          = _rand(rng, (dt_rank + 2 * d_state, d_inner), dtype)
        weights[f"blk.{il}.ssm_dt.weight"]         = _rand(rng, (d_inner, dt_rank), dtype)
        weights[f"blk.{il}.ssm_dt.bias"]           = _rand(rng, (d_inner,), dtype)
        # ssm_a / ssm_d have no "weight" suffix in mamba (see mamba.cpp).
        weights[f"blk.{il}.ssm_a"]                 = -_rand(rng, (d_inner, d_state), dtype, scale=0.5)
        weights[f"blk.{il}.ssm_d"]                 = _rand(rng, (d_inner,), dtype)
        weights[f"blk.{il}.ssm_out.weight"]        = _rand(rng, (n_embd, d_inner), dtype)
    extra_kv = [
        ("add_ssm_conv_kernel",     [d_conv]),
        ("add_ssm_inner_size",      [d_inner]),
        ("add_ssm_state_size",      [d_state]),
        ("add_ssm_time_step_rank",  [dt_rank]),
    ]
    return TinyModel(arch="mamba", hparams=h, weights=weights, extra_kv=extra_kv)


def tiny_qwen2vl(seed=46, *, dtype=np.float32) -> TinyModel:
    rng = np.random.default_rng(seed)
    h = dict(n_layer=2, n_embd=16, n_ff=32, n_head=4, n_head_kv=4, head_dim=4,
             vocab_size=32, rms_eps=1e-6, rope_base=1_000_000.0,
             # mrope: 4 sections (t, h, w, e) summing to n_rot/2 = head_dim/2 = 2.
             rope_sections=[1, 0, 1, 0])
    # qwen2vl uses Q/K/V biases by default (qwen2 family convention).
    weights = _llama_like_weights(rng, h, dtype)
    for il in range(h["n_layer"]):
        _qwen2_attn_biases(weights, rng, h, il, dtype)
    return TinyModel(arch="qwen2vl", hparams=h, weights=weights)


def tiny_qwen2(seed=48, *, dtype=np.float32) -> TinyModel:
    rng = np.random.default_rng(seed)
    h = dict(n_layer=2, n_embd=16, n_ff=32, n_head=4, n_head_kv=2, head_dim=4,
             vocab_size=32, rms_eps=1e-6, rope_base=1_000_000.0)
    weights = _llama_like_weights(rng, h, dtype)
    for il in range(h["n_layer"]):
        _qwen2_attn_biases(weights, rng, h, il, dtype)
    return TinyModel(arch="qwen2", hparams=h, weights=weights)


def tiny_phi3(seed=49, *, dtype=np.float32) -> TinyModel:
    rng = np.random.default_rng(seed)
    h = dict(n_layer=2, n_embd=16, n_ff=32, n_head=4, n_head_kv=4, head_dim=4,
             vocab_size=32, rms_eps=1e-5, rope_base=10_000.0)
    return TinyModel(arch="phi3", hparams=h, weights=_phi3_like_weights(rng, h, dtype))


def tiny_mixtral(seed=50, *, dtype=np.float32) -> TinyModel:
    """Mixtral-8x7B-style MoE; GGUF arch is ``llama`` (n_expert > 0)."""
    rng = np.random.default_rng(seed)
    h = dict(n_layer=2, n_embd=16, n_ff=32, n_head=4, n_head_kv=2, head_dim=4,
             vocab_size=32, rms_eps=1e-5, rope_base=1_000_000.0,
             n_expert=4, n_expert_used=2)
    return TinyModel(arch="llama", label="mixtral", hparams=h,
                     weights=_llama_like_weights(rng, h, dtype, with_moe=True, moe_style="llama"))


def tiny_pixtral(seed=53, *, dtype=np.float32) -> TinyModel:
    """Pixtral-12B text backbone (standard ``llama`` arch, no vision tensors)."""
    rng = np.random.default_rng(seed)
    h = dict(n_layer=2, n_embd=16, n_ff=32, n_head=4, n_head_kv=2, head_dim=4,
             vocab_size=32, rms_eps=1e-5, rope_base=1_000_000.0)
    return TinyModel(arch="llama", label="pixtral", hparams=h,
                     weights=_llama_like_weights(rng, h, dtype))


def tiny_internvl(seed=54, *, dtype=np.float32) -> TinyModel:
    """InternVL text backbone (``internlm2`` LM graph, no vision tensors)."""
    rng = np.random.default_rng(seed)
    h = dict(n_layer=2, n_embd=16, n_ff=32, n_head=4, n_head_kv=2, head_dim=4,
             vocab_size=32, rms_eps=1e-5, rope_base=10_000.0)
    weights = _llama_like_weights(rng, h, dtype)
    weights["output.weight"] = _rand(rng, (h["vocab_size"], h["n_embd"]), dtype)
    return TinyModel(arch="internlm2", label="internvl", hparams=h, weights=weights)


def tiny_internlm2(seed=51, *, dtype=np.float32) -> TinyModel:
    rng = np.random.default_rng(seed)
    h = dict(n_layer=2, n_embd=16, n_ff=32, n_head=4, n_head_kv=2, head_dim=4,
             vocab_size=32, rms_eps=1e-5, rope_base=10_000.0)
    weights = _llama_like_weights(rng, h, dtype)
    weights["output.weight"] = _rand(rng, (h["vocab_size"], h["n_embd"]), dtype)
    return TinyModel(arch="internlm2", hparams=h, weights=weights)


def tiny_gemma2(seed=52, *, dtype=np.float32) -> TinyModel:
    rng = np.random.default_rng(seed)
    h = dict(n_layer=2, n_embd=16, n_ff=32, n_head=4, n_head_kv=2, head_dim=4,
             vocab_size=32, rms_eps=1e-6, rope_base=10_000.0)
    extra_kv = [
        ("add_sliding_window", [64]),
        ("add_sliding_window_pattern", [2]),
    ]
    return TinyModel(arch="gemma2", hparams=h,
                     weights=_gemma2_like_weights(rng, h, dtype), extra_kv=extra_kv)


# ---------------------------------------------------------------------------
# VLM LM-side backends (text backbone only; no vision encoder tensors)
# ---------------------------------------------------------------------------


def tiny_qwen3vl(seed=56, *, dtype=np.float32) -> TinyModel:
    """Qwen3-VL text backbone (IM-RoPE + deepstack + Q/K norms)."""
    rng = np.random.default_rng(seed)
    h = dict(n_layer=2, n_embd=16, n_ff=32, n_head=4, n_head_kv=2, head_dim=4,
             vocab_size=32, rms_eps=1e-6, rope_base=1_000_000.0,
             rope_sections=[1, 0, 1, 0], n_deepstack_layers=1)
    extra_kv = [("add_num_deepstack_layers", [h["n_deepstack_layers"]])]
    return TinyModel(arch="qwen3vl", hparams=h,
                     weights=_llama_like_weights(rng, h, dtype, with_qk_norm=True),
                     extra_kv=extra_kv)


def _llama4_weights(rng, h, dtype):
    weights = {
        "token_embd.weight":   _rand(rng, (h["vocab_size"], h["n_embd"]), dtype),
        "output_norm.weight":  np.ones((h["n_embd"],), dtype=dtype),
    }
    n_exp = h["n_expert"]
    n_ff_exp = h["n_ff_exp"]
    moe_step = h["n_moe_layer_step"]
    for il in range(h["n_layer"]):
        weights[f"blk.{il}.attn_norm.weight"]   = np.ones((h["n_embd"],), dtype=dtype)
        weights[f"blk.{il}.attn_q.weight"]      = _rand(rng, (h["n_head"]    * h["head_dim"], h["n_embd"]), dtype)
        weights[f"blk.{il}.attn_k.weight"]      = _rand(rng, (h["n_head_kv"] * h["head_dim"], h["n_embd"]), dtype)
        weights[f"blk.{il}.attn_v.weight"]      = _rand(rng, (h["n_head_kv"] * h["head_dim"], h["n_embd"]), dtype)
        weights[f"blk.{il}.attn_output.weight"] = _rand(rng, (h["n_embd"], h["n_head"] * h["head_dim"]), dtype)
        weights[f"blk.{il}.ffn_norm.weight"]    = np.ones((h["n_embd"],), dtype=dtype)
        if moe_step > 0 and (il + 1) % moe_step == 0:
            weights[f"blk.{il}.ffn_gate_inp.weight"]   = _rand(rng, (n_exp, h["n_embd"]), dtype)
            weights[f"blk.{il}.ffn_gate_exps.weight"]  = _rand(rng, (n_exp, n_ff_exp, h["n_embd"]), dtype)
            weights[f"blk.{il}.ffn_up_exps.weight"]    = _rand(rng, (n_exp, n_ff_exp, h["n_embd"]), dtype)
            weights[f"blk.{il}.ffn_down_exps.weight"]  = _rand(rng, (n_exp, h["n_embd"], n_ff_exp), dtype)
            weights[f"blk.{il}.ffn_gate_shexp.weight"]  = _rand(rng, (n_ff_exp, h["n_embd"]), dtype)
            weights[f"blk.{il}.ffn_up_shexp.weight"]   = _rand(rng, (n_ff_exp, h["n_embd"]), dtype)
            weights[f"blk.{il}.ffn_down_shexp.weight"] = _rand(rng, (h["n_embd"], n_ff_exp), dtype)
        else:
            weights[f"blk.{il}.ffn_gate.weight"] = _rand(rng, (h["n_ff"], h["n_embd"]), dtype)
            weights[f"blk.{il}.ffn_up.weight"]   = _rand(rng, (h["n_ff"], h["n_embd"]), dtype)
            weights[f"blk.{il}.ffn_down.weight"] = _rand(rng, (h["n_embd"], h["n_ff"]), dtype)
    return weights


def tiny_llama4(seed=57, *, dtype=np.float32) -> TinyModel:
    """Llama 4 Scout/Maverick text backbone (interleaved MoE + shared experts)."""
    rng = np.random.default_rng(seed)
    h = dict(n_layer=4, n_embd=16, n_ff=32, n_head=4, n_head_kv=2, head_dim=4,
             vocab_size=32, rms_eps=1e-5, rope_base=500_000.0,
             n_expert=4, n_expert_used=2, n_ff_exp=8, n_moe_layer_step=2)
    extra_kv = [
        ("add_expert_feed_forward_length", [h["n_ff_exp"]]),
        ("add_interleave_moe_layer_step",  [h["n_moe_layer_step"]]),
        ("add_sliding_window",             [0]),
    ]
    return TinyModel(arch="llama4", hparams=h,
                     weights=_llama4_weights(rng, h, dtype), extra_kv=extra_kv)


def _cogvlm_weights(rng, h, dtype):
    n_qkv = h["n_head"] * h["head_dim"] * 3
    weights = {
        "token_embd.weight":   _rand(rng, (h["vocab_size"], h["n_embd"]), dtype),
        "output_norm.weight":  np.ones((h["n_embd"],), dtype=dtype),
    }
    for il in range(h["n_layer"]):
        weights[f"blk.{il}.attn_norm.weight"]       = np.ones((h["n_embd"],), dtype=dtype)
        weights[f"blk.{il}.attn_qkv.weight"]        = _rand(rng, (n_qkv, h["n_embd"]), dtype)
        weights[f"blk.{il}.attn_output.weight"]     = _rand(rng, (h["n_embd"], h["n_head"] * h["head_dim"]), dtype)
        weights[f"blk.{il}.vis_attn_qkv.weight"]    = _rand(rng, (n_qkv, h["n_embd"]), dtype)
        weights[f"blk.{il}.vis_attn_output.weight"] = _rand(rng, (h["n_embd"], h["n_head"] * h["head_dim"]), dtype)
        weights[f"blk.{il}.ffn_norm.weight"]        = np.ones((h["n_embd"],), dtype=dtype)
        weights[f"blk.{il}.ffn_gate.weight"]       = _rand(rng, (h["n_ff"], h["n_embd"]), dtype)
        weights[f"blk.{il}.ffn_up.weight"]         = _rand(rng, (h["n_ff"], h["n_embd"]), dtype)
        weights[f"blk.{il}.ffn_down.weight"]       = _rand(rng, (h["n_embd"], h["n_ff"]), dtype)
        weights[f"blk.{il}.vis_gate.weight"]       = _rand(rng, (h["n_ff"], h["n_embd"]), dtype)
        weights[f"blk.{il}.vis_up.weight"]         = _rand(rng, (h["n_ff"], h["n_embd"]), dtype)
        weights[f"blk.{il}.vis_down.weight"]       = _rand(rng, (h["n_embd"], h["n_ff"]), dtype)
    return weights


def tiny_cogvlm(seed=58, *, dtype=np.float32) -> TinyModel:
    """CogVLM LM graph (text + vis-expert weight paths)."""
    rng = np.random.default_rng(seed)
    h = dict(n_layer=2, n_embd=16, n_ff=32, n_head=4, n_head_kv=4, head_dim=4,
             vocab_size=32, rms_eps=1e-5, rope_base=10000.0)
    return TinyModel(arch="cogvlm", hparams=h, weights=_cogvlm_weights(rng, h, dtype))


def tiny_minicpm(seed=59, *, dtype=np.float32) -> TinyModel:
    """MiniCPM-V text backbone (granite-style embedding/residual/logit scaling)."""
    rng = np.random.default_rng(seed)
    n_layer = 2
    h = dict(n_layer=n_layer, n_embd=16, n_ff=32, n_head=4, n_head_kv=4, head_dim=4,
             vocab_size=32, rms_eps=1e-5, rope_base=10000.0)
    extra_kv = [
        ("add_embedding_scale", [12.0]),
        ("add_residual_scale",  [1.4 / (n_layer ** 0.5)]),
        ("add_logit_scale",     [256.0 / h["n_embd"]]),
    ]
    return TinyModel(arch="minicpm", label="minicpm-v", hparams=h,
                     weights=_llama_like_weights(rng, h, dtype), extra_kv=extra_kv)


def _glm4_weights(rng, h, dtype):
    weights = {
        "token_embd.weight":   _rand(rng, (h["vocab_size"], h["n_embd"]), dtype),
        "output_norm.weight":  np.ones((h["n_embd"],), dtype=dtype),
    }
    for il in range(h["n_layer"]):
        weights[f"blk.{il}.attn_norm.weight"]           = np.ones((h["n_embd"],), dtype=dtype)
        weights[f"blk.{il}.attn_q.weight"]              = _rand(rng, (h["n_head"]    * h["head_dim"], h["n_embd"]), dtype)
        weights[f"blk.{il}.attn_k.weight"]              = _rand(rng, (h["n_head_kv"] * h["head_dim"], h["n_embd"]), dtype)
        weights[f"blk.{il}.attn_v.weight"]              = _rand(rng, (h["n_head_kv"] * h["head_dim"], h["n_embd"]), dtype)
        weights[f"blk.{il}.attn_output.weight"]         = _rand(rng, (h["n_embd"], h["n_head"] * h["head_dim"]), dtype)
        weights[f"blk.{il}.post_attention_norm.weight"] = np.ones((h["n_embd"],), dtype=dtype)
        weights[f"blk.{il}.ffn_norm.weight"]              = np.ones((h["n_embd"],), dtype=dtype)
        weights[f"blk.{il}.ffn_up.weight"]              = _rand(rng, (h["n_ff"] * 2, h["n_embd"]), dtype)
        weights[f"blk.{il}.ffn_down.weight"]            = _rand(rng, (h["n_embd"], h["n_ff"]), dtype)
        weights[f"blk.{il}.post_ffw_norm.weight"]       = np.ones((h["n_embd"],), dtype=dtype)
    return weights


def tiny_glm4(seed=60, *, dtype=np.float32) -> TinyModel:
    """GLM-4 text backbone (non-multimodal)."""
    rng = np.random.default_rng(seed)
    h = dict(n_layer=2, n_embd=16, n_ff=32, n_head=4, n_head_kv=2, head_dim=4,
             vocab_size=32, rms_eps=1e-5, rope_base=10000.0)
    return TinyModel(arch="glm4", hparams=h, weights=_glm4_weights(rng, h, dtype))


def tiny_glm4v(seed=61, *, dtype=np.float32) -> TinyModel:
    """GLM-4V text backbone (glm4 arch with M-RoPE for image token positions)."""
    rng = np.random.default_rng(seed)
    h = dict(n_layer=2, n_embd=16, n_ff=32, n_head=4, n_head_kv=2, head_dim=4,
             vocab_size=32, rms_eps=1e-5, rope_base=10000.0, n_rot=2,
             rope_sections=[1, 1, 0, 0])
    return TinyModel(arch="glm4", label="glm4v", hparams=h, weights=_glm4_weights(rng, h, dtype))


def _minicpm3_weights(rng, h, dtype):
    n_embd = h["n_embd"]
    n_head = h["n_head"]
    head_dim = h["head_dim"]
    n_rot = h["n_rot"]
    q_rank = h["q_lora_rank"]
    kv_rank = h["kv_lora_rank"]
    n_qk_nope = head_dim - n_rot
    n_v = head_dim
    weights = {
        "token_embd.weight":   _rand(rng, (h["vocab_size"], n_embd), dtype),
        "output_norm.weight":  np.ones((n_embd,), dtype=dtype),
    }
    for il in range(h["n_layer"]):
        weights[f"blk.{il}.attn_norm.weight"]      = np.ones((n_embd,), dtype=dtype)
        weights[f"blk.{il}.attn_q_a_norm.weight"]  = np.ones((q_rank,), dtype=dtype)
        weights[f"blk.{il}.attn_kv_a_norm.weight"] = np.ones((kv_rank,), dtype=dtype)
        weights[f"blk.{il}.attn_q_a.weight"]       = _rand(rng, (q_rank, n_embd), dtype)
        weights[f"blk.{il}.attn_q_b.weight"]       = _rand(rng, (n_head * head_dim, q_rank), dtype)
        weights[f"blk.{il}.attn_kv_a_mqa.weight"]  = _rand(rng, (kv_rank + n_rot, n_embd), dtype)
        weights[f"blk.{il}.attn_kv_b.weight"]      = _rand(rng, (n_head * (n_qk_nope + n_v), kv_rank), dtype)
        weights[f"blk.{il}.attn_output.weight"]    = _rand(rng, (n_embd, n_head * n_v), dtype)
        weights[f"blk.{il}.ffn_norm.weight"]       = np.ones((n_embd,), dtype=dtype)
        weights[f"blk.{il}.ffn_gate.weight"]       = _rand(rng, (h["n_ff"], n_embd), dtype)
        weights[f"blk.{il}.ffn_up.weight"]         = _rand(rng, (h["n_ff"], n_embd), dtype)
        weights[f"blk.{il}.ffn_down.weight"]       = _rand(rng, (n_embd, h["n_ff"]), dtype)
    return weights


def tiny_minicpm3(seed=62, *, dtype=np.float32) -> TinyModel:
    """MiniCPM3 / MiniCPM-V 4.x text backbone (MLA-style attention)."""
    rng = np.random.default_rng(seed)
    h = dict(n_layer=2, n_embd=16, n_ff=32, n_head=4, n_head_kv=4, head_dim=4,
             vocab_size=32, rms_eps=1e-5, rope_base=10000.0,
             n_rot=4, q_lora_rank=4, kv_lora_rank=4)
    extra_kv = [
        ("add_q_lora_rank",  [h["q_lora_rank"]]),
        ("add_kv_lora_rank", [h["kv_lora_rank"]]),
    ]
    return TinyModel(arch="minicpm3", hparams=h,
                     weights=_minicpm3_weights(rng, h, dtype), extra_kv=extra_kv)
