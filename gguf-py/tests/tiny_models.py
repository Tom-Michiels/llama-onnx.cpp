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


def _rand(rng: np.random.Generator, shape, dtype=np.float32, scale=0.05):
    return (rng.standard_normal(shape).astype(np.float32) * scale).astype(dtype)


def _qkv_norm_tensors(weights, il, n_embd, head_dim, n_head, n_head_kv, dtype):
    weights[f"blk.{il}.attn_q_norm.weight"] = np.ones((head_dim,), dtype=dtype)
    weights[f"blk.{il}.attn_k_norm.weight"] = np.ones((head_dim,), dtype=dtype)


def write_gguf(path: Path, m: TinyModel, *, dtype=np.float32) -> None:
    w = GGUFWriter(path, m.arch)
    h = m.hparams
    w.add_block_count(h["n_layer"])
    w.add_embedding_length(h["n_embd"])
    w.add_feed_forward_length(h["n_ff"])
    w.add_head_count(h["n_head"])
    w.add_head_count_kv(h["n_head_kv"])
    w.add_key_length(h["head_dim"])
    w.add_value_length(h["head_dim"])
    w.add_layer_norm_rms_eps(h["rms_eps"])
    w.add_rope_dimension_count(h.get("n_rot", h["head_dim"]))
    w.add_rope_freq_base(h.get("rope_base", 10000.0))
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


def _llama_like_weights(rng, h, dtype, *, with_qk_norm=False, with_moe=False):
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


def tiny_qwen2vl(seed=46, *, dtype=np.float32) -> TinyModel:
    rng = np.random.default_rng(seed)
    h = dict(n_layer=2, n_embd=16, n_ff=32, n_head=4, n_head_kv=4, head_dim=4,
             vocab_size=32, rms_eps=1e-6, rope_base=1_000_000.0,
             # mrope: 4 sections (t, h, w, e) summing to n_rot/2 = head_dim/2 = 2.
             rope_sections=[1, 0, 1, 0])
    # qwen2vl uses Q/K/V biases by default (qwen2 family convention).
    weights = _llama_like_weights(rng, h, dtype)
    for il in range(h["n_layer"]):
        weights[f"blk.{il}.attn_q.bias"] = _rand(rng, (h["n_head"]    * h["head_dim"],), dtype)
        weights[f"blk.{il}.attn_k.bias"] = _rand(rng, (h["n_head_kv"] * h["head_dim"],), dtype)
        weights[f"blk.{il}.attn_v.bias"] = _rand(rng, (h["n_head_kv"] * h["head_dim"],), dtype)
    return TinyModel(arch="qwen2vl", hparams=h, weights=weights)
