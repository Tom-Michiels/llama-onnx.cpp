"""Reader / runner for the ``.gfxt`` verification fixtures.

A fixture file is the output of ``llama-onnx-export-dump --fixture <path>``
and contains:

* the input token IDs fed to ``llama_decode()``, and
* the reference logits llama.cpp computed for those tokens.

The rest of the graph's inputs (positions, KV-cache write indices, the
causal mask) are not stored in the fixture — they are derived in Python
from the gdump's graph structure (which input feeds which op) plus the
token count, so the test is bit-for-bit reproducible without needing
llama.cpp to expose its post-decode tensor state.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import gdump


_FIXTURE_MAGIC = 0x54584647  # "GFXT"
_FIXTURE_VERSION = 2


@dataclass
class Fixture:
    n_tokens: int
    n_vocab: int
    token_ids: np.ndarray   # int32, shape (n_tokens,)
    logits: np.ndarray      # float32, shape (n_tokens, n_vocab)


def _read_exact(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"short read: wanted {n}, got {len(b)}")
    return b


def load(path: str | Path) -> Fixture:
    with open(path, "rb") as f:
        magic, version = struct.unpack("<II", _read_exact(f, 8))
        if magic != _FIXTURE_MAGIC:
            raise ValueError(f"bad fixture magic 0x{magic:08x}")
        if version != _FIXTURE_VERSION:
            raise ValueError(f"unsupported fixture version {version} (this build understands {_FIXTURE_VERSION})")
        n_tokens, n_vocab = struct.unpack("<II", _read_exact(f, 8))
        tokens = np.frombuffer(_read_exact(f, n_tokens * 4), dtype=np.int32).copy()
        (logits_nbytes,) = struct.unpack("<Q", _read_exact(f, 8))
        expected = n_tokens * n_vocab * 4
        if logits_nbytes != expected:
            raise ValueError(f"logits size mismatch: got {logits_nbytes}, expected {expected}")
        logits = np.frombuffer(_read_exact(f, logits_nbytes), dtype=np.float32).copy()
        logits = logits.reshape(n_tokens, n_vocab)
        return Fixture(n_tokens=n_tokens, n_vocab=n_vocab, token_ids=tokens, logits=logits)


# --------------------------------------------------------------------------
# Role identification: walk the gdump and figure out which leaf plays which
# role in the prefill graph (positions, kq_mask, KV indices, out_ids, ...).
# The translator renames a few well-known leafs (inp_tokens, cache_*_l*);
# we mirror that mapping here so the synthesised inputs feed straight into
# the ONNX model's named ports.
# --------------------------------------------------------------------------


def fixture_name_to_onnx_input(ggml_name: str) -> str:
    if ggml_name == "inp_tokens":
        return "input_ids"
    if ggml_name.startswith("cache_k_l"):
        return f"past_key_values.{ggml_name[len('cache_k_l'):]}.key"
    if ggml_name.startswith("cache_v_l"):
        return f"past_key_values.{ggml_name[len('cache_v_l'):]}.value"
    return ggml_name


def _is_cache(name: str, kind: str) -> bool:
    return name.startswith(f"cache_{kind}_l")


def identify_roles(dump: gdump.GraphDump) -> dict[int, str]:
    """Return a mapping from leaf-tensor index -> role.

    Possible roles:
      "tokens"       -- input token IDs
      "positions"    -- RoPE position vector
      "k_idxs"       -- SET_ROWS column indices for the K cache
      "v_idxs"       -- SET_ROWS column indices for the V cache
      "out_ids"      -- GET_ROWS indices into a layer output
      "kq_mask"      -- additive attention mask
      "cache_k_<L>"  -- L-th layer K cache (initial zeros)
      "cache_v_<L>"  -- L-th layer V cache (initial zeros)
    """
    roles: dict[int, str] = {}
    for t in dump.tensors:
        op = t.ggml_op
        if op == gdump.GgmlOp.ROPE and len(t.sources) >= 2:
            roles[t.sources[1]] = "positions"
        elif op == gdump.GgmlOp.SET_ROWS and len(t.sources) >= 3:
            tgt = dump.tensors[t.sources[2]]
            if _is_cache(tgt.name, "k"):
                roles[t.sources[1]] = "k_idxs"
            elif _is_cache(tgt.name, "v"):
                roles[t.sources[1]] = "v_idxs"
        elif op in (gdump.GgmlOp.FLASH_ATTN_EXT, gdump.GgmlOp.SOFT_MAX):
            mask_idx = 3 if op == gdump.GgmlOp.FLASH_ATTN_EXT else 1
            if len(t.sources) > mask_idx:
                src = dump.tensors[t.sources[mask_idx]]
                # The mask can be wrapped in a CPY/CONT; walk through them.
                cur = src
                seen = set()
                while cur.ggml_op in (gdump.GgmlOp.CPY, gdump.GgmlOp.CONT, gdump.GgmlOp.VIEW) and cur.sources:
                    if cur.index in seen: break
                    seen.add(cur.index)
                    cur = dump.tensors[cur.sources[0]]
                if cur.is_leaf:
                    roles[cur.index] = "kq_mask"
        elif op == gdump.GgmlOp.GET_ROWS and len(t.sources) >= 2:
            # The embedding GET_ROWS uses inp_tokens (already named); any
            # other GET_ROWS whose index leaf isn't roled yet is out_ids
            # (the prefill graph uses one to pick which positions get logits).
            idx_src = dump.tensors[t.sources[1]]
            data_src = dump.tensors[t.sources[0]]
            if idx_src.is_leaf and idx_src.name != "inp_tokens" and data_src.name != "token_embd.weight":
                roles.setdefault(idx_src.index, "out_ids")

    for t in dump.tensors:
        if t.name == "inp_tokens":
            roles[t.index] = "tokens"
        elif _is_cache(t.name, "k"):
            roles[t.index] = f"cache_k_{t.name[len('cache_k_l'):]}"
        elif _is_cache(t.name, "v"):
            roles[t.index] = f"cache_v_{t.name[len('cache_v_l'):]}"

    return roles


def synthesize_inputs(
    dump: gdump.GraphDump,
    fixture: Fixture,
    onnx_input_names: list[str],
) -> dict[str, np.ndarray]:
    """Build an onnxruntime ``feeds`` dict from the fixture and gdump.

    The graph's input shapes come straight from the gdump (so we don't have
    to second-guess them); the values are derived from prefill semantics:

      tokens     := fixture.token_ids
      positions  := arange(n_tokens)
      k_idxs     := arange(n_tokens)            (writing into [0, n_tokens))
      v_idxs     := arange(n_tokens)
      out_ids    := arange(n_tokens)            (we asked for logits everywhere)
      kq_mask    := lower-triangular for [0, n_tokens), -inf for [n_tokens, kv_total)
      cache_k/v  := zeros (fresh prefill)
    """
    roles = identify_roles(dump)
    n_tokens = fixture.n_tokens

    feeds: dict[str, np.ndarray] = {}
    for leaf_idx, role in roles.items():
        t = dump.tensors[leaf_idx]
        onnx_name = fixture_name_to_onnx_input(t.name)
        if onnx_name not in onnx_input_names:
            continue
        ne = [d for d in t.ne if d > 1] or [1]
        shape = list(reversed(ne))
        np_dtype = _np_for(t.dtype)
        if role == "tokens":
            arr = fixture.token_ids.astype(np_dtype)
        elif role == "positions":
            # ggml may pass 1 or 4 positions per token (4 for mrope); the
            # gdump shape tells us which.
            n_per_token = max(1, int(np.prod(shape)) // n_tokens)
            arr = np.tile(np.arange(n_tokens, dtype=np_dtype), n_per_token).reshape(shape)
        elif role in ("k_idxs", "v_idxs", "out_ids"):
            arr = np.arange(n_tokens, dtype=np_dtype).reshape(shape)
        elif role == "kq_mask":
            kv_total, q_len = t.ne[0], t.ne[1]
            mask = np.full((q_len, kv_total), -np.inf, dtype=np.float32)
            q = np.arange(q_len)[:, None]
            k = np.arange(kv_total)[None, :]
            mask[(k <= q) & (k < n_tokens)] = 0.0
            arr = mask.reshape(shape).astype(np_dtype)
        elif role.startswith("cache_"):
            arr = np.zeros(shape, dtype=np_dtype)
        else:
            arr = np.zeros(shape, dtype=np_dtype)
        feeds[onnx_name] = arr
    return feeds


def _np_for(ggml_dtype: int) -> np.dtype:
    return {
        gdump.GgmlType.F32.value: np.dtype("float32"),
        gdump.GgmlType.F16.value: np.dtype("float16"),
        gdump.GgmlType.I32.value: np.dtype("int32"),
        gdump.GgmlType.I64.value: np.dtype("int64"),
    }.get(ggml_dtype, np.dtype("float32"))
