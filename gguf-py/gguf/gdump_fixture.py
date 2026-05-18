"""Reader / runner for the ``.gfxt`` verification fixtures.

A fixture file is the output of ``llama-onnx-export-dump --fixture <path>``
(LM fixtures) or ``--vision-fixture <path>`` (vision encoder fixtures), and
contains:

For LM fixtures (v2 / v3, ``kind="lm"``):

* the input token IDs fed to ``llama_decode()``, and
* the reference logits llama.cpp computed for those tokens.

  The rest of the graph's inputs (positions, KV-cache write indices, the
  causal mask) are not stored in the fixture — they are derived in Python
  from the gdump's graph structure (which input feeds which op) plus the
  token count, so the test is bit-for-bit reproducible without needing
  llama.cpp to expose its post-decode tensor state.

For vision fixtures (v4, ``kind="vis"``):

* the deterministic input image (random fp32 pixels in HWC layout), and
* the projected visual features the clip encoder produced for them.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from . import gdump


_FIXTURE_MAGIC = 0x54584647  # "GFXT"
_FIXTURE_VERSION_MIN = 2
_FIXTURE_VERSION_MAX = 4


@dataclass
class Intermediate:
    """One ggml-side captured intermediate tensor.

    `ne` is the ggml shape (length-4, fastest-varying first). `data` is row-
    major fp32 with the trailing singleton dims dropped (ready for direct
    comparison against the ONNX side, which produces row-major arrays).
    """
    name: str
    ne: list[int]
    data: np.ndarray  # fp32, row-major, shape = list(reversed(ne_nontrivial))


@dataclass
class Fixture:
    # Discriminator for which fields are populated. LM fixtures (v2 / v3) use
    # token_ids / logits / intermediates; vision fixtures (v4) use input_pixels
    # / output_features.
    kind: Literal["lm", "vision"] = "lm"

    # LM-specific fields.
    n_tokens: int = 0
    n_vocab: int = 0
    token_ids: Optional[np.ndarray] = None    # int32, (n_tokens,)
    logits: Optional[np.ndarray] = None       # float32, (n_tokens, n_vocab)
    intermediates: list["Intermediate"] = field(default_factory=list)

    # Vision-specific fields.
    input_pixels: Optional[np.ndarray] = None      # float32, (H, W, C) by default
    output_features: Optional[np.ndarray] = None   # float32, encoder output
    output_name: str = ""


def _read_exact(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"short read: wanted {n}, got {len(b)}")
    return b


def _read_shape(f) -> list[int]:
    (rank,) = struct.unpack("<I", _read_exact(f, 4))
    dims = list(struct.unpack(f"<{rank}Q", _read_exact(f, rank * 8))) if rank else []
    return dims


def load(path: str | Path) -> Fixture:
    with open(path, "rb") as f:
        magic, version = struct.unpack("<II", _read_exact(f, 8))
        if magic != _FIXTURE_MAGIC:
            raise ValueError(f"bad fixture magic 0x{magic:08x}")
        if not (_FIXTURE_VERSION_MIN <= version <= _FIXTURE_VERSION_MAX):
            raise ValueError(f"unsupported fixture version {version} (this build understands "
                             f"{_FIXTURE_VERSION_MIN}..{_FIXTURE_VERSION_MAX})")

        # v4 introduces an explicit 4-byte kind tag right after the version.
        # v2 / v3 are LM-only (no tag).
        if version >= 4:
            kind_tag = _read_exact(f, 4).rstrip(b"\x00").decode("ascii", errors="replace")
        else:
            kind_tag = "lm"

        if kind_tag in ("", "lm"):
            return _load_lm(f, version)
        if kind_tag == "vis":
            return _load_vision(f)
        raise ValueError(f"unknown fixture kind tag {kind_tag!r}")


def _load_lm(f, version: int) -> Fixture:
    n_tokens, n_vocab = struct.unpack("<II", _read_exact(f, 8))
    tokens = np.frombuffer(_read_exact(f, n_tokens * 4), dtype=np.int32).copy()
    (logits_nbytes,) = struct.unpack("<Q", _read_exact(f, 8))
    expected = n_tokens * n_vocab * 4
    if logits_nbytes != expected:
        raise ValueError(f"logits size mismatch: got {logits_nbytes}, expected {expected}")
    logits = np.frombuffer(_read_exact(f, logits_nbytes), dtype=np.float32).copy()
    logits = logits.reshape(n_tokens, n_vocab)

    intermediates: list[Intermediate] = []
    if version >= 3:
        (n_entries,) = struct.unpack("<I", _read_exact(f, 4))
        for _ in range(n_entries):
            (name_len,) = struct.unpack("<I", _read_exact(f, 4))
            name = _read_exact(f, name_len).decode("utf-8", errors="replace")
            ne = list(struct.unpack("<4q", _read_exact(f, 32)))
            (n_elements,) = struct.unpack("<Q", _read_exact(f, 8))
            data = np.frombuffer(_read_exact(f, n_elements * 4), dtype=np.float32).copy()
            shape = list(reversed([d for d in ne if d > 1])) or [1]
            data = data.reshape(shape)
            intermediates.append(Intermediate(name=name, ne=ne, data=data))

    return Fixture(kind="lm", n_tokens=n_tokens, n_vocab=n_vocab,
                   token_ids=tokens, logits=logits, intermediates=intermediates)


def _load_vision(f) -> Fixture:
    in_shape = _read_shape(f)
    in_count = 1
    for d in in_shape:
        in_count *= int(d)
    input_pixels = np.frombuffer(_read_exact(f, in_count * 4), dtype=np.float32).copy()
    input_pixels = input_pixels.reshape(in_shape)

    (name_len,) = struct.unpack("<I", _read_exact(f, 4))
    output_name = _read_exact(f, name_len).decode("utf-8", errors="replace")

    out_shape = _read_shape(f)
    out_count = 1
    for d in out_shape:
        out_count *= int(d)
    output_features = np.frombuffer(_read_exact(f, out_count * 4), dtype=np.float32).copy()
    output_features = output_features.reshape(out_shape)

    return Fixture(kind="vision",
                   input_pixels=input_pixels,
                   output_features=output_features,
                   output_name=output_name)


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
