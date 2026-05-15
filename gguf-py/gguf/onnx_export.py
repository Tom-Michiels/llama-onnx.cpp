"""
GGUF -> ONNX export for the LLaMA architecture.

Reads a GGUF file produced by llama.cpp's converters and emits an ONNX model
that implements the same forward pass:

  - Token embedding (Gather)
  - For each block: RMSNorm -> Q/K/V projections -> RoPE -> grouped-query
    self-attention with KV cache -> output projection -> residual -> RMSNorm
    -> SwiGLU FFN -> residual
  - Final RMSNorm -> LM head

The emitted graph takes ``input_ids`` and ``position_ids`` together with a pair
of KV cache tensors per layer, and returns logits plus the updated KV cache.
This is enough to drive incremental token generation from a runtime such as
onnxruntime.

Only floating-point coefficient types (F32, F16, BF16, F64) are supported.
BF16 is widened to the export dtype on load because most ONNX runtimes do not
implement BF16 ops.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .constants import GGMLQuantizationType, Keys, TENSOR_NAMES, MODEL_TENSOR
from .gguf_reader import GGUFReader, ReaderTensor


logger = logging.getLogger(__name__)


FLOAT_GGML_TYPES = {
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.F64,
    GGMLQuantizationType.BF16,
}


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


@dataclass
class LlamaHParams:
    arch: str
    n_layer: int
    n_embd: int
    n_ff: int
    n_head: int
    n_head_kv: int
    head_dim: int
    n_rot: int          # RoPE dim; equals head_dim for llama
    vocab_size: int
    rms_eps: float
    rope_base: float
    rope_scale: float   # linear scaling factor (1.0 for vanilla llama)
    context_length: int

    @property
    def n_kv_repeat(self) -> int:
        if self.n_head % self.n_head_kv != 0:
            raise ValueError(
                f"n_head ({self.n_head}) is not divisible by n_head_kv ({self.n_head_kv})"
            )
        return self.n_head // self.n_head_kv


def _required(reader: GGUFReader, key: str):
    field = reader.get_field(key)
    if field is None:
        raise KeyError(f"missing required GGUF key: {key}")
    return field.contents()


def _optional(reader: GGUFReader, key: str, default):
    field = reader.get_field(key)
    if field is None:
        return default
    return field.contents()


def read_llama_hparams(reader: GGUFReader) -> LlamaHParams:
    arch = _required(reader, Keys.General.ARCHITECTURE)
    if arch != "llama":
        raise ValueError(
            f"unsupported architecture {arch!r}; this exporter currently only handles 'llama'"
        )

    def k(template: str) -> str:
        return template.format(arch=arch)

    n_layer   = int(_required(reader, k(Keys.LLM.BLOCK_COUNT)))
    n_embd    = int(_required(reader, k(Keys.LLM.EMBEDDING_LENGTH)))
    n_ff      = int(_required(reader, k(Keys.LLM.FEED_FORWARD_LENGTH)))
    n_head    = int(_required(reader, k(Keys.Attention.HEAD_COUNT)))
    n_head_kv = int(_optional(reader, k(Keys.Attention.HEAD_COUNT_KV), n_head))

    key_length = _optional(reader, k(Keys.Attention.KEY_LENGTH), None)
    head_dim = int(key_length) if key_length is not None else n_embd // n_head
    n_rot = int(_optional(reader, k(Keys.Rope.DIMENSION_COUNT), head_dim))

    rms_eps = float(_required(reader, k(Keys.Attention.LAYERNORM_RMS_EPS)))
    rope_base = float(_optional(reader, k(Keys.Rope.FREQ_BASE), 10000.0))
    rope_scale_factor = float(_optional(reader, k(Keys.Rope.SCALING_FACTOR), 1.0))
    # GGUF stores the scaling *factor* (>= 1 stretches positions); the actual
    # frequency scale applied multiplicatively to theta is 1/factor.
    rope_scale = 1.0 / rope_scale_factor if rope_scale_factor != 0.0 else 1.0

    context_length = int(_optional(reader, k(Keys.LLM.CONTEXT_LENGTH), 0))

    # Vocab size: prefer the explicit key, fall back to the tokens array.
    vocab_size = _optional(reader, k(Keys.LLM.VOCAB_SIZE), None)
    if vocab_size is None:
        tok_list = reader.get_field(Keys.Tokenizer.LIST)
        if tok_list is None:
            raise KeyError("cannot determine vocab size: neither vocab_size key nor tokenizer.ggml.tokens present")
        vocab_size = len(tok_list.data)
    vocab_size = int(vocab_size)

    return LlamaHParams(
        arch=arch,
        n_layer=n_layer,
        n_embd=n_embd,
        n_ff=n_ff,
        n_head=n_head,
        n_head_kv=n_head_kv,
        head_dim=head_dim,
        n_rot=n_rot,
        vocab_size=vocab_size,
        rms_eps=rms_eps,
        rope_base=rope_base,
        rope_scale=rope_scale,
        context_length=context_length,
    )


# ---------------------------------------------------------------------------
# Tensor loading
# ---------------------------------------------------------------------------


def _bf16_to_f32(raw_bytes: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Convert a BF16 byte buffer to a float32 numpy array.

    BF16 is the upper 16 bits of an IEEE-754 float32, so we just shift the
    bits into place.
    """
    if raw_bytes.dtype != np.uint8:
        raw_bytes = raw_bytes.view(np.uint8)
    as_u16 = np.frombuffer(raw_bytes.tobytes(), dtype=np.uint16)
    as_u32 = as_u16.astype(np.uint32) << 16
    return as_u32.view(np.float32).reshape(shape).copy()


def tensor_to_numpy(t: ReaderTensor) -> np.ndarray:
    """Return a ReaderTensor's contents as a numpy float array in its natural dtype."""
    qt = t.tensor_type
    if qt not in FLOAT_GGML_TYPES:
        raise ValueError(
            f"tensor {t.name!r}: unsupported non-float dtype {qt!r}; "
            "this exporter currently requires floating-point coefficients"
        )
    if qt == GGMLQuantizationType.BF16:
        # ReaderTensor.data has the byte shape (last dim doubled); use the
        # logical shape from the GGUF file (reversed dims) for the result.
        logical_shape = tuple(reversed(t.shape.tolist()))
        return _bf16_to_f32(t.data, logical_shape)
    # F16 / F32 / F64 already arrive with the correct numpy dtype.
    return np.ascontiguousarray(t.data)


def collect_tensors(reader: GGUFReader) -> dict[str, ReaderTensor]:
    by_name: dict[str, ReaderTensor] = {}
    for t in reader.tensors:
        if t.name in by_name:
            raise ValueError(f"duplicate tensor {t.name}")
        by_name[t.name] = t
    return by_name


# ---------------------------------------------------------------------------
# ONNX graph helpers
# ---------------------------------------------------------------------------


# Mapping from numpy dtypes that we use to ONNX TensorProto types.
_NP_TO_ONNX_DTYPE = {
    np.dtype("float32"): TensorProto.FLOAT,
    np.dtype("float16"): TensorProto.FLOAT16,
    np.dtype("int64"):   TensorProto.INT64,
    np.dtype("int32"):   TensorProto.INT32,
    np.dtype("bool"):    TensorProto.BOOL,
}


def _np_to_onnx_dtype(dtype: np.dtype) -> int:
    try:
        return _NP_TO_ONNX_DTYPE[np.dtype(dtype)]
    except KeyError as e:
        raise ValueError(f"no ONNX TensorProto type for numpy dtype {dtype!r}") from e


class GraphBuilder:
    """Imperatively build an ONNX graph with auto-named intermediates."""

    def __init__(self):
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.inputs: list[onnx.ValueInfoProto] = []
        self.outputs: list[onnx.ValueInfoProto] = []
        self._counters: dict[str, int] = {}
        self._used_names: set[str] = set()

    def fresh(self, hint: str) -> str:
        n = self._counters.get(hint, 0)
        while True:
            name = f"{hint}_{n}"
            n += 1
            if name not in self._used_names:
                self._used_names.add(name)
                self._counters[hint] = n
                return name

    def reserve(self, name: str) -> str:
        if name in self._used_names:
            raise ValueError(f"name {name!r} already used")
        self._used_names.add(name)
        return name

    def initializer(self, name: str, array: np.ndarray) -> str:
        """Add an initializer with a caller-chosen name. The caller is
        responsible for ensuring the name is unique."""
        if name not in self._used_names:
            self.reserve(name)
        tp = numpy_helper.from_array(array, name=name)
        self.initializers.append(tp)
        return name

    def constant(self, hint: str, array: np.ndarray) -> str:
        """Create an initializer with an auto-generated name."""
        name = self.fresh(hint)
        tp = numpy_helper.from_array(array, name=name)
        self.initializers.append(tp)
        return name

    def input(self, name: str, dtype: int, shape: list) -> str:
        self.reserve(name)
        vi = helper.make_tensor_value_info(name, dtype, shape)
        self.inputs.append(vi)
        return name

    def output(self, name: str, dtype: int, shape: list) -> str:
        # Outputs of the graph are produced by some node; we just record the
        # ValueInfo here. Name must already be unique.
        vi = helper.make_tensor_value_info(name, dtype, shape)
        self.outputs.append(vi)
        return name

    def node(
        self,
        op_type: str,
        inputs: list[str],
        out_hint: str | list[str],
        *,
        name: str | None = None,
        **attrs,
    ) -> str | list[str]:
        """Append a node. ``out_hint`` may be a string (single output) or a list
        of strings (one hint per output). All extra kwargs become ONNX attributes.
        """
        if isinstance(out_hint, list):
            output_names = [self.fresh(h) for h in out_hint]
            multi = True
        else:
            output_names = [self.fresh(out_hint)]
            multi = False
        n = helper.make_node(
            op_type,
            inputs=inputs,
            outputs=output_names,
            name=name or self.fresh(f"{op_type}_node"),
            **attrs,
        )
        self.nodes.append(n)
        return output_names if multi else output_names[0]


# ---------------------------------------------------------------------------
# Building blocks of the llama graph
# ---------------------------------------------------------------------------


class LlamaOnnxBuilder:
    """Build the ONNX llama graph block by block.

    The graph operates in the precision selected via ``compute_dtype`` (FP32 by
    default). Weight initializers are also stored at that precision; non-matching
    GGUF tensors are cast on load.
    """

    OPSET = 17

    def __init__(self, hparams: LlamaHParams, tensors: dict[str, ReaderTensor], compute_dtype: np.dtype):
        self.h = hparams
        self.tensors = tensors
        self.dtype = np.dtype(compute_dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float16")):
            raise ValueError(f"compute_dtype must be float32 or float16, not {self.dtype}")
        self.onnx_dtype = _np_to_onnx_dtype(self.dtype)
        self.g = GraphBuilder()

        # Useful scalar / shape constants reused everywhere.
        self._scalar_cache: dict[tuple, str] = {}
        self._shape_cache: dict[tuple, str] = {}

    # -- weight access ----------------------------------------------------

    def _get_tensor(self, name: str, required: bool = True) -> np.ndarray | None:
        t = self.tensors.get(name)
        if t is None:
            if required:
                raise KeyError(f"missing required tensor {name!r}")
            return None
        arr = tensor_to_numpy(t)
        return arr.astype(self.dtype, copy=False)

    def _weight_initializer(self, gguf_name: str, *, required: bool = True) -> str | None:
        arr = self._get_tensor(gguf_name, required=required)
        if arr is None:
            return None
        return self.g.initializer(gguf_name.replace(".", "_"), arr)

    # -- small constants --------------------------------------------------

    def _scalar(self, value: float | int, np_dtype: np.dtype) -> str:
        key = (np_dtype.kind, np_dtype.itemsize, value)
        if key in self._scalar_cache:
            return self._scalar_cache[key]
        arr = np.array(value, dtype=np_dtype)
        name = self.g.constant("scalar", arr)
        self._scalar_cache[key] = name
        return name

    def _const_int64(self, values: list[int] | tuple[int, ...]) -> str:
        key = ("i64", tuple(values))
        if key in self._shape_cache:
            return self._shape_cache[key]
        arr = np.array(list(values), dtype=np.int64)
        name = self.g.constant("shape", arr)
        self._shape_cache[key] = name
        return name

    # -- ops --------------------------------------------------------------

    def _rms_norm(self, x: str, weight_name: str, eps: float) -> str:
        """Apply x = x * rsqrt(mean(x^2) + eps) * weight, computed in fp32 for stability."""
        # The eps add and rsqrt are sensitive to precision, so do them in fp32
        # even when the surrounding compute is fp16. Mirrors llama.cpp's behaviour.
        x_f32 = self.g.node("Cast", [x], "x_f32", to=TensorProto.FLOAT)
        sq = self.g.node("Mul", [x_f32, x_f32], "rmsnorm_sq")
        ms = self.g.node("ReduceMean", [sq], "rmsnorm_mean", axes=[-1], keepdims=1)
        eps_c = self._scalar(eps, np.dtype("float32"))
        ms_eps = self.g.node("Add", [ms, eps_c], "rmsnorm_msEps")
        inv = self.g.node("Reciprocal",
                          [self.g.node("Sqrt", [ms_eps], "rmsnorm_rms")],
                          "rmsnorm_inv")
        normed_f32 = self.g.node("Mul", [x_f32, inv], "rmsnorm_normed")
        normed = self.g.node("Cast", [normed_f32], "rmsnorm_cast", to=self.onnx_dtype)
        return self.g.node("Mul", [normed, weight_name], "rmsnorm_out")

    def _matmul_proj(self, x: str, weight_2d: np.ndarray, name_hint: str) -> str:
        """MatMul a [..., in_features] tensor by a [out, in] weight.

        The GGUF tensor for a linear layer has shape ``[out_features, in_features]``.
        We transpose to ``[in, out]`` and store as an initializer, then MatMul.
        """
        w = np.ascontiguousarray(weight_2d.T)
        w_name = self.g.constant(name_hint, w)
        return self.g.node("MatMul", [x, w_name], f"{name_hint}_mm")

    def _matmul_proj_named(self, x: str, gguf_name: str) -> str:
        arr = self._get_tensor(gguf_name)
        return self._matmul_proj(x, arr, gguf_name.replace(".", "_"))

    # -- RoPE -------------------------------------------------------------

    def _build_rope_inv_freqs(self) -> str:
        """Pre-compute the per-pair inverse frequency vector (length n_rot/2).

        Includes the Llama-3 style frequency rescaling (``rope_freqs.weight``)
        when that tensor is present in the GGUF file.
        """
        n_rot = self.h.n_rot
        if n_rot % 2 != 0:
            raise ValueError(f"n_rot must be even, got {n_rot}")
        base = float(self.h.rope_base)
        inv_freqs = 1.0 / (base ** (np.arange(0, n_rot, 2, dtype=np.float64) / n_rot))
        inv_freqs = inv_freqs * float(self.h.rope_scale)
        rope_factors_name = TENSOR_NAMES[MODEL_TENSOR.ROPE_FREQS]
        # GGUF stores it as `rope_freqs.weight` at the top level.
        rope_factors = self.tensors.get(f"{rope_factors_name}.weight")
        if rope_factors is not None:
            rf = tensor_to_numpy(rope_factors).astype(np.float64)
            if rf.shape != (n_rot // 2,):
                raise ValueError(
                    f"rope_freqs.weight has shape {rf.shape}, expected ({n_rot // 2},)"
                )
            inv_freqs = inv_freqs / rf
        return self.g.constant("rope_inv_freqs", inv_freqs.astype(np.float32))

    def _build_rope_cos_sin(self, position_ids: str, inv_freqs_name: str) -> tuple[str, str]:
        """Return cos/sin tensors of shape [B, S, 1, n_rot/2] in compute dtype."""
        pos_f32 = self.g.node("Cast", [position_ids], "pos_f32", to=TensorProto.FLOAT)
        # pos_f32: [B, S]
        # inv_freqs: [n_rot/2]
        # angles: [B, S, n_rot/2] = pos[..., None] * inv_freqs[None, None, :]
        pos_exp = self.g.node("Unsqueeze", [pos_f32, self._const_int64([-1])], "pos_exp")
        inv_exp = self.g.node("Unsqueeze", [inv_freqs_name, self._const_int64([0, 1])], "inv_exp")
        angles = self.g.node("Mul", [pos_exp, inv_exp], "rope_angles")
        cos = self.g.node("Cos", [angles], "rope_cos_f32")
        sin = self.g.node("Sin", [angles], "rope_sin_f32")
        # Unsqueeze a head-axis: [B, S, 1, n_rot/2]
        axes_h = self._const_int64([2])
        cos = self.g.node("Unsqueeze", [cos, axes_h], "rope_cos_bsh")
        sin = self.g.node("Unsqueeze", [sin, axes_h], "rope_sin_bsh")
        if self.onnx_dtype != TensorProto.FLOAT:
            cos = self.g.node("Cast", [cos], "rope_cos", to=self.onnx_dtype)
            sin = self.g.node("Cast", [sin], "rope_sin", to=self.onnx_dtype)
        return cos, sin

    def _apply_rope_norm(self, x: str, cos: str, sin: str, n_heads: int) -> str:
        """Apply NORM-style (consecutive-pair) RoPE to x of shape [B, S, H, D].

        Pairs are ``(x[..., 2i], x[..., 2i+1])`` for ``i = 0..D/2-1``. This is
        what ``LLAMA_ROPE_TYPE_NORM`` does in llama.cpp.
        """
        head_dim = self.h.head_dim
        half = self.h.n_rot // 2
        # Reshape last dim from [D] to [D/2, 2] (assumes n_rot == head_dim, which
        # holds for llama).
        if self.h.n_rot != head_dim:
            raise NotImplementedError(
                "partial RoPE (n_rot != head_dim) is not implemented for llama"
            )
        new_shape = self._const_int64([0, 0, n_heads, half, 2])
        x_pairs = self.g.node("Reshape", [x, new_shape], "rope_x_pairs")
        # Split last axis into two halves (opset 17 derives count from outputs).
        x_even, x_odd = self.g.node(
            "Split",
            [x_pairs],
            ["x_even", "x_odd"],
            axis=-1,
        )
        # Squeeze the trailing length-1 axis.
        sq_axis = self._const_int64([-1])
        x_even = self.g.node("Squeeze", [x_even, sq_axis], "x_even_sq")
        x_odd  = self.g.node("Squeeze", [x_odd,  sq_axis], "x_odd_sq")
        # cos/sin: [B, S, 1, D/2]; broadcast over heads.
        e_cos = self.g.node("Mul", [x_even, cos], "rope_e_cos")
        o_sin = self.g.node("Mul", [x_odd,  sin], "rope_o_sin")
        e_sin = self.g.node("Mul", [x_even, sin], "rope_e_sin")
        o_cos = self.g.node("Mul", [x_odd,  cos], "rope_o_cos")
        new_even = self.g.node("Sub", [e_cos, o_sin], "rope_new_even")
        new_odd  = self.g.node("Add", [e_sin, o_cos], "rope_new_odd")
        # Stack to [B, S, H, D/2, 2] then reshape back to [B, S, H, D].
        ne_u = self.g.node("Unsqueeze", [new_even, sq_axis], "ne_u")
        no_u = self.g.node("Unsqueeze", [new_odd,  sq_axis], "no_u")
        stacked = self.g.node("Concat", [ne_u, no_u], "rope_stacked", axis=-1)
        flat_shape = self._const_int64([0, 0, n_heads, head_dim])
        return self.g.node("Reshape", [stacked, flat_shape], "rope_out")

    # -- attention --------------------------------------------------------

    def _repeat_kv(self, t: str, n_rep: int) -> str:
        """Replicate the KV head axis ``n_rep`` times.

        Input shape:  ``[B, H_kv, T, D]``
        Output shape: ``[B, H_kv * n_rep, T, D]``
        """
        if n_rep == 1:
            return t
        # Unsqueeze head axis to [B, H_kv, 1, T, D], expand to [B, H_kv, n_rep, T, D]
        # via tile, then reshape to [B, H_kv*n_rep, T, D]. Using Tile keeps shapes
        # symbolic at the batch/seq dimensions.
        u = self.g.node("Unsqueeze", [t, self._const_int64([2])], "kv_u")
        repeats = self._const_int64([1, 1, n_rep, 1, 1])
        tiled = self.g.node("Tile", [u, repeats], "kv_tile")
        # Final shape: merge axis 1 and 2.
        # Use a Shape/Slice trick so batch, T, D stay dynamic.
        shape_t = self.g.node("Shape", [t], "kv_shape")
        # We need [B, H_kv*n_rep, T, D]; build it from shape_t.
        b   = self.g.node("Gather", [shape_t, self._const_int64([0])], "b", axis=0)
        h   = self.g.node("Gather", [shape_t, self._const_int64([1])], "hkv", axis=0)
        seq = self.g.node("Gather", [shape_t, self._const_int64([2])], "tlen", axis=0)
        d   = self.g.node("Gather", [shape_t, self._const_int64([3])], "d", axis=0)
        nrep = self._const_int64([n_rep])
        hnew = self.g.node("Mul", [h, nrep], "h_new")
        new_shape = self.g.node("Concat", [b, hnew, seq, d], "kv_new_shape", axis=0)
        return self.g.node("Reshape", [tiled, new_shape], "kv_rep_out")

    def _causal_mask(self, q_seq_len_name: str, total_seq_len_name: str) -> str:
        """Build an additive causal mask of shape ``[1, 1, q_len, total_len]``.

        The query at index ``q`` (0-indexed within the new tokens) is allowed to
        attend to keys at indices ``[0, past_len + q]`` where
        ``past_len = total_len - q_len``.
        """
        # past_len = total_len - q_len
        past_len = self.g.node("Sub", [total_seq_len_name, q_seq_len_name], "past_len")
        zero = self._const_int64([0])
        one  = self._const_int64([1])
        # We need scalar inputs to Range, so reshape to scalar shape [].
        scalar_shape = self._const_int64([])
        past_len_s = self.g.node("Reshape", [past_len, scalar_shape], "past_len_s")
        total_s = self.g.node("Reshape", [total_seq_len_name, scalar_shape], "total_s")
        zero_s = self.g.node("Reshape", [self._const_int64([0]), scalar_shape], "zero_s")
        one_s = self.g.node("Reshape", [one, scalar_shape], "one_s")
        # q_pos = arange(past_len, total_len)   -> [q_len]
        # k_pos = arange(0, total_len)          -> [total_len]
        q_pos = self.g.node("Range", [past_len_s, total_s, one_s], "q_pos")
        k_pos = self.g.node("Range", [zero_s, total_s, one_s], "k_pos")
        # mask[q, k] = (k > past_len + q)  --> True means "masked"
        q_pos_u = self.g.node("Unsqueeze", [q_pos, self._const_int64([1])], "q_pos_u")  # [q_len, 1]
        k_pos_u = self.g.node("Unsqueeze", [k_pos, self._const_int64([0])], "k_pos_u")  # [1, total_len]
        masked = self.g.node("Greater", [k_pos_u, q_pos_u], "masked")
        # masked: [q_len, total_len] bool
        masked_f = self.g.node("Cast", [masked], "masked_f", to=self.onnx_dtype)
        # finfo().min keeps softmax happy in fp16 (-1e30 overflows to -inf and
        # then softmax produces NaN).
        neg_inf = self._scalar(float(np.finfo(self.dtype).min), self.dtype)
        add_mask = self.g.node("Mul", [masked_f, neg_inf], "add_mask")
        # Unsqueeze to [1, 1, q_len, total_len].
        add_mask = self.g.node("Unsqueeze", [add_mask, self._const_int64([0, 1])], "add_mask_b")
        return add_mask

    def _attention(
        self,
        q: str, k: str, v: str,
        past_k: str, past_v: str,
    ) -> tuple[str, str, str]:
        """Run the GQA self-attention block.

        Inputs:
          ``q``: [B, S, n_head, head_dim]
          ``k``, ``v``: [B, S, n_kv_head, head_dim]
          ``past_k``, ``past_v``: [B, n_kv_head, past_len, head_dim]
        Outputs: ``(attn_out [B, S, n_head*head_dim], present_k, present_v)``.
        """
        # Transpose to [B, H, S, D].
        perm = [0, 2, 1, 3]
        q_t = self.g.node("Transpose", [q], "q_t", perm=perm)
        k_t = self.g.node("Transpose", [k], "k_t", perm=perm)
        v_t = self.g.node("Transpose", [v], "v_t", perm=perm)
        # Append to KV cache (axis=2 = time).
        present_k = self.g.node("Concat", [past_k, k_t], "present_k", axis=2)
        present_v = self.g.node("Concat", [past_v, v_t], "present_v", axis=2)
        # Repeat for GQA.
        k_rep = self._repeat_kv(present_k, self.h.n_kv_repeat)
        v_rep = self._repeat_kv(present_v, self.h.n_kv_repeat)
        # Scores: Q @ K^T / sqrt(D)
        k_rep_t = self.g.node("Transpose", [k_rep], "k_rep_T", perm=[0, 1, 3, 2])
        scores = self.g.node("MatMul", [q_t, k_rep_t], "scores")
        inv_sqrt_d = self._scalar(1.0 / math.sqrt(self.h.head_dim), self.dtype)
        scores = self.g.node("Mul", [scores, inv_sqrt_d], "scores_scaled")
        # Causal mask.
        q_shape = self.g.node("Shape", [q_t], "q_shape")
        k_shape = self.g.node("Shape", [present_k], "k_shape")
        q_len = self.g.node("Gather", [q_shape, self._const_int64([2])], "q_len", axis=0)
        total_len = self.g.node("Gather", [k_shape, self._const_int64([2])], "total_len", axis=0)
        add_mask = self._causal_mask(q_len, total_len)
        scores = self.g.node("Add", [scores, add_mask], "scores_masked")
        attn = self.g.node("Softmax", [scores], "attn_softmax", axis=-1)
        # Multiply with V.
        ctx = self.g.node("MatMul", [attn, v_rep], "ctx")  # [B, H, S, D]
        # Transpose back to [B, S, H, D] and merge to [B, S, H*D].
        ctx = self.g.node("Transpose", [ctx], "ctx_BSHD", perm=[0, 2, 1, 3])
        merged_shape = self._const_int64([0, 0, self.h.n_head * self.h.head_dim])
        ctx = self.g.node("Reshape", [ctx, merged_shape], "ctx_merged")
        return ctx, present_k, present_v

    # -- transformer block ------------------------------------------------

    def _block(
        self,
        hidden: str,
        il: int,
        position_ids: str,
        inv_freqs: str,
        past_k: str,
        past_v: str,
    ) -> tuple[str, str, str]:
        h = self.h
        attn_norm_w = self._weight_initializer(f"blk.{il}.attn_norm.weight")
        x = self._rms_norm(hidden, attn_norm_w, h.rms_eps)
        # Q/K/V projections.
        q = self._matmul_proj_named(x, f"blk.{il}.attn_q.weight")
        k = self._matmul_proj_named(x, f"blk.{il}.attn_k.weight")
        v = self._matmul_proj_named(x, f"blk.{il}.attn_v.weight")
        # Reshape to [B, S, H, D].
        q_shape = self._const_int64([0, 0, h.n_head, h.head_dim])
        kv_shape = self._const_int64([0, 0, h.n_head_kv, h.head_dim])
        q = self.g.node("Reshape", [q, q_shape], f"l{il}_q_resh")
        k = self.g.node("Reshape", [k, kv_shape], f"l{il}_k_resh")
        v = self.g.node("Reshape", [v, kv_shape], f"l{il}_v_resh")
        # RoPE.
        cos, sin = self._build_rope_cos_sin(position_ids, inv_freqs)
        q = self._apply_rope_norm(q, cos, sin, h.n_head)
        k = self._apply_rope_norm(k, cos, sin, h.n_head_kv)
        # Attention.
        attn_out, present_k, present_v = self._attention(q, k, v, past_k, past_v)
        # Output projection and residual.
        attn_out = self._matmul_proj_named(attn_out, f"blk.{il}.attn_output.weight")
        x = self.g.node("Add", [hidden, attn_out], f"l{il}_attn_residual")
        # FFN.
        ffn_norm_w = self._weight_initializer(f"blk.{il}.ffn_norm.weight")
        ffn_in = self._rms_norm(x, ffn_norm_w, h.rms_eps)
        gate = self._matmul_proj_named(ffn_in, f"blk.{il}.ffn_gate.weight")
        up = self._matmul_proj_named(ffn_in, f"blk.{il}.ffn_up.weight")
        # SwiGLU: silu(gate) * up
        silu_gate = self.g.node("Mul",
                                [gate, self.g.node("Sigmoid", [gate], f"l{il}_silu_sig")],
                                f"l{il}_silu")
        glu = self.g.node("Mul", [silu_gate, up], f"l{il}_glu")
        ffn_out = self._matmul_proj_named(glu, f"blk.{il}.ffn_down.weight")
        out = self.g.node("Add", [x, ffn_out], f"l{il}_block_out")
        return out, present_k, present_v

    # -- whole model ------------------------------------------------------

    def build(self, *, model_name: str = "llama_from_gguf") -> onnx.ModelProto:
        h = self.h
        g = self.g
        # Inputs.
        input_ids = g.input("input_ids", TensorProto.INT64, ["batch", "seq_len"])
        position_ids = g.input("position_ids", TensorProto.INT64, ["batch", "seq_len"])
        past_k_names: list[str] = []
        past_v_names: list[str] = []
        kv_shape = ["batch", h.n_head_kv, "past_len", h.head_dim]
        for il in range(h.n_layer):
            past_k_names.append(g.input(f"past_key_values.{il}.key", self.onnx_dtype, list(kv_shape)))
            past_v_names.append(g.input(f"past_key_values.{il}.value", self.onnx_dtype, list(kv_shape)))

        # Embedding.
        tok_embd_name = self._weight_initializer("token_embd.weight")
        # token_embd has shape [vocab_size, n_embd]; Gather along axis=0.
        hidden = g.node("Gather", [tok_embd_name, input_ids], "tok_embed", axis=0)

        # Pre-compute RoPE inverse frequencies once and re-use for all layers.
        inv_freqs = self._build_rope_inv_freqs()

        present_k_names: list[str] = []
        present_v_names: list[str] = []
        for il in range(h.n_layer):
            hidden, pk, pv = self._block(
                hidden, il, position_ids, inv_freqs,
                past_k_names[il], past_v_names[il],
            )
            present_k_names.append(pk)
            present_v_names.append(pv)

        # Final norm and LM head.
        out_norm_w = self._weight_initializer("output_norm.weight")
        hidden = self._rms_norm(hidden, out_norm_w, h.rms_eps)
        if self.tensors.get("output.weight") is not None:
            # Use the dedicated LM-head weight from the GGUF file.
            logits = self._matmul_proj_named(hidden, "output.weight")
        else:
            # Llama 3.2 1B (and most 1B/3B models) tie the LM head to the token
            # embedding. Reuse the initializer via a Transpose so we don't have
            # to keep two copies of a 250M-parameter table in the file.
            lm_head_T = g.node("Transpose", [tok_embd_name], "lm_head_T", perm=[1, 0])
            logits = g.node("MatMul", [hidden, lm_head_T], "logits")

        # Outputs.
        g.output("logits", self.onnx_dtype, ["batch", "seq_len", h.vocab_size])
        # Rename the last logits node output to "logits".
        # Trick: emit an Identity to give it the canonical output name.
        ident = helper.make_node("Identity", inputs=[logits], outputs=["logits"], name="logits_out")
        g.nodes.append(ident)

        for il in range(h.n_layer):
            pk_out = f"present.{il}.key"
            pv_out = f"present.{il}.value"
            g.nodes.append(helper.make_node("Identity", [present_k_names[il]], [pk_out], name=f"pk_out_{il}"))
            g.nodes.append(helper.make_node("Identity", [present_v_names[il]], [pv_out], name=f"pv_out_{il}"))
            g.output(pk_out, self.onnx_dtype, ["batch", h.n_head_kv, "total_len", h.head_dim])
            g.output(pv_out, self.onnx_dtype, ["batch", h.n_head_kv, "total_len", h.head_dim])

        graph = helper.make_graph(
            nodes=g.nodes,
            name=model_name,
            inputs=g.inputs,
            outputs=g.outputs,
            initializer=g.initializers,
        )
        opset = helper.make_opsetid("", self.OPSET)
        model = helper.make_model(graph, opset_imports=[opset], producer_name="llama-onnx.cpp")
        model.ir_version = 8  # broadly compatible with onnxruntime 1.15+
        return model


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def convert(
    gguf_path: str | Path,
    onnx_path: str | Path,
    *,
    dtype: str = "float32",
    external_threshold: int = 1024,
) -> None:
    gguf_path = Path(gguf_path)
    onnx_path = Path(onnx_path)
    if not gguf_path.is_file():
        raise FileNotFoundError(gguf_path)

    logger.info("loading GGUF: %s", gguf_path)
    reader = GGUFReader(str(gguf_path))

    hparams = read_llama_hparams(reader)
    logger.info(
        "llama: n_layer=%d n_embd=%d n_head=%d n_kv=%d head_dim=%d vocab=%d ctx=%d",
        hparams.n_layer, hparams.n_embd, hparams.n_head, hparams.n_head_kv,
        hparams.head_dim, hparams.vocab_size, hparams.context_length,
    )

    tensors = collect_tensors(reader)
    # Sanity-check: all tensors we'll use must be floating-point.
    needed_prefixes = ("blk.",)
    needed_globals = (
        "token_embd.weight", "output_norm.weight",
        "output.weight", "rope_freqs.weight",
    )
    for name, t in tensors.items():
        used = name in needed_globals or any(name.startswith(p) for p in needed_prefixes)
        if not used:
            continue
        if t.tensor_type not in FLOAT_GGML_TYPES:
            raise ValueError(
                f"tensor {name!r} has non-float dtype {t.tensor_type!r}; "
                "this exporter is restricted to floating-point GGUF files for now"
            )

    np_dtype = {"float32": np.float32, "fp32": np.float32, "f32": np.float32,
                "float16": np.float16, "fp16": np.float16, "f16": np.float16}.get(dtype.lower())
    if np_dtype is None:
        raise ValueError(f"unsupported --dtype {dtype!r}; choose float32 or float16")

    logger.info("building ONNX graph (compute dtype=%s)...", np_dtype.__name__)
    builder = LlamaOnnxBuilder(hparams, tensors, np_dtype)
    model = builder.build()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    data_location = onnx_path.name + ".data"
    logger.info("saving ONNX to %s (external data: %s)", onnx_path, data_location)
    onnx.save_model(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_location,
        size_threshold=external_threshold,
        convert_attribute=False,
    )
    logger.info("done.")
