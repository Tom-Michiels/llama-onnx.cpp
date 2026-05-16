"""Translate a ``.gdump`` produced by ``llama-onnx-export-dump`` to ONNX.

The dump is the output of `llama_graph_reserve()` (i.e. exactly what
``src/models/<arch>.cpp`` builds at runtime). This module walks the topo
list and emits one or more ONNX nodes per ggml op.

Status: first cut. Targets `llama` end-to-end on the synthetic test;
documents (and warns about) the ops it does not yet know how to translate.
"""

from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from . import gdump


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _ggml_type_to_onnx(t: int) -> int:
    """Return the ONNX TensorProto type for a ggml_type."""
    gt = gdump.GgmlType(t) if t in {x.value for x in gdump.GgmlType} else None
    return {
        gdump.GgmlType.F32: TensorProto.FLOAT,
        gdump.GgmlType.F16: TensorProto.FLOAT16,
        gdump.GgmlType.F64: TensorProto.DOUBLE,
        gdump.GgmlType.I8:  TensorProto.INT8,
        gdump.GgmlType.I16: TensorProto.INT16,
        gdump.GgmlType.I32: TensorProto.INT32,
        gdump.GgmlType.I64: TensorProto.INT64,
        gdump.GgmlType.BF16: TensorProto.FLOAT,  # widened on load
    }.get(gt, TensorProto.UNDEFINED)


def _ggml_shape_to_logical(ne: list[int]) -> list[int]:
    """Convert ggml's ne (ne[0] = fastest-varying) to a "row-major" logical
    shape suitable for numpy / ONNX. We strip trailing 1-dimensions but keep
    at least rank 1.
    """
    rank = 4
    while rank > 1 and ne[rank - 1] == 1:
        rank -= 1
    return list(reversed(ne[:rank]))


# ----------------------------------------------------------------------
# Translator
# ----------------------------------------------------------------------


class Translator:
    OPSET = 17

    def __init__(self, dump: gdump.GraphDump):
        self.dump = dump
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.inputs: list[onnx.ValueInfoProto] = []
        self.outputs: list[onnx.ValueInfoProto] = []
        # Map from gdump tensor index -> ONNX value name that holds it.
        self.value: dict[int, str] = {}
        self._name_counter = 0
        # Detect which tensors are actually consumed (so we can prune unused
        # inputs like inp_embd, when the token path is selected).
        self._consumers: dict[int, int] = {i: 0 for i in range(len(dump.tensors))}
        for t in dump.tensors:
            for s in t.sources:
                self._consumers[s] = self._consumers.get(s, 0) + 1

    # -- name allocation -------------------------------------------------

    def _fresh(self, hint: str) -> str:
        n = self._name_counter
        self._name_counter += 1
        # Sanitise so ONNX is happy (no spaces, parens, etc.).
        safe = "".join(c if c.isalnum() or c in "._" else "_" for c in hint)
        return f"{safe}_{n}"

    def _reserve(self, name: str) -> str:
        """Allocate a specific name. The caller guarantees uniqueness."""
        return name

    def _name_for(self, t: gdump.Tensor) -> str:
        return self._fresh(t.name or f"t{t.index}")

    # -- low-level ONNX emit --------------------------------------------

    def _emit(self, op: str, inputs: list[str], out_hint: str, **attrs) -> str:
        out = self._fresh(out_hint)
        self.nodes.append(helper.make_node(op, inputs=inputs, outputs=[out], **attrs))
        return out

    def _const(self, hint: str, array: np.ndarray) -> str:
        name = self._fresh(hint)
        self.initializers.append(numpy_helper.from_array(array, name=name))
        return name

    def _const_i64(self, hint: str, values: list[int]) -> str:
        return self._const(hint, np.array(values, dtype=np.int64))

    # -- top level ------------------------------------------------------

    def translate(self) -> onnx.ModelProto:
        for t in self.dump.tensors:
            self._handle(t)

        # Expose each layer's final cache state ("present_K/present_V") as an
        # ONNX output. The dump puts SET_ROWS results on tensors with a name
        # like "cache_k_l0 (view)" — we find the *latest* such write per
        # layer and rename its output to follow the standard HF naming.
        for layer in self._present_layers():
            for which in ("k", "v"):
                src_t = self._latest_cache_write(layer, which)
                if src_t is None:
                    continue
                kind = "key" if which == "k" else "value"
                name = self._reserve(f"present.{layer}.{kind}")
                self.nodes.append(helper.make_node(
                    "Identity", [self.value[src_t.index]], [name], name=f"present_{layer}_{kind}_id"))
                self.outputs.append(helper.make_tensor_value_info(
                    name, _ggml_type_to_onnx(src_t.dtype), _ggml_shape_to_logical(src_t.ne)))

        graph = helper.make_graph(
            nodes=self.nodes,
            name=f"{self.dump.arch}_from_gguf",
            inputs=self.inputs,
            outputs=self.outputs,
            initializer=self.initializers,
        )
        opset = helper.make_opsetid("", self.OPSET)
        m = helper.make_model(graph, opset_imports=[opset], producer_name="llama-onnx.cpp")
        m.ir_version = 8
        return m

    def _present_layers(self) -> list[str]:
        """Return the layer ids (as strings) that have a cache_*_l{N} leaf."""
        seen: dict[str, None] = {}
        for t in self.dump.tensors:
            if not t.is_leaf or not (t.name.startswith("cache_k_l") or t.name.startswith("cache_v_l")):
                continue
            _, layer = t.name.split("_l", 1)
            seen[layer] = None
        return list(seen.keys())

    def _latest_cache_write(self, layer: str, which: str) -> gdump.Tensor | None:
        """Find the last SET_ROWS that targets the cache for layer/which."""
        target_name = f"cache_{which}_l{layer}"
        last: gdump.Tensor | None = None
        for t in self.dump.tensors:
            if t.ggml_op != gdump.GgmlOp.SET_ROWS:
                continue
            # SET_ROWS sources: [data, idxs, target] (see ggml.c).
            target_src = self.dump.tensors[t.sources[2]]
            if target_src.name == target_name:
                last = t
        return last

    # -- per-tensor dispatch --------------------------------------------

    def _handle(self, t: gdump.Tensor) -> None:
        if t.is_leaf:
            self._handle_leaf(t)
            return

        op = t.ggml_op
        handler = _OP_HANDLERS.get(op)
        if handler is None:
            raise NotImplementedError(
                f"no ONNX translation for ggml op {op.name if op != gdump.GgmlOp.NONE else t.op}"
                f" (tensor {t.name!r})"
            )
        handler(self, t)

        # Mark outputs.
        if t.is_output:
            out = self._reserve("logits")
            # Identity to give it the canonical name.
            self.nodes.append(helper.make_node("Identity", [self.value[t.index]], [out], name="logits_out"))
            self.value[t.index] = out
            self.outputs.append(helper.make_tensor_value_info(
                out, _ggml_type_to_onnx(t.dtype), _ggml_shape_to_logical(t.ne)))

    def _handle_leaf(self, t: gdump.Tensor) -> None:
        is_cache = t.name.startswith("cache_k_l") or t.name.startswith("cache_v_l")

        if is_cache:
            # Caches are stored in the dump as zero-filled weight buffers but
            # we want them as ONNX *inputs* (past_K / past_V). We rename them
            # so the ONNX I/O surface follows the standard HF convention.
            kind, layer = t.name.split("_l", 1)
            which = "key" if kind == "cache_k" else "value"
            name = self._reserve(f"past_key_values.{layer}.{which}")
            shape = _ggml_shape_to_logical(t.ne)
            self.inputs.append(helper.make_tensor_value_info(
                name, _ggml_type_to_onnx(t.dtype), shape))
            self.value[t.index] = name
            return

        if t.has_data:
            # Weight initializer.
            name = self._fresh(t.name)
            arr = t.data.astype(np.float32) if t.dtype == gdump.GgmlType.BF16 else t.data
            self.initializers.append(numpy_helper.from_array(arr, name=name))
            self.value[t.index] = name
            return

        # Otherwise this is a graph input (inp_tokens, positions, masks, ...).
        # Some inputs may be entirely unused (e.g. `inp_embd` when the token
        # path is selected); we skip those to keep the ONNX I/O clean.
        if self._consumers.get(t.index, 0) == 0:
            logger.debug("skipping unused input %r", t.name)
            return

        # Translate the well-known ggml input names to the standard HF naming
        # so callers don't have to know about ggml's internal labels.
        rename = {
            "inp_tokens": "input_ids",
        }
        canonical = rename.get(t.name, t.name)
        name = self._reserve(canonical) if canonical != t.name or canonical not in self._used_input_names() else self._fresh(t.name)
        shape = _ggml_shape_to_logical(t.ne)
        self.inputs.append(helper.make_tensor_value_info(
            name, _ggml_type_to_onnx(t.dtype), shape))
        self.value[t.index] = name

    def _used_input_names(self) -> set[str]:
        return {i.name for i in self.inputs}

    # -- per-op handlers ------------------------------------------------
    #
    # Each handler takes the result tensor `t` and is responsible for emitting
    # whatever ONNX nodes implement it, then calling `self.value[t.index] = ...`
    # with the name of the value that holds the result.

    def src_value(self, t: gdump.Tensor, i: int) -> str:
        return self.value[t.sources[i]]

    def src(self, t: gdump.Tensor, i: int) -> gdump.Tensor:
        return self.dump.tensors[t.sources[i]]


# ----------------------------------------------------------------------
# Op handlers
# ----------------------------------------------------------------------


_OP_HANDLERS: dict[gdump.GgmlOp, Callable[[Translator, gdump.Tensor], None]] = {}


def _op(*ops):
    def deco(fn):
        for o in ops:
            _OP_HANDLERS[o] = fn
        return fn
    return deco


@_op(gdump.GgmlOp.ADD)
def _h_add(tr: Translator, t: gdump.Tensor) -> None:
    out = tr._emit("Add", [tr.src_value(t, 0), tr.src_value(t, 1)], t.name or "add")
    tr.value[t.index] = out


@_op(gdump.GgmlOp.SUB)
def _h_sub(tr: Translator, t: gdump.Tensor) -> None:
    tr.value[t.index] = tr._emit("Sub", [tr.src_value(t, 0), tr.src_value(t, 1)], t.name or "sub")


@_op(gdump.GgmlOp.MUL)
def _h_mul(tr: Translator, t: gdump.Tensor) -> None:
    tr.value[t.index] = tr._emit("Mul", [tr.src_value(t, 0), tr.src_value(t, 1)], t.name or "mul")


@_op(gdump.GgmlOp.DIV)
def _h_div(tr: Translator, t: gdump.Tensor) -> None:
    tr.value[t.index] = tr._emit("Div", [tr.src_value(t, 0), tr.src_value(t, 1)], t.name or "div")


@_op(gdump.GgmlOp.SCALE)
def _h_scale(tr: Translator, t: gdump.Tensor) -> None:
    # op_params: float scale (f32 at offset 0), float bias (f32 at offset 4).
    scale, bias = t.op_params_f32(0, 2)
    x = tr.src_value(t, 0)
    s_const = tr._const("scale_const", np.array(scale, dtype=np.float32))
    scaled = tr._emit("Mul", [x, s_const], "scaled")
    if bias != 0.0:
        b_const = tr._const("bias_const", np.array(bias, dtype=np.float32))
        scaled = tr._emit("Add", [scaled, b_const], "scaled_b")
    tr.value[t.index] = scaled


@_op(gdump.GgmlOp.MUL_MAT)
def _h_mul_mat(tr: Translator, t: gdump.Tensor) -> None:
    # ggml: result = b @ a.T (a = weight, b = activation)
    a_name = tr.src_value(t, 0)  # weight or upstream
    b_name = tr.src_value(t, 1)  # activation
    # If the weight is a constant initializer, we pre-transpose it to avoid
    # an explicit Transpose node. Otherwise we emit a Transpose at runtime.
    a_t = tr.src(t, 0)
    if a_t.has_data:
        # Replace the initializer with its transpose.
        arr = a_t.data
        arr_t = np.ascontiguousarray(arr.T)
        # Re-add as a fresh initializer; the old one (a_name) still exists but
        # is harmless (ONNX prunes unused initializers).
        a_T_name = tr._const(a_t.name + "_T", arr_t)
        tr.value[t.index] = tr._emit("MatMul", [b_name, a_T_name], t.name or "matmul")
    else:
        # Symbolic weight: transpose at runtime. We transpose the LAST two
        # dims so this works for batched matmul shapes too.
        a_T = tr._emit("Transpose", [a_name], "wT", perm=[1, 0])
        tr.value[t.index] = tr._emit("MatMul", [b_name, a_T], t.name or "matmul")


@_op(gdump.GgmlOp.GET_ROWS)
def _h_get_rows(tr: Translator, t: gdump.Tensor) -> None:
    # ggml GET_ROWS has two distinct flavours sharing the same op:
    #
    # (a) "embedding lookup": data is 2D (M, N), indices is 1D (K).
    #     Result is (K, N). This is a plain ONNX Gather on axis 0.
    #
    # (b) "per-batch row pick" (used by MoE for routing): data is (..., B, M)
    #     and indices is (..., B, K), where ``data.ne[2..3]`` align with
    #     ``b.ne[1..2]``. Each row of the batch picks K columns. This is
    #     ONNX GatherElements with axis = row axis.
    data_t = tr.src(t, 0)
    idx_t  = tr.src(t, 1)
    data   = tr.src_value(t, 0)
    idx    = tr.src_value(t, 1)
    idx_i64 = tr._emit("Cast", [idx], "idx_i64", to=TensorProto.INT64)

    # Flavour (a): all batch dims (ne[1..3] for data, ne[1..2] for idx) are
    # trivial. Use Gather axis=0.
    if data_t.ne[2] == 1 and data_t.ne[3] == 1 and idx_t.ne[1] == 1 and idx_t.ne[2] == 1:
        tr.value[t.index] = tr._emit("Gather", [data, idx_i64], t.name or "gather", axis=0)
        return

    # Flavour (b): per-batch column pick. Both data and idx may carry trailing
    # singleton axes that we need to drop so they line up.
    data_logical = _ggml_shape_to_logical(data_t.ne)
    idx_logical  = _ggml_shape_to_logical(idx_t.ne)

    def _squeeze_to(arr_name: str, logical: list[int], target_rank: int) -> tuple[str, list[int]]:
        # Drop any trailing or leading 1-dims until we hit target_rank.
        while len(logical) > target_rank and logical[0] == 1:
            logical = logical[1:]
        while len(logical) > target_rank and logical[-1] == 1:
            logical = logical[:-1]
        if logical == _ggml_shape_to_logical([0] * 4):
            return arr_name, logical
        return tr._emit("Reshape", [arr_name, tr._const_i64("get_rows_squeeze", logical)], "get_rows_sq"), logical

    # Decide a common rank. For per-batch column pick we expect data and idx
    # to share all leading axes; idx's trailing axis is the K we pick.
    target_rank = min(len(data_logical), len(idx_logical))
    if target_rank < 2:
        raise NotImplementedError(
            f"GET_ROWS general case unsupported: data {data_logical}, idx {idx_logical}"
        )
    data, data_logical = _squeeze_to(data, data_logical, target_rank)
    idx_i64, idx_logical = _squeeze_to(idx_i64, idx_logical, target_rank)
    if data_logical[:-1] != idx_logical[:-1]:
        raise NotImplementedError(
            f"GET_ROWS shape mismatch after squeeze: data {data_logical}, idx {idx_logical}"
        )
    axis = len(data_logical) - 1
    tr.value[t.index] = tr._emit(
        "GatherElements", [data, idx_i64], t.name or "gather_elements", axis=axis,
    )


@_op(gdump.GgmlOp.RMS_NORM)
def _h_rms_norm(tr: Translator, t: gdump.Tensor) -> None:
    # op_params: f32 eps at offset 0.
    eps = t.op_params_f32(0, 1)[0]
    x = tr.src_value(t, 0)
    # Compute in float32 (mirrors ggml's behaviour).
    x_f = tr._emit("Cast", [x], "x_f32", to=TensorProto.FLOAT)
    sq = tr._emit("Mul", [x_f, x_f], "sq")
    ms = tr._emit("ReduceMean", [sq], "ms", axes=[-1], keepdims=1)
    eps_c = tr._const("eps", np.array(eps, dtype=np.float32))
    ms_eps = tr._emit("Add", [ms, eps_c], "ms_eps")
    rms = tr._emit("Sqrt", [ms_eps], "rms")
    inv = tr._emit("Reciprocal", [rms], "inv")
    out = tr._emit("Mul", [x_f, inv], "rms_normed")
    tr.value[t.index] = out


@_op(gdump.GgmlOp.RESHAPE)
def _h_reshape(tr: Translator, t: gdump.Tensor) -> None:
    # ggml RESHAPE is just a view with a new shape. The result shape is in t.ne.
    src = tr.src_value(t, 0)
    target = _ggml_shape_to_logical(t.ne)
    shape_name = tr._const_i64("reshape_shape", target)
    tr.value[t.index] = tr._emit("Reshape", [src, shape_name], t.name or "reshape")


@_op(gdump.GgmlOp.VIEW, gdump.GgmlOp.CONT)
def _h_view(tr: Translator, t: gdump.Tensor) -> None:
    # Both VIEW and CONT change layout / contiguity but not values.
    src = tr.src(t, 0)
    src_v = tr.src_value(t, 0)
    target = _ggml_shape_to_logical(t.ne)
    src_logical = _ggml_shape_to_logical(src.ne)
    if target == src_logical:
        tr.value[t.index] = src_v
        return
    # Same element count -> simple Reshape.
    if int(np.prod(target)) == int(np.prod(src_logical)):
        shape_name = tr._const_i64("view_shape", target)
        tr.value[t.index] = tr._emit("Reshape", [src_v, shape_name], t.name or "view")
        return

    # Strided view: try to express as a contiguous Slice (optionally followed
    # by a Squeeze) along a single axis. ggml stores the view's byte offset
    # in op_params[0..7] (size_t).
    if t.ggml_op == gdump.GgmlOp.VIEW:
        offset = struct.unpack_from("<Q", t.op_params, 0)[0]
        if _try_slice_view(tr, t, src, src_v, offset):
            return
        if _try_squeeze_view(tr, t, src, src_v, offset):
            return

    raise NotImplementedError(
        f"VIEW from {src_logical} to {target} "
        f"(src.ne={src.ne}, view.ne={t.ne}, view.nb={t.nb}, offset={offset if t.ggml_op == gdump.GgmlOp.VIEW else 'n/a'}) "
        f"is not supported yet (tensor {t.name!r})"
    )


def _try_slice_view(tr: "Translator", t: gdump.Tensor, src: gdump.Tensor, src_v: str, offset: int) -> bool:
    """Slice along exactly one ggml axis where view.ne[ax] < src.ne[ax]."""
    diffs = [i for i in range(4) if t.ne[i] != src.ne[i]]
    if len(diffs) != 1:
        return False
    ax_ggml = diffs[0]
    if src.ne[ax_ggml] == 0:
        return False
    # Translate offset to a start index along this axis.
    stride = src.nb[ax_ggml] if src.nb[ax_ggml] else t.nb[0] or 1
    start = offset // stride
    end = start + t.ne[ax_ggml]
    rank = max(t.ndim, src.ndim)
    ax_np = rank - 1 - ax_ggml
    if ax_np < 0:
        return False
    tr.value[t.index] = tr._emit(
        "Slice",
        [src_v,
         tr._const_i64("slice_starts", [start]),
         tr._const_i64("slice_ends", [end]),
         tr._const_i64("slice_axes", [ax_np])],
        t.name or "view_slice",
    )
    return True


def _try_squeeze_view(tr: "Translator", t: gdump.Tensor, src: gdump.Tensor, src_v: str, offset: int) -> bool:
    """View that picks one slot along an axis and drops that axis.

    Pattern: source ne[ax] > 1, view does not have that axis (its remaining
    axes match the other source axes in order). E.g. (n_tokens, n_used, n_embd)
    -> (n_tokens, n_embd) by selecting one expert slot.
    """
    # Find a ggml axis in source whose size > 1 and which, if removed, would
    # make the remaining source.ne match the view's ne (axes preserved in order).
    for ax_drop in range(4):
        if src.ne[ax_drop] <= 1:
            continue
        # Build the candidate source ne with ax_drop removed (insert a 1 at the
        # end to keep length 4).
        remaining = [src.ne[i] for i in range(4) if i != ax_drop] + [1]
        if remaining != t.ne:
            continue
        # Found it. Compute slot index from offset.
        slot = offset // src.nb[ax_drop]
        rank = src.ndim
        ax_np = rank - 1 - ax_drop
        sliced = tr._emit(
            "Slice",
            [src_v,
             tr._const_i64("slice_starts", [slot]),
             tr._const_i64("slice_ends", [slot + 1]),
             tr._const_i64("slice_axes", [ax_np])],
            "view_slot",
        )
        tr.value[t.index] = tr._emit(
            "Squeeze", [sliced, tr._const_i64("squeeze_axes", [ax_np])],
            t.name or "view_squeeze",
        )
        return True
    return False


@_op(gdump.GgmlOp.PERMUTE, gdump.GgmlOp.TRANSPOSE)
def _h_permute(tr: Translator, t: gdump.Tensor) -> None:
    # op_params: int32[4] permutation. PERMUTE reorders ggml's ne axes; we
    # have to flip the perm because logical (numpy) axes are reversed from
    # ggml ne axes.
    perm_ggml = t.op_params_i32(4)
    rank = max(t.ndim, tr.src(t, 0).ndim)
    perm_ggml = perm_ggml[:rank]
    # Map ggml axis i (fastest-varying first) to numpy axis (rank-1-i).
    perm_np = [rank - 1 - perm_ggml[rank - 1 - i] for i in range(rank)]
    tr.value[t.index] = tr._emit("Transpose", [tr.src_value(t, 0)], t.name or "permute", perm=perm_np)


@_op(gdump.GgmlOp.CPY)
def _h_cpy(tr: Translator, t: gdump.Tensor) -> None:
    # ggml CPY copies src[0] into src[1]'s layout/dtype. For our purposes
    # (a no-op semantically; just dtype-casts when types differ) we either
    # pass through or cast.
    src = tr.src(t, 0)
    dst = tr.src(t, 1)
    src_v = tr.src_value(t, 0)
    if src.dtype != dst.dtype:
        tr.value[t.index] = tr._emit("Cast", [src_v], t.name or "cpy_cast", to=_ggml_type_to_onnx(dst.dtype))
    else:
        tr.value[t.index] = src_v


@_op(gdump.GgmlOp.GLU)
def _h_glu(tr: Translator, t: gdump.Tensor) -> None:
    # op_params[0] = ggml_glu_op, op_params[1] = swapped (0 or 1).
    # With two non-null sources, gate = src[0], up = src[1] (and swapped is
    # already baked into the source order, so we don't need to swap again).
    op_i = t.op_params_i32(2)
    glu_op = op_i[0]
    gate = tr.src_value(t, 0)
    up = tr.src_value(t, 1)
    if glu_op == gdump.GgmlGluOp.SWIGLU.value:
        sig = tr._emit("Sigmoid", [gate], "silu_sigmoid")
        act = tr._emit("Mul", [gate, sig], "silu")
    elif glu_op in (gdump.GgmlGluOp.GEGLU.value, gdump.GgmlGluOp.GEGLU_ERF.value):
        act = _emit_gelu(tr, gate, "gelu")
    elif glu_op == gdump.GgmlGluOp.REGLU.value:
        act = tr._emit("Relu", [gate], "relu")
    else:
        raise NotImplementedError(f"GLU variant {gdump.GgmlGluOp(glu_op).name} not implemented")
    tr.value[t.index] = tr._emit("Mul", [act, up], t.name or "glu_out")


def _emit_gelu(tr: "Translator", x: str, hint: str) -> str:
    """Erf-based GELU built from primitive ops (opset 17 has no Gelu)."""
    inv_sqrt2 = tr._const("inv_sqrt2", np.array(1.0 / math.sqrt(2.0), dtype=np.float32))
    half = tr._const("half", np.array(0.5, dtype=np.float32))
    one  = tr._const("one",  np.array(1.0, dtype=np.float32))
    arg = tr._emit("Mul", [x, inv_sqrt2], f"{hint}_arg")
    er  = tr._emit("Erf", [arg], f"{hint}_erf")
    one_plus = tr._emit("Add", [er, one], f"{hint}_1pErf")
    half_x = tr._emit("Mul", [x, half], f"{hint}_xH")
    return tr._emit("Mul", [half_x, one_plus], hint)


@_op(gdump.GgmlOp.SOFT_MAX)
def _h_softmax(tr: Translator, t: gdump.Tensor) -> None:
    # op_params: f32 scale at offset 0, f32 max_bias at offset 4.
    scale, max_bias = t.op_params_f32(0, 2)
    if max_bias != 0.0:
        raise NotImplementedError("SOFT_MAX with ALiBi (max_bias > 0) not supported")
    x = tr.src_value(t, 0)
    if scale != 1.0:
        s = tr._const("softmax_scale", np.array(scale, dtype=np.float32))
        x = tr._emit("Mul", [x, s], "scaled_for_softmax")
    if len(t.sources) >= 2:
        # Optional additive mask (e.g. attention causal mask).
        mask = tr.src_value(t, 1)
        m_f = tr._emit("Cast", [mask], "mask_f32", to=TensorProto.FLOAT)
        x = tr._emit("Add", [x, m_f], "softmax_masked")
    tr.value[t.index] = tr._emit("Softmax", [x], t.name or "softmax", axis=-1)


@_op(gdump.GgmlOp.CONCAT)
def _h_concat(tr: Translator, t: gdump.Tensor) -> None:
    # op_params[0] = ggml axis (in ggml's ne ordering, fastest-varying = 0);
    # we flip it for numpy's row-major ordering.
    ax_ggml = t.op_params_i32(1)[0]
    rank = max(tr.src(t, 0).ndim, t.ndim)
    ax_np = rank - 1 - ax_ggml
    tr.value[t.index] = tr._emit(
        "Concat", [tr.src_value(t, 0), tr.src_value(t, 1)],
        t.name or "concat", axis=ax_np,
    )


@_op(gdump.GgmlOp.UNARY)
def _h_unary(tr: Translator, t: gdump.Tensor) -> None:
    uop = t.op_params_i32(1)[0]
    x = tr.src_value(t, 0)
    if uop == gdump.GgmlUnaryOp.SILU.value:
        sig = tr._emit("Sigmoid", [x], "silu_sig")
        tr.value[t.index] = tr._emit("Mul", [x, sig], t.name or "silu")
        return
    if uop in (gdump.GgmlUnaryOp.GELU.value, gdump.GgmlUnaryOp.GELU_ERF.value):
        tr.value[t.index] = _emit_gelu(tr, x, t.name or "gelu")
        return
    table = {
        gdump.GgmlUnaryOp.RELU.value:   ("Relu", {}),
        gdump.GgmlUnaryOp.SIGMOID.value:("Sigmoid", {}),
        gdump.GgmlUnaryOp.TANH.value:   ("Tanh", {}),
        gdump.GgmlUnaryOp.NEG.value:    ("Neg", {}),
        gdump.GgmlUnaryOp.EXP.value:    ("Exp", {}),
        gdump.GgmlUnaryOp.HARDSWISH.value: ("HardSwish", {}),
        gdump.GgmlUnaryOp.HARDSIGMOID.value:("HardSigmoid", {}),
    }
    if uop not in table:
        raise NotImplementedError(f"UNARY op {gdump.GgmlUnaryOp(uop).name} not implemented")
    op, attrs = table[uop]
    tr.value[t.index] = tr._emit(op, [x], t.name or op.lower(), **attrs)


@_op(gdump.GgmlOp.ARGSORT)
def _h_argsort(tr: Translator, t: gdump.Tensor) -> None:
    # op_params[0] = sort_order (0 = ascending, 1 = descending).
    order = t.op_params_i32(1)[0]
    x = tr.src_value(t, 0)
    # Use TopK on the last axis to get a sort. K = last-dim size of input.
    k = tr.src(t, 0).ne[0]
    k_const = tr._const_i64("topk_k", [k])
    vals = tr._fresh("argsort_vals")
    idxs = tr._fresh("argsort_idxs")
    tr.nodes.append(helper.make_node(
        "TopK", [x, k_const], [vals, idxs], axis=-1,
        largest=1 if order == 1 else 0, sorted=1,
    ))
    # ggml returns the indices as i32; ONNX TopK returns i64.
    tr.value[t.index] = tr._emit("Cast", [idxs], t.name or "argsort", to=TensorProto.INT32)


@_op(gdump.GgmlOp.TOP_K)
def _h_topk(tr: Translator, t: gdump.Tensor) -> None:
    # op_params[0] = k.
    k = t.op_params_i32(1)[0]
    x = tr.src_value(t, 0)
    k_const = tr._const_i64("topk_k", [k])
    vals = tr._fresh("topk_vals")
    idxs = tr._fresh("topk_idxs")
    tr.nodes.append(helper.make_node(
        "TopK", [x, k_const], [vals, idxs], axis=-1, largest=1, sorted=1,
    ))
    # ggml stores TopK as just indices; ONNX returns both.
    tr.value[t.index] = tr._emit("Cast", [idxs], t.name or "topk", to=TensorProto.INT32)


@_op(gdump.GgmlOp.SUM_ROWS)
def _h_sum_rows(tr: Translator, t: gdump.Tensor) -> None:
    # ggml SUM_ROWS reduces along ne[0] (the fastest-varying axis = numpy's
    # last axis), keeping that axis with size 1.
    x = tr.src_value(t, 0)
    tr.value[t.index] = tr._emit("ReduceSum", [x, tr._const_i64("sumrows_axes", [-1])],
                                 t.name or "sum_rows", keepdims=1)


@_op(gdump.GgmlOp.SUM)
def _h_sum(tr: Translator, t: gdump.Tensor) -> None:
    # Reduce-sum over all axes, keep dims. (Rare in transformer prefill graphs.)
    x = tr.src_value(t, 0)
    tr.value[t.index] = tr._emit("ReduceSum", [x], t.name or "sum", keepdims=1)


@_op(gdump.GgmlOp.MEAN)
def _h_mean(tr: Translator, t: gdump.Tensor) -> None:
    x = tr.src_value(t, 0)
    tr.value[t.index] = tr._emit("ReduceMean", [x, tr._const_i64("mean_axes", [-1])],
                                 t.name or "mean", keepdims=1)


@_op(gdump.GgmlOp.NORM)
def _h_norm(tr: Translator, t: gdump.Tensor) -> None:
    # ggml NORM = (x - mean(x)) / sqrt(var(x) + eps). LayerNormalization in
    # ONNX does exactly this, but without learnable affine — and the input
    # to gemma's `cb(cur, "ffn_norm", ...)` is later multiplied by a weight
    # *outside* this op, so plain centered/scaled is the right semantics.
    eps = t.op_params_f32(0, 1)[0]
    x = tr.src_value(t, 0)
    x_f = tr._emit("Cast", [x], "norm_x_f32", to=TensorProto.FLOAT)
    mean = tr._emit("ReduceMean", [x_f], "norm_mean", axes=[-1], keepdims=1)
    centered = tr._emit("Sub", [x_f, mean], "norm_centered")
    var = tr._emit("ReduceMean",
                   [tr._emit("Mul", [centered, centered], "norm_sq")],
                   "norm_var", axes=[-1], keepdims=1)
    eps_c = tr._const("norm_eps", np.array(eps, dtype=np.float32))
    std = tr._emit("Sqrt", [tr._emit("Add", [var, eps_c], "norm_var_eps")], "norm_std")
    inv = tr._emit("Reciprocal", [std], "norm_inv")
    tr.value[t.index] = tr._emit("Mul", [centered, inv], t.name or "norm")


@_op(gdump.GgmlOp.SQR)
def _h_sqr(tr: Translator, t: gdump.Tensor) -> None:
    x = tr.src_value(t, 0)
    tr.value[t.index] = tr._emit("Mul", [x, x], t.name or "sqr")


@_op(gdump.GgmlOp.SQRT)
def _h_sqrt(tr: Translator, t: gdump.Tensor) -> None:
    tr.value[t.index] = tr._emit("Sqrt", [tr.src_value(t, 0)], t.name or "sqrt")


@_op(gdump.GgmlOp.CLAMP)
def _h_clamp(tr: Translator, t: gdump.Tensor) -> None:
    # op_params: f32 min at +0, f32 max at +4.
    lo, hi = t.op_params_f32(0, 2)
    lo_c = tr._const("clamp_min", np.array(lo, dtype=np.float32))
    hi_c = tr._const("clamp_max", np.array(hi, dtype=np.float32))
    tr.value[t.index] = tr._emit("Clip", [tr.src_value(t, 0), lo_c, hi_c],
                                 t.name or "clamp")


@_op(gdump.GgmlOp.SIN)
def _h_sin(tr: Translator, t: gdump.Tensor) -> None:
    tr.value[t.index] = tr._emit("Sin", [tr.src_value(t, 0)], t.name or "sin")


@_op(gdump.GgmlOp.COS)
def _h_cos(tr: Translator, t: gdump.Tensor) -> None:
    tr.value[t.index] = tr._emit("Cos", [tr.src_value(t, 0)], t.name or "cos")


@_op(gdump.GgmlOp.LOG)
def _h_log(tr: Translator, t: gdump.Tensor) -> None:
    tr.value[t.index] = tr._emit("Log", [tr.src_value(t, 0)], t.name or "log")


@_op(gdump.GgmlOp.REPEAT)
def _h_repeat(tr: Translator, t: gdump.Tensor) -> None:
    # ggml REPEAT replicates src to a larger target shape using broadcast
    # semantics. ONNX Tile takes repeats per axis.
    src = tr.src(t, 0)
    src_shape = _ggml_shape_to_logical(src.ne)
    tgt_shape = _ggml_shape_to_logical(t.ne)
    # Pad src_shape to the same rank as tgt_shape (broadcast over leading dims).
    while len(src_shape) < len(tgt_shape):
        src_shape.insert(0, 1)
    repeats = [int(tgt_shape[i] // src_shape[i]) for i in range(len(tgt_shape))]
    repeats_c = tr._const_i64("repeat_counts", repeats)
    tr.value[t.index] = tr._emit("Tile", [tr.src_value(t, 0), repeats_c],
                                 t.name or "repeat")


@_op(gdump.GgmlOp.MUL_MAT_ID)
def _h_mul_mat_id(tr: Translator, t: gdump.Tensor) -> None:
    """ggml MUL_MAT_ID: per-(token, slot) routed matmul for MoE.

    Source layout (numpy):
      experts: (n_expert, out_features, in_features)
      acts:    (n_tokens, [1 or n_groups], in_features)
      ids:     (n_tokens, n_experts_used)              -- int32

    Per ``(t, k)`` pair we want:
      out[t, k, :] = experts[ids[t, k], :, :] @ acts[t, 0, :]

    We emit that as: gather the expert rows by id (yielding a 4-D tensor),
    broadcast the activation along the experts_used axis, multiply, and
    sum over the in_features axis.
    """
    experts_t = tr.src(t, 0)
    acts_t    = tr.src(t, 1)
    ids_t     = tr.src(t, 2)
    experts   = tr.src_value(t, 0)
    acts      = tr.src_value(t, 1)
    ids       = tr.src_value(t, 2)

    n_expert    = experts_t.ne[2]
    out_feat    = experts_t.ne[1]
    in_feat     = experts_t.ne[0]
    n_tokens    = acts_t.ne[2]
    # acts.ne[1] is either 1 (gate/up: one activation per token, broadcast over
    # the n_used selected experts) or n_used (down: a distinct activation per
    # selected expert, already produced by the SwiGLU). We pick the right
    # broadcast shape based on which case we're in.
    n_used_in_acts = acts_t.ne[1]
    n_used      = ids_t.ne[0]

    # Gather: (n_expert, out, in) along axis=0 with int indices of shape
    # (n_tokens, n_used) -> (n_tokens, n_used, out, in).
    ids_i64 = tr._emit("Cast", [ids], "expert_ids_i64", to=TensorProto.INT64)
    ids_2d = tr._emit("Reshape", [ids_i64, tr._const_i64("ids_shape_2d", [n_tokens, n_used])], "ids_2d")
    gathered = tr._emit("Gather", [experts, ids_2d], "experts_gathered", axis=0)

    if n_used_in_acts == 1:
        # gate/up case: acts is (n_tokens, 1, in_feat). Broadcast over n_used.
        acts_4d = tr._emit("Reshape", [acts, tr._const_i64("acts_shape_4d_b", [n_tokens, 1, 1, in_feat])], "acts_4d_b")
    else:
        # down case: acts is (n_tokens, n_used, in_feat). One activation per slot.
        acts_4d = tr._emit("Reshape", [acts, tr._const_i64("acts_shape_4d_s", [n_tokens, n_used_in_acts, 1, in_feat])], "acts_4d_s")
    prod = tr._emit("Mul", [gathered, acts_4d], "moe_prod")
    summed = tr._emit("ReduceSum", [prod, tr._const_i64("moe_sum_axes", [-1])],
                      "moe_summed", keepdims=0)
    tr.value[t.index] = summed


def _h_rope_mrope(
    tr: "Translator", t: gdump.Tensor,
    n_dims: int, sections: list[int],
    freq_base: float, freq_scale: float, attn_factor: float,
) -> None:
    """qwen2vl-style multimodal RoPE.

    The position tensor has ``4 * n_tokens`` entries laid out as four
    concatenated sections (t, h, w, e). The op_params encode how many of the
    ``n_dims/2`` rotation pairs come from each section. The rotation itself is
    NEOX-style (halves split).
    """
    x_t = tr.src(t, 0)
    head_dim = x_t.ne[0]
    n_head   = x_t.ne[1]
    n_tokens = x_t.ne[2]
    assert head_dim == n_dims, "partial mrope not supported"
    half = head_dim // 2
    sum_sections = sum(sections)
    assert sum_sections > 0, "mrope sections cannot be all zero"

    # Build the section index for each pair: which of the four sections it
    # belongs to. Mirrors ggml_mrope_cache_init.
    section_idx = []
    sec_w = sections[0] + sections[1]
    sec_e = sec_w + sections[2]
    for i in range(half):
        sector = i % sum_sections
        if sector < sections[0]:
            k = 0
        elif sector < sec_w:
            k = 1
        elif sector < sec_e:
            k = 2
        else:
            k = 3
        section_idx.append(k)
    sec_idx_const = tr._const("mrope_section_idx", np.array(section_idx, dtype=np.int64))

    x_v = tr.src_value(t, 0)
    pos_v = tr.src_value(t, 1)
    freq_factors_v = tr.src_value(t, 2) if len(t.sources) >= 3 and t.sources[2] != t.sources[0] else None

    # positions is [4, n_tokens] in logical layout (4 sections × n_tokens).
    pos_f = tr._emit("Cast", [pos_v], "pos_f", to=TensorProto.FLOAT)
    pos_4xN = tr._emit("Reshape", [pos_f, tr._const_i64("mrope_pos_shape", [4, n_tokens])], "pos_4xN")
    # Gather per-pair positions: shape [half, n_tokens]
    pos_per_pair = tr._emit("Gather", [pos_4xN, sec_idx_const], "pos_per_pair", axis=0)
    # Transpose to [n_tokens, half] for downstream broadcasting.
    pos_per_pair = tr._emit("Transpose", [pos_per_pair], "pos_per_pair_T", perm=[1, 0])

    inv = 1.0 / (freq_base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    inv = inv * float(freq_scale) * float(attn_factor or 1.0)
    inv_const = tr._const("mrope_inv_freqs", inv.astype(np.float32))
    inv_u = tr._emit("Unsqueeze", [inv_const, tr._const_i64("axes_0_mrope", [0])], "inv_u")
    if freq_factors_v is not None:
        inv_u = tr._emit("Div", [inv_u, tr._emit("Unsqueeze", [freq_factors_v, tr._const_i64("axes0c", [0])], "ff_u")], "inv_u_scaled")

    angles = tr._emit("Mul", [pos_per_pair, inv_u], "mrope_angles")
    cos = tr._emit("Cos", [angles], "mrope_cos")
    sin = tr._emit("Sin", [angles], "mrope_sin")
    cos_b = tr._emit("Unsqueeze", [cos, tr._const_i64("axes_1_mrope", [1])], "mrope_cos_b")
    sin_b = tr._emit("Unsqueeze", [sin, tr._const_i64("axes_1b_mrope", [1])], "mrope_sin_b")

    # NEOX-style halves split.
    x_resh = tr._emit("Reshape", [x_v, tr._const_i64("mrope_x_shape", [n_tokens, n_head, 2, half])], "mrope_x")
    x_e_name, x_o_name = tr._fresh("mrope_x_even"), tr._fresh("mrope_x_odd")
    tr.nodes.append(helper.make_node("Split", [x_resh], [x_e_name, x_o_name], axis=-2))
    x_e = tr._emit("Squeeze", [x_e_name, tr._const_i64("axes_-2s_mrope", [-2])], "mrope_x_e")
    x_o = tr._emit("Squeeze", [x_o_name, tr._const_i64("axes_-2s2_mrope", [-2])], "mrope_x_o")
    new_e = tr._emit("Sub",
                     [tr._emit("Mul", [x_e, cos_b], "mrope_ec"),
                      tr._emit("Mul", [x_o, sin_b], "mrope_os")], "mrope_ne")
    new_o = tr._emit("Add",
                     [tr._emit("Mul", [x_e, sin_b], "mrope_es"),
                      tr._emit("Mul", [x_o, cos_b], "mrope_oc")], "mrope_no")
    ne_u = tr._emit("Unsqueeze", [new_e, tr._const_i64("axes_-2u_mrope", [-2])], "mrope_ne_u")
    no_u = tr._emit("Unsqueeze", [new_o, tr._const_i64("axes_-2u2_mrope", [-2])], "mrope_no_u")
    stacked = tr._fresh("mrope_stacked")
    tr.nodes.append(helper.make_node("Concat", [ne_u, no_u], [stacked], axis=-2))
    out = tr._emit("Reshape", [stacked, tr._const_i64("mrope_out_shape", [n_tokens, n_head, head_dim])], t.name or "mrope_out")
    tr.value[t.index] = out


@_op(gdump.GgmlOp.ROPE)
def _h_rope(tr: Translator, t: gdump.Tensor) -> None:
    # op_params:
    #   i32[0]      = n_past   (unused at exec time)
    #   i32[1]      = n_dims
    #   i32[2]      = mode (0 = NORMAL, 2 = NEOX, 8 = MROPE, ...)
    #   i32[3]      = n_ctx
    #   i32[4]      = n_ctx_orig
    #   f32 at +20  = freq_base, freq_scale, ext_factor, attn_factor,
    #                 beta_fast, beta_slow
    #   i32 at +44  = sections[4] (only when MROPE bit is set)
    params_i = t.op_params_i32(5)
    n_dims = params_i[1]
    mode   = params_i[2]
    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow = t.op_params_f32(20, 6)
    sections = list(struct.unpack_from("<4i", t.op_params, 44))

    NORMAL_MODE = 0
    NEOX_MODE   = 2
    MROPE_MODE  = 8
    IMROPE_MODE = 40
    if mode not in (NORMAL_MODE, NEOX_MODE, MROPE_MODE):
        raise NotImplementedError(f"RoPE mode {mode} not implemented")
    if ext_factor != 0.0:
        raise NotImplementedError("YaRN-style RoPE (ext_factor != 0) not implemented")
    if mode == MROPE_MODE:
        _h_rope_mrope(tr, t, n_dims, sections, freq_base, freq_scale, attn_factor)
        return

    x_v = tr.src_value(t, 0)
    pos_v = tr.src_value(t, 1)
    # Optional third source: rope_freqs.weight (per-pair frequency scaling).
    freq_factors_v = tr.src_value(t, 2) if len(t.sources) >= 3 else None

    # x logical shape: ggml ne = [head_dim, n_head, n_tokens, 1] -> numpy
    # (1, n_tokens, n_head, head_dim) or rank-3 if leading dim is 1.
    x_t = tr.src(t, 0)
    head_dim = x_t.ne[0]
    n_head   = x_t.ne[1]
    n_tokens = x_t.ne[2]
    assert head_dim == n_dims, f"partial RoPE not supported (head_dim={head_dim}, n_dims={n_dims})"
    half = head_dim // 2

    # inv_freqs as a constant initializer.
    inv = 1.0 / (freq_base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    inv = inv * float(freq_scale)
    if attn_factor != 1.0:
        inv = inv * float(attn_factor)
    inv_const = tr._const("inv_freqs", inv.astype(np.float32))

    # Build cos/sin tables in float32.
    pos_f = tr._emit("Cast", [pos_v], "pos_f", to=TensorProto.FLOAT)
    pos_u = tr._emit("Unsqueeze", [pos_f, tr._const_i64("axes_-1", [-1])], "pos_u")
    inv_u = tr._emit("Unsqueeze", [inv_const, tr._const_i64("axes_0", [0])], "inv_u")
    if freq_factors_v is not None:
        # inv_freqs /= rope_factors elementwise.
        inv_u = tr._emit("Div", [inv_u, tr._emit("Unsqueeze", [freq_factors_v, tr._const_i64("axes0b", [0])], "ff_u")], "inv_u_scaled")
    angles = tr._emit("Mul", [pos_u, inv_u], "rope_angles")
    cos = tr._emit("Cos", [angles], "rope_cos")
    sin = tr._emit("Sin", [angles], "rope_sin")
    # Make cos/sin broadcastable over n_head: shape [n_tokens, 1, head_dim/2].
    cos_b = tr._emit("Unsqueeze", [cos, tr._const_i64("axes_1", [1])], "rope_cos_b")
    sin_b = tr._emit("Unsqueeze", [sin, tr._const_i64("axes_1b", [1])], "rope_sin_b")

    # Reshape x to [n_tokens, n_head, head_dim/2, 2] for NORMAL or
    # [n_tokens, n_head, 2, head_dim/2] for NEOX.
    if mode == NORMAL_MODE:
        x_pairs = tr._emit("Reshape", [x_v, tr._const_i64("rope_pairs_shape", [n_tokens, n_head, half, 2])], "x_pairs")
        # Split into even / odd along last axis.
        x_e_name, x_o_name = tr._fresh("x_even"), tr._fresh("x_odd")
        tr.nodes.append(helper.make_node("Split", [x_pairs], [x_e_name, x_o_name], axis=-1))
        x_e = tr._emit("Squeeze", [x_e_name, tr._const_i64("axes_-1s", [-1])], "x_e_sq")
        x_o = tr._emit("Squeeze", [x_o_name, tr._const_i64("axes_-1s2", [-1])], "x_o_sq")
        new_e = tr._emit("Sub", [tr._emit("Mul", [x_e, cos_b], "ec"), tr._emit("Mul", [x_o, sin_b], "os")], "new_e")
        new_o = tr._emit("Add", [tr._emit("Mul", [x_e, sin_b], "es"), tr._emit("Mul", [x_o, cos_b], "oc")], "new_o")
        ne_u = tr._emit("Unsqueeze", [new_e, tr._const_i64("axes_-1u", [-1])], "ne_u")
        no_u = tr._emit("Unsqueeze", [new_o, tr._const_i64("axes_-1u2", [-1])], "no_u")
        stacked = tr._fresh("rope_stacked")
        tr.nodes.append(helper.make_node("Concat", [ne_u, no_u], [stacked], axis=-1))
        out = tr._emit("Reshape", [stacked, tr._const_i64("rope_out_shape", [n_tokens, n_head, head_dim])], t.name or "rope")
        tr.value[t.index] = out
    else:
        # NEOX: split halves.
        x_resh = tr._emit("Reshape", [x_v, tr._const_i64("rope_neox_shape", [n_tokens, n_head, 2, half])], "x_neox")
        x_e_name, x_o_name = tr._fresh("x_even"), tr._fresh("x_odd")
        tr.nodes.append(helper.make_node("Split", [x_resh], [x_e_name, x_o_name], axis=-2))
        x_e = tr._emit("Squeeze", [x_e_name, tr._const_i64("axes_-2s", [-2])], "x_e_sq")
        x_o = tr._emit("Squeeze", [x_o_name, tr._const_i64("axes_-2s2", [-2])], "x_o_sq")
        new_e = tr._emit("Sub", [tr._emit("Mul", [x_e, cos_b], "ec"), tr._emit("Mul", [x_o, sin_b], "os")], "new_e")
        new_o = tr._emit("Add", [tr._emit("Mul", [x_e, sin_b], "es"), tr._emit("Mul", [x_o, cos_b], "oc")], "new_o")
        ne_u = tr._emit("Unsqueeze", [new_e, tr._const_i64("axes_-2u", [-2])], "ne_u")
        no_u = tr._emit("Unsqueeze", [new_o, tr._const_i64("axes_-2u2", [-2])], "no_u")
        stacked = tr._fresh("rope_stacked")
        tr.nodes.append(helper.make_node("Concat", [ne_u, no_u], [stacked], axis=-2))
        out = tr._emit("Reshape", [stacked, tr._const_i64("rope_neox_out", [n_tokens, n_head, head_dim])], t.name or "rope_neox")
        tr.value[t.index] = out


@_op(gdump.GgmlOp.SET_ROWS)
def _h_set_rows(tr: Translator, t: gdump.Tensor) -> None:
    # ggml: set_rows(target=src[2], data=src[0], idxs=src[1])
    data = tr.src_value(t, 0)
    idx  = tr.src_value(t, 1)
    tgt  = tr.src_value(t, 2)
    # Use ScatterND. Indices need shape [N, 1] for a 2D target.
    # The dtype of the target may differ from the data — cast first.
    tgt_t = tr.src(t, 2)
    data_t = tr.src(t, 0)
    if data_t.dtype != tgt_t.dtype:
        data = tr._emit("Cast", [data], "cast_to_cache", to=_ggml_type_to_onnx(tgt_t.dtype))
    idx_i64 = tr._emit("Cast", [idx], "idx_i64", to=TensorProto.INT64)
    idx_u = tr._emit("Unsqueeze", [idx_i64, tr._const_i64("axes_-1u", [-1])], "idx_u")
    tr.value[t.index] = tr._emit("ScatterND", [tgt, idx_u, data], t.name or "set_rows")


@_op(gdump.GgmlOp.FLASH_ATTN_EXT)
def _h_flash_attn(tr: Translator, t: gdump.Tensor) -> None:
    # op_params: f32[3] = scale, max_bias, logit_softcap.
    scale, max_bias, logit_softcap = t.op_params_f32(0, 3)
    if logit_softcap != 0.0:
        raise NotImplementedError("FLASH_ATTN_EXT with logit softcap not implemented")
    if max_bias != 0.0:
        raise NotImplementedError("FLASH_ATTN_EXT with ALiBi (max_bias > 0) not implemented")

    q = tr.src_value(t, 0)
    k = tr.src_value(t, 1)
    v = tr.src_value(t, 2)
    mask = tr.src_value(t, 3) if len(t.sources) >= 4 else None

    # After the PERMUTE(0,2,1,3) just above us, the logical (numpy) shapes are
    #   q: (n_head,   n_tokens, head_dim)
    #   k: (n_head_kv, kv_total, head_dim)
    #   v: (n_head_kv, kv_total, head_dim)
    # (the trailing batch dim of 1 has been squeezed away by our shape conversion)
    q_t = tr.src(t, 0)
    k_t = tr.src(t, 1)
    rank = max(q_t.ndim, k_t.ndim)
    # head axis is the leading axis among the non-batch dims.
    n_head = q_t.ne[2]
    n_head_kv = k_t.ne[2]
    if n_head != n_head_kv:
        n_rep = n_head // n_head_kv
        # Repeat along the head axis (axis 0 for our rank-3 layout, axis 1 for rank-4).
        repeats = [1] * rank
        head_axis = rank - 3  # 0 if rank==3 else 1
        repeats[head_axis] = n_rep
        k = tr._emit("Tile", [k, tr._const_i64("tile_kv", repeats)], "k_rep")
        v = tr._emit("Tile", [v, tr._const_i64("tile_kv2", repeats)], "v_rep")
    # Q is fp32, K/V may be fp16 in the cache. Promote to fp32.
    k = tr._emit("Cast", [k], "k_f32", to=TensorProto.FLOAT)
    v = tr._emit("Cast", [v], "v_f32", to=TensorProto.FLOAT)
    # Scores: Q @ K^T  (transpose the last two dims of K).
    perm = list(range(rank))
    perm[-1], perm[-2] = perm[-2], perm[-1]
    kT = tr._emit("Transpose", [k], "kT", perm=perm)
    scores = tr._emit("MatMul", [q, kT], "scores")
    s_const = tr._const("attn_scale", np.array(scale, dtype=np.float32))
    scores = tr._emit("Mul", [scores, s_const], "scores_scaled")
    if mask is not None:
        mask_f32 = tr._emit("Cast", [mask], "mask_f32", to=TensorProto.FLOAT)
        scores = tr._emit("Add", [scores, mask_f32], "scores_masked")
    attn = tr._emit("Softmax", [scores], "attn_softmax", axis=-1)
    ctx = tr._emit("MatMul", [attn, v], "ctx")
    # ggml flash-attn output ne: [head_dim, n_head, n_tokens, 1]; numpy
    # (1, n_tokens, n_head, head_dim) → rank-3 (n_tokens, n_head, head_dim)
    # after stripping the leading 1. We have ctx in (n_head, n_tokens, head_dim);
    # swap the first two axes.
    perm2 = list(range(rank))
    perm2[-2], perm2[-3] = perm2[-3], perm2[-2]
    out = tr._emit("Transpose", [ctx], t.name or "fattn_out", perm=perm2)
    tr.value[t.index] = out


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def convert(dump_path: str | Path, onnx_path: str | Path, *, external_threshold: int = 1024) -> None:
    dump_path = Path(dump_path)
    onnx_path = Path(onnx_path)
    logger.info("loading dump: %s", dump_path)
    d = gdump.load(dump_path)
    logger.info("arch=%s n_tokens=%d n_seqs=%d tensors=%d", d.arch, d.n_tokens, d.n_seqs, len(d.tensors))

    tr = Translator(d)
    model = tr.translate()

    data_location = onnx_path.name + ".data"
    logger.info("saving to %s (external data: %s)", onnx_path, data_location)
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
