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

    def __init__(self, dump: gdump.GraphDump, *, weight_dtype: str = "float16",
                 extra_output_names: list[str] | None = None):
        self.dump = dump
        # Optional list of ggml tensor names to expose as additional ONNX
        # outputs (cast to fp32). Used by the per-layer drift investigator
        # in tools/onnx-export/per_layer_diff.py to compare against
        # llama.cpp's eval-callback captures.
        self._extra_output_names = list(extra_output_names or [])
        # Weights from quantised GGUF tensors are dequantised to float32 on
        # load (see gdump._decode_data); we cast them to ``weight_dtype``
        # before stamping them as ONNX initializers. Float16 is the default
        # because a fp32 copy of a real-size LLM doesn't usually fit on
        # consumer hardware, and most ONNX runtimes execute fp16 natively.
        # The original ggml type is preserved as TensorProto.doc_string so
        # downstream tools can still see how the weights were stored.
        wd = weight_dtype.lower()
        if wd in ("float16", "fp16", "f16"):
            self._weight_np_dtype = np.dtype("float16")
            self._compute_onnx_dtype = TensorProto.FLOAT16
        elif wd in ("float32", "fp32", "f32"):
            self._weight_np_dtype = np.dtype("float32")
            self._compute_onnx_dtype = TensorProto.FLOAT
        else:
            raise ValueError(f"weight_dtype must be float16 or float32, got {weight_dtype!r}")
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.inputs: list[onnx.ValueInfoProto] = []
        self.outputs: list[onnx.ValueInfoProto] = []
        # Map from gdump tensor index -> ONNX value name that holds it.
        self.value: dict[int, str] = {}
        # Original ggml type per ONNX initializer name (recorded for the
        # model-level metadata + per-tensor doc_string).
        self._original_dtypes: dict[str, str] = {}
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

    def _axes_const(self, axes: tuple[int, ...] | list[int]) -> str:
        """Memoised i64 axes constant. The RoPE / mrope handlers emit dozens
        of single-axis sentinels (`[-1]`, `[0]`, `[1]`, `[-2]`, ...) per layer;
        without this cache each one becomes a fresh initializer in the ONNX
        model, bloating the file with duplicates.
        Use this for "small literal axis arrays known at translate time";
        keep `_const_i64` for shape tensors, slice bounds, and other values
        you genuinely want as separate initializers."""
        key = tuple(int(a) for a in axes)
        cache = getattr(self, "_axes_consts", None)
        if cache is None:
            cache = {}
            self._axes_consts = cache
        if key not in cache:
            label = "axes_" + "_".join(("n" + str(-a) if a < 0 else str(a)) for a in key)
            cache[key] = self._const_i64(label, list(key))
        return cache[key]

    def _to_compute(self, value_name: str, hint: str) -> str:
        """Insert a Cast to the chosen activation dtype (matches weight_dtype).
        Used at the boundaries of explicitly-fp32 regions like RMSNorm."""
        return self._emit("Cast", [value_name], hint, to=self._compute_onnx_dtype)

    def _add_weight_initializer(
        self, t: gdump.Tensor, *,
        transpose: bool = False, name_hint: str | None = None, bind_value: bool = True,
    ) -> str:
        """Cast a dequantised weight to weight_dtype, add it as an ONNX
        initializer, and stamp the original ggml type on its doc_string.

        Set ``bind_value=False`` when this is an auxiliary copy (e.g. a
        pre-transposed initializer used by a single MatMul) — in that case
        we don't replace ``self.value[t.index]``, so other consumers of
        the same weight still see the canonical untransposed initializer.
        """
        arr = t.data
        if transpose:
            arr = np.ascontiguousarray(arr.T)
        arr = arr.astype(self._weight_np_dtype, copy=False)
        name = self._fresh(name_hint or t.name)
        tp = numpy_helper.from_array(arr, name=name)
        # Resolve the ggml type name. `gdump.GgmlType` only enumerates a small
        # subset (F32, F16, BF16, F64, Q4_0, Q4_1, I8..I64); fall through to
        # `gguf.constants.GGMLQuantizationType` for the full set (Q4_K, Q5_K,
        # Q6_K, Q8_0, all IQ*, …) so quantised initializers carry a human-
        # readable original_ggml_type instead of a cryptic `type_<int>`.
        orig: str
        try:
            orig = gdump.GgmlType(t.dtype).name
        except ValueError:
            try:
                from .constants import GGMLQuantizationType
                orig = GGMLQuantizationType(t.dtype).name
            except (ValueError, ImportError):
                orig = f"type_{t.dtype}"
        tp.doc_string = f"original_ggml_type={orig}"
        self.initializers.append(tp)
        self._original_dtypes[name] = orig
        if bind_value:
            self.value[t.index] = name
        return name

    # -- top level ------------------------------------------------------

    def translate(self) -> onnx.ModelProto:
        for t in self.dump.tensors:
            self._handle(t)

        # Expose any user-requested intermediates as extra ONNX outputs. We
        # look up by ggml name; ggml names aren't unique (e.g. "Qcur-0"
        # appears on the MUL_MAT, the RESHAPE, and the ROPE), and the model
        # code (src/models/llama.cpp) calls cb() *after* the final transform
        # at each cb()-site, which corresponds to the LAST node with that
        # name in topo order. Pick that one.
        for want in self._extra_output_names:
            tensor_idx = None
            for t in self.dump.tensors:
                if t.name == want and t.index in self.value:
                    tensor_idx = t.index
            if tensor_idx is None:
                logger.warning("extra output %r not found in dump; skipping", want)
                continue
            t = self.dump.tensors[tensor_idx]
            # Cast to fp32 for portability of the comparison; we always
            # compare in fp32 on the Python side.
            f_name = self._fresh("dbg_" + want)
            self.nodes.append(helper.make_node(
                "Cast", [self.value[tensor_idx]], [f_name],
                name=f"dbg_{want}_cast", to=TensorProto.FLOAT))
            # ONNX disallows certain chars in I/O names; replace with '_'.
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in want)
            out_name = f"dbg__{safe}"
            self.nodes.append(helper.make_node(
                "Identity", [f_name], [out_name], name=f"dbg_{safe}_id"))
            self.outputs.append(helper.make_tensor_value_info(
                out_name, TensorProto.FLOAT, _ggml_shape_to_logical(t.ne)))

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
                name = f"present.{layer}.{kind}"
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

        # Stamp metadata so downstream tooling can see what we did. ONNX
        # provides two well-known surfaces:
        #   - model.metadata_props (model-level free-form key/value)
        #   - TensorProto.doc_string (per-initializer free-form string)
        # We populate both with the original ggml type so a reader can
        # always recover how each weight was stored.
        from collections import Counter
        type_counts = Counter(self._original_dtypes.values())
        type_summary = ",".join(f"{k}:{v}" for k, v in sorted(type_counts.items()))
        for k, v in [
            ("llama_onnx.architecture", self.dump.arch),
            ("llama_onnx.weight_dtype", str(self._weight_np_dtype)),
            ("llama_onnx.original_dtype_counts", type_summary),
            ("llama_onnx.n_tokens", str(self.dump.n_tokens)),
            ("llama_onnx.n_seqs", str(self.dump.n_seqs)),
        ]:
            e = m.metadata_props.add()
            e.key, e.value = k, v

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
            # The logits flow through the graph in compute_dtype (fp16 by
            # default); cast back to fp32 so downstream tooling gets the
            # conventional precision regardless of how the weights were stored.
            out = "logits"
            self.nodes.append(helper.make_node(
                "Cast", [self.value[t.index]], [out], name="logits_to_f32",
                to=TensorProto.FLOAT))
            self.value[t.index] = out
            self.outputs.append(helper.make_tensor_value_info(
                out, TensorProto.FLOAT, _ggml_shape_to_logical(t.ne)))

    def _handle_leaf(self, t: gdump.Tensor) -> None:
        is_cache = t.name.startswith("cache_k_l") or t.name.startswith("cache_v_l")

        if is_cache:
            # Caches are stored in the dump as zero-filled weight buffers but
            # we want them as ONNX *inputs* (past_K / past_V). We rename them
            # so the ONNX I/O surface follows the standard HF convention.
            kind, layer = t.name.split("_l", 1)
            which = "key" if kind == "cache_k" else "value"
            name = f"past_key_values.{layer}.{which}"
            shape = _ggml_shape_to_logical(t.ne)
            self.inputs.append(helper.make_tensor_value_info(
                name, _ggml_type_to_onnx(t.dtype), shape))
            self.value[t.index] = name
            return

        if t.has_data:
            # Weight initializer. The data was already dequantised to float32
            # on load (see gdump._decode_data); we now cast to the chosen
            # weight_dtype and record the original ggml type as metadata so
            # downstream tools can see how the weights were stored.
            self._add_weight_initializer(t)
            return

        # Otherwise this is a graph input (inp_tokens, positions, masks, ...).
        # Some inputs may be entirely unused (e.g. `inp_embd` when the token
        # path is selected); we skip those to keep the ONNX I/O clean.
        if self._consumers.get(t.index, 0) == 0:
            logger.debug("skipping unused input %r", t.name)
            return

        # Translate the well-known ggml input names to the standard HF naming
        # so callers don't have to know about ggml's internal labels. For other
        # leaves, prefer the *literal* ggml name on first use (downstream
        # tooling like gdump_fixture.synthesize_inputs identifies leaves by
        # their unaltered names) and only fall back to a counter-suffixed
        # fresh name if a collision would occur.
        if t.name == "inp_tokens":
            name = "input_ids"
        elif t.name not in {i.name for i in self.inputs}:
            name = t.name
        else:
            name = self._fresh(t.name)
        shape = _ggml_shape_to_logical(t.ne)
        self.inputs.append(helper.make_tensor_value_info(
            name, _ggml_type_to_onnx(t.dtype), shape))
        self.value[t.index] = name

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


@_op(gdump.GgmlOp.ADD_ID)
def _h_add_id(tr: Translator, t: gdump.Tensor) -> None:
    # ggml ADD_ID (see ggml_compute_forward_add_id_f32):
    #   dst[i3, i2, i1, :] = src0[i3, i2, i1, :] + src1[ids[i1, i2], :]
    # i.e. a *gather* of bias rows from src1 along axis 0 by the i32 ids, then
    # an elementwise Add against src0. Used by shared MoE bias / per-token-id
    # bias paths in llama-graph.cpp.
    base   = tr.src_value(t, 0)
    bias   = tr.src_value(t, 1)
    ids    = tr.src_value(t, 2)
    # ONNX Gather needs int64 indices; ggml stores them as int32.
    ids_i64 = tr._emit("Cast", [ids], "addid_idx_i64", to=TensorProto.INT64)
    gathered = tr._emit("Gather", [bias, ids_i64], "addid_bias_gather", axis=0)
    tr.value[t.index] = tr._emit("Add", [base, gathered], t.name or "add_id")


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
    s_const = tr._const("scale_const", np.array(scale, dtype=tr._weight_np_dtype))
    scaled = tr._emit("Mul", [x, s_const], "scaled")
    if bias != 0.0:
        b_const = tr._const("bias_const", np.array(bias, dtype=tr._weight_np_dtype))
        scaled = tr._emit("Add", [scaled, b_const], "scaled_b")
    tr.value[t.index] = scaled


@_op(gdump.GgmlOp.MUL_MAT)
def _h_mul_mat(tr: Translator, t: gdump.Tensor) -> None:
    # ggml: result = b @ a.T (a = weight, b = activation)
    b_name = tr.src_value(t, 1)  # activation
    a_t = tr.src(t, 0)
    if a_t.has_data:
        # Constant weight: pre-transpose so we don't need a Transpose node.
        # Goes through the central weight pipeline so it carries the original
        # ggml-type metadata and gets cast to the translator's weight_dtype.
        a_T_name = tr._add_weight_initializer(a_t, transpose=True, name_hint=a_t.name + "_T", bind_value=False)
        tr.value[t.index] = tr._emit("MatMul", [b_name, a_T_name], t.name or "matmul")
    else:
        # Symbolic weight: transpose at runtime. We transpose the LAST two
        # dims so this works for batched matmul shapes too (e.g. Q @ K^T in
        # vision attention, where both operands are batched (heads, seq,
        # head_dim) and the "weight" position is itself a runtime tensor).
        # The rank is taken from the logical shape of src 0 (which has already
        # had trailing-1 dims stripped by ``_ggml_shape_to_logical``); a perm
        # of [0, 1, ..., rank-2, rank-1] with the last two swapped is the
        # ONNX equivalent of "transpose the last two axes". Using a fixed
        # [1, 0] perm here would silently misbehave on rank>2 inputs.
        a_name = tr.src_value(t, 0)
        a_rank = max(len(_ggml_shape_to_logical(a_t.ne)), 2)
        perm = list(range(a_rank))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        a_T = tr._emit("Transpose", [a_name], "wT", perm=perm)
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
        # ggml result is shaped (data.ne[0], idx.ne[0]) in ne order, which is
        # (idx.ne[0], data.ne[0]) in numpy order. If data appears rank-1 in
        # numpy (data.ne[1] == 1, so _ggml_shape_to_logical stripped it), an
        # ONNX Gather(axis=0) would index along the row-width axis instead of
        # the rows axis. Reshape data to add the leading "rows" axis so the
        # Gather axis aligns with ggml's semantics, then drop it again when
        # the consumer expects the stripped logical shape.
        if data_t.ne[1] == 1:
            # data is rank-1 (length data.ne[0]); idx selects rows of a 1-row
            # matrix. The only valid idx value is 0, so the result is "data
            # repeated idx.ne[0] times". We emit it as Reshape→Gather.
            data_2d = tr._emit(
                "Reshape",
                [data, tr._const_i64("get_rows_data_2d", [1, data_t.ne[0]])],
                "get_rows_data_unsq",
            )
            gathered = tr._emit("Gather", [data_2d, idx_i64], "gather_2d", axis=0)
            target = _ggml_shape_to_logical(t.ne)
            if target == [data_t.ne[0]]:
                # Stripped logical shape matches a single row; squeeze the
                # idx axis we just added.
                tr.value[t.index] = tr._emit(
                    "Reshape",
                    [gathered, tr._const_i64("get_rows_squeeze", target)],
                    t.name or "gather",
                )
            else:
                tr.value[t.index] = gathered
            return
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
    gathered = tr._emit(
        "GatherElements", [data, idx_i64], t.name or "gather_elements", axis=axis,
    )
    target = _ggml_shape_to_logical(t.ne)
    if target != data_logical[:-1] + [idx_logical[-1]]:
        gathered = tr._emit(
            "Reshape", [gathered, tr._const_i64("get_rows_target", target)],
            t.name or "gather_elements_reshaped",
        )
    tr.value[t.index] = gathered


@_op(gdump.GgmlOp.RMS_NORM)
def _h_rms_norm(tr: Translator, t: gdump.Tensor) -> None:
    # op_params: f32 eps at offset 0. Reduction/reciprocal in fp32 even when
    # compute is fp16 (matches llama.cpp's behaviour); cast back at the end.
    eps = t.op_params_f32(0, 1)[0]
    x = tr.src_value(t, 0)
    x_f = tr._emit("Cast", [x], "x_f32", to=TensorProto.FLOAT)
    sq = tr._emit("Mul", [x_f, x_f], "sq")
    ms = tr._emit("ReduceMean", [sq], "ms", axes=[-1], keepdims=1)
    eps_c = tr._const("eps", np.array(eps, dtype=np.float32))
    ms_eps = tr._emit("Add", [ms, eps_c], "ms_eps")
    rms = tr._emit("Sqrt", [ms_eps], "rms")
    inv = tr._emit("Reciprocal", [rms], "inv")
    out_f = tr._emit("Mul", [x_f, inv], "rms_normed")
    tr.value[t.index] = tr._to_compute(out_f, "rms_normed_cast")


@_op(gdump.GgmlOp.RESHAPE)
def _h_reshape(tr: Translator, t: gdump.Tensor) -> None:
    # ggml RESHAPE is just a view with a new shape. The result shape is in t.ne.
    src = tr.src_value(t, 0)
    target = _ggml_shape_to_logical(t.ne)
    shape_name = tr._const_i64("reshape_shape", target)
    tr.value[t.index] = tr._emit("Reshape", [src, shape_name], t.name or "reshape")


def _resolve_view_root(dump: gdump.GraphDump, t: gdump.Tensor) -> tuple[gdump.Tensor, int]:
    """Follow a chain of VIEW ops to the underlying tensor and sum byte offsets."""
    total_offset = 0
    cur = t
    seen: set[int] = set()
    while cur.ggml_op == gdump.GgmlOp.VIEW and cur.sources:
        if cur.index in seen:
            break
        seen.add(cur.index)
        if cur.op_params and len(cur.op_params) >= 8:
            total_offset += struct.unpack_from("<Q", cur.op_params, 0)[0]
        cur = dump.tensors[cur.sources[0]]
    return cur, total_offset


@_op(gdump.GgmlOp.VIEW, gdump.GgmlOp.CONT)
def _h_view(tr: Translator, t: gdump.Tensor) -> None:
    # Both VIEW and CONT change layout / contiguity but not values.
    if t.ggml_op == gdump.GgmlOp.VIEW:
        src, offset = _resolve_view_root(tr.dump, t)
        src_v = tr.value.get(src.index)
        if src_v is None:
            src_v = tr.src_value(t, 0)
    else:
        src = tr.src(t, 0)
        src_v = tr.src_value(t, 0)
        offset = 0
    target = _ggml_shape_to_logical(t.ne)
    src_logical = _ggml_shape_to_logical(src.ne)
    if target == src_logical:
        tr.value[t.index] = src_v
        return
    # Zero-element view: emit an empty constant of the target shape. ggml
    # uses these in mamba/recurrent caches at the n_seq_tokens=0 boundary,
    # where a downstream CPY then writes 0 elements back; the values are
    # never read. ConstantOfShape with a 0 in the shape is well-defined in
    # ONNX and propagates as an empty tensor through any consumer.
    if int(np.prod(target)) == 0:
        shape_name = tr._const_i64("view_empty_shape", target)
        onnx_dt = _ggml_type_to_onnx(src.dtype)
        if onnx_dt == TensorProto.UNDEFINED:
            onnx_dt = tr._compute_onnx_dtype
        fill_value = helper.make_tensor("v", onnx_dt, [1], [0])
        tr.value[t.index] = tr._emit(
            "ConstantOfShape", [shape_name], t.name or "view_empty",
            value=fill_value,
        )
        return
    # Same element count -> simple Reshape.
    if int(np.prod(target)) == int(np.prod(src_logical)):
        shape_name = tr._const_i64("view_shape", target)
        tr.value[t.index] = tr._emit("Reshape", [src_v, shape_name], t.name or "view")
        return

    # Strided view: try to express as a contiguous Slice (optionally followed
    # by a Squeeze) along a single axis. ggml stores the view's byte offset
    # in op_params[0..7] (size_t); chained views accumulate into ``offset``.
    if t.ggml_op == gdump.GgmlOp.VIEW:
        if _try_slice_view(tr, t, src, src_v, offset):
            return
        if _try_squeeze_view(tr, t, src, src_v, offset):
            return
        if _try_slice_reshape_view(tr, t, src, src_v, offset):
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


def _try_slice_reshape_view(tr: "Translator", t: gdump.Tensor, src: gdump.Tensor, src_v: str, offset: int) -> bool:
    """View that flattens N consecutive inner ggml axes into a contiguous slice
    of src's innermost axis, with the outer axes preserved.

    Typical use case: fused QKV slicing in Phi-3-style models.
        src.ne  = [9216, 32, 1, 1]    (numpy (32, 9216))
        view.ne = [96, 32, 32, 1]      (numpy (32, 32, 96))
    The view's two inner ggml dims (96, 32) collapse into a 3072-long contiguous
    slice of src's innermost axis; the outer dim (32) matches src's row axis.
    Detected via the view's strides:
        view.nb[0] == src.nb[0]                  (inner element stride match)
        view.nb[k] == src.nb[1] for some k>=1    (boundary to next src row)
        view.ne[k..] match src.ne[1..]
    Emits ONNX `Slice(axis=-1, start=offset/elem, end=start+slice_len) + Reshape`.
    """
    elem = src.nb[0]
    if not elem or t.nb[0] != elem:
        return False
    if offset % elem != 0:
        return False
    # Flat-source special case: src is rank-1 (ne[1..3] all 1). The view
    # reshapes a contiguous slice of src into a multi-dim tensor, but there
    # is no "outer src row stride" to align against, so the general loop
    # below would bail out. Detect that the view's strides are themselves
    # contiguous (nb[i+1] == nb[i] * ne[i]) and emit Slice+Reshape directly.
    if src.ne[1] == 1 and src.ne[2] == 1 and src.ne[3] == 1:
        slice_len = 1
        expected = elem
        for ax in range(4):
            if t.ne[ax] <= 0:
                return False
            if t.nb[ax] != expected and t.ne[ax] > 1:
                return False
            slice_len *= t.ne[ax]
            expected *= t.ne[ax]
        slice_start = offset // elem
        if slice_start + slice_len > src.ne[0]:
            return False
        target = _ggml_shape_to_logical(t.ne)
        sliced = tr._emit(
            "Slice",
            [src_v,
             tr._const_i64("slice_starts", [slice_start]),
             tr._const_i64("slice_ends",   [slice_start + slice_len]),
             tr._const_i64("slice_axes",   [0])],
            "view_flat_slice",
        )
        if target == [slice_len]:
            tr.value[t.index] = sliced
        else:
            shape_name = tr._const_i64("view_flat_shape", target)
            tr.value[t.index] = tr._emit("Reshape", [sliced, shape_name], t.name or "view_flat")
        return True
    # Find the view ggml-axis k where the stride jumps to src's next-row stride.
    src_row_stride = src.nb[1] if src.ne[1] > 1 else 0
    if not src_row_stride:
        return False
    k = None
    for ax in range(1, 4):
        if t.nb[ax] == src_row_stride and t.ne[ax] > 1:
            k = ax
            break
    if k is None:
        return False
    # The view dims at indices >= k must align with src's outer dims [1..],
    # in order (skipping trailing 1s).
    view_outer = [t.ne[i] for i in range(k, 4) if t.ne[i] > 1]
    src_outer  = [src.ne[i] for i in range(1, 4) if src.ne[i] > 1]
    if view_outer != src_outer:
        return False
    # The product of view dims [0..k-1] must fit in src.ne[0].
    slice_len = 1
    for i in range(k):
        slice_len *= t.ne[i]
    if slice_len <= 0 or slice_len > src.ne[0]:
        return False
    slice_start = offset // elem
    if slice_start + slice_len > src.ne[0]:
        return False
    # Emit Slice on numpy's last axis (= ggml axis 0).
    target = _ggml_shape_to_logical(t.ne)
    sliced = tr._emit(
        "Slice",
        [src_v,
         tr._const_i64("slice_starts", [slice_start]),
         tr._const_i64("slice_ends",   [slice_start + slice_len]),
         tr._const_i64("slice_axes",   [-1])],
        "view_qkv_slice",
    )
    shape_name = tr._const_i64("view_qkv_shape", target)
    tr.value[t.index] = tr._emit("Reshape", [sliced, shape_name], t.name or "view_qkv")
    return True


@_op(gdump.GgmlOp.PERMUTE, gdump.GgmlOp.TRANSPOSE)
def _h_permute(tr: Translator, t: gdump.Tensor) -> None:
    # op_params: int32[4] permutation. The ggml convention is unusual:
    # ``ggml_permute(a, ax0, ax1, ax2, ax3)`` sets ``result.ne[ax_i] = a.ne[i]``,
    # i.e. ``ax_i`` is the *destination* ggml-axis for input ggml-axis ``i``.
    # This is the INVERSE of the standard numpy/ONNX perm convention (where
    # ``perm[out_axis] = in_axis``). For self-inverse perms (e.g. (0,2,1,3),
    # (1,0,2,3) used by attention QKV head splits) the distinction is
    # invisible, but for true cycles (e.g. (1,2,0,3) used by clip.cpp's V
    # permute and by NCHW/HWC reshuffles) the difference matters.
    #
    # The ggml TRANSPOSE op is implemented as a special-case view that swaps
    # ne[0] and ne[1] but does *not* write op_params (see ggml_transpose in
    # ggml.c). Detect that and use the implicit perm [1, 0, 2, 3] instead.
    if t.ggml_op == gdump.GgmlOp.TRANSPOSE:
        perm_ggml = [1, 0, 2, 3]
    else:
        perm_ggml = t.op_params_i32(4)
    rank = max(t.ndim, tr.src(t, 0).ndim)
    perm_ggml = perm_ggml[:rank]
    # Invert the ggml perm so it becomes a standard "output-axis ← input-axis"
    # mapping in ggml axis order.
    inv_perm_ggml = [0] * rank
    for i, ax in enumerate(perm_ggml):
        if ax >= rank:
            # Stripping op_params past rank can leave gaps; this is only
            # well-defined when the trailing dims are identity. Fall back to
            # the un-inverted perm in that case (matches the legacy behaviour
            # for trailing-1 dims).
            inv_perm_ggml = list(perm_ggml)
            break
        inv_perm_ggml[ax] = i
    # Reverse axis order to go from ggml (fastest first) to numpy (slowest
    # first): numpy output axis j corresponds to ggml output axis r-1-j;
    # ggml input axis = inv_perm_ggml[r-1-j]; numpy input axis = r-1-that.
    perm_np = [rank - 1 - inv_perm_ggml[rank - 1 - i] for i in range(rank)]
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
    # Two flavours of source layout:
    #   * Two sources: gate = src[0], up = src[1] (most archs).
    #   * One source:  src[0] carries [gate ; up] packed on its innermost axis
    #     (split halfway; `swapped` flips which half is gate). Phi-3 family
    #     and gpt-oss style fused FFNs use this.
    op_i = t.op_params_i32(2)
    glu_op  = op_i[0]
    swapped = op_i[1]
    if len(t.sources) >= 2:
        gate = tr.src_value(t, 0)
        up   = tr.src_value(t, 1)
    else:
        # Single-source packed: split last axis in two equal halves. Opset 17's
        # Split needs an explicit `split` sizes input (num_outputs is opset 18+).
        packed = tr.src_value(t, 0)
        src0 = tr.src(t, 0)
        half = src0.ne[0] // 2  # ggml innermost = numpy last axis
        split_sizes = tr._const_i64("glu_split_sizes", [half, half])
        a_name, b_name = tr._fresh("glu_a"), tr._fresh("glu_b")
        tr.nodes.append(helper.make_node(
            "Split", [packed, split_sizes], [a_name, b_name], axis=-1,
        ))
        if swapped:
            gate, up = b_name, a_name
        else:
            gate, up = a_name, b_name
    if glu_op == gdump.GgmlGluOp.SWIGLU.value:
        sig = tr._emit("Sigmoid", [gate], "silu_sigmoid")
        act = tr._emit("Mul", [gate, sig], "silu")
    elif glu_op in (gdump.GgmlGluOp.GEGLU.value, gdump.GgmlGluOp.GEGLU_ERF.value):
        act = _emit_gelu(tr, gate, "gelu")
    elif glu_op == gdump.GgmlGluOp.REGLU.value:
        act = tr._emit("Relu", [gate], "relu")
    elif glu_op == gdump.GgmlGluOp.SWIGLU_OAI.value:
        # gpt-oss variant (ggml_compute_forward_swiglu_oai_f32):
        #   x  = min(gate, limit)
        #   y  = clip(up,  -limit, +limit)
        #   out = (x / (1 + exp(-alpha * x))) * (y + 1)
        # = (x * Sigmoid(alpha*x)) * (y + 1).
        # alpha/limit live in op_params_f32 at indices 2 / 3 (the f32 array, so
        # byte offsets 8 / 12).
        alpha, limit = t.op_params_f32(8, 2)
        dt = tr._weight_np_dtype
        limit_c     = tr._const("oai_limit",     np.array( limit, dtype=dt))
        neg_limit_c = tr._const("oai_neg_limit", np.array(-limit, dtype=dt))
        alpha_c     = tr._const("oai_alpha",     np.array( alpha, dtype=dt))
        one_c       = tr._const("oai_one",       np.array(   1.0, dtype=dt))
        gate_clipped = tr._emit("Min", [gate, limit_c], "oai_gate_min")
        up_clipped = tr._emit("Max",
                              [tr._emit("Min", [up, limit_c], "oai_up_min"),
                               neg_limit_c],
                              "oai_up_clip")
        sig = tr._emit("Sigmoid", [tr._emit("Mul", [gate_clipped, alpha_c], "oai_alpha_x")], "oai_sig")
        act_g = tr._emit("Mul", [gate_clipped, sig], "oai_silu_alpha")
        y_plus_1 = tr._emit("Add", [up_clipped, one_c], "oai_y1")
        tr.value[t.index] = tr._emit("Mul", [act_g, y_plus_1], t.name or "swiglu_oai_out")
        return
    else:
        raise NotImplementedError(f"GLU variant {gdump.GgmlGluOp(glu_op).name} not implemented")
    tr.value[t.index] = tr._emit("Mul", [act, up], t.name or "glu_out")


def _emit_gelu(tr: "Translator", x: str, hint: str) -> str:
    """Erf-based GELU built from primitive ops (opset 17 has no Gelu)."""
    inv_sqrt2 = tr._const("inv_sqrt2", np.array(1.0 / math.sqrt(2.0), dtype=tr._weight_np_dtype))
    half = tr._const("half", np.array(0.5, dtype=tr._weight_np_dtype))
    one  = tr._const("one",  np.array(1.0, dtype=tr._weight_np_dtype))
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
        s = tr._const("softmax_scale", np.array(scale, dtype=tr._weight_np_dtype))
        x = tr._emit("Mul", [x, s], "scaled_for_softmax")
    if len(t.sources) >= 2:
        # Optional additive mask (e.g. attention causal mask). The mask in
        # ggml is f32; cast to the compute dtype so it broadcasts cleanly.
        mask = tr.src_value(t, 1)
        m = tr._emit("Cast", [mask], "softmax_mask_cast", to=tr._compute_onnx_dtype)
        x = tr._emit("Add", [x, m], "softmax_masked")
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
    # Composed variants — ONNX has no direct equivalent, so build from primitives.
    if uop == gdump.GgmlUnaryOp.STEP.value:
        # ggml: step(x) = (x > 0) ? 1 : 0, dtype preserved.
        zero = tr._const("step_zero", np.array(0, dtype=tr._weight_np_dtype))
        gt = tr._emit("Greater", [x, zero], "step_gt")
        tr.value[t.index] = tr._emit(
            "Cast", [gt], t.name or "step", to=_ggml_type_to_onnx(t.dtype))
        return
    if uop == gdump.GgmlUnaryOp.GELU_QUICK.value:
        # ggml: gelu_quick(x) = x / (1 + exp(-1.702 * x)) = x * sigmoid(1.702 * x).
        c = tr._const("gelu_quick_coef", np.array(1.702, dtype=tr._weight_np_dtype))
        scaled = tr._emit("Mul", [x, c], "gelu_quick_scaled")
        sig = tr._emit("Sigmoid", [scaled], "gelu_quick_sig")
        tr.value[t.index] = tr._emit("Mul", [x, sig], t.name or "gelu_quick")
        return
    if uop == gdump.GgmlUnaryOp.XIELU.value:
        # ggml op_params (4-byte f32 slots starting at byte 4):
        #   [1] = beta + softplus(alpha_n)   == effective alpha_n
        #   [2] = softplus(alpha_p)          == effective alpha_p
        #   [3] = beta
        #   [4] = eps
        # ggml: xielu(x) = (x>0) ? alpha_p*x*x + beta*x
        #                : (expm1(min(x, eps)) - x) * alpha_n + beta*x
        alpha_n, alpha_p, beta, eps = t.op_params_f32(4, 4)
        dt = tr._weight_np_dtype
        zero = tr._const("xielu_zero", np.array(0, dtype=dt))
        an_c = tr._const("xielu_alpha_n", np.array(alpha_n, dtype=dt))
        ap_c = tr._const("xielu_alpha_p", np.array(alpha_p, dtype=dt))
        b_c  = tr._const("xielu_beta",    np.array(beta,    dtype=dt))
        eps_c = tr._const("xielu_eps",    np.array(eps,     dtype=dt))
        bx = tr._emit("Mul", [x, b_c], "xielu_bx")
        # positive branch: alpha_p * x*x + beta*x
        xx = tr._emit("Mul", [x, x], "xielu_xx")
        pos = tr._emit("Mul", [xx, ap_c], "xielu_pos_a")
        pos = tr._emit("Add", [pos, bx], "xielu_pos")
        # negative branch: (expm1(min(x, eps)) - x) * alpha_n + beta*x
        mn = tr._emit("Min", [x, eps_c], "xielu_minxeps")
        em1 = tr._emit("Exp", [mn], "xielu_exp")
        one_c = tr._const("xielu_one", np.array(1, dtype=dt))
        em1 = tr._emit("Sub", [em1, one_c], "xielu_expm1")
        diff = tr._emit("Sub", [em1, x], "xielu_diff")
        neg = tr._emit("Mul", [diff, an_c], "xielu_neg_a")
        neg = tr._emit("Add", [neg, bx], "xielu_neg")
        cond = tr._emit("Greater", [x, zero], "xielu_cond")
        tr.value[t.index] = tr._emit("Where", [cond, pos, neg], t.name or "xielu")
        return
    if uop == gdump.GgmlUnaryOp.TRUNC.value:
        # ggml: truncf(x) -> round toward zero. Compose as Sign(x) * Floor(Abs(x)).
        a = tr._emit("Abs", [x], "trunc_abs")
        fl = tr._emit("Floor", [a], "trunc_floor")
        sg = tr._emit("Sign", [x], "trunc_sign")
        tr.value[t.index] = tr._emit("Mul", [sg, fl], t.name or "trunc")
        return
    table = {
        gdump.GgmlUnaryOp.RELU.value:   ("Relu", {}),
        gdump.GgmlUnaryOp.SIGMOID.value:("Sigmoid", {}),
        gdump.GgmlUnaryOp.TANH.value:   ("Tanh", {}),
        gdump.GgmlUnaryOp.NEG.value:    ("Neg", {}),
        gdump.GgmlUnaryOp.EXP.value:    ("Exp", {}),
        gdump.GgmlUnaryOp.HARDSWISH.value: ("HardSwish", {}),
        gdump.GgmlUnaryOp.HARDSIGMOID.value:("HardSigmoid", {}),
        gdump.GgmlUnaryOp.ABS.value:    ("Abs", {}),
        gdump.GgmlUnaryOp.SGN.value:    ("Sign", {}),
        gdump.GgmlUnaryOp.ELU.value:    ("Elu", {"alpha": 1.0}),
        gdump.GgmlUnaryOp.FLOOR.value:  ("Floor", {}),
        gdump.GgmlUnaryOp.CEIL.value:   ("Ceil", {}),
        gdump.GgmlUnaryOp.ROUND.value:  ("Round", {}),
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
    normed_f = tr._emit("Mul", [centered, inv], "norm_centered_scaled")
    tr.value[t.index] = tr._to_compute(normed_f, t.name or "norm")


@_op(gdump.GgmlOp.L2_NORM)
def _h_l2_norm(tr: Translator, t: gdump.Tensor) -> None:
    # ggml L2_NORM: y = x / max(sqrt(sum(x*x)), eps)   (note: max, not additive eps)
    # Reduction in fp32 even when compute is fp16, like RMS_NORM.
    eps = t.op_params_f32(0, 1)[0]
    x = tr.src_value(t, 0)
    x_f = tr._emit("Cast", [x], "l2_x_f32", to=TensorProto.FLOAT)
    sq = tr._emit("Mul", [x_f, x_f], "l2_sq")
    ss = tr._emit("ReduceSum", [sq], "l2_sumsq", axes=[-1], keepdims=1)
    norm = tr._emit("Sqrt", [ss], "l2_norm")
    eps_c = tr._const("l2_eps", np.array(eps, dtype=np.float32))
    denom = tr._emit("Max", [norm, eps_c], "l2_denom")
    out_f = tr._emit("Div", [x_f, denom], "l2_normed")
    tr.value[t.index] = tr._to_compute(out_f, t.name or "l2_norm_out")


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
    lo_c = tr._const("clamp_min", np.array(lo, dtype=tr._weight_np_dtype))
    hi_c = tr._const("clamp_max", np.array(hi, dtype=tr._weight_np_dtype))
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
    *, interleaved: bool = False,
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
    n_rope = n_dims
    n_pass = head_dim - n_rope
    half = n_rope // 2
    sum_sections = sum(sections)
    assert sum_sections > 0, "mrope sections cannot be all zero"

    # Build the section index for each pair: which of the four sections it
    # belongs to. Mirrors ggml_mrope_cache_init.
    section_idx = []
    sec_w = sections[0] + sections[1]
    sec_e = sec_w + sections[2]
    for i in range(half):
        sector = i % sum_sections
        if interleaved:
            if sector % 3 == 1 and sector < 3 * sections[1]:
                k = 1
            elif sector % 3 == 2 and sector < 3 * sections[2]:
                k = 2
            elif sector % 3 == 0 and sector < 3 * sections[0]:
                k = 0
            else:
                k = 3
        elif sector < sections[0]:
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

    inv = 1.0 / (freq_base ** (np.arange(0, n_rope, 2, dtype=np.float64) / float(n_rope)))
    inv = inv * float(freq_scale) * float(attn_factor or 1.0)
    inv_const = tr._const("mrope_inv_freqs", inv.astype(np.float32))
    inv_u = tr._emit("Unsqueeze", [inv_const, tr._axes_const([0])], "inv_u")
    if freq_factors_v is not None:
        # Cast freq_factors to fp32 (same reason as in _h_rope: avoid Div type-mismatch
        # when weight_dtype is fp16).
        ff_u = tr._emit("Unsqueeze", [freq_factors_v, tr._const_i64("axes0c", [0])], "ff_u_mrope")
        ff_u_f32 = tr._emit("Cast", [ff_u], "ff_u_f32_mrope", to=TensorProto.FLOAT)
        inv_u = tr._emit("Div", [inv_u, ff_u_f32], "inv_u_scaled")

    angles = tr._emit("Mul", [pos_per_pair, inv_u], "mrope_angles")
    cos = tr._emit("Cos", [angles], "mrope_cos")
    sin = tr._emit("Sin", [angles], "mrope_sin")
    cos_b = tr._to_compute(
        tr._emit("Unsqueeze", [cos, tr._axes_const([1])], "mrope_cos_b_f"),
        "mrope_cos_b")
    sin_b = tr._to_compute(
        tr._emit("Unsqueeze", [sin, tr._axes_const([1])], "mrope_sin_b_f"),
        "mrope_sin_b")

    # NEOX-style halves split on the rotated prefix only.
    if n_pass > 0:
        x_flat = tr._emit("Reshape", [x_v, tr._const_i64("mrope_x_flat", [n_tokens, n_head, head_dim])], "mrope_x_flat")
        rot_name, pass_name = tr._fresh("mrope_rot"), tr._fresh("mrope_pass")
        split_sizes = tr._const_i64("mrope_split_sizes", [n_rope, n_pass])
        tr.nodes.append(helper.make_node(
            "Split", [x_flat, split_sizes], [rot_name, pass_name], axis=-1,
        ))
        x_rot = rot_name
        x_pass = pass_name
    else:
        x_rot = tr._emit("Reshape", [x_v, tr._const_i64("mrope_x_flat", [n_tokens, n_head, head_dim])], "mrope_x_flat")
        x_pass = None
    x_resh = tr._emit("Reshape", [x_rot, tr._const_i64("mrope_x_shape", [n_tokens, n_head, 2, half])], "mrope_x")
    x_e_name, x_o_name = tr._fresh("mrope_x_even"), tr._fresh("mrope_x_odd")
    tr.nodes.append(helper.make_node("Split", [x_resh], [x_e_name, x_o_name], axis=-2))
    x_e = tr._emit("Squeeze", [x_e_name, tr._axes_const([-2])], "mrope_x_e")
    x_o = tr._emit("Squeeze", [x_o_name, tr._axes_const([-2])], "mrope_x_o")
    new_e = tr._emit("Sub",
                     [tr._emit("Mul", [x_e, cos_b], "mrope_ec"),
                      tr._emit("Mul", [x_o, sin_b], "mrope_os")], "mrope_ne")
    new_o = tr._emit("Add",
                     [tr._emit("Mul", [x_e, sin_b], "mrope_es"),
                      tr._emit("Mul", [x_o, cos_b], "mrope_oc")], "mrope_no")
    ne_u = tr._emit("Unsqueeze", [new_e, tr._axes_const([-2])], "mrope_ne_u")
    no_u = tr._emit("Unsqueeze", [new_o, tr._axes_const([-2])], "mrope_no_u")
    stacked = tr._fresh("mrope_stacked")
    tr.nodes.append(helper.make_node("Concat", [ne_u, no_u], [stacked], axis=-2))
    rot_out = tr._emit("Reshape", [stacked, tr._const_i64("mrope_rot_shape", [n_tokens, n_head, n_rope])], "mrope_rot_out")
    if x_pass is not None:
        out = tr._emit("Concat", [rot_out, x_pass], t.name or "mrope_out", axis=-1)
    else:
        out = rot_out
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
    n_dims     = params_i[1]
    mode       = params_i[2]
    n_ctx_orig = params_i[4]
    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow = t.op_params_f32(20, 6)
    sections = list(struct.unpack_from("<4i", t.op_params, 44))

    NORMAL_MODE = 0
    NEOX_MODE   = 2
    MROPE_MODE  = 8
    IMROPE_MODE = 40
    if mode not in (NORMAL_MODE, NEOX_MODE, MROPE_MODE, IMROPE_MODE):
        raise NotImplementedError(f"RoPE mode {mode} not implemented")
    if mode in (MROPE_MODE, IMROPE_MODE):
        if ext_factor != 0.0:
            raise NotImplementedError("YaRN-style MROPE (ext_factor != 0) not implemented")
        _h_rope_mrope(tr, t, n_dims, sections, freq_base, freq_scale, attn_factor,
                      interleaved=(mode == IMROPE_MODE))
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
    n_rope = n_dims
    n_pass = head_dim - n_rope
    half = n_rope // 2

    # inv_freqs as a constant initializer.
    # Pre-YaRN behaviour bakes attn_factor and freq_scale into the angle
    # (cos(attn_factor * freq_scale * pos / base^(2i/d))). For YaRN (ext_factor != 0)
    # the math is different: angles use freq_scale-mixed extrapolation/interpolation
    # of theta, and attn_factor turns into an *amplitude* scaling on cos/sin.
    # See ggml/src/ggml-cpu/ops.cpp:rope_yarn for the reference impl.
    i0 = np.arange(0, n_rope, 2, dtype=np.float64)
    inv_extrap = 1.0 / (freq_base ** (i0 / float(n_rope)))
    if ext_factor != 0.0:
        # YaRN: mix extrapolation/interpolation theta over a ramp on i0/2.
        def _yarn_corr_dim(rot: float) -> float:
            return head_dim * math.log(n_ctx_orig / (rot * 2 * math.pi)) / (2 * math.log(freq_base))
        low  = max(0.0,              math.floor(_yarn_corr_dim(beta_fast)))
        high = min(float(head_dim - 1), math.ceil (_yarn_corr_dim(beta_slow)))
        denom = max(0.001, high - low)
        y = (i0 / 2.0 - low) / denom
        ramp = 1.0 - np.clip(y, 0.0, 1.0)
        ramp_mix = ramp * float(ext_factor)
        inv_interp = inv_extrap * float(freq_scale)
        inv = inv_interp * (1.0 - ramp_mix) + inv_extrap * ramp_mix
        mscale = float(attn_factor) * (1.0 + 0.1 * math.log(1.0 / max(float(freq_scale), 1e-12)))
    else:
        # Non-YaRN: keep historical behaviour — attn_factor scales the angle, not
        # the amplitude. (For attn_factor == 1 this is equivalent to applying
        # mscale=1 after Cos/Sin.)
        inv = inv_extrap * float(freq_scale)
        if attn_factor != 1.0:
            inv = inv * float(attn_factor)
        mscale = 1.0
    inv_const = tr._const("inv_freqs", inv.astype(np.float32))

    # Build cos/sin tables in float32.
    pos_f = tr._emit("Cast", [pos_v], "pos_f", to=TensorProto.FLOAT)
    pos_u = tr._emit("Unsqueeze", [pos_f, tr._axes_const([-1])], "pos_u")
    inv_u = tr._emit("Unsqueeze", [inv_const, tr._axes_const([0])], "inv_u")
    if freq_factors_v is not None:
        # inv_freqs /= rope_factors elementwise. inv_u is fp32; rope_freqs.weight
        # is stored in weight_dtype (often fp16) — cast to fp32 first so the
        # Div doesn't get mixed-type-bound by ONNX shape inference.
        ff_u = tr._emit("Unsqueeze", [freq_factors_v, tr._const_i64("axes0b", [0])], "ff_u")
        ff_u_f32 = tr._emit("Cast", [ff_u], "ff_u_f32", to=TensorProto.FLOAT)
        inv_u = tr._emit("Div", [inv_u, ff_u_f32], "inv_u_scaled")
    angles = tr._emit("Mul", [pos_u, inv_u], "rope_angles")
    cos = tr._emit("Cos", [angles], "rope_cos")
    sin = tr._emit("Sin", [angles], "rope_sin")
    if mscale != 1.0:
        mscale_c = tr._const("rope_mscale", np.array(mscale, dtype=np.float32))
        cos = tr._emit("Mul", [cos, mscale_c], "rope_cos_mscale")
        sin = tr._emit("Mul", [sin, mscale_c], "rope_sin_mscale")
    # Make cos/sin broadcastable over n_head: shape [n_tokens, 1, head_dim/2].
    # Cast to the compute dtype so the downstream Muls don't mix f32 with f16.
    cos_b = tr._to_compute(
        tr._emit("Unsqueeze", [cos, tr._axes_const([1])], "rope_cos_b_f"),
        "rope_cos_b")
    sin_b = tr._to_compute(
        tr._emit("Unsqueeze", [sin, tr._axes_const([1])], "rope_sin_b_f"),
        "rope_sin_b")

    if n_pass > 0:
        x_flat = tr._emit("Reshape", [x_v, tr._const_i64("rope_x_flat", [n_tokens, n_head, head_dim])], "rope_x_flat")
        rot_name, pass_name = tr._fresh("rope_rot"), tr._fresh("rope_pass")
        split_sizes = tr._const_i64("rope_split_sizes", [n_rope, n_pass])
        tr.nodes.append(helper.make_node(
            "Split", [x_flat, split_sizes], [rot_name, pass_name], axis=-1,
        ))
        x_work = rot_name
        x_pass = pass_name
    else:
        x_work = x_v
        x_pass = None

    # Reshape x to [n_tokens, n_head, n_rope/2, 2] for NORMAL or
    # [n_tokens, n_head, 2, n_rope/2] for NEOX.
    if mode == NORMAL_MODE:
        x_pairs = tr._emit("Reshape", [x_work, tr._const_i64("rope_pairs_shape", [n_tokens, n_head, half, 2])], "x_pairs")
        x_e_name, x_o_name = tr._fresh("x_even"), tr._fresh("x_odd")
        tr.nodes.append(helper.make_node("Split", [x_pairs], [x_e_name, x_o_name], axis=-1))
        x_e = tr._emit("Squeeze", [x_e_name, tr._axes_const([-1])], "x_e_sq")
        x_o = tr._emit("Squeeze", [x_o_name, tr._axes_const([-1])], "x_o_sq")
        new_e = tr._emit("Sub", [tr._emit("Mul", [x_e, cos_b], "ec"), tr._emit("Mul", [x_o, sin_b], "os")], "new_e")
        new_o = tr._emit("Add", [tr._emit("Mul", [x_e, sin_b], "es"), tr._emit("Mul", [x_o, cos_b], "oc")], "new_o")
        ne_u = tr._emit("Unsqueeze", [new_e, tr._axes_const([-1])], "ne_u")
        no_u = tr._emit("Unsqueeze", [new_o, tr._axes_const([-1])], "no_u")
        stacked = tr._fresh("rope_stacked")
        tr.nodes.append(helper.make_node("Concat", [ne_u, no_u], [stacked], axis=-1))
        rot_out = tr._emit("Reshape", [stacked, tr._const_i64("rope_rot_shape", [n_tokens, n_head, n_rope])], "rope_rot_out")
    else:
        x_resh = tr._emit("Reshape", [x_work, tr._const_i64("rope_neox_shape", [n_tokens, n_head, 2, half])], "x_neox")
        x_e_name, x_o_name = tr._fresh("x_even"), tr._fresh("x_odd")
        tr.nodes.append(helper.make_node("Split", [x_resh], [x_e_name, x_o_name], axis=-2))
        x_e = tr._emit("Squeeze", [x_e_name, tr._axes_const([-2])], "x_e_sq")
        x_o = tr._emit("Squeeze", [x_o_name, tr._axes_const([-2])], "x_o_sq")
        new_e = tr._emit("Sub", [tr._emit("Mul", [x_e, cos_b], "ec"), tr._emit("Mul", [x_o, sin_b], "os")], "new_e")
        new_o = tr._emit("Add", [tr._emit("Mul", [x_e, sin_b], "es"), tr._emit("Mul", [x_o, cos_b], "oc")], "new_o")
        ne_u = tr._emit("Unsqueeze", [new_e, tr._axes_const([-2])], "ne_u")
        no_u = tr._emit("Unsqueeze", [new_o, tr._axes_const([-2])], "no_u")
        stacked = tr._fresh("rope_stacked")
        tr.nodes.append(helper.make_node("Concat", [ne_u, no_u], [stacked], axis=-2))
        rot_out = tr._emit("Reshape", [stacked, tr._const_i64("rope_rot_shape", [n_tokens, n_head, n_rope])], "rope_rot_out")

    if x_pass is not None:
        out = tr._emit("Concat", [rot_out, x_pass], t.name or "rope_out", axis=-1)
    else:
        out = rot_out
    tr.value[t.index] = out


@_op(gdump.GgmlOp.SET_ROWS)
def _h_set_rows(tr: Translator, t: gdump.Tensor) -> None:
    # ggml: set_rows(target=src[2], data=src[0], idxs=src[1]). In ggml this
    # is in-place: subsequent reads of `target` see the new values. ONNX is
    # functional, so we emit a ScatterND that produces a fresh tensor and
    # *redirect* the value binding for `target`'s gdump index to that fresh
    # tensor. Any later VIEW/PERMUTE that references the same leaf then
    # picks up the scattered version.
    data = tr.src_value(t, 0)
    idx  = tr.src_value(t, 1)
    tgt  = tr.src_value(t, 2)
    tgt_idx = t.sources[2]
    tgt_t = tr.src(t, 2)
    data_t = tr.src(t, 0)
    if data_t.dtype != tgt_t.dtype:
        data = tr._emit("Cast", [data], "cast_to_cache", to=_ggml_type_to_onnx(tgt_t.dtype))
    idx_i64 = tr._emit("Cast", [idx], "idx_i64", to=TensorProto.INT64)
    idx_u = tr._emit("Unsqueeze", [idx_i64, tr._axes_const([-1])], "idx_u")
    scattered = tr._emit("ScatterND", [tgt, idx_u, data], t.name or "set_rows")
    tr.value[t.index] = scattered
    # Redirect future reads of the cache leaf to the scattered tensor.
    tr.value[tgt_idx] = scattered


@_op(gdump.GgmlOp.FLASH_ATTN_EXT)
def _h_flash_attn(tr: Translator, t: gdump.Tensor) -> None:
    # op_params: f32[3] = scale, max_bias, logit_softcap.
    scale, max_bias, logit_softcap = t.op_params_f32(0, 3)
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
        # GQA expansion: each KV head is shared by n_rep consecutive Q heads.
        # A plain ONNX Tile would interleave (-> [H0, H1, H0, H1, ...]); we need
        # consecutive repetition (-> [H0, H0, H1, H1, ...]). The standard
        # unsqueeze+tile+reshape trick does that: insert a fresh axis right
        # after the head axis, tile that axis by n_rep, then fold it back in.
        head_axis = rank - 3  # 0 for rank-3 inputs, 1 for rank-4
        u_axis = head_axis + 1
        u_axes = tr._const_i64("gqa_u_axes", [u_axis])
        k_u = tr._emit("Unsqueeze", [k, u_axes], "k_unsq")
        v_u = tr._emit("Unsqueeze", [v, u_axes], "v_unsq")
        tile_shape = [1] * (rank + 1)
        tile_shape[u_axis] = n_rep
        k_t = tr._emit("Tile", [k_u, tr._const_i64("gqa_tile", tile_shape)], "k_tile")
        v_t = tr._emit("Tile", [v_u, tr._const_i64("gqa_tile2", tile_shape)], "v_tile")
        # Fold the new axis back into the head axis: shape stays rank, but
        # head_axis now has n_head_kv * n_rep == n_head entries.
        # Build the target shape dynamically from the input shape so the batch
        # dimensions stay symbolic.
        shape_in = tr._emit("Shape", [k], "k_shape")
        # head_axis dim becomes n_head; others unchanged.
        # We construct it as a Concat of single-element slices.
        parts = []
        for ax in range(rank):
            if ax == head_axis:
                parts.append(tr._const_i64("gqa_new_head", [n_head]))
            else:
                start = tr._const_i64(f"gqa_ax_{ax}_s", [ax])
                end   = tr._const_i64(f"gqa_ax_{ax}_e", [ax + 1])
                parts.append(tr._emit("Slice", [shape_in, start, end], f"gqa_dim_{ax}"))
        new_shape = tr._emit("Concat", parts, "gqa_new_shape", axis=0)
        k = tr._emit("Reshape", [k_t, new_shape], "k_rep")
        v = tr._emit("Reshape", [v_t, new_shape], "v_rep")
    # Bring K and V into the compute dtype so the MatMul has matching types.
    # The KV cache is stored fp16 in ggml; the activation flow is whatever
    # weight_dtype the user picked.
    k = tr._emit("Cast", [k], "k_cast", to=tr._compute_onnx_dtype)
    v = tr._emit("Cast", [v], "v_cast", to=tr._compute_onnx_dtype)
    # Scores: Q @ K^T  (transpose the last two dims of K).
    perm = list(range(rank))
    perm[-1], perm[-2] = perm[-2], perm[-1]
    kT = tr._emit("Transpose", [k], "kT", perm=perm)
    scores = tr._emit("MatMul", [q, kT], "scores")
    if logit_softcap != 0.0:
        scale = scale / logit_softcap
    s_const = tr._const("attn_scale", np.array(scale, dtype=tr._weight_np_dtype))
    scores = tr._emit("Mul", [scores, s_const], "scores_scaled")
    if logit_softcap != 0.0:
        cap_const = tr._const("attn_logit_softcap", np.array(logit_softcap, dtype=tr._weight_np_dtype))
        scores = tr._emit("Tanh", [scores], "scores_tanh")
        scores = tr._emit("Mul", [scores, cap_const], "scores_softcap")
    if mask is not None:
        mask_cast = tr._emit("Cast", [mask], "mask_cast", to=tr._compute_onnx_dtype)
        scores = tr._emit("Add", [scores, mask_cast], "scores_masked")
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
# Phase B: vision-encoder ops
# ----------------------------------------------------------------------
#
# These handlers cover the ops a typical ViT / CLIP / SigLIP encoder uses
# that the LM path doesn't already exercise: LeakyRelu, Pad, MaxPool /
# AvgPool (1D & 2D), Upscale (Resize), GroupNorm (opset-17 hand-roll),
# Conv2D / Conv2DDw, and IM2COL. Patch-embedding convolutions and
# pooling-based projectors are the main consumers.


@_op(gdump.GgmlOp.LEAKY_RELU)
def _h_leaky_relu(tr: Translator, t: gdump.Tensor) -> None:
    # op_params: f32 negative_slope at byte 0.
    alpha = t.op_params_f32(0, 1)[0]
    x = tr.src_value(t, 0)
    tr.value[t.index] = tr._emit("LeakyRelu", [x], t.name or "leaky_relu", alpha=float(alpha))


@_op(gdump.GgmlOp.PAD)
def _h_pad(tr: Translator, t: gdump.Tensor) -> None:
    # ggml_pad_ext stores nine i32 values:
    #   [lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3, circular_flag]
    # interleaved per-ggml-dim. ONNX `Pad` takes a (2 * rank) i64 vector of
    # the form [begin_d0, begin_d1, ..., end_d0, end_d1, ...] in the *numpy*
    # axis order (which is reversed from ggml's ne ordering).
    p = t.op_params_i32(9)
    circular = p[8]
    if circular:
        raise NotImplementedError("PAD with circular wrap not implemented")
    # Build [(lp_ggml_axis_i, rp_ggml_axis_i) for i in 0..3].
    lp = [p[0], p[2], p[4], p[6]]  # ggml-axis fastest-first
    rp = [p[1], p[3], p[5], p[7]]
    # Convert to numpy axis order: numpy axis = rank - 1 - ggml_axis. We pick
    # rank from the result tensor (after stripping trailing 1s).
    rank = max(t.ndim, tr.src(t, 0).ndim)
    begins = [0] * rank
    ends   = [0] * rank
    for ggml_ax in range(rank):
        np_ax = rank - 1 - ggml_ax
        begins[np_ax] = lp[ggml_ax]
        ends  [np_ax] = rp[ggml_ax]
    pads = begins + ends
    pads_const = tr._const_i64("pad_pads", pads)
    # Value as a 0-d tensor of the same dtype as the input (ggml pads with 0).
    val_dt = _ggml_type_to_onnx(t.dtype)
    val_np = {TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16}.get(val_dt, np.float32)
    val_const = tr._const("pad_value", np.array(0, dtype=val_np))
    tr.value[t.index] = tr._emit(
        "Pad", [tr.src_value(t, 0), pads_const, val_const], t.name or "pad", mode="constant",
    )


def _h_pool_common(tr: Translator, t: gdump.Tensor, *,
                   kernel: list[int], strides: list[int], pads: list[int]) -> None:
    """Common dispatch for POOL_1D / POOL_2D. ``kernel``/``strides``/``pads``
    are in numpy ordering (matches ONNX MaxPool/AveragePool conventions).
    op_params[0] selects max (0) vs avg (1). ONNX MaxPool only emits one
    output here (the values); we ignore the optional indices output.
    """
    op = t.op_params_i32(1)[0]  # ggml_op_pool: 0 = MAX, 1 = AVG
    x = tr.src_value(t, 0)
    if op == 0:
        # MaxPool: pads layout in ONNX is [x1_begin, ..., x_n_begin, x1_end, ..., x_n_end].
        # Our pads are already [b0, b1, ..., e0, e1, ...].
        tr.value[t.index] = tr._emit(
            "MaxPool", [x], t.name or "maxpool",
            kernel_shape=kernel, strides=strides, pads=pads,
        )
    elif op == 1:
        tr.value[t.index] = tr._emit(
            "AveragePool", [x], t.name or "avgpool",
            kernel_shape=kernel, strides=strides, pads=pads,
            count_include_pad=0,
        )
    else:
        raise NotImplementedError(f"POOL op variant {op} not implemented")


@_op(gdump.GgmlOp.POOL_2D)
def _h_pool_2d(tr: Translator, t: gdump.Tensor) -> None:
    # ggml_pool_2d stores: [op, k0, k1, s0, s1, p0, p1] (i32 layout in op_params,
    # but the trailing two values are originally float for p0/p1 — we cast to
    # int because integer padding is by far the common case in vision encoders
    # and ONNX MaxPool/AveragePool require integer pads anyway).
    p = t.op_params_i32(7)
    k0, k1, s0, s1, p0, p1 = p[1], p[2], p[3], p[4], p[5], p[6]
    # ggml 0 = width (fastest), 1 = height. Numpy/ONNX convention is (H, W).
    kernel  = [k1, k0]
    strides = [s1, s0]
    pads    = [p1, p0, p1, p0]  # [begin_H, begin_W, end_H, end_W]
    # ONNX MaxPool/AveragePool require rank-4 input for 2D; reshape if needed.
    inp_t = tr.src(t, 0)
    inp_logical = _ggml_shape_to_logical(inp_t.ne)
    if len(inp_logical) < 4:
        new_shape = [1] * (4 - len(inp_logical)) + list(inp_logical)
        padded = tr._emit(
            "Reshape", [tr.src_value(t, 0), tr._const_i64("pool_inp_pad", new_shape)],
            "pool_inp_pad_r",
        )
        # Temporarily rebind the source value for the duration of this handler.
        orig = tr.value[t.sources[0]]
        tr.value[t.sources[0]] = padded
        _h_pool_common(tr, t, kernel=kernel, strides=strides, pads=pads)
        tr.value[t.sources[0]] = orig
        return
    _h_pool_common(tr, t, kernel=kernel, strides=strides, pads=pads)


@_op(gdump.GgmlOp.POOL_1D)
def _h_pool_1d(tr: Translator, t: gdump.Tensor) -> None:
    # ggml_pool_1d stores: [op, k0, s0, p0].
    p = t.op_params_i32(4)
    k0, s0, p0 = p[1], p[2], p[3]
    kernel  = [k0]
    strides = [s0]
    pads    = [p0, p0]
    _h_pool_common(tr, t, kernel=kernel, strides=strides, pads=pads)


@_op(gdump.GgmlOp.UPSCALE)
def _h_upscale(tr: Translator, t: gdump.Tensor) -> None:
    # ggml UPSCALE / ggml_interpolate:
    #   op_params_i32(0) low byte = mode (0=nearest, 1=bilinear, 2=bicubic)
    #   high bits encode flags: ALIGN_CORNERS (1<<8), ANTIALIAS (1<<9).
    raw_mode = t.op_params_i32(1)[0]
    mode = raw_mode & 0xFF
    align_corners = bool(raw_mode & (1 << 8))
    antialias = bool(raw_mode & (1 << 9))

    mode_str = {0: "nearest", 1: "linear", 2: "cubic"}.get(mode)
    if mode_str is None:
        raise NotImplementedError(f"UPSCALE mode {mode} not implemented")

    src_t = tr.src(t, 0)
    # Both shapes are known statically; use the explicit ``sizes`` form for
    # the most deterministic ONNX behaviour. Numpy/ONNX expects sizes in
    # (N, C, H, W, ...) order. We have to match the *source* rank exactly
    # (ONNX Resize requires len(sizes) == rank(X)); ``_ggml_shape_to_logical``
    # strips trailing 1s independently per tensor, which can leave the
    # source and target with different ranks. Pad the shorter shape with
    # leading 1s to align.
    target_shape = _ggml_shape_to_logical(t.ne)
    src_shape    = _ggml_shape_to_logical(src_t.ne)
    while len(target_shape) < len(src_shape):
        target_shape.insert(0, 1)
    while len(src_shape) < len(target_shape):
        src_shape.insert(0, 1)
    sizes_const = tr._const_i64("upscale_sizes", target_shape)
    # Resize takes (X, roi, scales, sizes). With sizes we pass empty roi and
    # empty scales (use empty-string input names — onnx supports omitted
    # optional inputs as the empty string).
    empty = ""
    attrs = dict(mode=mode_str)
    if mode_str == "nearest":
        # Match ggml's "round half to floor" (which is the default 'floor' in onnx).
        attrs["nearest_mode"] = "floor"
    if align_corners:
        attrs["coordinate_transformation_mode"] = "align_corners"
    else:
        # ggml's default coord transform behaves like "asymmetric" for nearest
        # and "half_pixel" for bilinear/bicubic; pick the ONNX default that
        # matches the closest. (Some ggml backends use other variants; if a
        # test fails we'll revisit.)
        attrs["coordinate_transformation_mode"] = "asymmetric" if mode_str == "nearest" else "half_pixel"
    if antialias and mode_str == "linear":
        attrs["antialias"] = 1
    tr.value[t.index] = tr._emit(
        "Resize", [tr.src_value(t, 0), empty, empty, sizes_const],
        t.name or "upscale", **attrs,
    )


@_op(gdump.GgmlOp.GROUP_NORM)
def _h_group_norm(tr: Translator, t: gdump.Tensor) -> None:
    # op_params: i32[0] = n_groups, f32[1] = eps (the f32 lives at byte 4 since
    # ggml_set_op_params_f32 indexes in 4-byte units).
    n_groups = t.op_params_i32(1)[0]
    eps = t.op_params_f32(4, 1)[0]
    # We don't go through ONNX GroupNormalization (opset-18+); hand-roll it
    # against opset 17 primitives, in fp32, the same way RMS_NORM does.
    src_t = tr.src(t, 0)
    src_v = tr.src_value(t, 0)

    # Logical shape (numpy axis order): typically (N, C, H, W) for vision. We
    # support an arbitrary trailing-dim count; the channel axis is index 1
    # in numpy convention (which matches ggml's ne[2] for vision activations
    # because ne is fastest-first: ne = [W, H, C, N]).
    logical = _ggml_shape_to_logical(src_t.ne)
    rank = len(logical)
    if rank < 2:
        raise NotImplementedError(f"GROUP_NORM with rank {rank} not implemented")
    N = logical[0] if rank >= 4 else 1
    if rank >= 4:
        C = logical[1]
        spatial = logical[2:]
    elif rank == 3:
        # (C, H, W) — treat N=1.
        C = logical[0]
        spatial = logical[1:]
    else:
        # (C, K)
        C = logical[0]
        spatial = logical[1:]
    G = n_groups
    if C % G != 0:
        raise NotImplementedError(f"GROUP_NORM: channels {C} not divisible by n_groups {G}")
    cpg = C // G  # channels per group
    spatial_size = int(np.prod(spatial)) if len(spatial) > 0 else 1

    # Fold to (N, G, cpg * spatial) and reduce over the last axis.
    fold_shape = [N, G, cpg * spatial_size]
    x_f = tr._emit("Cast", [src_v], "gn_x_f32", to=TensorProto.FLOAT)
    x_fold = tr._emit("Reshape", [x_f, tr._const_i64("gn_fold_shape", fold_shape)], "gn_x_fold")
    mean = tr._emit("ReduceMean", [x_fold], "gn_mean", axes=[-1], keepdims=1)
    centered = tr._emit("Sub", [x_fold, mean], "gn_centered")
    var = tr._emit("ReduceMean",
                   [tr._emit("Mul", [centered, centered], "gn_sq")],
                   "gn_var", axes=[-1], keepdims=1)
    eps_c = tr._const("gn_eps", np.array(eps, dtype=np.float32))
    std = tr._emit("Sqrt", [tr._emit("Add", [var, eps_c], "gn_var_eps")], "gn_std")
    inv = tr._emit("Reciprocal", [std], "gn_inv")
    normed = tr._emit("Mul", [centered, inv], "gn_normed")

    # Reshape back to the original logical shape.
    out_f = tr._emit("Reshape", [normed, tr._const_i64("gn_unfold_shape", logical)], "gn_unfold")

    # Optional affine: ggml's GROUP_NORM op itself doesn't take gamma/beta as
    # sources (those are usually applied via separate MUL/ADD ops downstream),
    # so we don't multiply by anything here. Cast back to compute dtype.
    tr.value[t.index] = tr._to_compute(out_f, t.name or "group_norm_out")


def _conv2d_weight(tr: "Translator", w_t: gdump.Tensor, *, depthwise: bool = False) -> str:
    """Stamp a CONV_2D weight as an ONNX initializer in (OC, IC/G, KH, KW) order.

    ggml stores Conv2D weights in (KW, KH, IC, OC) layout (ne[0..3] fastest-first).
    On the numpy side that row-major-decodes as (OC, IC, KH, KW), which is
    exactly the ONNX layout — no transpose needed. For depthwise convs,
    ggml's weight is (KW, KH, 1, Cin); numpy decode is (Cin, KH, KW) once the
    singleton IC dim is stripped, and ONNX wants (Cin, 1, KH, KW), so we
    insert the singleton axis back at position 1.
    """
    arr = w_t.data
    if depthwise:
        # ggml ne = [KW, KH, 1, Cin]; numpy decode dropped the 1 -> (Cin, KH, KW).
        if arr.ndim == 3:
            arr = arr[:, None, :, :]
        elif arr.ndim == 4 and arr.shape[1] != 1:
            # Already 4D but in (Cin, IC, KH, KW) with IC != 1 — unexpected.
            raise NotImplementedError(f"CONV_2D_DW weight shape {arr.shape} not understood")
    arr = arr.astype(tr._weight_np_dtype, copy=False)
    name = tr._fresh(w_t.name + "_conv2dw")
    tp = numpy_helper.from_array(arr, name=name)
    try:
        orig = gdump.GgmlType(w_t.dtype).name
    except ValueError:
        orig = f"type_{w_t.dtype}"
    tp.doc_string = f"original_ggml_type={orig}"
    tr.initializers.append(tp)
    tr._original_dtypes[name] = orig
    return name


def _ensure_4d_for_conv(tr: "Translator", inp: str, inp_t: gdump.Tensor) -> str:
    """ONNX Conv requires rank-4 input (N, C, H, W). ``_ggml_shape_to_logical``
    strips trailing-1 dims, so for a (W, H, C, 1) input the ONNX binding has
    rank 3 (C, H, W). Reshape with a leading 1 so Conv's shape inference
    is happy.
    """
    inp_logical = _ggml_shape_to_logical(inp_t.ne)
    if len(inp_logical) < 4:
        new_shape = [1] * (4 - len(inp_logical)) + list(inp_logical)
        inp = tr._emit("Reshape", [inp, tr._const_i64("conv_inp_pad", new_shape)], "conv_inp_pad_r")
    return inp


@_op(gdump.GgmlOp.CONV_2D)
def _h_conv_2d(tr: Translator, t: gdump.Tensor) -> None:
    # op_params (i32): s0, s1, p0, p1, d0, d1.
    p = t.op_params_i32(6)
    s0, s1, p0, p1, d0, d1 = p
    weight_t = tr.src(t, 0)
    inp_t = tr.src(t, 1)
    inp = _ensure_4d_for_conv(tr, tr.src_value(t, 1), inp_t)
    # ggml input b: ne=[W, H, C, N] -> numpy (N, C, H, W). That's ONNX layout,
    # no permute needed (modulo possibly stripping trailing N=1, handled above).
    if weight_t.has_data:
        w_name = _conv2d_weight(tr, weight_t)
    else:
        # Symbolic weight — numpy decode of (KW, KH, IC, OC) is (OC, IC, KH, KW)
        # which already matches ONNX layout.
        w_name = tr.src_value(t, 0)
    # ONNX Conv attribute layout is (H, W) for both 2D kernel/stride/pad/dilation.
    tr.value[t.index] = tr._emit(
        "Conv", [inp, w_name], t.name or "conv2d",
        kernel_shape=[weight_t.ne[1], weight_t.ne[0]],  # (KH, KW)
        strides=[s1, s0],
        pads=[p1, p0, p1, p0],  # [begin_H, begin_W, end_H, end_W]
        dilations=[d1, d0],
        group=1,
    )


@_op(gdump.GgmlOp.CONV_2D_DW)
def _h_conv_2d_dw(tr: Translator, t: gdump.Tensor) -> None:
    # ggml_conv_2d_dw_direct stores the same 6 i32 params as CONV_2D:
    # [s0, s1, p0, p1, d0, d1].
    p = t.op_params_i32(6)
    s0, s1, p0, p1, d0, d1 = p
    weight_t = tr.src(t, 0)
    inp_t = tr.src(t, 1)
    inp = _ensure_4d_for_conv(tr, tr.src_value(t, 1), inp_t)
    # Depthwise: ggml weight ne = [KW, KH, 1, Cin]. Numpy decode = (Cin, KH, KW)
    # (after the 1 is stripped); _conv2d_weight inserts the missing IC=1 axis
    # to make it (Cin, 1, KH, KW) as ONNX expects.
    if weight_t.has_data:
        w_name = _conv2d_weight(tr, weight_t, depthwise=True)
    else:
        w_name = tr.src_value(t, 0)
    # group = Cin = src0.ne[3] in ggml (the "OC" slot, which is set to Cin
    # for depthwise convs).
    cin = weight_t.ne[3]
    tr.value[t.index] = tr._emit(
        "Conv", [inp, w_name], t.name or "conv2d_dw",
        kernel_shape=[weight_t.ne[1], weight_t.ne[0]],
        strides=[s1, s0],
        pads=[p1, p0, p1, p0],
        dilations=[d1, d0],
        group=cin,
    )


@_op(gdump.GgmlOp.IM2COL)
def _h_im2col(tr: Translator, t: gdump.Tensor) -> None:
    # op_params (i32): s0, s1, p0, p1, d0, d1, is_2d.
    p = t.op_params_i32(7)
    s0, s1, p0, p1, d0, d1, is_2d_i = p
    weight_t = tr.src(t, 0)  # a: kernel (gives us KH/KW)
    inp_t    = tr.src(t, 1)  # b: input
    inp_v    = tr.src_value(t, 1)
    is_2d = bool(is_2d_i)

    # ggml IM2COL's output has ne:
    #   2D:  ne[0] = IC * KH * KW, ne[1] = OW, ne[2] = OH,    ne[3] = N
    #   1D:  ne[0] = IC * K,       ne[1] = OW, ne[2] = b->ne[2], ne[3] = 1
    # Numpy row-major decode is the reverse: (N, OH, OW, IC*KH*KW) or 1D
    # equivalent. We synthesise the op via Gather + Concat. The input may
    # have rank 3 or 4 depending on whether N was elided by
    # _ggml_shape_to_logical (which strips trailing-1 dims); we normalise to
    # rank-4 (N, IC, IH, IW) up front by inserting any missing leading 1s.

    # Determine the actual numpy rank of inp_v by looking at the logical
    # shape we'd compute for inp_t.
    inp_logical = _ggml_shape_to_logical(inp_t.ne)

    if not is_2d:
        # 1D im2col: implement with the same logic with KH=1.
        KW = weight_t.ne[0]
        IC = inp_t.ne[1]
        N  = inp_t.ne[2]   # batch axis in 1D
        IW = inp_t.ne[0]
        OW = t.ne[1]

        # Normalise the input to numpy shape (N, IC, IW) regardless of how
        # _ggml_shape_to_logical squeezed trailing-1 dims.
        target_rank = 3
        if len(inp_logical) < target_rank:
            # Pad with leading 1s.
            new_shape = [1] * (target_rank - len(inp_logical)) + list(inp_logical)
            inp_v = tr._emit("Reshape", [inp_v, tr._const_i64("im2col1d_inp_pad", new_shape)], "im2col1d_inp_pad_r")

        cols = []
        for kw in range(KW):
            start = -p0 + kw * d0
            idxs = np.array([start + s0 * o for o in range(OW)], dtype=np.int64)
            valid = (idxs >= 0) & (idxs < IW)
            safe_idxs = np.clip(idxs, 0, IW - 1)
            idx_const = tr._const_i64(f"im2col_idx_{kw}", safe_idxs.tolist())
            gathered = tr._emit("Gather", [inp_v, idx_const], f"im2col_g_{kw}", axis=-1)
            if not valid.all():
                mask = valid.astype(np.float32)
                mask_const = tr._const(f"im2col_mask_{kw}", mask.astype(np.float32))
                mask_cast = tr._emit("Cast", [mask_const], f"im2col_mask_c_{kw}", to=tr._compute_onnx_dtype)
                gathered = tr._emit("Mul", [gathered, mask_cast], f"im2col_masked_{kw}")
            # gathered: (N, IC, OW). Add a leading kw axis -> (1, N, IC, OW).
            u = tr._emit("Unsqueeze", [gathered, tr._const_i64(f"im2col_u_{kw}", [0])], f"im2col_u_{kw}")
            cols.append(u)
        # Concat along axis 0: (KW, N, IC, OW)
        if len(cols) > 1:
            stacked_name = tr._fresh("im2col_stack")
            tr.nodes.append(helper.make_node("Concat", cols, [stacked_name], axis=0))
        else:
            stacked_name = cols[0]
        # Permute to (N, OW, IC, KW) and reshape to (N, OW, IC*KW).
        perm = tr._emit("Transpose", [stacked_name], "im2col_perm", perm=[1, 3, 2, 0])
        out_shape = [N, OW, IC * KW]
        tr.value[t.index] = tr._emit(
            "Reshape", [perm, tr._const_i64("im2col_out_shape", out_shape)],
            t.name or "im2col",
        )
        return

    # 2D im2col.
    KW = weight_t.ne[0]
    KH = weight_t.ne[1]
    IC = inp_t.ne[2]
    N  = inp_t.ne[3]
    IH = inp_t.ne[1]
    IW = inp_t.ne[0]
    OW = t.ne[1]
    OH = t.ne[2]

    # Normalise the input to numpy shape (N, IC, IH, IW) regardless of how
    # _ggml_shape_to_logical squeezed trailing-1 dims. When N is elided we
    # end up with logical shape (IC, IH, IW); we reshape to (1, IC, IH, IW)
    # so the remaining axis arithmetic doesn't shift.
    target_rank = 4
    if len(inp_logical) < target_rank:
        new_shape = [1] * (target_rank - len(inp_logical)) + list(inp_logical)
        inp_v = tr._emit("Reshape", [inp_v, tr._const_i64("im2col2d_inp_pad", new_shape)], "im2col2d_inp_pad_r")

    # Build the full grid of valid indices per (kh, kw). Output should have
    # numpy shape (N, OH, OW, IC*KH*KW) — that's the row-major decode of
    # ggml ne = [IC*KH*KW, OW, OH, N].
    cols = []
    for kh in range(KH):
        start_h = -p1 + kh * d1
        row_h = np.array([start_h + s1 * oh for oh in range(OH)], dtype=np.int64)
        valid_h = (row_h >= 0) & (row_h < IH)
        safe_h = np.clip(row_h, 0, IH - 1)
        for kw in range(KW):
            start_w = -p0 + kw * d0
            row_w = np.array([start_w + s0 * ow for ow in range(OW)], dtype=np.int64)
            valid_w = (row_w >= 0) & (row_w < IW)
            safe_w = np.clip(row_w, 0, IW - 1)
            h_idx_c = tr._const_i64(f"im2col_h_{kh}_{kw}", safe_h.tolist())
            g_h = tr._emit("Gather", [inp_v, h_idx_c], f"im2col_gh_{kh}_{kw}", axis=-2)
            w_idx_c = tr._const_i64(f"im2col_w_{kh}_{kw}", safe_w.tolist())
            g_hw = tr._emit("Gather", [g_h, w_idx_c], f"im2col_ghw_{kh}_{kw}", axis=-1)
            if not (valid_h.all() and valid_w.all()):
                m = (valid_h[:, None] & valid_w[None, :]).astype(np.float32)
                m_const = tr._const(f"im2col_m_{kh}_{kw}", m)
                m_cast = tr._emit("Cast", [m_const], f"im2col_m_c_{kh}_{kw}", to=tr._compute_onnx_dtype)
                g_hw = tr._emit("Mul", [g_hw, m_cast], f"im2col_masked_{kh}_{kw}")
            # Insert a leading "kk" axis so we can stack: (1, N, IC, OH, OW)
            u = tr._emit("Unsqueeze", [g_hw, tr._const_i64(f"im2col_u_{kh}_{kw}", [0])],
                         f"im2col_u_{kh}_{kw}")
            cols.append(u)
    # Concat along axis 0: shape (KH*KW, N, IC, OH, OW).
    if len(cols) > 1:
        stacked_name = tr._fresh("im2col_stack")
        tr.nodes.append(helper.make_node("Concat", cols, [stacked_name], axis=0))
    else:
        stacked_name = cols[0]
    # Permute to (N, OH, OW, IC, KH*KW), then reshape to (N, OH, OW, IC * KH * KW).
    perm = tr._emit("Transpose", [stacked_name], "im2col_perm", perm=[1, 3, 4, 2, 0])
    out_shape = [N, OH, OW, IC * KH * KW]
    tr.value[t.index] = tr._emit(
        "Reshape", [perm, tr._const_i64("im2col_out_shape", out_shape)],
        t.name or "im2col",
    )


# ----------------------------------------------------------------------
# Phase C: static-shape unrolled recurrent ops (Mamba SSM, RWKV-7)
# ----------------------------------------------------------------------
#
# Each of these ops is a `for t in range(n_tokens)` recurrence over a
# hidden state. Since the GGUF->ONNX exporter operates on static-shape
# prefill graphs (n_tokens is known at export time), we unroll the time
# axis: one copy of the step body per token, threading the state through
# a chain of ONNX nodes. This avoids needing ONNX `Scan` / `Loop` (which
# many runtimes implement poorly) and keeps the result fully amenable to
# graph optimisation.


def _unroll_recurrent(tr: "Translator", n_tokens: int, init_state: str,
                      step_fn) -> tuple[str, list[str]]:
    """Generic unroll harness for an op that produces a per-token output and
    threads a state. ``step_fn(state, t)`` should return ``(new_state, y_t)``
    where ``y_t`` already has a leading-token axis of 1 ready to Concat.

    Returns ``(final_state, [y_0, y_1, ...])``. The caller does the final
    Concat / packing of the per-token outputs.
    """
    state = init_state
    ys: list[str] = []
    for ti in range(n_tokens):
        state, y_t = step_fn(state, ti)
        ys.append(y_t)
    return state, ys


def _gather_token(tr: "Translator", v: str, ti: int, axis: int, hint: str) -> str:
    """Slice a single token out of an activation along ``axis`` and squeeze
    that axis. Returns the rank-reduced tensor name.
    """
    sl = tr._emit(
        "Slice",
        [v,
         tr._const_i64(f"{hint}_start", [ti]),
         tr._const_i64(f"{hint}_end",   [ti + 1]),
         tr._const_i64(f"{hint}_axes",  [axis])],
        f"{hint}_slice",
    )
    return tr._emit("Squeeze", [sl, tr._const_i64(f"{hint}_sq_ax", [axis])], f"{hint}_sq")


@_op(gdump.GgmlOp.SSM_CONV)
def _h_ssm_conv(tr: "Translator", t: gdump.Tensor) -> None:
    """Mamba 1D causal convolution.

    Source layout (per ``ggml_ssm_conv`` in ggml.c + the CPU forward):
      sx (src0):  ne=[d_conv-1+n_t, d_inner, n_s, 1]
        -> numpy logical (n_s, d_inner, d_conv-1+n_t)
      c  (src1):  ne=[d_conv, d_inner, 1, 1]      (weight, already padded by the caller)
        -> numpy logical (d_inner, d_conv)
      result: ne=[d_inner, n_t, n_s, 1]
        -> numpy logical (n_s, n_t, d_inner)

    For each ``(i3, i2, i1)``:
        out[i3, i2, i1] = sum_{i0=0}^{d_conv-1} sx[i3, i1, i0+i2] * c[i1, i0]

    This is a per-channel (depthwise) 1D correlation. We emit it as
    a single ONNX ``Conv`` node with ``group=d_inner``: the weight is
    reshaped to ``(d_inner, 1, d_conv)`` and the input is already in the
    expected ``(N, C, L)`` layout. The output is then transposed from
    ``(N, C, n_t)`` back to ``(N, n_t, C)`` to match ggml's output layout.

    Note: the caller has already pre-padded ``sx`` with the previous d_conv-1
    samples (see ``mamba-base.cpp`` -> ``ggml_concat``), so the convolution
    itself is unpadded (valid mode).
    """
    sx_t = tr.src(t, 0)
    c_t  = tr.src(t, 1)
    sx_v = tr.src_value(t, 0)
    c_v  = tr.src_value(t, 1)

    d_conv  = c_t.ne[0]
    d_inner = c_t.ne[1]
    n_t     = t.ne[1]
    n_s     = t.ne[2] if t.ne[2] > 0 else 1

    # Reshape weight from (d_inner, d_conv) to (d_inner, 1, d_conv) for
    # depthwise Conv. Use a dynamic Reshape so the symbolic d_inner/d_conv
    # in the dump don't need to match a literal initializer shape.
    w_shape = tr._const_i64("ssm_conv_w_shape", [d_inner, 1, d_conv])
    w = tr._emit("Reshape", [c_v, w_shape], "ssm_conv_w")

    # Input may come in as either rank-3 (n_s, d_inner, L) or rank-2 if
    # n_s was squeezed earlier. Force it to rank-3 with an explicit reshape.
    L = sx_t.ne[0]
    x_shape = tr._const_i64("ssm_conv_x_shape", [n_s, d_inner, L])
    x = tr._emit("Reshape", [sx_v, x_shape], "ssm_conv_x")

    y = tr._emit(
        "Conv", [x, w], "ssm_conv_y",
        group=d_inner, kernel_shape=[d_conv], pads=[0, 0], strides=[1], dilations=[1],
    )
    # ONNX gives us (n_s, d_inner, n_t). ggml wants logical (n_s, n_t, d_inner).
    y = tr._emit("Transpose", [y], "ssm_conv_yT", perm=[0, 2, 1])
    # Match the dump's reported output rank (strip the leading 1 if n_s == 1).
    if n_s == 1:
        # logical shape (n_t, d_inner) after squeezing the leading batch axis.
        y = tr._emit("Squeeze", [y, tr._const_i64("ssm_conv_sq", [0])], "ssm_conv_y2d")
    tr.value[t.index] = y


@_op(gdump.GgmlOp.SSM_SCAN)
def _h_ssm_scan(tr: "Translator", t: gdump.Tensor) -> None:
    """Mamba / Mamba-2 selective scan, time-axis unrolled.

    Source ordering (per ``ggml_ssm_scan`` in ggml.c):
        s   (src0): ne=[d_state, head_dim, n_head, n_seqs+]
              -> numpy (n_seqs+, n_head, head_dim, d_state)
        x   (src1): ne=[head_dim, n_head, n_seq_tokens, n_seqs]
              -> numpy (n_seqs, n_seq_tokens, n_head, head_dim)
        dt  (src2): ne=[n_head, n_seq_tokens, n_seqs]
              -> numpy (n_seqs, n_seq_tokens, n_head)
        A   (src3): ne=[d_state, n_head] or [1, n_head]
              -> numpy (n_head, d_state) or (n_head, 1)
        B   (src4): ne=[d_state, n_group, n_seq_tokens, n_seqs]
              -> numpy (n_seqs, n_seq_tokens, n_group, d_state)
        C   (src5): same shape as B
        ids (src6): ne=[n_seqs] (i32) -- selects which seq's state to read

    Step body per token i2 (CPU ref: ``ggml_compute_forward_ssm_scan_f32``):
        dt_sp = softplus(dt[i3, i2, h])
        x_dt  = x[i3, i2, h, i1] * dt_sp
        # Mamba-2: dA is a scalar per (h,)
        dA    = exp(dt_sp * A[h, 0])
        # Mamba-1: dA is per (h, d_state)
        dA    = exp(dt_sp * A[h, i0])
        state[i3, h, i1, i0] = s_prev[ids[i3], h, i1, i0] * dA
                             + B[i3, i2, g, i0] * x_dt
        y[i3, i2, h, i1] = sum_{i0} state[..., i0] * C[i3, i2, g, i0]
    where ``g = h // (n_head / n_group)``.

    The result tensor is a flat 1D buffer of ``nelements(x) + d_state*head_dim*n_head*n_seqs``
    floats, with the y region first (ne=[head_dim, n_head, n_seq_tokens, n_seqs]
    in ggml order, contiguous) followed by the new state (ne=[d_state, head_dim,
    n_head, n_seqs]).
    """
    s_t  = tr.src(t, 0)
    x_t  = tr.src(t, 1)
    dt_t = tr.src(t, 2)
    A_t  = tr.src(t, 3)
    B_t  = tr.src(t, 4)
    C_t  = tr.src(t, 5)
    ids_t = tr.src(t, 6)

    s_v  = tr.src_value(t, 0)
    x_v  = tr.src_value(t, 1)
    dt_v = tr.src_value(t, 2)
    A_v  = tr.src_value(t, 3)
    B_v  = tr.src_value(t, 4)
    C_v  = tr.src_value(t, 5)
    ids_v = tr.src_value(t, 6)

    d_state = s_t.ne[0]
    head_dim = x_t.ne[0]
    n_head = x_t.ne[1]
    n_seq_tokens = x_t.ne[2]
    n_seqs = x_t.ne[3] if x_t.ne[3] > 0 else 1
    n_group = B_t.ne[1] if B_t.ne[1] > 0 else 1
    is_mamba2 = (A_t.ne[0] == 1)
    heads_per_group = n_head // n_group  # repeat_interleave divisor

    # Bring everything to fp32 for the scan to match the CPU reference
    # exactly. The recurrence accumulates over d_state which is small (16)
    # but the loss of precision otherwise compounds across n_seq_tokens.
    s_f  = tr._emit("Cast", [s_v],  "ssm_s_f32",  to=TensorProto.FLOAT)
    x_f  = tr._emit("Cast", [x_v],  "ssm_x_f32",  to=TensorProto.FLOAT)
    dt_f = tr._emit("Cast", [dt_v], "ssm_dt_f32", to=TensorProto.FLOAT)
    A_f  = tr._emit("Cast", [A_v],  "ssm_A_f32",  to=TensorProto.FLOAT)
    B_f  = tr._emit("Cast", [B_v],  "ssm_B_f32",  to=TensorProto.FLOAT)
    C_f  = tr._emit("Cast", [C_v],  "ssm_C_f32",  to=TensorProto.FLOAT)
    ids_i64 = tr._emit("Cast", [ids_v], "ssm_ids_i64", to=TensorProto.INT64)

    # Reshape inputs to a canonical layout:
    #   x:  (n_seqs, n_seq_tokens, n_head, head_dim)
    #   dt: (n_seqs, n_seq_tokens, n_head)
    #   B:  (n_seqs, n_seq_tokens, n_group, d_state)
    #   C:  same as B
    #   s:  (n_seqs+, n_head, head_dim, d_state)
    #   A:  (n_head, d_state) or (n_head, 1)
    x_r  = tr._emit("Reshape", [x_f,  tr._const_i64("ssm_x_shape",  [n_seqs, n_seq_tokens, n_head, head_dim])], "ssm_x")
    dt_r = tr._emit("Reshape", [dt_f, tr._const_i64("ssm_dt_shape", [n_seqs, n_seq_tokens, n_head])],            "ssm_dt")
    B_r  = tr._emit("Reshape", [B_f,  tr._const_i64("ssm_B_shape",  [n_seqs, n_seq_tokens, n_group, d_state])], "ssm_B")
    C_r  = tr._emit("Reshape", [C_f,  tr._const_i64("ssm_C_shape",  [n_seqs, n_seq_tokens, n_group, d_state])], "ssm_C")
    n_seq_state = s_t.ne[3] if s_t.ne[3] > 0 else 1
    s_r  = tr._emit("Reshape", [s_f,  tr._const_i64("ssm_s_shape",  [n_seq_state, n_head, head_dim, d_state])], "ssm_s_init")
    if is_mamba2:
        A_r = tr._emit("Reshape", [A_f, tr._const_i64("ssm_A_shape", [n_head, 1])], "ssm_A")
    else:
        A_r = tr._emit("Reshape", [A_f, tr._const_i64("ssm_A_shape", [n_head, d_state])], "ssm_A")

    # Gather the initial state for each active sequence: ids has shape (n_seqs,).
    # state_cur: (n_seqs, n_head, head_dim, d_state)
    state = tr._emit("Gather", [s_r, ids_i64], "ssm_s_gathered", axis=0)

    # A and the group-broadcast structure are time-invariant; precompute as
    # broadcastable shapes (1, n_head, ...) so the step body just multiplies.
    # For the group expansion B/C[..., g, :] -> [..., h, :], we use a Gather
    # along the group axis with indices = h // heads_per_group.
    group_idx_np = np.array([h // heads_per_group for h in range(n_head)], dtype=np.int64)
    group_idx = tr._const("ssm_group_idx", group_idx_np)

    # Expand B / C from (n_seqs, n_seq_tokens, n_group, d_state) to
    # (n_seqs, n_seq_tokens, n_head, d_state) once, then slice per-token.
    B_expanded = tr._emit("Gather", [B_r, group_idx], "ssm_B_exp", axis=2)
    C_expanded = tr._emit("Gather", [C_r, group_idx], "ssm_C_exp", axis=2)

    # Helpers for softplus / one_const reused across step bodies.
    one_f32 = tr._const("ssm_one", np.array(1.0, dtype=np.float32))

    ys: list[str] = []
    for ti in range(n_seq_tokens):
        # Per-token inputs, shapes after squeeze:
        #   dt_t: (n_seqs, n_head)
        #   x_t : (n_seqs, n_head, head_dim)
        #   B_t : (n_seqs, n_head, d_state)
        #   C_t : (n_seqs, n_head, d_state)
        dt_ti = _gather_token(tr, dt_r,        ti, axis=1, hint=f"ssm_dt_t{ti}")
        x_ti  = _gather_token(tr, x_r,         ti, axis=1, hint=f"ssm_x_t{ti}")
        B_ti  = _gather_token(tr, B_expanded,  ti, axis=1, hint=f"ssm_B_t{ti}")
        C_ti  = _gather_token(tr, C_expanded,  ti, axis=1, hint=f"ssm_C_t{ti}")

        # softplus(dt) = log(1 + exp(dt)). ONNX opset 17 doesn't have Softplus
        # by spec but does have it as an alias; we use the explicit form for
        # safety across runtimes.
        dt_exp = tr._emit("Exp", [dt_ti], f"ssm_dt_exp_{ti}")
        dt_sp  = tr._emit("Log", [tr._emit("Add", [dt_exp, one_f32], f"ssm_dt_exp1_{ti}")], f"ssm_dt_sp_{ti}")
        # Broadcastable shapes:
        #   dt_sp     : (n_seqs, n_head)              -> unsqueeze for (..., 1, 1)
        #   A_r       : (n_head, d_state) or (n_head, 1)
        #   x_ti      : (n_seqs, n_head, head_dim)    -> unsqueeze to (..., 1)
        #   state     : (n_seqs, n_head, head_dim, d_state)
        #   B_ti      : (n_seqs, n_head, d_state)     -> unsqueeze to (..., 1, d_state)
        #   C_ti      : same as B_ti
        dt_sp_u = tr._emit("Unsqueeze", [dt_sp, tr._const_i64(f"ssm_dt_u_{ti}", [-1])], f"ssm_dt_sp_u_{ti}")  # (n_seqs, n_head, 1)
        # dA: in mamba2, A is (n_head, 1); broadcast over head_dim and d_state.
        #     dA shape = (n_seqs, n_head, 1) after exp(dt_sp_u * A[h, 0]).
        # in mamba1, A is (n_head, d_state); dA shape = (n_seqs, n_head, d_state).
        dA_arg = tr._emit("Mul", [dt_sp_u, A_r], f"ssm_dA_arg_{ti}")
        dA = tr._emit("Exp", [dA_arg], f"ssm_dA_{ti}")
        # Bring dA up to (..., 1, d_state_or_1) for broadcasting against state's
        # last two dims (head_dim, d_state).
        dA_b = tr._emit("Unsqueeze", [dA, tr._const_i64(f"ssm_dA_u_{ti}", [-2])], f"ssm_dA_b_{ti}")
        # state * dA, broadcasting over head_dim (and d_state if mamba2).
        state_decay = tr._emit("Mul", [state, dA_b], f"ssm_state_decay_{ti}")
        # x_dt = x * dt_sp, shape (n_seqs, n_head, head_dim) -> unsqueeze last
        # for broadcasting against d_state.
        dt_sp_b = tr._emit("Unsqueeze", [dt_sp, tr._const_i64(f"ssm_dtb_u_{ti}", [-1])], f"ssm_dt_sp_b_{ti}")
        x_dt = tr._emit("Mul", [x_ti, dt_sp_b], f"ssm_x_dt_{ti}")
        x_dt_u = tr._emit("Unsqueeze", [x_dt, tr._const_i64(f"ssm_xdt_u_{ti}", [-1])], f"ssm_x_dt_u_{ti}")  # (n_seqs, n_head, head_dim, 1)
        B_ti_u = tr._emit("Unsqueeze", [B_ti, tr._const_i64(f"ssm_Bu_{ti}", [-2])], f"ssm_B_u_{ti}")        # (n_seqs, n_head, 1, d_state)
        dB_x = tr._emit("Mul", [B_ti_u, x_dt_u], f"ssm_dB_x_{ti}")
        # New state.
        state = tr._emit("Add", [state_decay, dB_x], f"ssm_state_{ti}")
        # y_t = state @ C: sum over d_state of state * C (broadcast). Result
        # has shape (n_seqs, n_head, head_dim).
        C_ti_u = tr._emit("Unsqueeze", [C_ti, tr._const_i64(f"ssm_Cu_{ti}", [-2])], f"ssm_C_u_{ti}")        # (n_seqs, n_head, 1, d_state)
        y_prod = tr._emit("Mul", [state, C_ti_u], f"ssm_y_prod_{ti}")
        y_t = tr._emit("ReduceSum", [y_prod, tr._const_i64(f"ssm_y_axes_{ti}", [-1])], f"ssm_y_t{ti}", keepdims=0)
        # Stack along a fresh token axis = 1 (between n_seqs and n_head).
        y_t_u = tr._emit("Unsqueeze", [y_t, tr._const_i64(f"ssm_yu_{ti}", [1])], f"ssm_y_t{ti}_u")
        ys.append(y_t_u)

    # Concatenate per-token y outputs along the token axis.
    if n_seq_tokens == 1:
        y_concat = ys[0]
    else:
        y_concat = tr._fresh("ssm_y_concat")
        tr.nodes.append(helper.make_node("Concat", ys, [y_concat], axis=1))
    # y_concat now has shape (n_seqs, n_seq_tokens, n_head, head_dim) — matches
    # the ggml y layout. Flatten to 1D (ggml's contiguous (dim, nh, nt, ns)
    # innermost-first order is identical to our (ns, nt, nh, dim) row-major
    # flatten — bytes line up).
    n_y = head_dim * n_head * n_seq_tokens * n_seqs
    y_flat = tr._emit("Reshape", [y_concat, tr._const_i64("ssm_y_flat_shape", [n_y])], "ssm_y_flat")
    # state has shape (n_seqs, n_head, head_dim, d_state) -- same as the
    # ggml in-memory layout for the new state region.
    n_s = d_state * head_dim * n_head * n_seqs
    state_flat = tr._emit("Reshape", [state, tr._const_i64("ssm_state_flat_shape", [n_s])], "ssm_state_flat")
    # Concatenate y || state to match the flat 1D output buffer ggml produces.
    out = tr._fresh("ssm_scan_out")
    tr.nodes.append(helper.make_node("Concat", [y_flat, state_flat], [out], axis=0))
    # Cast back to the activation dtype so downstream consumers (which run
    # in compute_dtype) don't see an unexpected fp32 boundary.
    out_cast = tr._to_compute(out, t.name or "ssm_scan_out_cast")
    tr.value[t.index] = out_cast


@_op(gdump.GgmlOp.RWKV_WKV7)
def _h_rwkv_wkv7(tr: "Translator", t: gdump.Tensor) -> None:
    """RWKV-7 (Goose) WKV recurrence, time-axis unrolled.

    Source ordering (per ``ggml_rwkv_wkv7`` in ggml.c):
        r (src0), w (src1), k (src2), v (src3), a (src4), b (src5): ne=[S, H, T]
            -> numpy (T, H, S)
        state (src6): a flat buffer with ``S*S*H*n_seqs`` elements

    Per-token step body (CPU ref: ``ggml_compute_forward_rwkv_wkv7_f32``):
        for h in range(H):
            for i in range(S):
                sa[h, i] = sum_j a[t, h, j] * state_prev[h, i, j]
                for j in range(S):
                    state_cur[h, i, j] =
                        state_prev[h, i, j] * w[t, h, j]   # decay
                      + v[t, h, i] * k[t, h, j]            # kv outer product
                      + sa[h, i]   * b[t, h, j]            # sa*b outer product
                y[t, h, i] = sum_j state_cur[h, i, j] * r[t, h, j]

    Output is a 2D tensor with ne=[S*H, T + S*n_seqs] (logical
    (T + S*n_seqs, S*H)), where the first T rows are the y outputs
    and the remaining S*n_seqs rows are the final state laid out in the
    same ``(seq, h, i, j)`` memory order ggml uses.

    Note: ``n_seqs == 1`` in all current callers; the prefill graph has a
    single sequence. We still keep the seq axis explicit so the layout
    matches if someone exports a multi-sequence dump.
    """
    r_t = tr.src(t, 0)
    state_t = tr.src(t, 6)

    r_v = tr.src_value(t, 0)
    w_v = tr.src_value(t, 1)
    k_v = tr.src_value(t, 2)
    v_v = tr.src_value(t, 3)
    a_v = tr.src_value(t, 4)
    b_v = tr.src_value(t, 5)
    state_v = tr.src_value(t, 6)

    S = r_t.ne[0]           # head_size
    H = r_t.ne[1]           # head_count
    T = r_t.ne[2]           # n_tokens
    C = S * H
    # state's total elements / (S*S*H) gives n_seqs. The state's gdump shape
    # is unfortunately a bit overloaded; rely on the dump-level n_seqs as a
    # tiebreaker, but compute from the buffer size when available.
    n_seqs = max(1, int(np.prod(state_t.ne)) // (S * S * H))

    # Bring all inputs to fp32 for numerical fidelity with the CPU path.
    r_f = tr._emit("Cast", [r_v], "wkv_r_f32", to=TensorProto.FLOAT)
    w_f = tr._emit("Cast", [w_v], "wkv_w_f32", to=TensorProto.FLOAT)
    k_f = tr._emit("Cast", [k_v], "wkv_k_f32", to=TensorProto.FLOAT)
    v_f = tr._emit("Cast", [v_v], "wkv_v_f32", to=TensorProto.FLOAT)
    a_f = tr._emit("Cast", [a_v], "wkv_a_f32", to=TensorProto.FLOAT)
    b_f = tr._emit("Cast", [b_v], "wkv_b_f32", to=TensorProto.FLOAT)
    s_f = tr._emit("Cast", [state_v], "wkv_s_f32", to=TensorProto.FLOAT)

    # Canonicalise input shapes:
    #   r/w/k/v/a/b -> (T, H, S)
    #   state       -> (n_seqs, H, S, S)
    def _resh(name: str, shape: list[int], hint: str) -> str:
        return tr._emit("Reshape", [name, tr._const_i64(f"{hint}_sh", shape)], hint)
    r_r = _resh(r_f, [T, H, S], "wkv_r")
    w_r = _resh(w_f, [T, H, S], "wkv_w")
    k_r = _resh(k_f, [T, H, S], "wkv_k")
    v_r = _resh(v_f, [T, H, S], "wkv_v")
    a_r = _resh(a_f, [T, H, S], "wkv_a")
    b_r = _resh(b_f, [T, H, S], "wkv_b")
    state = _resh(s_f, [n_seqs, H, S, S], "wkv_s")

    # We only support n_seqs == 1 for unrolling — the C code separates state
    # per sequence on chunk boundaries, but llama.cpp prefill always uses one
    # sequence per ubatch.
    if n_seqs != 1:
        raise NotImplementedError(
            f"RWKV_WKV7 with n_seqs={n_seqs} not yet supported; prefill graphs "
            f"are expected to have n_seqs=1"
        )

    # Squeeze the seq axis so state is (H, S, S); we'll re-add it at the end.
    state = tr._emit("Squeeze", [state, tr._const_i64("wkv_s_sq0", [0])], "wkv_state_init")

    ys: list[str] = []
    for ti in range(T):
        # Per-token slices, shape (H, S). All produced from (T, H, S) tensors
        # via Slice + Squeeze along axis 0.
        r_t_ = _gather_token(tr, r_r, ti, axis=0, hint=f"wkv_r_t{ti}")
        w_t_ = _gather_token(tr, w_r, ti, axis=0, hint=f"wkv_w_t{ti}")
        k_t_ = _gather_token(tr, k_r, ti, axis=0, hint=f"wkv_k_t{ti}")
        v_t_ = _gather_token(tr, v_r, ti, axis=0, hint=f"wkv_v_t{ti}")
        a_t_ = _gather_token(tr, a_r, ti, axis=0, hint=f"wkv_a_t{ti}")
        b_t_ = _gather_token(tr, b_r, ti, axis=0, hint=f"wkv_b_t{ti}")
        # sa[h, i] = sum_j a[h, j] * state[h, i, j].
        # Broadcast a from (H, S) to (H, 1, S) and multiply with state (H, S, S).
        a_u = tr._emit("Unsqueeze", [a_t_, tr._const_i64(f"wkv_a_u_{ti}", [1])], f"wkv_a_u_{ti}")  # (H, 1, S)
        sa = tr._emit(
            "ReduceSum",
            [tr._emit("Mul", [state, a_u], f"wkv_sa_prod_{ti}"),
             tr._const_i64(f"wkv_sa_ax_{ti}", [-1])],
            f"wkv_sa_{ti}", keepdims=0,
        )  # (H, S)

        # kv[h, i, j] = v[h, i] * k[h, j]  -- outer product per head.
        v_u = tr._emit("Unsqueeze", [v_t_, tr._const_i64(f"wkv_v_u_{ti}", [-1])], f"wkv_v_u_{ti}")  # (H, S, 1)
        k_u = tr._emit("Unsqueeze", [k_t_, tr._const_i64(f"wkv_k_u_{ti}", [1])], f"wkv_k_u_{ti}")  # (H, 1, S)
        kv = tr._emit("Mul", [v_u, k_u], f"wkv_kv_{ti}")  # (H, S, S)
        # sab[h, i, j] = sa[h, i] * b[h, j]  -- another outer product.
        sa_u = tr._emit("Unsqueeze", [sa, tr._const_i64(f"wkv_sa_u_{ti}", [-1])], f"wkv_sa_u_{ti}")  # (H, S, 1)
        b_u  = tr._emit("Unsqueeze", [b_t_, tr._const_i64(f"wkv_b_u_{ti}", [1])], f"wkv_b_u_{ti}")    # (H, 1, S)
        sab  = tr._emit("Mul", [sa_u, b_u], f"wkv_sab_{ti}")  # (H, S, S)
        # decay[h, i, j] = state[h, i, j] * w[h, j].
        w_u = tr._emit("Unsqueeze", [w_t_, tr._const_i64(f"wkv_w_u_{ti}", [1])], f"wkv_w_u_{ti}")    # (H, 1, S)
        decay = tr._emit("Mul", [state, w_u], f"wkv_decay_{ti}")
        # New state.
        new_state = tr._emit(
            "Add",
            [tr._emit("Add", [decay, kv], f"wkv_decay_kv_{ti}"), sab],
            f"wkv_state_{ti}",
        )
        # y[h, i] = sum_j new_state[h, i, j] * r[h, j].
        r_u = tr._emit("Unsqueeze", [r_t_, tr._const_i64(f"wkv_r_u_{ti}", [1])], f"wkv_r_u_{ti}")    # (H, 1, S)
        y_prod = tr._emit("Mul", [new_state, r_u], f"wkv_y_prod_{ti}")
        y = tr._emit(
            "ReduceSum",
            [y_prod, tr._const_i64(f"wkv_y_ax_{ti}", [-1])],
            f"wkv_y_t{ti}", keepdims=0,
        )  # (H, S)
        # Stash y for the t-axis concat; reshape to (1, H*S) so we can stack
        # along a fresh leading axis.
        y_flat = tr._emit("Reshape", [y, tr._const_i64(f"wkv_y_flat_{ti}", [1, C])], f"wkv_y_flat_{ti}")
        ys.append(y_flat)
        state = new_state

    # Stack per-token outputs along the token axis: (T, S*H).
    if T == 1:
        y_all = ys[0]
    else:
        y_all = tr._fresh("wkv_y_concat")
        tr.nodes.append(helper.make_node("Concat", ys, [y_all], axis=0))

    # Final state: (H, S, S) -> flatten to row-major then reshape to (S*n_seqs, S*H).
    # ggml's in-memory order is (seq, h, i, j) with n_seqs=1 so equivalent to
    # (h, i, j) row-major. We want the combined buffer to satisfy
    #   buffer_row[T + r] = state_in_seq_at_row_r where r in [0, S*n_seqs)
    # i.e. buffer is reshape((T + S*n_seqs, S*H)) of a 1D float buffer that
    # matches ggml's memory layout. The state portion has S*S*H elements, so
    # S*n_seqs rows of S*H = S*S*H elements -- the math checks out.
    state_flat = tr._emit("Reshape", [state, tr._const_i64("wkv_s_flat", [S * H * S])], "wkv_state_flat_1d")
    state_rows = tr._emit("Reshape", [state_flat, tr._const_i64("wkv_s_rows", [S * n_seqs, C])], "wkv_state_rows")

    # Concat the y and state regions along axis 0 to produce the combined
    # (T + S*n_seqs, S*H) tensor.
    combined = tr._fresh("wkv_combined")
    tr.nodes.append(helper.make_node("Concat", [y_all, state_rows], [combined], axis=0))
    # Cast back to compute dtype.
    out = tr._to_compute(combined, t.name or "rwkv_wkv7_out")
    tr.value[t.index] = out


@_op(gdump.GgmlOp.RWKV_WKV6)
def _h_rwkv_wkv6(tr: "Translator", t: gdump.Tensor) -> None:
    """RWKV-6 WKV recurrence, time-axis unrolled.

    Source ordering (per ``ggml_rwkv_wkv6`` in ggml.c):
        k  (src0):     ne=[S, H, T, 1]    -> numpy (T, H, S)
        v  (src1):     ne=[S, H, T, 1]    -> numpy (T, H, S)
        r  (src2):     ne=[S, H, T, 1]    -> numpy (T, H, S)
        tf (src3):     ne=[S, H, 1, 1]    -> numpy (H, S)         (time_first / time_faaaa, NOT per-T)
        td (src4):     ne=[S, H, T, 1]    -> numpy (T, H, S)      (time_decay, IS per-T -- RWKV-6 differs from older variants)
        state (src5):  flat buffer with S*S*H*n_seqs elements

    Per-token, per-head step body (CPU ref: ``ggml_compute_forward_rwkv_wkv6_f32``):
        for i in range(S):
            for j in range(S):
                kv[i, j]        = k[t, h, i] * v[t, h, j]
                y[t, h, j]     += (kv[i, j] * tf[h, i] + state_prev[h, i, j]) * r[t, h, i]
                state[h, i, j]  = state_prev[h, i, j] * td[t, h, i] + kv[i, j]

    In vectorised form per (t, h), with shapes (S,) for k, v, r and (S, S)
    for state, (H, S) for tf, (T, H, S) for td:
        kv[i, j]    = k[i] * v[j]                            # outer product, (S, S)
        # Note: tf indexes along i (the "key" axis), td likewise:
        temp[i, j]  = kv[i, j] * tf[h, i] + state_prev[i, j] # (S, S)
        y[j]        = sum_i temp[i, j] * r[i]                # (S,)
        state'[i, j]= state_prev[i, j] * td[h, i] + kv[i, j] # (S, S)

    Output layout matches RWKV-7: a 2D (T + S*n_seqs, S*H) tensor where the
    first T rows hold the y outputs and the remaining S*n_seqs rows hold the
    final state in (seq, h, i, j) row-major order.
    """
    k_t = tr.src(t, 0)
    state_t = tr.src(t, 5)

    k_v  = tr.src_value(t, 0)
    v_v  = tr.src_value(t, 1)
    r_v  = tr.src_value(t, 2)
    tf_v = tr.src_value(t, 3)
    td_v = tr.src_value(t, 4)
    state_v = tr.src_value(t, 5)

    S = k_t.ne[0]           # head_size
    H = k_t.ne[1]           # head_count
    T = k_t.ne[2]           # n_tokens
    C = S * H
    # Compute n_seqs from the state buffer total size: state has S*S*H*n_seqs floats.
    n_seqs = max(1, int(np.prod(state_t.ne)) // (S * S * H))

    # Bring all inputs to fp32 for numerical fidelity with the CPU path.
    k_f  = tr._emit("Cast", [k_v],     "wkv6_k_f32",  to=TensorProto.FLOAT)
    v_f  = tr._emit("Cast", [v_v],     "wkv6_v_f32",  to=TensorProto.FLOAT)
    r_f  = tr._emit("Cast", [r_v],     "wkv6_r_f32",  to=TensorProto.FLOAT)
    tf_f = tr._emit("Cast", [tf_v],    "wkv6_tf_f32", to=TensorProto.FLOAT)
    td_f = tr._emit("Cast", [td_v],    "wkv6_td_f32", to=TensorProto.FLOAT)
    s_f  = tr._emit("Cast", [state_v], "wkv6_s_f32",  to=TensorProto.FLOAT)

    # Canonicalise input shapes:
    #   k/v/r/td -> (T, H, S)
    #   tf       -> (H, S)            (time-invariant — no T axis in WKV-6)
    #   state    -> (n_seqs, H, S, S)
    k_r  = tr._emit("Reshape", [k_f,  tr._const_i64("wkv6_k_sh",  [T, H, S])],         "wkv6_k")
    v_r  = tr._emit("Reshape", [v_f,  tr._const_i64("wkv6_v_sh",  [T, H, S])],         "wkv6_v")
    r_r  = tr._emit("Reshape", [r_f,  tr._const_i64("wkv6_r_sh",  [T, H, S])],         "wkv6_r")
    td_r = tr._emit("Reshape", [td_f, tr._const_i64("wkv6_td_sh", [T, H, S])],         "wkv6_td")
    tf_r = tr._emit("Reshape", [tf_f, tr._const_i64("wkv6_tf_sh", [H, S])],            "wkv6_tf")
    state = tr._emit("Reshape", [s_f, tr._const_i64("wkv6_s_sh",  [n_seqs, H, S, S])], "wkv6_s_init")

    if n_seqs != 1:
        raise NotImplementedError(
            f"RWKV_WKV6 with n_seqs={n_seqs} not yet supported; prefill graphs "
            f"are expected to have n_seqs=1"
        )

    # Squeeze the seq axis so state is (H, S, S).
    state = tr._emit("Squeeze", [state, tr._const_i64("wkv6_s_sq0", [0])], "wkv6_state_init")

    # tf is time-invariant: precompute its broadcast shape for the (H, S, 1) axis.
    tf_u = tr._emit("Unsqueeze", [tf_r, tr._const_i64("wkv6_tf_u", [-1])], "wkv6_tf_u")  # (H, S, 1)

    ys: list[str] = []
    for ti in range(T):
        # Per-token slices, all (H, S).
        k_t_  = _gather_token(tr, k_r,  ti, axis=0, hint=f"wkv6_k_t{ti}")
        v_t_  = _gather_token(tr, v_r,  ti, axis=0, hint=f"wkv6_v_t{ti}")
        r_t_  = _gather_token(tr, r_r,  ti, axis=0, hint=f"wkv6_r_t{ti}")
        td_t_ = _gather_token(tr, td_r, ti, axis=0, hint=f"wkv6_td_t{ti}")

        # kv[h, i, j] = k[h, i] * v[h, j]  -- outer product per head.
        k_u = tr._emit("Unsqueeze", [k_t_, tr._const_i64(f"wkv6_k_u_{ti}", [-1])], f"wkv6_k_u_{ti}")  # (H, S, 1)
        v_u = tr._emit("Unsqueeze", [v_t_, tr._const_i64(f"wkv6_v_u_{ti}", [1])],  f"wkv6_v_u_{ti}")  # (H, 1, S)
        kv  = tr._emit("Mul", [k_u, v_u], f"wkv6_kv_{ti}")  # (H, S, S)

        # temp[h, i, j] = kv[h, i, j] * tf[h, i] + state_prev[h, i, j]
        kv_tf = tr._emit("Mul", [kv, tf_u], f"wkv6_kv_tf_{ti}")  # (H, S, S)
        temp  = tr._emit("Add", [kv_tf, state], f"wkv6_temp_{ti}")  # (H, S, S)

        # y[h, j] = sum_i temp[h, i, j] * r[h, i]
        r_u = tr._emit("Unsqueeze", [r_t_, tr._const_i64(f"wkv6_r_u_{ti}", [-1])], f"wkv6_r_u_{ti}")  # (H, S, 1)
        y_prod = tr._emit("Mul", [temp, r_u], f"wkv6_y_prod_{ti}")  # (H, S, S)
        y = tr._emit(
            "ReduceSum",
            [y_prod, tr._const_i64(f"wkv6_y_ax_{ti}", [-2])],
            f"wkv6_y_t{ti}", keepdims=0,
        )  # (H, S)

        # state'[h, i, j] = state_prev[h, i, j] * td[h, i] + kv[h, i, j]
        td_u = tr._emit("Unsqueeze", [td_t_, tr._const_i64(f"wkv6_td_u_{ti}", [-1])], f"wkv6_td_u_{ti}")  # (H, S, 1)
        decay = tr._emit("Mul", [state, td_u], f"wkv6_decay_{ti}")  # (H, S, S)
        state = tr._emit("Add", [decay, kv], f"wkv6_state_{ti}")  # (H, S, S)

        # Stash y for the t-axis concat; reshape to (1, H*S).
        y_flat = tr._emit("Reshape", [y, tr._const_i64(f"wkv6_y_flat_{ti}", [1, C])], f"wkv6_y_flat_{ti}")
        ys.append(y_flat)

    # Stack per-token outputs along the token axis: (T, S*H).
    if T == 1:
        y_all = ys[0]
    else:
        y_all = tr._fresh("wkv6_y_concat")
        tr.nodes.append(helper.make_node("Concat", ys, [y_all], axis=0))

    # Final state: (H, S, S) flattened, then reshape to (S*n_seqs, S*H).
    state_flat = tr._emit("Reshape", [state, tr._const_i64("wkv6_s_flat", [S * H * S])], "wkv6_state_flat_1d")
    state_rows = tr._emit("Reshape", [state_flat, tr._const_i64("wkv6_s_rows", [S * n_seqs, C])], "wkv6_state_rows")

    combined = tr._fresh("wkv6_combined")
    tr.nodes.append(helper.make_node("Concat", [y_all, state_rows], [combined], axis=0))
    out = tr._to_compute(combined, t.name or "rwkv_wkv6_out")
    tr.value[t.index] = out


@_op(gdump.GgmlOp.GATED_LINEAR_ATTN)
def _h_gla(tr: "Translator", t: gdump.Tensor) -> None:
    """Gated linear attention (RWKV-6 hybrid path), time-axis unrolled.

    Source ordering (per ``ggml_gated_linear_attn`` in ggml.c):
        k  (src0):     ne=[S, H, T, 1]   -> numpy (T, H, S)
        v  (src1):     ne=[S, H, T, 1]   -> numpy (T, H, S)
        q  (src2):     ne=[S, H, T, 1]   -> numpy (T, H, S)
        g  (src3):     ne=[S, H, T, 1]   -> numpy (T, H, S)   (gate, per-i decay)
        state (src4):  flat buffer with S*S*H*n_seqs elements

    op_params: float ``scale`` at f32 offset 0 (applied to the output of each step).

    Per-token, per-head step body (CPU ref: ``ggml_compute_forward_gla_f32``):
        for i in range(S):
            for j in range(S):
                kv[i, j]        = k[i] * v[j]
                temp[i, j]      = state_prev[i, j] * g[i] + kv[i, j]
                y[j]           += temp[i, j] * (q[i] * scale)
                state[i, j]     = temp[i, j]

    Vectorised per (t, h), with state (S, S):
        kv      = outer(k, v)                                  # (S, S)
        temp    = state_prev * g[:, None] + kv                 # (S, S)
        y[j]    = sum_i temp[i, j] * q[i] * scale              # (S,)
        state'  = temp

    Output layout mirrors RWKV-6/7: 2D (T + S*n_seqs, S*H), y rows first.
    """
    k_t = tr.src(t, 0)
    state_t = tr.src(t, 4)

    k_v = tr.src_value(t, 0)
    v_v = tr.src_value(t, 1)
    q_v = tr.src_value(t, 2)
    g_v = tr.src_value(t, 3)
    state_v = tr.src_value(t, 4)

    S = k_t.ne[0]
    H = k_t.ne[1]
    T = k_t.ne[2]
    C = S * H
    n_seqs = max(1, int(np.prod(state_t.ne)) // (S * S * H))
    scale = t.op_params_f32(0, 1)[0]

    k_f = tr._emit("Cast", [k_v],     "gla_k_f32", to=TensorProto.FLOAT)
    v_f = tr._emit("Cast", [v_v],     "gla_v_f32", to=TensorProto.FLOAT)
    q_f = tr._emit("Cast", [q_v],     "gla_q_f32", to=TensorProto.FLOAT)
    g_f = tr._emit("Cast", [g_v],     "gla_g_f32", to=TensorProto.FLOAT)
    s_f = tr._emit("Cast", [state_v], "gla_s_f32", to=TensorProto.FLOAT)

    k_r = tr._emit("Reshape", [k_f, tr._const_i64("gla_k_sh", [T, H, S])], "gla_k")
    v_r = tr._emit("Reshape", [v_f, tr._const_i64("gla_v_sh", [T, H, S])], "gla_v")
    q_r = tr._emit("Reshape", [q_f, tr._const_i64("gla_q_sh", [T, H, S])], "gla_q")
    g_r = tr._emit("Reshape", [g_f, tr._const_i64("gla_g_sh", [T, H, S])], "gla_g")
    state = tr._emit("Reshape", [s_f, tr._const_i64("gla_s_sh", [n_seqs, H, S, S])], "gla_s_init")

    if n_seqs != 1:
        raise NotImplementedError(
            f"GATED_LINEAR_ATTN with n_seqs={n_seqs} not yet supported; prefill "
            f"graphs are expected to have n_seqs=1"
        )

    state = tr._emit("Squeeze", [state, tr._const_i64("gla_s_sq0", [0])], "gla_state_init")

    scale_const = tr._const("gla_scale", np.array(scale, dtype=np.float32))

    ys: list[str] = []
    for ti in range(T):
        k_t_ = _gather_token(tr, k_r, ti, axis=0, hint=f"gla_k_t{ti}")  # (H, S)
        v_t_ = _gather_token(tr, v_r, ti, axis=0, hint=f"gla_v_t{ti}")
        q_t_ = _gather_token(tr, q_r, ti, axis=0, hint=f"gla_q_t{ti}")
        g_t_ = _gather_token(tr, g_r, ti, axis=0, hint=f"gla_g_t{ti}")

        # kv[h, i, j] = k[h, i] * v[h, j]
        k_u = tr._emit("Unsqueeze", [k_t_, tr._const_i64(f"gla_k_u_{ti}", [-1])], f"gla_k_u_{ti}")  # (H, S, 1)
        v_u = tr._emit("Unsqueeze", [v_t_, tr._const_i64(f"gla_v_u_{ti}", [1])],  f"gla_v_u_{ti}")  # (H, 1, S)
        kv  = tr._emit("Mul", [k_u, v_u], f"gla_kv_{ti}")  # (H, S, S)

        # temp = state_prev * g[:, :, None] + kv     -- g decays state along i axis.
        g_u = tr._emit("Unsqueeze", [g_t_, tr._const_i64(f"gla_g_u_{ti}", [-1])], f"gla_g_u_{ti}")  # (H, S, 1)
        decay = tr._emit("Mul", [state, g_u], f"gla_decay_{ti}")  # (H, S, S)
        temp = tr._emit("Add", [decay, kv], f"gla_temp_{ti}")  # (H, S, S)

        # y[h, j] = scale * sum_i temp[h, i, j] * q[h, i]
        q_u = tr._emit("Unsqueeze", [q_t_, tr._const_i64(f"gla_q_u_{ti}", [-1])], f"gla_q_u_{ti}")  # (H, S, 1)
        y_prod = tr._emit("Mul", [temp, q_u], f"gla_y_prod_{ti}")  # (H, S, S)
        y_sum = tr._emit(
            "ReduceSum",
            [y_prod, tr._const_i64(f"gla_y_ax_{ti}", [-2])],
            f"gla_y_sum_{ti}", keepdims=0,
        )  # (H, S)
        y = tr._emit("Mul", [y_sum, scale_const], f"gla_y_t{ti}")

        # State carries forward as ``temp`` (no kv update beyond temp).
        state = temp

        y_flat = tr._emit("Reshape", [y, tr._const_i64(f"gla_y_flat_{ti}", [1, C])], f"gla_y_flat_{ti}")
        ys.append(y_flat)

    if T == 1:
        y_all = ys[0]
    else:
        y_all = tr._fresh("gla_y_concat")
        tr.nodes.append(helper.make_node("Concat", ys, [y_all], axis=0))

    state_flat = tr._emit("Reshape", [state, tr._const_i64("gla_s_flat", [S * H * S])], "gla_state_flat_1d")
    state_rows = tr._emit("Reshape", [state_flat, tr._const_i64("gla_s_rows", [S * n_seqs, C])], "gla_state_rows")

    combined = tr._fresh("gla_combined")
    tr.nodes.append(helper.make_node("Concat", [y_all, state_rows], [combined], axis=0))
    out = tr._to_compute(combined, t.name or "gla_out")
    tr.value[t.index] = out


@_op(gdump.GgmlOp.GATED_DELTA_NET)
def _h_gdn(tr: "Translator", t: gdump.Tensor) -> None:
    """Gated delta-net (Qwen3-Next style) delta-rule recurrence, time-axis unrolled.

    Source ordering (per ``ggml_gated_delta_net`` in ggml.c):
        q     (src0): ne=[S, H, T, n_seqs] -> numpy (n_seqs, T, H, S)
        k     (src1): ne=[S, H, T, n_seqs]
        v     (src2): ne=[S, H, T, n_seqs]
        g     (src3): ne=[1 or S, H, T, n_seqs]   (kda if ne[0]==S, else scalar gate)
        beta  (src4): ne=[1, H, T, n_seqs]
        state (src5): flat buffer with S*S*H*n_seqs floats

    Scale is the hard-coded value 1/sqrt(S) (NOT taken from op_params).

    The CPU code stores state TRANSPOSED in memory: a contiguous (S, S)
    block where memory layout is ``M[j, i] = S[i][j]``. In the in-memory
    view (which is what we work with in ONNX), S[i][j] lives at flat index
    ``j*S + i``. We work in the *logical* (i, j) view:

        state_logical[i, j] = state_memory[j * S + i]

    i.e. ``state_logical = state_memory.transpose()``.

    Per-token, per-head step body (CPU ref:
    ``ggml_compute_forward_gated_delta_net_one_chunk``):
        1. Gate-decay: state_logical[i, j] *= exp(g[i])   (kda; if scalar, *= exp(g))
        2. delta[j] = (v[j] - sum_i state_logical[i, j] * k[i]) * beta
        3. state_logical[i, j] += k[i] * delta[j]
        4. y[j] = scale * sum_i state_logical[i, j] * q[i]   where scale = 1/sqrt(S)

    Output layout: 2D (n_tokens*n_seqs + S*n_seqs, S*H), per the
    constructor's ne; for n_seqs==1 this is (T + S, S*H). The state portion
    is written back in the same transposed memory layout the CPU code uses.
    """
    q_t = tr.src(t, 0)
    g_t = tr.src(t, 3)
    state_t = tr.src(t, 5)

    q_v = tr.src_value(t, 0)
    k_v = tr.src_value(t, 1)
    v_v = tr.src_value(t, 2)
    g_v = tr.src_value(t, 3)
    beta_v = tr.src_value(t, 4)
    state_v = tr.src_value(t, 5)

    S = q_t.ne[0]
    H = q_t.ne[1]
    T = q_t.ne[2]
    n_seqs = q_t.ne[3] if q_t.ne[3] > 0 else 1
    C = S * H
    kda = (g_t.ne[0] == S)
    scale = 1.0 / float(np.sqrt(S))

    q_f    = tr._emit("Cast", [q_v],     "gdn_q_f32", to=TensorProto.FLOAT)
    k_f    = tr._emit("Cast", [k_v],     "gdn_k_f32", to=TensorProto.FLOAT)
    v_f    = tr._emit("Cast", [v_v],     "gdn_v_f32", to=TensorProto.FLOAT)
    g_f    = tr._emit("Cast", [g_v],     "gdn_g_f32", to=TensorProto.FLOAT)
    beta_f = tr._emit("Cast", [beta_v],  "gdn_b_f32", to=TensorProto.FLOAT)
    s_f    = tr._emit("Cast", [state_v], "gdn_s_f32", to=TensorProto.FLOAT)

    # Canonical shapes:
    #   q/k/v:    (n_seqs, T, H, S)
    #   g:        (n_seqs, T, H, S) if kda else (n_seqs, T, H, 1)
    #   beta:     (n_seqs, T, H, 1)
    #   state:    in-memory (n_seqs, H, S, S) -- logical[i, j] = memory[j, i],
    #             so reshape to memory layout then transpose the last two axes
    #             to get the logical view.
    q_r    = tr._emit("Reshape", [q_f,    tr._const_i64("gdn_q_sh", [n_seqs, T, H, S])], "gdn_q")
    k_r    = tr._emit("Reshape", [k_f,    tr._const_i64("gdn_k_sh", [n_seqs, T, H, S])], "gdn_k")
    v_r    = tr._emit("Reshape", [v_f,    tr._const_i64("gdn_v_sh", [n_seqs, T, H, S])], "gdn_v")
    g_dim  = S if kda else 1
    g_r    = tr._emit("Reshape", [g_f,    tr._const_i64("gdn_g_sh", [n_seqs, T, H, g_dim])], "gdn_g")
    beta_r = tr._emit("Reshape", [beta_f, tr._const_i64("gdn_b_sh", [n_seqs, T, H, 1])], "gdn_b")
    # state memory: (n_seqs, H, S, S) row-major; logical = transpose last two.
    state_mem = tr._emit("Reshape", [s_f, tr._const_i64("gdn_s_sh", [n_seqs, H, S, S])], "gdn_s_mem")
    state = tr._emit("Transpose", [state_mem], "gdn_s_init", perm=[0, 1, 3, 2])  # (n_seqs, H, S, S) logical

    if n_seqs != 1:
        raise NotImplementedError(
            f"GATED_DELTA_NET with n_seqs={n_seqs} not yet supported; prefill "
            f"graphs are expected to have n_seqs=1"
        )

    # Squeeze the seq axis so state is (H, S, S) logical.
    state = tr._emit("Squeeze", [state, tr._const_i64("gdn_s_sq0", [0])], "gdn_state_init")
    # Squeeze the seq axis from inputs as well.
    q_r2    = tr._emit("Squeeze", [q_r,    tr._const_i64("gdn_q_sq", [0])], "gdn_q_2d")
    k_r2    = tr._emit("Squeeze", [k_r,    tr._const_i64("gdn_k_sq", [0])], "gdn_k_2d")
    v_r2    = tr._emit("Squeeze", [v_r,    tr._const_i64("gdn_v_sq", [0])], "gdn_v_2d")
    g_r2    = tr._emit("Squeeze", [g_r,    tr._const_i64("gdn_g_sq", [0])], "gdn_g_2d")
    beta_r2 = tr._emit("Squeeze", [beta_r, tr._const_i64("gdn_b_sq", [0])], "gdn_b_2d")
    # Now: q/k/v: (T, H, S); g: (T, H, g_dim); beta: (T, H, 1).

    scale_const = tr._const("gdn_scale", np.array(scale, dtype=np.float32))

    ys: list[str] = []
    for ti in range(T):
        q_t_ = _gather_token(tr, q_r2, ti, axis=0, hint=f"gdn_q_t{ti}")  # (H, S)
        k_t_ = _gather_token(tr, k_r2, ti, axis=0, hint=f"gdn_k_t{ti}")
        v_t_ = _gather_token(tr, v_r2, ti, axis=0, hint=f"gdn_v_t{ti}")
        g_t_ = _gather_token(tr, g_r2, ti, axis=0, hint=f"gdn_g_t{ti}")  # (H, g_dim)
        b_t_ = _gather_token(tr, beta_r2, ti, axis=0, hint=f"gdn_b_t{ti}")  # (H, 1)

        # Step 1: gate-decay along the i axis of state.
        # state_logical[h, i, j] *= exp(g[h, i])  (kda) or *= exp(g[h, 0]) (scalar)
        exp_g = tr._emit("Exp", [g_t_], f"gdn_exp_g_{ti}")  # (H, g_dim)
        # In both kda and scalar cases we unsqueeze along the last axis to get
        # (H, S, 1) or (H, 1, 1) which broadcasts correctly across the j axis.
        exp_g_u = tr._emit("Unsqueeze", [exp_g, tr._const_i64(f"gdn_eg_u_{ti}", [-1])], f"gdn_eg_u_{ti}")
        state_decayed = tr._emit("Mul", [state, exp_g_u], f"gdn_state_dec_{ti}")  # (H, S, S)

        # Step 2: delta[h, j] = (v[h, j] - sum_i state_decayed[h, i, j] * k[h, i]) * beta[h, 0]
        k_u = tr._emit("Unsqueeze", [k_t_, tr._const_i64(f"gdn_k_u_{ti}", [-1])], f"gdn_k_u_{ti}")  # (H, S, 1)
        kdot_prod = tr._emit("Mul", [state_decayed, k_u], f"gdn_kdot_prod_{ti}")  # (H, S, S)
        kdot_sum = tr._emit(
            "ReduceSum",
            [kdot_prod, tr._const_i64(f"gdn_kdot_ax_{ti}", [-2])],
            f"gdn_kdot_{ti}", keepdims=0,
        )  # (H, S)
        v_minus = tr._emit("Sub", [v_t_, kdot_sum], f"gdn_vmk_{ti}")  # (H, S)
        delta = tr._emit("Mul", [v_minus, b_t_], f"gdn_delta_{ti}")  # (H, S)  -- b_t_ is (H, 1)

        # Step 3: state[h, i, j] += k[h, i] * delta[h, j]
        delta_u = tr._emit("Unsqueeze", [delta, tr._const_i64(f"gdn_d_u_{ti}", [1])], f"gdn_d_u_{ti}")  # (H, 1, S)
        outer_kd = tr._emit("Mul", [k_u, delta_u], f"gdn_kd_{ti}")  # (H, S, S)
        state = tr._emit("Add", [state_decayed, outer_kd], f"gdn_state_{ti}")  # (H, S, S)

        # Step 4: y[h, j] = scale * sum_i state[h, i, j] * q[h, i]
        q_u = tr._emit("Unsqueeze", [q_t_, tr._const_i64(f"gdn_q_u_{ti}", [-1])], f"gdn_q_u_{ti}")  # (H, S, 1)
        y_prod = tr._emit("Mul", [state, q_u], f"gdn_y_prod_{ti}")  # (H, S, S)
        y_sum = tr._emit(
            "ReduceSum",
            [y_prod, tr._const_i64(f"gdn_y_ax_{ti}", [-2])],
            f"gdn_y_sum_{ti}", keepdims=0,
        )  # (H, S)
        y = tr._emit("Mul", [y_sum, scale_const], f"gdn_y_t{ti}")  # (H, S)

        y_flat = tr._emit("Reshape", [y, tr._const_i64(f"gdn_y_flat_{ti}", [1, C])], f"gdn_y_flat_{ti}")
        ys.append(y_flat)

    if T == 1:
        y_all = ys[0]
    else:
        y_all = tr._fresh("gdn_y_concat")
        tr.nodes.append(helper.make_node("Concat", ys, [y_all], axis=0))

    # state is logical (H, S, S). Convert back to memory layout (transpose
    # last two so memory[h, j, i] = logical[h, i, j]) and flatten.
    state_mem_out = tr._emit("Transpose", [state], "gdn_s_out_mem", perm=[0, 2, 1])  # (H, S_j, S_i) memory
    state_flat = tr._emit("Reshape", [state_mem_out, tr._const_i64("gdn_s_flat", [S * H * S])], "gdn_state_flat_1d")
    state_rows = tr._emit("Reshape", [state_flat, tr._const_i64("gdn_s_rows", [S * n_seqs, C])], "gdn_state_rows")

    combined = tr._fresh("gdn_combined")
    tr.nodes.append(helper.make_node("Concat", [y_all, state_rows], [combined], axis=0))
    out = tr._to_compute(combined, t.name or "gdn_out")
    tr.value[t.index] = out


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def convert(
    dump_path: str | Path, onnx_path: str | Path, *,
    weight_dtype: str = "float16",
    extra_output_names: list[str] | None = None,
) -> None:
    dump_path = Path(dump_path)
    onnx_path = Path(onnx_path)
    logger.info("loading dump: %s", dump_path)
    d = gdump.load(dump_path)
    logger.info("arch=%s n_tokens=%d n_seqs=%d tensors=%d", d.arch, d.n_tokens, d.n_seqs, len(d.tensors))

    tr = Translator(d, weight_dtype=weight_dtype, extra_output_names=extra_output_names)
    model = tr.translate()

    data_location = onnx_path.name + ".data"
    logger.info("saving to %s (external data: %s)", onnx_path, data_location)
    onnx.save_model(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_location,
        size_threshold=1024,
        convert_attribute=False,
    )
    logger.info("done.")
