#!/usr/bin/env python3
"""Numerical unit tests for Phase-C recurrent op handlers.

Each test:
  1. Constructs a synthetic gdump-like ``Tensor`` set in-memory.
  2. Drives the handler to emit ONNX nodes.
  3. Wraps the emitted nodes into a minimal ONNX model and runs it
     through onnxruntime.
  4. Compares the result against a hand-written numpy reference
     translated directly from ggml/src/ggml-cpu/ops.cpp.

We exercise SSM_CONV, SSM_SCAN (both Mamba-1 and Mamba-2 forms),
and RWKV_WKV7.
"""

import math
import struct
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gguf-py"))

from gguf import gdump
from gguf.gdump_to_onnx import Translator


def _make_tensor(name, op, dtype, ne, sources=(), op_params=b"", flags=0,
                 data=None, index=0):
    ne_full = list(ne) + [1] * (4 - len(ne))
    nb = [4]
    for i in range(1, 4):
        nb.append(nb[-1] * ne_full[i - 1])
    return gdump.Tensor(
        name=name,
        op=int(op) if op is not None else 0,
        dtype=dtype,
        flags=flags,
        ne=ne_full,
        nb=nb,
        op_params=op_params + b"\0" * max(0, 64 - len(op_params)),
        sources=list(sources),
        data=data,
        index=index,
    )


def _run_handler(handler_op: gdump.GgmlOp, tensors: list, result_t: "gdump.Tensor",
                 input_feeds: dict, expected_output_shape):
    """Run a single handler in isolation against the given inputs."""
    d = gdump.GraphDump(arch="test", n_tokens=1, n_seqs=1, tensors=tensors)
    tr = Translator(d, weight_dtype="float32")
    # Bind inputs as ONNX inputs (skip leaves with data — those become
    # initializers).
    for t in tensors:
        if t.index == result_t.index:
            continue
        if t.has_data:
            tr._add_weight_initializer(t)
        else:
            name = tr._fresh(t.name or f"t{t.index}")
            shape = [d for d in t.ne if d > 0]
            while len(shape) > 1 and shape[-1] == 1:
                shape = shape[:-1]
            tr.inputs.append(helper.make_tensor_value_info(
                name, _ggml_to_onnx(t.dtype), shape))
            tr.value[t.index] = name
    # Run the handler.
    from gguf.gdump_to_onnx import _OP_HANDLERS
    _OP_HANDLERS[handler_op](tr, result_t)
    out_name = tr.value[result_t.index]
    # Cast result to fp32 in case the handler put us in compute_dtype.
    f_name = tr._fresh("dbg_out")
    tr.nodes.append(helper.make_node("Cast", [out_name], [f_name], to=TensorProto.FLOAT))
    tr.outputs.append(helper.make_tensor_value_info(f_name, TensorProto.FLOAT,
                                                    list(expected_output_shape)))
    graph = helper.make_graph(tr.nodes, "test", tr.inputs, tr.outputs, tr.initializers)
    opset = helper.make_opsetid("", 17)
    m = helper.make_model(graph, opset_imports=[opset])
    m.ir_version = 8
    onnx.checker.check_model(m)

    import onnxruntime as ort
    sess = ort.InferenceSession(m.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    outs = sess.run([f_name], input_feeds)
    return outs[0]


def _ggml_to_onnx(t):
    return {0: TensorProto.FLOAT, 26: TensorProto.INT32}[t]


# -------------------------------------------------------------------- SSM_CONV
def ssm_conv_ref(sx, c):
    """Mirror of ggml_compute_forward_ssm_conv_f32.

    sx logical (n_s, d_inner, d_conv-1+n_t), c logical (d_inner, d_conv).
    Returns (n_s, n_t, d_inner).
    """
    n_s, d_inner, ncs = sx.shape
    d_inner_w, d_conv = c.shape
    assert d_inner == d_inner_w
    n_t = ncs - d_conv + 1
    out = np.zeros((n_s, n_t, d_inner), dtype=np.float32)
    for i3 in range(n_s):
        for i2 in range(n_t):
            for i1 in range(d_inner):
                s = 0.0
                for i0 in range(d_conv):
                    s += sx[i3, i1, i0 + i2] * c[i1, i0]
                out[i3, i2, i1] = s
    return out


def test_ssm_conv():
    rng = np.random.default_rng(0)
    d_conv, d_inner, n_t, n_s = 4, 6, 8, 1
    sx = rng.standard_normal((n_s, d_inner, d_conv - 1 + n_t)).astype(np.float32)
    c  = rng.standard_normal((d_inner, d_conv)).astype(np.float32)
    ref = ssm_conv_ref(sx, c)

    tensors = []
    sx_t = _make_tensor("sx", None, 0, [d_conv - 1 + n_t, d_inner, n_s], index=0,
                        flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT)
    c_t  = _make_tensor("c",  None, 0, [d_conv, d_inner],                index=1,
                        flags=gdump.FLAG_IS_LEAF | gdump.FLAG_HAS_DATA,
                        data=c)
    # Result tensor: ne=[d_inner, n_t, n_s].
    result_t = _make_tensor("ssm_conv", gdump.GgmlOp.SSM_CONV, 0,
                            [d_inner, n_t, n_s], sources=[0, 1], index=2)
    tensors = [sx_t, c_t, result_t]
    # Find the input name onnxruntime expects.
    d = gdump.GraphDump(arch="test", n_tokens=1, n_seqs=1, tensors=tensors)
    tr = Translator(d, weight_dtype="float32")
    tr._add_weight_initializer(c_t)
    sx_name = tr._fresh(sx_t.name)
    tr.inputs.append(helper.make_tensor_value_info(sx_name, TensorProto.FLOAT,
                                                    [n_s, d_inner, d_conv - 1 + n_t]))
    tr.value[0] = sx_name
    from gguf.gdump_to_onnx import _OP_HANDLERS
    _OP_HANDLERS[gdump.GgmlOp.SSM_CONV](tr, result_t)
    out_name = tr.value[result_t.index]
    expected_shape = [n_t, d_inner] if n_s == 1 else [n_s, n_t, d_inner]
    tr.outputs.append(helper.make_tensor_value_info(out_name, TensorProto.FLOAT,
                                                     expected_shape))
    graph = helper.make_graph(tr.nodes, "test", tr.inputs, tr.outputs, tr.initializers)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    import onnxruntime as ort
    sess = ort.InferenceSession(m.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    (got,) = sess.run([out_name], {sx_name: sx})
    if n_s == 1:
        got = got.reshape(n_s, n_t, d_inner)
    diff = np.abs(got - ref).max()
    assert diff < 1e-4, f"SSM_CONV max diff {diff}"
    print(f"SSM_CONV: ok  (max diff {diff:.2e})")


# -------------------------------------------------------------------- SSM_SCAN
def ssm_scan_ref(s0, x, dt, A, B, C, ids, is_mamba2: bool):
    """Mirror of ggml_compute_forward_ssm_scan_f32 (the non-SIMD scalar path).

    Logical shapes:
      s0: (n_seqs+, n_head, head_dim, d_state)
      x:  (n_seqs, n_seq_tokens, n_head, head_dim)
      dt: (n_seqs, n_seq_tokens, n_head)
      A:  (n_head, d_state) or (n_head, 1)
      B:  (n_seqs, n_seq_tokens, n_group, d_state)
      C:  same as B
      ids: (n_seqs,) int32

    Returns y (n_seqs, n_seq_tokens, n_head, head_dim) and
            new_state (n_seqs, n_head, head_dim, d_state).
    """
    n_seqs, n_seq_tokens, n_head, head_dim = x.shape
    d_state = s0.shape[-1]
    n_group = B.shape[2]
    heads_per_group = n_head // n_group

    y = np.zeros_like(x, dtype=np.float32)
    state = np.zeros((n_seqs, n_head, head_dim, d_state), dtype=np.float32)

    for i3 in range(n_seqs):
        s_prev = s0[ids[i3]].copy()
        for i2 in range(n_seq_tokens):
            for h in range(n_head):
                g = h // heads_per_group
                dt_sp = math.log1p(math.exp(dt[i3, i2, h]))  # softplus
                if is_mamba2:
                    dA = math.exp(dt_sp * A[h, 0])
                for i1 in range(head_dim):
                    x_dt = x[i3, i2, h, i1] * dt_sp
                    s_acc = 0.0
                    for i0 in range(d_state):
                        if is_mamba2:
                            new = s_prev[h, i1, i0] * dA + B[i3, i2, g, i0] * x_dt
                        else:
                            new = s_prev[h, i1, i0] * math.exp(dt_sp * A[h, i0]) + B[i3, i2, g, i0] * x_dt
                        s_acc += new * C[i3, i2, g, i0]
                        s_prev[h, i1, i0] = new
                    y[i3, i2, h, i1] = s_acc
        state[i3] = s_prev
    return y, state


def _ssm_scan_case(*, is_mamba2: bool, d_state, head_dim, n_head, n_seq_tokens,
                   n_seqs, n_group, label):
    rng = np.random.default_rng(0)
    s0 = (rng.standard_normal((n_seqs, n_head, head_dim, d_state)) * 0.1).astype(np.float32)
    x  = rng.standard_normal((n_seqs, n_seq_tokens, n_head, head_dim)).astype(np.float32) * 0.1
    dt = rng.standard_normal((n_seqs, n_seq_tokens, n_head)).astype(np.float32) * 0.5 - 4.0
    if is_mamba2:
        A = -np.abs(rng.standard_normal((n_head, 1))).astype(np.float32)
    else:
        A = -np.abs(rng.standard_normal((n_head, d_state))).astype(np.float32)
    B = rng.standard_normal((n_seqs, n_seq_tokens, n_group, d_state)).astype(np.float32) * 0.3
    C = rng.standard_normal((n_seqs, n_seq_tokens, n_group, d_state)).astype(np.float32) * 0.3
    ids = np.arange(n_seqs, dtype=np.int32)

    y_ref, state_ref = ssm_scan_ref(s0, x, dt, A, B, C, ids, is_mamba2=is_mamba2)

    # Build a fake gdump with these as inputs.
    n_seq_state = n_seqs  # for simplicity
    # ne layouts (ggml order: innermost first).
    s_ne  = [d_state, head_dim, n_head, n_seq_state]
    x_ne  = [head_dim, n_head, n_seq_tokens, n_seqs]
    dt_ne = [n_head, n_seq_tokens, n_seqs, 1]
    A_ne  = [1 if is_mamba2 else d_state, n_head, 1, 1]
    B_ne  = [d_state, n_group, n_seq_tokens, n_seqs]
    C_ne  = B_ne[:]
    ids_ne = [n_seqs, 1, 1, 1]

    tensors = []
    s_t   = _make_tensor("s",  None, 0, s_ne,  index=0, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT)
    x_t   = _make_tensor("x",  None, 0, x_ne,  index=1, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT)
    dt_t  = _make_tensor("dt", None, 0, dt_ne, index=2, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT)
    A_t   = _make_tensor("A",  None, 0, A_ne,  index=3, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_HAS_DATA, data=A)
    B_t   = _make_tensor("B",  None, 0, B_ne,  index=4, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT)
    C_t   = _make_tensor("C",  None, 0, C_ne,  index=5, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT)
    ids_t = _make_tensor("ids", None, 26, ids_ne, index=6, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT)
    result_ne = [head_dim * n_head * n_seq_tokens * n_seqs + d_state * head_dim * n_head * n_seqs, 1, 1, 1]
    result_t = _make_tensor("ssm_scan", gdump.GgmlOp.SSM_SCAN, 0, result_ne,
                            sources=[0, 1, 2, 3, 4, 5, 6], index=7)
    tensors = [s_t, x_t, dt_t, A_t, B_t, C_t, ids_t, result_t]

    d = gdump.GraphDump(arch="test", n_tokens=1, n_seqs=1, tensors=tensors)
    tr = Translator(d, weight_dtype="float32")
    # A is a weight (has data); add as initializer.
    tr._add_weight_initializer(A_t)
    # Other inputs: register as ONNX inputs in their logical shapes.
    def _reg(t, logical_shape, onnx_dtype):
        name = tr._fresh(t.name)
        tr.inputs.append(helper.make_tensor_value_info(name, onnx_dtype, logical_shape))
        tr.value[t.index] = name
        return name
    s_name  = _reg(s_t,  [n_seq_state, n_head, head_dim, d_state], TensorProto.FLOAT)
    x_name  = _reg(x_t,  [n_seqs, n_seq_tokens, n_head, head_dim], TensorProto.FLOAT)
    dt_name = _reg(dt_t, [n_seqs, n_seq_tokens, n_head],           TensorProto.FLOAT)
    B_name  = _reg(B_t,  [n_seqs, n_seq_tokens, n_group, d_state], TensorProto.FLOAT)
    C_name  = _reg(C_t,  [n_seqs, n_seq_tokens, n_group, d_state], TensorProto.FLOAT)
    ids_name = _reg(ids_t, [n_seqs], TensorProto.INT32)
    from gguf.gdump_to_onnx import _OP_HANDLERS
    _OP_HANDLERS[gdump.GgmlOp.SSM_SCAN](tr, result_t)
    out_name = tr.value[result_t.index]
    tr.outputs.append(helper.make_tensor_value_info(out_name, TensorProto.FLOAT,
                                                     [result_ne[0]]))
    graph = helper.make_graph(tr.nodes, "test", tr.inputs, tr.outputs, tr.initializers)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)

    import onnxruntime as ort
    sess = ort.InferenceSession(m.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    feeds = {s_name: s0, x_name: x, dt_name: dt, B_name: B, C_name: C,
             ids_name: ids}
    (got,) = sess.run([out_name], feeds)
    # Split y / state.
    n_y = head_dim * n_head * n_seq_tokens * n_seqs
    got_y = got[:n_y].reshape(n_seqs, n_seq_tokens, n_head, head_dim)
    got_state = got[n_y:].reshape(n_seqs, n_head, head_dim, d_state)
    diff_y = np.abs(got_y - y_ref).max()
    diff_state = np.abs(got_state - state_ref).max()
    assert diff_y < 1e-3, f"{label}: y mismatch max abs diff {diff_y}"
    assert diff_state < 1e-3, f"{label}: state mismatch max abs diff {diff_state}"
    print(f"{label}: ok  (y max diff {diff_y:.2e}, state max diff {diff_state:.2e})")


def test_ssm_scan_mamba2():
    _ssm_scan_case(is_mamba2=True, d_state=4, head_dim=2, n_head=4,
                   n_seq_tokens=3, n_seqs=1, n_group=2,
                   label="SSM_SCAN(Mamba-2)")


def test_ssm_scan_mamba1():
    _ssm_scan_case(is_mamba2=False, d_state=4, head_dim=2, n_head=2,
                   n_seq_tokens=3, n_seqs=1, n_group=1,
                   label="SSM_SCAN(Mamba-1)")


# ----------------------------------------------------------------- RWKV_WKV7
def rwkv_wkv7_ref(r, w, k, v, a, b, state):
    """Mirror of ggml_compute_forward_rwkv_wkv7_f32 (scalar path).

    r/w/k/v/a/b: (T, H, S). state: (n_seqs, H, S, S).
    Returns y (T, H, S) and new_state (n_seqs, H, S, S).
    """
    T, H, S = r.shape
    n_seqs = state.shape[0]
    y = np.zeros((T, H, S), dtype=np.float32)
    new_state = state.copy().astype(np.float32)
    C = H * S
    # The CPU C code routes each token to seq = t // (T / n_seqs).
    chunk = T // n_seqs
    for t in range(T):
        seq = min(n_seqs - 1, t // max(1, chunk))
        for h in range(H):
            for i in range(S):
                sa = 0.0
                for j in range(S):
                    sa += a[t, h, j] * new_state[seq, h, i, j]
                yt = 0.0
                for j in range(S):
                    kv = v[t, h, i] * k[t, h, j]
                    new = new_state[seq, h, i, j] * w[t, h, j] + kv + sa * b[t, h, j]
                    new_state[seq, h, i, j] = new
                    yt += new * r[t, h, j]
                y[t, h, i] = yt
    return y, new_state


def test_rwkv_wkv7():
    rng = np.random.default_rng(0)
    S, H, T, n_seqs = 4, 3, 5, 1
    r = rng.standard_normal((T, H, S)).astype(np.float32) * 0.3
    w = (np.tanh(rng.standard_normal((T, H, S)).astype(np.float32)) + 1.0) * 0.4   # in (0, 0.8)
    k = rng.standard_normal((T, H, S)).astype(np.float32) * 0.3
    v = rng.standard_normal((T, H, S)).astype(np.float32) * 0.3
    a = rng.standard_normal((T, H, S)).astype(np.float32) * 0.3
    b_ = rng.standard_normal((T, H, S)).astype(np.float32) * 0.3
    state = (rng.standard_normal((n_seqs, H, S, S)) * 0.1).astype(np.float32)
    y_ref, state_ref = rwkv_wkv7_ref(r, w, k, v, a, b_, state)

    # Build gdump.
    r_ne = w_ne = k_ne = v_ne = a_ne = b_ne = [S, H, T, 1]
    state_ne = [S * S * H * n_seqs, 1, 1, 1]
    result_ne = [S * H, T + S * n_seqs, 1, 1]
    tensors = [
        _make_tensor("r", None, 0, r_ne, index=0, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("w", None, 0, w_ne, index=1, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("k", None, 0, k_ne, index=2, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("v", None, 0, v_ne, index=3, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("a", None, 0, a_ne, index=4, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("b", None, 0, b_ne, index=5, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("state", None, 0, state_ne, index=6, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
    ]
    result_t = _make_tensor("rwkv_wkv7", gdump.GgmlOp.RWKV_WKV7, 0, result_ne,
                            sources=[0, 1, 2, 3, 4, 5, 6], index=7)
    tensors.append(result_t)
    d = gdump.GraphDump(arch="test", n_tokens=1, n_seqs=1, tensors=tensors)
    tr = Translator(d, weight_dtype="float32")
    names = []
    for i, label in enumerate("rwkvab"):
        name = tr._fresh(label)
        tr.inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT,
                                                       [T, H, S]))
        tr.value[i] = name
        names.append(name)
    state_name = tr._fresh("state")
    tr.inputs.append(helper.make_tensor_value_info(
        state_name, TensorProto.FLOAT, [S * S * H * n_seqs]))
    tr.value[6] = state_name
    from gguf.gdump_to_onnx import _OP_HANDLERS
    _OP_HANDLERS[gdump.GgmlOp.RWKV_WKV7](tr, result_t)
    out_name = tr.value[result_t.index]
    tr.outputs.append(helper.make_tensor_value_info(out_name, TensorProto.FLOAT,
                                                    [T + S * n_seqs, S * H]))
    graph = helper.make_graph(tr.nodes, "test", tr.inputs, tr.outputs, tr.initializers)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    import onnxruntime as ort
    sess = ort.InferenceSession(m.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    feeds = {names[0]: r, names[1]: w, names[2]: k, names[3]: v,
             names[4]: a, names[5]: b_,
             state_name: state.reshape(-1)}
    (got,) = sess.run([out_name], feeds)
    # Split y and state.
    got_y = got[:T].reshape(T, H, S)
    got_state = got[T:].reshape(n_seqs, H, S, S)
    diff_y = np.abs(got_y - y_ref).max()
    diff_state = np.abs(got_state - state_ref).max()
    assert diff_y < 1e-3, f"RWKV_WKV7: y mismatch {diff_y}"
    assert diff_state < 1e-3, f"RWKV_WKV7: state mismatch {diff_state}"
    print(f"RWKV_WKV7: ok  (y max diff {diff_y:.2e}, state max diff {diff_state:.2e})")


if __name__ == "__main__":
    test_ssm_conv()
    test_ssm_scan_mamba2()
    test_ssm_scan_mamba1()
    test_rwkv_wkv7()
    print("\nAll Phase-C math tests passed.")
