#!/usr/bin/env python3
"""Numerical unit tests for the three recurrent / state-space op handlers
added on top of Phase C: RWKV_WKV6, GATED_LINEAR_ATTN, GATED_DELTA_NET.

For each op:
  1. Construct a synthetic gdump-like ``Tensor`` set in-memory.
  2. Drive the handler to emit ONNX nodes.
  3. Wrap the emitted nodes into a minimal ONNX model and run it through
     onnxruntime.
  4. Compare against a scalar numpy reference translated directly from
     ggml/src/ggml-cpu/ops.cpp.
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

from gguf import gdump  # noqa: E402
from gguf.gdump_to_onnx import Translator  # noqa: E402


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


# --------------------------------------------------------------------- RWKV_WKV6
def rwkv_wkv6_ref(k, v, r, tf, td, state):
    """Scalar mirror of ggml_compute_forward_rwkv_wkv6_f32 (non-SIMD path).

    Logical shapes:
      k/v/r/td: (T, H, S)
      tf:       (H, S)
      state:    (n_seqs, H, S, S)
    Returns y (T, H, S), new_state (n_seqs, H, S, S).
    """
    T, H, S = k.shape
    n_seqs = state.shape[0]
    y = np.zeros((T, H, S), dtype=np.float32)
    new_state = state.copy().astype(np.float32)
    # The CPU code routes each token to seq = t // (T / n_seqs).
    chunk = max(1, T // n_seqs)
    for t in range(T):
        seq = min(n_seqs - 1, t // chunk)
        for h in range(H):
            for i in range(S):
                k_val = k[t, h, i]
                r_val = r[t, h, i]
                tf_val = tf[h, i]
                td_val = td[t, h, i]
                for j in range(S):
                    v_val = v[t, h, j]
                    kv = v_val * k_val
                    prev = new_state[seq, h, i, j]
                    temp = kv * tf_val + prev
                    y[t, h, j] += temp * r_val
                    new_state[seq, h, i, j] = prev * td_val + kv
    return y, new_state


def test_rwkv_wkv6():
    rng = np.random.default_rng(123)
    S, H, T, n_seqs = 8, 4, 6, 1
    k  = rng.standard_normal((T, H, S)).astype(np.float32) * 0.3
    v  = rng.standard_normal((T, H, S)).astype(np.float32) * 0.3
    r  = rng.standard_normal((T, H, S)).astype(np.float32) * 0.3
    tf = rng.standard_normal((H, S)).astype(np.float32) * 0.5
    td = (np.tanh(rng.standard_normal((T, H, S)).astype(np.float32)) + 1.0) * 0.4
    state = (rng.standard_normal((n_seqs, H, S, S)) * 0.1).astype(np.float32)

    y_ref, state_ref = rwkv_wkv6_ref(k, v, r, tf, td, state)

    # Build a gdump-style record. Sources: (k, v, r, tf, td, state).
    k_ne = v_ne = r_ne = td_ne = [S, H, T, 1]
    tf_ne = [S, H, 1, 1]
    state_ne = [S * S * H * n_seqs, 1, 1, 1]
    result_ne = [S * H, T + S * n_seqs, 1, 1]
    tensors = [
        _make_tensor("k",  None, 0, k_ne,  index=0, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("v",  None, 0, v_ne,  index=1, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("r",  None, 0, r_ne,  index=2, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("tf", None, 0, tf_ne, index=3, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("td", None, 0, td_ne, index=4, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("state", None, 0, state_ne, index=5, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
    ]
    result_t = _make_tensor("rwkv_wkv6", gdump.GgmlOp.RWKV_WKV6, 0, result_ne,
                            sources=[0, 1, 2, 3, 4, 5], index=6)
    tensors.append(result_t)
    d = gdump.GraphDump(arch="test", n_tokens=1, n_seqs=1, tensors=tensors)
    tr = Translator(d, weight_dtype="float32")
    # Register each input.
    name_map = {}
    k_name  = tr._fresh("k");  tr.inputs.append(helper.make_tensor_value_info(k_name,  TensorProto.FLOAT, [T, H, S]));  tr.value[0] = k_name;  name_map["k"]  = k_name
    v_name  = tr._fresh("v");  tr.inputs.append(helper.make_tensor_value_info(v_name,  TensorProto.FLOAT, [T, H, S]));  tr.value[1] = v_name;  name_map["v"]  = v_name
    r_name  = tr._fresh("r");  tr.inputs.append(helper.make_tensor_value_info(r_name,  TensorProto.FLOAT, [T, H, S]));  tr.value[2] = r_name;  name_map["r"]  = r_name
    tf_name = tr._fresh("tf"); tr.inputs.append(helper.make_tensor_value_info(tf_name, TensorProto.FLOAT, [H, S]));     tr.value[3] = tf_name; name_map["tf"] = tf_name
    td_name = tr._fresh("td"); tr.inputs.append(helper.make_tensor_value_info(td_name, TensorProto.FLOAT, [T, H, S]));  tr.value[4] = td_name; name_map["td"] = td_name
    s_name  = tr._fresh("s");  tr.inputs.append(helper.make_tensor_value_info(s_name,  TensorProto.FLOAT, [S * S * H * n_seqs])); tr.value[5] = s_name; name_map["s"] = s_name

    from gguf.gdump_to_onnx import _OP_HANDLERS
    _OP_HANDLERS[gdump.GgmlOp.RWKV_WKV6](tr, result_t)
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
    feeds = {
        k_name:  k,
        v_name:  v,
        r_name:  r,
        tf_name: tf,
        td_name: td,
        s_name:  state.reshape(-1),
    }
    (got,) = sess.run([out_name], feeds)
    got_y = got[:T].reshape(T, H, S)
    got_state = got[T:].reshape(n_seqs, H, S, S)
    diff_y = np.abs(got_y - y_ref).max()
    diff_state = np.abs(got_state - state_ref).max()
    tol = 5e-8 * max(1.0, np.abs(y_ref).max() + np.abs(state_ref).max())
    # Allow a modest relaxation: fp32 inner products of length S accumulate
    # noise on the order of S * eps * |op|, and we have nested loops.
    abs_tol = 5e-5
    assert diff_y < abs_tol, f"RWKV_WKV6: y mismatch {diff_y}"
    assert diff_state < abs_tol, f"RWKV_WKV6: state mismatch {diff_state}"
    print(f"RWKV_WKV6: ok  (y max diff {diff_y:.2e}, state max diff {diff_state:.2e})")


# --------------------------------------------------------------------- GLA
def gla_ref(k, v, q, g, state, scale):
    """Scalar mirror of ggml_compute_forward_gla_f32 (non-SIMD path).

    Logical shapes:
      k/v/q/g:  (T, H, S)
      state:    (n_seqs, H, S, S)
    Returns y (T, H, S), new_state (n_seqs, H, S, S).
    """
    T, H, S = k.shape
    n_seqs = state.shape[0]
    y = np.zeros((T, H, S), dtype=np.float32)
    new_state = state.copy().astype(np.float32)
    chunk = max(1, T // n_seqs)
    for t in range(T):
        seq = min(n_seqs - 1, t // chunk)
        for h in range(H):
            for i in range(S):
                k_val = k[t, h, i]
                q_val = q[t, h, i] * scale
                g_val = g[t, h, i]
                for j in range(S):
                    v_val = v[t, h, j]
                    kv = v_val * k_val
                    prev = new_state[seq, h, i, j]
                    temp = prev * g_val + kv
                    y[t, h, j] += temp * q_val
                    new_state[seq, h, i, j] = temp
    return y, new_state


def test_gla():
    rng = np.random.default_rng(456)
    S, H, T, n_seqs = 8, 3, 4, 1
    scale = 1.0 / math.sqrt(S)
    k = rng.standard_normal((T, H, S)).astype(np.float32) * 0.3
    v = rng.standard_normal((T, H, S)).astype(np.float32) * 0.3
    q = rng.standard_normal((T, H, S)).astype(np.float32) * 0.3
    g = (np.tanh(rng.standard_normal((T, H, S)).astype(np.float32)) + 1.0) * 0.4
    state = (rng.standard_normal((n_seqs, H, S, S)) * 0.1).astype(np.float32)
    y_ref, state_ref = gla_ref(k, v, q, g, state, scale)

    # Sources: (k, v, q, g, state).
    k_ne = v_ne = q_ne = g_ne = [S, H, T, 1]
    state_ne = [S * S * H * n_seqs, 1, 1, 1]
    result_ne = [S * H, T + S * n_seqs, 1, 1]
    op_params = struct.pack("<f", scale)
    tensors = [
        _make_tensor("k", None, 0, k_ne, index=0, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("v", None, 0, v_ne, index=1, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("q", None, 0, q_ne, index=2, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("g", None, 0, g_ne, index=3, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("state", None, 0, state_ne, index=4, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
    ]
    result_t = _make_tensor("gla", gdump.GgmlOp.GATED_LINEAR_ATTN, 0, result_ne,
                            sources=[0, 1, 2, 3, 4], index=5, op_params=op_params)
    tensors.append(result_t)
    d = gdump.GraphDump(arch="test", n_tokens=1, n_seqs=1, tensors=tensors)
    tr = Translator(d, weight_dtype="float32")
    names = {}
    for i, (lab, ne_) in enumerate([("k", [T, H, S]), ("v", [T, H, S]),
                                     ("q", [T, H, S]), ("g", [T, H, S])]):
        nm = tr._fresh(lab)
        tr.inputs.append(helper.make_tensor_value_info(nm, TensorProto.FLOAT, ne_))
        tr.value[i] = nm
        names[lab] = nm
    s_name = tr._fresh("s")
    tr.inputs.append(helper.make_tensor_value_info(s_name, TensorProto.FLOAT, [S * S * H * n_seqs]))
    tr.value[4] = s_name
    from gguf.gdump_to_onnx import _OP_HANDLERS
    _OP_HANDLERS[gdump.GgmlOp.GATED_LINEAR_ATTN](tr, result_t)
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
    feeds = {names["k"]: k, names["v"]: v, names["q"]: q, names["g"]: g,
             s_name: state.reshape(-1)}
    (got,) = sess.run([out_name], feeds)
    got_y = got[:T].reshape(T, H, S)
    got_state = got[T:].reshape(n_seqs, H, S, S)
    diff_y = np.abs(got_y - y_ref).max()
    diff_state = np.abs(got_state - state_ref).max()
    abs_tol = 5e-5
    assert diff_y < abs_tol, f"GLA: y mismatch {diff_y}"
    assert diff_state < abs_tol, f"GLA: state mismatch {diff_state}"
    print(f"GATED_LINEAR_ATTN: ok  (y max diff {diff_y:.2e}, state max diff {diff_state:.2e})")


# --------------------------------------------------------------------- GDN
def gdn_ref(q, k, v, g, beta, state, kda):
    """Scalar mirror of ggml_compute_forward_gated_delta_net_one_chunk.

    Inputs:
      q/k/v:  (n_seqs, T, H, S) -- we mostly use (T, H, S) with implicit n_seqs=1
      g:      (n_seqs, T, H, S) if kda else (n_seqs, T, H, 1)
      beta:   (n_seqs, T, H, 1)
      state:  in-memory layout (n_seqs, H, S, S) where M[..., j, i] = S[i][j]
    Returns y (n_seqs, T, H, S), new_state in memory layout (n_seqs, H, S, S).
    """
    n_seqs, T, H, S = q.shape
    scale = 1.0 / math.sqrt(S)
    M = state.copy().astype(np.float32)
    y = np.zeros((n_seqs, T, H, S), dtype=np.float32)
    delta = np.zeros((S,), dtype=np.float32)
    for ns in range(n_seqs):
        for h in range(H):
            # M[h] is (S, S) memory; we work in-place on a copy.
            for t in range(T):
                q_d = q[ns, t, h]
                k_d = k[ns, t, h]
                v_d = v[ns, t, h]
                g_d = g[ns, t, h]
                beta_val = beta[ns, t, h, 0]

                # 1. Gate decay.
                if kda:
                    # M[j, :] *= exp(g)  (per-i)
                    exp_g = np.exp(g_d).astype(np.float32)  # (S,)
                    for j in range(S):
                        M[ns, h, j, :] *= exp_g
                else:
                    M[ns, h] *= math.exp(g_d[0])

                # 2. delta[j] = (v[j] - dot(M[j, :], k)) * beta
                for j in range(S):
                    s = 0.0
                    for i in range(S):
                        s += M[ns, h, j, i] * k_d[i]
                    delta[j] = (v_d[j] - s) * beta_val

                # 3. M[j, :] += delta[j] * k
                for j in range(S):
                    for i in range(S):
                        M[ns, h, j, i] += k_d[i] * delta[j]

                # 4. y[j] = scale * dot(M[j, :], q)
                for j in range(S):
                    s = 0.0
                    for i in range(S):
                        s += M[ns, h, j, i] * q_d[i]
                    y[ns, t, h, j] = s * scale
    return y, M


def test_gdn_kda():
    _gdn_case(kda=True)


def test_gdn_scalar():
    _gdn_case(kda=False)


def _gdn_case(*, kda: bool):
    rng = np.random.default_rng(789 + (1 if kda else 0))
    S, H, T, n_seqs = 8, 3, 4, 1
    q = rng.standard_normal((n_seqs, T, H, S)).astype(np.float32) * 0.3
    k = rng.standard_normal((n_seqs, T, H, S)).astype(np.float32) * 0.3
    v = rng.standard_normal((n_seqs, T, H, S)).astype(np.float32) * 0.3
    g_dim = S if kda else 1
    g = rng.standard_normal((n_seqs, T, H, g_dim)).astype(np.float32) * 0.3
    beta = rng.standard_normal((n_seqs, T, H, 1)).astype(np.float32) * 0.5
    state_mem = rng.standard_normal((n_seqs, H, S, S)).astype(np.float32) * 0.1

    y_ref, state_ref_mem = gdn_ref(q, k, v, g, beta, state_mem, kda)

    # Sources: (q, k, v, g, beta, state).
    qkv_ne = [S, H, T, n_seqs]
    g_ne   = [g_dim, H, T, n_seqs]
    beta_ne = [1, H, T, n_seqs]
    state_ne_buf = [S * S * H * n_seqs, 1, 1, 1]
    result_ne = [S * H, T * n_seqs + S * n_seqs, 1, 1]
    tensors = [
        _make_tensor("q", None, 0, qkv_ne, index=0, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("k", None, 0, qkv_ne, index=1, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("v", None, 0, qkv_ne, index=2, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("g", None, 0, g_ne,   index=3, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("beta", None, 0, beta_ne, index=4, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
        _make_tensor("state", None, 0, state_ne_buf, index=5, flags=gdump.FLAG_IS_LEAF | gdump.FLAG_IS_INPUT),
    ]
    result_t = _make_tensor("gdn", gdump.GgmlOp.GATED_DELTA_NET, 0, result_ne,
                            sources=[0, 1, 2, 3, 4, 5], index=6)
    tensors.append(result_t)
    d = gdump.GraphDump(arch="test", n_tokens=1, n_seqs=1, tensors=tensors)
    tr = Translator(d, weight_dtype="float32")
    # Inputs in their logical (numpy/onnx) shapes:
    #   q/k/v: (n_seqs, T, H, S)
    #   g:     (n_seqs, T, H, g_dim)
    #   beta:  (n_seqs, T, H, 1)
    #   state: flat
    names = {}
    qkv_logical = [n_seqs, T, H, S]
    g_logical   = [n_seqs, T, H, g_dim]
    b_logical   = [n_seqs, T, H, 1]
    for lab, ne_ in [("q", qkv_logical), ("k", qkv_logical), ("v", qkv_logical),
                     ("g", g_logical),  ("b", b_logical)]:
        nm = tr._fresh(lab)
        tr.inputs.append(helper.make_tensor_value_info(nm, TensorProto.FLOAT, ne_))
        names[lab] = nm
    tr.value[0] = names["q"]
    tr.value[1] = names["k"]
    tr.value[2] = names["v"]
    tr.value[3] = names["g"]
    tr.value[4] = names["b"]
    s_name = tr._fresh("s")
    tr.inputs.append(helper.make_tensor_value_info(s_name, TensorProto.FLOAT, [S * S * H * n_seqs]))
    tr.value[5] = s_name

    from gguf.gdump_to_onnx import _OP_HANDLERS
    _OP_HANDLERS[gdump.GgmlOp.GATED_DELTA_NET](tr, result_t)
    out_name = tr.value[result_t.index]
    tr.outputs.append(helper.make_tensor_value_info(
        out_name, TensorProto.FLOAT, [T * n_seqs + S * n_seqs, S * H]))
    graph = helper.make_graph(tr.nodes, "test", tr.inputs, tr.outputs, tr.initializers)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    import onnxruntime as ort
    sess = ort.InferenceSession(m.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    feeds = {names["q"]: q, names["k"]: k, names["v"]: v,
             names["g"]: g, names["b"]: beta,
             s_name: state_mem.reshape(-1)}
    (got,) = sess.run([out_name], feeds)
    # Output layout: first T*n_seqs rows are y in (n_seqs, T, H, S) row-major,
    # then S*n_seqs rows are state in memory layout (n_seqs, H, S, S).
    n_y = T * n_seqs
    got_y = got[:n_y].reshape(n_seqs, T, H, S)
    got_state_mem = got[n_y:].reshape(n_seqs, H, S, S)
    diff_y = np.abs(got_y - y_ref).max()
    diff_state = np.abs(got_state_mem - state_ref_mem).max()
    abs_tol = 5e-5
    assert diff_y < abs_tol, f"GDN(kda={kda}): y mismatch {diff_y}"
    assert diff_state < abs_tol, f"GDN(kda={kda}): state mismatch {diff_state}"
    print(f"GATED_DELTA_NET(kda={kda}): ok  (y max diff {diff_y:.2e}, state max diff {diff_state:.2e})")


if __name__ == "__main__":
    test_rwkv_wkv6()
    test_gla()
    test_gdn_kda()
    test_gdn_scalar()
    print("\nAll recurrent-op math tests passed.")
