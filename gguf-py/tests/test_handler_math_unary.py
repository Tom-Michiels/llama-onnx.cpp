#!/usr/bin/env python3
"""Math tests for the 10 newly-added UNARY sub-op handlers in
``gguf-py/gguf/gdump_to_onnx.py`` (ABS, SGN, STEP, ELU, GELU_QUICK, XIELU,
FLOOR, CEIL, ROUND, TRUNC).

For each op we build a tiny ONNX graph mirroring exactly what ``_h_unary``
would emit, run it through onnxruntime, and compare against a numpy
translation of the ggml CPU reference math in
``ggml/src/ggml-cpu/unary-ops.cpp``.
"""
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper


def _run(nodes, initializers, x: np.ndarray) -> np.ndarray:
    """Wrap nodes into a single-input ONNX model, run on x, return output."""
    dtype_onnx = TensorProto.FLOAT
    g = helper.make_graph(
        nodes=nodes,
        name="t",
        inputs=[helper.make_tensor_value_info("x", dtype_onnx, list(x.shape))],
        outputs=[helper.make_tensor_value_info("y", dtype_onnx, list(x.shape))],
        initializer=initializers,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(["y"], {"x": x})[0]


# --- ggml CPU reference math (translated from unary-ops.cpp) ---------------
def ref_abs(x):    return np.abs(x).astype(np.float32)
def ref_sgn(x):    return np.where(x > 0, 1.0, np.where(x < 0, -1.0, 0.0)).astype(np.float32)
def ref_step(x):   return np.where(x > 0, 1.0, 0.0).astype(np.float32)
def ref_elu(x):    return np.where(x > 0, x, np.expm1(x)).astype(np.float32)
def ref_gelu_quick(x):
    return (x / (1.0 + np.exp(-1.702 * x))).astype(np.float32)
def ref_xielu(x, alpha_n, alpha_p, beta, eps):
    pos = alpha_p * x * x + beta * x
    mn = np.minimum(x, np.float32(eps))
    neg = (np.expm1(mn) - x) * alpha_n + beta * x
    return np.where(x > 0, pos, neg).astype(np.float32)
def ref_floor(x):  return np.floor(x).astype(np.float32)
def ref_ceil(x):   return np.ceil(x).astype(np.float32)
def ref_round(x):  return np.round(x).astype(np.float32)
def ref_trunc(x):  return np.trunc(x).astype(np.float32)


# --- ONNX graphs mirroring _h_unary emission -------------------------------
def _direct(op, **attrs):
    return [helper.make_node(op, ["x"], ["y"], **attrs)], []


def _graph_step():
    zero = numpy_helper.from_array(np.array(0, dtype=np.float32), "z")
    return [
        helper.make_node("Greater", ["x", "z"], ["gt"]),
        helper.make_node("Cast", ["gt"], ["y"], to=TensorProto.FLOAT),
    ], [zero]


def _graph_gelu_quick():
    c = numpy_helper.from_array(np.array(1.702, dtype=np.float32), "c")
    return [
        helper.make_node("Mul", ["x", "c"], ["sx"]),
        helper.make_node("Sigmoid", ["sx"], ["sg"]),
        helper.make_node("Mul", ["x", "sg"], ["y"]),
    ], [c]


def _graph_trunc():
    return [
        helper.make_node("Abs", ["x"], ["a"]),
        helper.make_node("Floor", ["a"], ["fl"]),
        helper.make_node("Sign", ["x"], ["sg"]),
        helper.make_node("Mul", ["sg", "fl"], ["y"]),
    ], []


def _graph_xielu(alpha_n, alpha_p, beta, eps):
    inits = [
        numpy_helper.from_array(np.array(0,       dtype=np.float32), "z"),
        numpy_helper.from_array(np.array(alpha_n, dtype=np.float32), "an"),
        numpy_helper.from_array(np.array(alpha_p, dtype=np.float32), "ap"),
        numpy_helper.from_array(np.array(beta,    dtype=np.float32), "b"),
        numpy_helper.from_array(np.array(eps,     dtype=np.float32), "e"),
        numpy_helper.from_array(np.array(1,       dtype=np.float32), "one"),
    ]
    nodes = [
        helper.make_node("Mul", ["x", "b"], ["bx"]),
        helper.make_node("Mul", ["x", "x"], ["xx"]),
        helper.make_node("Mul", ["xx", "ap"], ["posA"]),
        helper.make_node("Add", ["posA", "bx"], ["pos"]),
        helper.make_node("Min", ["x", "e"], ["mn"]),
        helper.make_node("Exp", ["mn"], ["expmn"]),
        helper.make_node("Sub", ["expmn", "one"], ["em1"]),
        helper.make_node("Sub", ["em1", "x"], ["diff"]),
        helper.make_node("Mul", ["diff", "an"], ["negA"]),
        helper.make_node("Add", ["negA", "bx"], ["neg"]),
        helper.make_node("Greater", ["x", "z"], ["cond"]),
        helper.make_node("Where", ["cond", "pos", "neg"], ["y"]),
    ]
    return nodes, inits


# --- the actual tests -------------------------------------------------------
def _x_sample():
    # Mix: negatives, zero, positives, small fractionals, integers, large.
    return np.array(
        [-3.7, -1.5, -0.5, -0.001, 0.0, 0.001, 0.5, 1.5, 2.0, 3.7, 10.25, -10.25],
        dtype=np.float32)


def test_direct():
    x = _x_sample()
    cases = [
        ("Abs",   {},               ref_abs,   1e-6),
        ("Sign",  {},               ref_sgn,   1e-6),
        ("Elu",   {"alpha": 1.0},   ref_elu,   1e-6),
        ("Floor", {},               ref_floor, 1e-6),
        ("Ceil",  {},               ref_ceil,  1e-6),
        ("Round", {},               ref_round, 1e-6),
    ]
    for op, attrs, ref, tol in cases:
        nodes, inits = _direct(op, **attrs)
        got = _run(nodes, inits, x)
        diff = float(np.abs(got - ref(x)).max())
        assert diff <= tol, f"{op}: max|diff|={diff} > {tol}"
        print(f"{op:9s}: ok (max|diff|={diff:.2e})")


def test_step():
    x = _x_sample()
    nodes, inits = _graph_step()
    got = _run(nodes, inits, x)
    diff = float(np.abs(got - ref_step(x)).max())
    assert diff <= 1e-6, f"STEP: max|diff|={diff}"
    # also confirm ggml's >0 semantics (zero -> 0)
    assert got[x.tolist().index(0.0)] == 0.0
    print(f"STEP     : ok (max|diff|={diff:.2e})")


def test_gelu_quick():
    x = _x_sample()
    nodes, inits = _graph_gelu_quick()
    got = _run(nodes, inits, x)
    diff = float(np.abs(got - ref_gelu_quick(x)).max())
    assert diff <= 1e-5, f"GELU_QUICK: max|diff|={diff}"
    print(f"GELU_QUICK: ok (max|diff|={diff:.2e})")


def test_trunc():
    # Include tricky edge cases for round-toward-zero
    x = np.array([-3.7, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 3.7, 1e-7, -1e-7],
                 dtype=np.float32)
    nodes, inits = _graph_trunc()
    got = _run(nodes, inits, x)
    diff = float(np.abs(got - ref_trunc(x)).max())
    assert diff <= 1e-5, f"TRUNC: max|diff|={diff}"
    # Sign(0)*Floor(Abs(0)) = 0*0 = 0  — matches truncf(0) = 0
    z_idx = int(np.where(x == 0.0)[0][0])
    assert got[z_idx] == 0.0
    print(f"TRUNC    : ok (max|diff|={diff:.2e})")


def test_xielu():
    # A few (alpha_n, alpha_p, beta, eps) tuples typical for Apertus.
    cases = [
        (0.8, 0.5, 1.0, -1e-3),
        (1.5, 0.25, 0.0, -0.1),
        (0.1, 0.1, 0.5, 0.0),
    ]
    x = _x_sample()
    for an, ap, b, eps in cases:
        nodes, inits = _graph_xielu(an, ap, b, eps)
        got = _run(nodes, inits, x)
        ref = ref_xielu(x, an, ap, b, eps)
        diff = float(np.abs(got - ref).max())
        assert diff <= 1e-5, f"XIELU(an={an},ap={ap},b={b},eps={eps}): max|diff|={diff}"
        print(f"XIELU    : ok (an={an}, ap={ap}, b={b}, eps={eps}, max|diff|={diff:.2e})")


if __name__ == "__main__":
    test_direct()
    test_step()
    test_gelu_quick()
    test_trunc()
    test_xielu()
    print("\nAll UNARY sub-op math tests pass against the ggml CPU reference.")
