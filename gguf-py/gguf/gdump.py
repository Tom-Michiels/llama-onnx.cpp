"""Reader for the binary graph dump format produced by ``llama-onnx-export-dump``.

The C++ tool walks the ggml compute graph that llama.cpp's own
``src/models/<arch>.cpp`` builders produce, and serialises every tensor (op,
type, ne, nb, op_params, sources, name, data) in topological order. This
module is the Python counterpart that turns that byte stream into a list of
:class:`Tensor` objects ready to be translated to ONNX.

See ``tools/onnx-export/README.md`` for the on-disk layout.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import BinaryIO

import numpy as np


_MAGIC = 0x504D4447  # "GDMP"
_VERSION = 1

FLAG_HAS_DATA = 1 << 0
FLAG_IS_INPUT = 1 << 1
FLAG_IS_OUTPUT = 1 << 2
FLAG_IS_LEAF = 1 << 3


# Mirror of ggml.h's enum ggml_type. We only enumerate the float types we
# actually support; everything else stays as an integer.
class GgmlType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    BF16 = 30
    F64 = 28
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27


# Mirror of ggml.h's enum ggml_op. Values match the on-disk enum exactly so
# we can interpret raw bytes from the dump file; the C++ side never remaps.
class GgmlOp(IntEnum):
    NONE = 0
    DUP = 1
    ADD = 2
    ADD_ID = 3
    ADD1 = 4
    ACC = 5
    SUB = 6
    MUL = 7
    DIV = 8
    SQR = 9
    SQRT = 10
    LOG = 11
    SIN = 12
    COS = 13
    SUM = 14
    SUM_ROWS = 15
    CUMSUM = 16
    MEAN = 17
    ARGMAX = 18
    COUNT_EQUAL = 19
    REPEAT = 20
    REPEAT_BACK = 21
    CONCAT = 22
    SILU_BACK = 23
    NORM = 24
    RMS_NORM = 25
    RMS_NORM_BACK = 26
    GROUP_NORM = 27
    L2_NORM = 28
    MUL_MAT = 29
    MUL_MAT_ID = 30
    OUT_PROD = 31
    SCALE = 32
    SET = 33
    CPY = 34
    CONT = 35
    RESHAPE = 36
    VIEW = 37
    PERMUTE = 38
    TRANSPOSE = 39
    GET_ROWS = 40
    GET_ROWS_BACK = 41
    SET_ROWS = 42
    DIAG = 43
    DIAG_MASK_INF = 44
    DIAG_MASK_ZERO = 45
    SOFT_MAX = 46
    SOFT_MAX_BACK = 47
    ROPE = 48
    ROPE_BACK = 49
    CLAMP = 50
    CONV_TRANSPOSE_1D = 51
    IM2COL = 52
    IM2COL_BACK = 53
    IM2COL_3D = 54
    CONV_2D = 55
    CONV_3D = 56
    CONV_2D_DW = 57
    CONV_TRANSPOSE_2D = 58
    POOL_1D = 59
    POOL_2D = 60
    POOL_2D_BACK = 61
    UPSCALE = 62
    PAD = 63
    PAD_REFLECT_1D = 64
    ROLL = 65
    ARANGE = 66
    TIMESTEP_EMBEDDING = 67
    ARGSORT = 68
    TOP_K = 69
    LEAKY_RELU = 70
    TRI = 71
    FILL = 72
    FLASH_ATTN_EXT = 73
    FLASH_ATTN_BACK = 74
    SSM_CONV = 75
    SSM_SCAN = 76
    WIN_PART = 77
    WIN_UNPART = 78
    GET_REL_POS = 79
    ADD_REL_POS = 80
    RWKV_WKV6 = 81
    GATED_LINEAR_ATTN = 82
    RWKV_WKV7 = 83
    SOLVE_TRI = 84
    GATED_DELTA_NET = 85
    UNARY = 86
    MAP_CUSTOM1 = 87
    MAP_CUSTOM2 = 88
    MAP_CUSTOM3 = 89
    CUSTOM = 90
    CROSS_ENTROPY_LOSS = 91
    CROSS_ENTROPY_LOSS_BACK = 92
    OPT_STEP_ADAMW = 93
    OPT_STEP_SGD = 94
    GLU = 95


# ggml unary op variants (op_params[0] when op == UNARY).
class GgmlUnaryOp(IntEnum):
    ABS = 0
    SGN = 1
    NEG = 2
    STEP = 3
    TANH = 4
    ELU = 5
    RELU = 6
    SIGMOID = 7
    GELU = 8
    GELU_QUICK = 9
    SILU = 10
    HARDSWISH = 11
    HARDSIGMOID = 12
    EXP = 13
    GELU_ERF = 14
    XIELU = 15
    FLOOR = 16
    CEIL = 17
    ROUND = 18
    TRUNC = 19


# ggml GLU variants (op_params[0] when op == GLU).
class GgmlGluOp(IntEnum):
    REGLU = 0
    GEGLU = 1
    SWIGLU = 2
    SWIGLU_OAI = 3
    GEGLU_ERF = 4
    GEGLU_QUICK = 5


# numpy dtype for a ggml_type, if any. Quantised types have no direct numpy
# representation; we keep their raw bytes around but the translator will
# refuse to emit them.
_GGML_TO_NUMPY = {
    GgmlType.F32: np.dtype("float32"),
    GgmlType.F16: np.dtype("float16"),
    GgmlType.F64: np.dtype("float64"),
    GgmlType.I8:  np.dtype("int8"),
    GgmlType.I16: np.dtype("int16"),
    GgmlType.I32: np.dtype("int32"),
    GgmlType.I64: np.dtype("int64"),
    # BF16 is decoded specially (widened to F32).
}


def ggml_dtype_to_numpy(t: int) -> np.dtype | None:
    return _GGML_TO_NUMPY.get(GgmlType(t)) if t in {x.value for x in _GGML_TO_NUMPY} else None


@dataclass
class Tensor:
    name: str
    op: int             # raw enum value
    dtype: int          # raw ggml_type value
    flags: int
    ne: list[int]       # length 4
    nb: list[int]       # length 4 (strides in bytes)
    op_params: bytes    # raw GGML_MAX_OP_PARAMS bytes
    sources: list[int]  # indices into the dump's tensor list
    data: np.ndarray | None = None  # only for leafs with FLAG_HAS_DATA
    index: int = -1     # position in the dump

    @property
    def is_leaf(self) -> bool:
        return bool(self.flags & FLAG_IS_LEAF)

    @property
    def is_input(self) -> bool:
        return bool(self.flags & FLAG_IS_INPUT)

    @property
    def is_output(self) -> bool:
        return bool(self.flags & FLAG_IS_OUTPUT)

    @property
    def has_data(self) -> bool:
        return bool(self.flags & FLAG_HAS_DATA)

    @property
    def ggml_op(self) -> GgmlOp:
        try:
            return GgmlOp(self.op)
        except ValueError:
            return GgmlOp.NONE  # caller should check `op` directly

    @property
    def ndim(self) -> int:
        # ggml stores trailing dims as 1; the effective rank is the number of
        # leading non-trivial dims, but at least 1.
        for r in range(4, 0, -1):
            if self.ne[r - 1] > 1:
                return r
        return 1

    @property
    def shape(self) -> list[int]:
        return list(self.ne[: self.ndim])

    def op_params_i32(self, n: int) -> list[int]:
        """Return the first ``n`` int32 values of op_params."""
        return list(struct.unpack_from(f"<{n}i", self.op_params, 0))

    def op_params_f32(self, off: int, n: int = 1) -> list[float]:
        """Return ``n`` float32 values starting at byte offset ``off``."""
        return list(struct.unpack_from(f"<{n}f", self.op_params, off))


@dataclass
class GraphDump:
    arch: str
    n_tokens: int
    n_seqs: int
    tensors: list[Tensor] = field(default_factory=list)

    def by_name(self, name: str) -> Tensor | None:
        # Names aren't unique in ggml (e.g. several "node_NN" exist); we return
        # the first match. For unique names like "result_output" this is fine.
        for t in self.tensors:
            if t.name == name:
                return t
        return None

    def all_named(self, name: str) -> list[Tensor]:
        return [t for t in self.tensors if t.name == name]


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------


def _read_u32(f: BinaryIO) -> int:
    return struct.unpack("<I", f.read(4))[0]


def _read_u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def _read_i64(f: BinaryIO) -> int:
    return struct.unpack("<q", f.read(8))[0]


def _read_bytes(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"short read: wanted {n} bytes, got {len(b)}")
    return b


def _decode_data(dtype: int, ne: list[int], nbytes: int, raw: bytes) -> np.ndarray:
    """Convert raw bytes from the dump into a numpy array with the logical
    shape (ne reversed — ggml's ne[0] is the fastest-varying dim).
    """
    np_shape = list(reversed([d for d in ne if d > 1]))
    if not np_shape:
        np_shape = [1]
    gt = GgmlType(dtype)
    if gt == GgmlType.BF16:
        # bf16 widened to float32
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        arr = u32.view(np.float32)
    else:
        npt = ggml_dtype_to_numpy(dtype)
        if npt is None:
            raise ValueError(f"unsupported tensor dtype {gt} for data decode")
        arr = np.frombuffer(raw, dtype=npt).copy()
    return arr.reshape(np_shape)


def load(path: str | Path) -> GraphDump:
    path = Path(path)
    with open(path, "rb") as f:
        magic = _read_u32(f)
        if magic != _MAGIC:
            raise ValueError(f"not a gdump file: bad magic 0x{magic:08x}")
        version = _read_u32(f)
        if version != _VERSION:
            raise ValueError(f"unsupported gdump version {version}")

        arch_len = _read_u32(f)
        arch = _read_bytes(f, arch_len).decode("utf-8")
        n_tokens = _read_u32(f)
        n_seqs = _read_u32(f)

        n_tensors = _read_u64(f)
        tensors: list[Tensor] = []
        for i in range(n_tensors):
            name_len = _read_u32(f)
            name = _read_bytes(f, name_len).decode("utf-8", errors="replace")
            op = _read_u32(f)
            dtype = _read_u32(f)
            flags = _read_u32(f)
            ne = [_read_i64(f) for _ in range(4)]
            nb = [_read_i64(f) for _ in range(4)]
            op_params_size = _read_u32(f)
            op_params = _read_bytes(f, op_params_size)
            n_src = _read_u32(f)
            sources = [_read_u32(f) for _ in range(n_src)]

            data = None
            if flags & FLAG_HAS_DATA:
                nbytes = _read_u64(f)
                raw = _read_bytes(f, nbytes)
                data = _decode_data(dtype, ne, nbytes, raw)
            elif flags & FLAG_IS_LEAF:
                # leaf with no data (input) — no extra payload follows.
                pass

            tensors.append(Tensor(
                name=name,
                op=op,
                dtype=dtype,
                flags=flags,
                ne=ne,
                nb=nb,
                op_params=op_params,
                sources=sources,
                data=data,
                index=i,
            ))

        return GraphDump(arch=arch, n_tokens=n_tokens, n_seqs=n_seqs, tensors=tensors)
