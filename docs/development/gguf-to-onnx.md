# Exporting a GGUF model to ONNX

This page documents the GGUF -> ONNX export pipeline introduced in
`tools/onnx-export/` and `gguf-py/gguf/gdump*.py`.

The exporter does **not** duplicate the model forward pass in Python. It
asks llama.cpp itself, via `llama_graph_reserve()`, to build the prefill
`ggml_cgraph` exactly as `src/models/<arch>.cpp` defines it. A small C++
binary then serialises that graph (plus the weight tensor data) to a
portable `.gdump` file, and a Python script translates the dump to ONNX.

## Pipeline

```
                build/bin/llama-onnx-export-dump
                ┌────────────────────────────────┐
   model.gguf ─▶│  load model                     │
                │  llama_graph_reserve()  ◀──── src/models/<arch>.cpp
                │  walk cgraph, dump tensors      │
                └──────────────┬─────────────────┘
                               │  model.gdump  (portable binary, see
                               │                tools/onnx-export/README.md)
                               ▼
                gguf.gdump_to_onnx (Python, in gguf-py/)
                ┌────────────────────────────────┐
                │  read dump                      │
                │  ggml op -> ONNX op translation │
                │  emit ONNX                      │
                └──────────────┬─────────────────┘
                               ▼
                          model.onnx + model.onnx.data
                               │
                               ▼
                          onnxruntime
```

## Building and running

```bash
# One-time build of the C++ dumper.
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-onnx-export-dump -j

# Convert (defaults to float16 weight initializers).
python convert_gguf_to_onnx.py path/to/model.gguf \
    --outfile path/to/model.onnx

# Use float32 weight initializers instead (larger, but matches fp32 hosts).
python convert_gguf_to_onnx.py path/to/model.gguf \
    --outfile path/to/model.onnx --weight-dtype float32
```

The driver script finds `llama-onnx-export-dump` under `build/bin/` by
default; pass `--dumper` to override.

If you want to inspect the intermediate dump, use `--keep-dump` (or run the
dumper directly).

## Quantised inputs

GGUF tensors of any ggml quantisation type (`Q4_0`, `Q4_K_M`, `Q8_0`, `Q2_K`,
... — anything `gguf.quants.dequantize` understands) are supported. The
weights are dequantised to float32 in Python on load and then cast to the
chosen `--weight-dtype` (float16 by default) before being stamped as ONNX
initializers. ONNX itself has no representation for ggml's block-quantised
formats, so we have to widen at the boundary.

Two complementary pieces of metadata record what the source GGUF actually
contained, so downstream tooling (e.g. a quantisation-aware runtime, an
inspector, or an analyser) can still see how each weight was originally
stored:

* The model carries a few `metadata_props` entries:
  - `llama_onnx.architecture`            — the GGUF `general.architecture`
  - `llama_onnx.weight_dtype`            — what the ONNX initializers were
    cast to (`float16` or `float32`)
  - `llama_onnx.original_dtype_counts`   — a histogram of the original
    ggml types across all initializers, e.g.
    `F32:18,Q4_0:14,Q8_0:1`
* Every `TensorProto` initializer carries its original ggml type on its
  `doc_string`, like `original_ggml_type=Q4_0`.

The exporter takes care of one subtlety transparently: llama.cpp's CPU
backend re-packs Q4_0 (and several other quantised types) into a
SIMD-friendly layout when the model is loaded. The C++ dumper therefore
reads the canonical GGUF bytes for quantised tensors directly from the
file (via `gguf_init_from_file`) rather than from the in-memory
`tensor->data` pointer, which the CPU backend may have rewritten.

## Verifying against `llama_decode`

The dumper accepts a `--fixture <path>` flag that, after writing the
`.gdump`, runs `llama_decode()` on a deterministic prompt and saves the
input token IDs together with the reference logits. The companion Python
module `gguf.gdump_fixture` reads that fixture and synthesises the rest
of the graph inputs (positions, KV-cache write indices, the causal mask)
from the gdump's graph structure plus the token count, so the test can
feed the exact same inputs through onnxruntime and compare.

## ONNX I/O contract

The exporter currently emits a **static-shape** ONNX model that matches the
shapes the cgraph was reserved for (`n_tokens` and `n_seqs` chosen by the
C++ binary, currently `n_tokens = min(n_ctx, n_ubatch)` and `n_seqs = 1`).
Inputs:

| name                              | dtype  | shape                                  |
| --------------------------------- | ------ | -------------------------------------- |
| `input_ids`                       | int32  | `[n_tokens]`                           |
| `leaf_N` (positions)              | int32  | `[n_tokens]`                           |
| `leaf_N` (KV-cache write idxs)    | int64  | `[n_tokens]`                           |
| `attn_inp_kq_mask`                | float  | `[n_tokens, kv_total]`                 |
| `past_key_values.{i}.key`         | f16    | `[kv_total, n_embd_kv]`                |
| `past_key_values.{i}.value`       | f16    | `[kv_total, n_embd_kv]`                |
| `leaf_N` (output ids)             | int32  | `[n_tokens]`                           |

Outputs:

| name                | dtype  | shape                                   |
| ------------------- | ------ | --------------------------------------- |
| `logits`            | float  | `[n_tokens, vocab_size]`                |
| `present.{i}.key`   | f16    | `[kv_total, n_embd_kv]`                 |
| `present.{i}.value` | f16    | `[kv_total, n_embd_kv]`                 |

The `leaf_N` inputs come straight from llama.cpp's graph builder — that
code does not call `ggml_set_name()` on positions / out_ids / KV idxs, so
they fall back to ggml's default `leaf_N` placeholder. A follow-up will
add the corresponding `ggml_set_name()` calls upstream (and the translator
will then surface them as `position_ids`, `out_ids`, etc.).

## Architecture coverage

The synthetic test in `gguf-py/tests/test_onnx_export.py` round-trips a tiny
random model through the full pipeline (build GGUF → C++ dumper →
Python translator → onnxruntime forward pass) for each of these. The
forward pass is checked to produce finite (non-NaN/Inf) logits; bit-exact
agreement against `llama-cli` running the same GGUF is the next step.

| arch       | end-to-end | notes |
| ---------- | ---------- | ----- |
| `llama`    | yes        | GQA, NORMAL-style RoPE |
| `gemma`    | yes        | input-embedding scaling, GEGLU FFN (built from primitive ops since opset 17 has no `Gelu`) |
| `qwen3`    | yes        | adds Q/K RMSNorm before RoPE |
| `qwen3moe` | yes        | TopK expert routing + `MUL_MAT_ID` translated as Gather + broadcast `Mul` + `ReduceSum`; per-batch `GET_ROWS` via `GatherElements` |
| `qwen2vl`  | yes (LM)   | uses mrope (multimodal RoPE with 4 position sections). The accompanying vision encoder is built by the separate `tools/mtmd` pipeline and is **not** dumped by `llama_graph_reserve` — that's a TODO if/when full multimodal export is needed. |
| any other  | depends    | adding a new architecture costs nothing on the dump side (C++ graph is the source of truth); the Python translator just needs to know any new ops it pulls in |

Adding a new architecture costs us nothing on the dump side — the C++
graph builder is the source of truth. The Python side just needs to know
how to translate any new ggml op that arch happens to use; the running
table is in `gguf-py/gguf/gdump_to_onnx.py` under `_OP_HANDLERS`.

## ggml op coverage in the translator

Implemented today:

```
ADD, SUB, MUL, DIV, SCALE, SQR, SQRT, SIN, COS, LOG, CLAMP, REPEAT
MUL_MAT, MUL_MAT_ID, SCALE
GET_ROWS (Gather + GatherElements), SET_ROWS (ScatterND)
RMS_NORM, NORM (LayerNorm without affine)
RESHAPE, VIEW (contiguous slice + squeeze patterns), CONT, PERMUTE,
TRANSPOSE, CPY (dtype cast), CONCAT
GLU (SWIGLU / GEGLU / GEGLU_ERF / REGLU)
ROPE (NORMAL, NEOX, MROPE)
SOFT_MAX (with optional additive mask + scale)
UNARY (SILU, RELU, SIGMOID, TANH, NEG, EXP, GELU, GELU_ERF, HARDSWISH, HARDSIGMOID)
ARGSORT, TOP_K (via TopK; indices cast to int32)
SUM, SUM_ROWS, MEAN (ReduceSum / ReduceMean along last axis)
FLASH_ATTN_EXT (no softcap, no ALiBi)
```

Not yet implemented (will raise `NotImplementedError`):

```
GROUP_NORM, ROPE with YaRN scaling, DIAG_MASK_INF, ADD_ID, FILL,
SSM_*, FLASH_ATTN_BACK, ROPE_TYPE_VISION / IMROPE, CONV_*, POOL_*,
strided VIEW patterns beyond contiguous-slice and slot-squeeze.
```

Adding one is a small change: an `@_op(GgmlOp.X)` handler that translates
the op_params (see `ggml.h` enum + `ggml.c` constructors for layout) into
the equivalent ONNX nodes.

## Known limitations

* **Static shapes.** The dumped graph has concrete `ne` for every tensor.
  To run a different `n_tokens` you currently re-export. A follow-up will
  symbolify the time-axis dimensions.
* **FP16 KV cache always.** The C++ side reserves the graph with the cache
  type llama.cpp would use by default (f16); the ONNX model therefore takes
  f16 past_K/past_V inputs. Casting on the fly is up to the caller.
* **`inp_embd` input is pruned.** The graph builder registers both
  `inp_tokens` and `inp_embd` as candidates and selects one at runtime; the
  translator drops the unused one to keep the ONNX I/O clean.
* **No numerical comparison test.** The synthetic test in
  `gguf-py/tests/test_onnx_export.py` only checks that a forward pass runs
  without NaNs. Bitwise (or low-tolerance) verification against
  `llama-cli`'s logits for the same input is the next thing to add.

## Test

```bash
python gguf-py/tests/test_onnx_export.py
```

It auto-skips if `llama-onnx-export-dump` isn't built or `onnxruntime`
isn't installed.
