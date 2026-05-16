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

# Convert.
python convert_gguf_to_onnx.py path/to/model.gguf \
    --outfile path/to/model.onnx
```

The driver script finds `llama-onnx-export-dump` under `build/bin/` by
default; pass `--dumper` to override.

If you want to inspect the intermediate dump, use `--keep-dump` (or run the
dumper directly).

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

| arch              | C++ graph (`src/models/`) | translator |
| ----------------- | ------------------------- | ---------- |
| `llama`           | shared with the runtime   | yes, end-to-end on the synthetic test |
| `qwen` (text)     | shared with the runtime   | most ops covered; the few extra (Q/K bias, sliding-window mask, qkv-fused matmul) need verification on real qwen GGUFs |
| `qwen2vl`         | shared with the runtime   | the language model goes through the same path; the vision encoder is built by the separate `tools/mtmd` pipeline and is **not** yet dumped — TODO |
| anything else     | shared with the runtime   | depends on which ops the model uses |

Adding a new architecture costs us nothing on the dump side — the C++
graph builder is the source of truth. The Python side just needs to know
how to translate any new ggml op that arch happens to use; the running
table is in `gguf-py/gguf/gdump_to_onnx.py` under `_OP_HANDLERS`.

## ggml op coverage in the translator

Implemented today:

```
ADD, SUB, MUL, DIV, SCALE, MUL_MAT, GET_ROWS, RMS_NORM, RESHAPE, VIEW,
CONT, PERMUTE, TRANSPOSE, CPY, GLU (SWIGLU only), ROPE (NORMAL + NEOX),
SET_ROWS, FLASH_ATTN_EXT (no softcap, no ALiBi)
```

Not yet implemented (will raise `NotImplementedError`):

```
SOFT_MAX standalone, NORM (LayerNorm), GROUP_NORM, ROPE with YaRN scaling
or mrope, UNARY ops (GELU, ReLU, ...), DIAG_MASK_INF, MUL_MAT_ID (MoE
routing), CONCAT, CLAMP, SUM/MEAN/ARGMAX along non-trivial axes,
ADD_ID, FILL, TOP_K, SSM_*, FLASH_ATTN_BACK
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
