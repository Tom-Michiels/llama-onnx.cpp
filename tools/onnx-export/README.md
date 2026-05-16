# GGUF -> ONNX export

This directory contains the C++ half of the GGUF -> ONNX export pipeline.

The C++ side (`llama-onnx-export-dump`) loads a GGUF model exactly the way
`llama-cli` would, calls `llama_graph_reserve()` to ask llama.cpp's own
`src/models/*.cpp` graph builder to produce the prefill `ggml_cgraph`, and
then serialises that graph plus the weight tensor data to a portable binary
dump file (`*.gdump`).

The Python side (`convert_gguf_to_onnx.py`, in the repo root) reads the
`.gdump` file and translates each ggml op to ONNX ops, producing an ONNX
model that can be run with `onnxruntime`.

Splitting the work this way means:

- We never duplicate the model algorithm in Python. The forward pass is
  defined in exactly one place — the C++ `src/models/<arch>.cpp` files that
  llama.cpp already maintains. Adding a new architecture costs us nothing on
  the export side: as soon as llama.cpp can run a model, the dumper can dump
  it.
- The translation table (ggml op -> ONNX op) lives in Python where it's easy
  to iterate, debug, and unit-test against onnxruntime. The C++ binary has
  no ONNX dependency.

## Dump format

```
[4 bytes]  magic       "GDMP"
[4 bytes]  version     1
[8 bytes]  n_tensors   (leafs + nodes, in topological order)

For each tensor i = 0 .. n_tensors-1:
    [4 bytes]   name_len
    [bytes]     name (UTF-8, not NUL-terminated)
    [4 bytes]   ggml_op            (GGML_OP_NONE for leafs)
    [4 bytes]   ggml_type
    [4 bytes]   flags              (bit 0: HAS_DATA, bit 1: IS_INPUT, bit 2: IS_OUTPUT)
    [4 * 8]     ne[4]              (logical shape, int64)
    [4 * 8]     nb[4]              (strides in bytes, int64)
    [4 bytes]   op_params_size     (always GGML_MAX_OP_PARAMS == 64 today)
    [bytes]     op_params raw bytes
    [4 bytes]   n_sources
    [n_src * 4] source tensor indices (uint32, indices into this list)
    [8 bytes]   data_size           (only if HAS_DATA is set)
    [bytes]     data                (only if HAS_DATA is set, ne_total * sizeof(type))
```

Tensors are written in topological order — every tensor's `sources` only
reference earlier indices. Leaf tensors with backing memory (model weights)
are written with `HAS_DATA` set; leafs without data are inputs to the graph
(token ids, positions, KV-cache slots, etc.) and are marked `IS_INPUT`. The
last node in the graph is also marked `IS_OUTPUT`.

## Build & run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-onnx-export-dump -j
./build/bin/llama-onnx-export-dump -m model.gguf -o model.gdump
python convert_gguf_to_onnx.py model.gdump --outfile model.onnx
```

## Status

This is the first cut. It is verified for the `llama` architecture (Llama 1
/ 2 / 3 family). `qwen` (text-only) is expected to work once the Python
translator covers the few extra ops it uses (Q/K bias, slightly different
RoPE config); see `docs/development/gguf-to-onnx.md` for the running op
coverage table. `qwen-vl` adds a multimodal vision encoder that is built
through llama.cpp's separate `mtmd` pipeline rather than `llama_graph_reserve`,
so it is documented as a TODO.
