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

### Shape flags (prefill vs decode)

`n_tokens` (new tokens per forward pass), `n_ctx` (KV-cache total length),
and `n_seqs` (parallel sequences) are baked into the exported ONNX file.
The dumper inherits llama.cpp's standard CLI arg parser, so these can be
overridden directly:

| flag                | what it sets             | default |
| ------------------- | ------------------------ | ------- |
| `-b, --batch-size`  | `params.n_batch`         | 32      |
| `-ub, --ubatch-size`| `params.n_ubatch`        | 32      |
| `-c, --ctx-size`    | `params.n_ctx`           | 512     |

The dumper uses `n_tokens = min(n_ctx, n_ubatch)` and `n_seqs = 1` (the
latter is currently hard-coded — all six recurrent op handlers raise
`NotImplementedError` for `n_seqs > 1` anyway).

For autoregressive serving you typically need two exports of the same
weights — one **prefill** shape and one **decode** shape:

```bash
# Prefill: 256 new tokens, 4096-token KV cache.
build/bin/llama-onnx-export-dump -m model.gguf -o prefill.gdump \
    --batch-size 256 --ubatch-size 256 --ctx-size 4096

# Decode (one token at a time): 1 new token, 4096-token KV cache.
build/bin/llama-onnx-export-dump -m model.gguf -o decode.gdump \
    --batch-size 1   --ubatch-size 1   --ctx-size 4096
```

The big external `.onnx.data` file (weights) can be reused between
prefill and decode shapes; only the small `.onnx` graph shell differs.

### Vision encoders (CLIP / ViT / SigLIP)

Multimodal models split into two graphs: a language model and a vision (or
audio) encoder that converts pixels (or mel spectrograms) into embeddings.
The LM graph is built through `llama_graph_reserve` like any other text
model; the vision graph lives inside `tools/mtmd/clip.cpp` and is built
through a separate pipeline.

The dumper can serialise both. Pass `--include-vision` to additionally
produce a `.vision.gdump` next to the main output; or pass
`--vision-dump <path>` to write the vision graph to a specific file. For a
vision-only mmproj GGUF (one that contains no LM at all, such as a
stand-alone `clip-vit-base-patch16.gguf`), pass `--vision-only` to skip the
LM step entirely.

```bash
# Combined LM + vision mmproj GGUF: dumps both graphs.
./build/bin/llama-onnx-export-dump -m model.gguf -o model.gdump --include-vision
# -> model.gdump (LM), model.vision.gdump (vision encoder)

# Vision-only mmproj:
./build/bin/llama-onnx-export-dump -m mmproj-clip.gguf -o /dev/null \
    --vision-only --vision-dump clip.gdump
```

The vision dump uses the same on-disk format as the LM dump (gdump v1) and
is read by the same `convert_gguf_to_onnx.py` Python tool. The `arch`
metadata field is set to `clip-vision` (or `clip-audio`) so downstream
tooling can tell apart a vision dump from an LM dump.

#### Vision verification fixture

`--vision-fixture <path>` is the vision counterpart to `--fixture`: alongside
the vision `.gdump`, it runs the clip CPU reference on a deterministic input
image and serialises `(input_pixels, output_features)` into a `.gfxt` file
so the ONNX-translated graph can be numerically compared against ggml.

The input image is a `(H, W, 3)` interleaved RGB float buffer filled by
`std::mt19937(42)` with `uniform_real_distribution<float>(-1, 1)`; `H` and
`W` come from the model's `clip_hparams.image_size`. The output is whatever
`clip_image_encode` writes — i.e. the projected visual features that
normally feed the LM.

```bash
build/bin/llama-onnx-export-dump --vision-only \
    -m /tmp/smolvlm/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
    -o /tmp/vlm.gdump --vision-fixture /tmp/vlm.gfxt
```

The fixture file uses the **v4** `.gfxt` layout (LM fixtures stay at v3);
both versions are read by `gguf.gdump_fixture.load(...)`, which now returns
a `Fixture` whose `kind` field is either `"lm"` or `"vision"`. The vision
variant populates `input_pixels` / `output_features` / `output_name`
instead of `token_ids` / `logits` / `intermediates`.

Manual comparison recipe (no driver script yet):

```python
from gguf.gdump_fixture import load
import onnxruntime as ort
import numpy as np

fx = load("/tmp/vlm.gfxt")
assert fx.kind == "vision"
# Translate vlm.vision.gdump -> vlm.vision.onnx using convert_gguf_to_onnx.py
sess = ort.InferenceSession("/tmp/vlm.vision.onnx", providers=["CPUExecutionProvider"])
# Feed the captured pixels under whatever name the ONNX model uses for the
# pixel input (typically inp_raw). clip's set_input rearranges HWC into the
# CHW unrolling the cgraph expects, so the ONNX model may want the same
# transpose — adapt to whatever the translated graph asks for.
inp = fx.input_pixels  # (H, W, 3) interleaved RGB
chw = inp.transpose(2, 0, 1).reshape(1, 3, *inp.shape[:2])  # (1, C, H, W)
out = sess.run(None, {sess.get_inputs()[0].name: chw})[0]
ref = fx.output_features
cos = (out.ravel() @ ref.ravel()) / (np.linalg.norm(out) * np.linalg.norm(ref))
print("cosine:", cos, "max-abs:", np.abs(out.reshape(ref.shape) - ref).max())
```

## Inspecting / validating dtype metadata

`inspect_weights.py` is a static inspector for the resulting `.onnx` file.
It lists every initializer with its in-ONNX dtype, the *original* ggml type
parsed from the `original_ggml_type=...` annotation stamped by the
translator, shape, and byte size. It also prints the model-level
`metadata_props` block and an observed-vs-recorded histogram diff. No
`onnxruntime` dependency.

```bash
python3 tools/onnx-export/inspect_weights.py model.onnx [--top 20] [--sort name|bytes|dtype] [--summary-only]
```

`validate_weight_dtypes.py` cross-checks an exported ONNX file against its
source GGUF: every weight initializer must carry an `original_ggml_type=`
annotation matching the GGUF tensor's actual ggml type, the model-level
histogram must match the per-initializer histogram, all initializers must
match the chosen `--weight-dtype`, and every GGUF tensor must be exported.

```bash
python3 tools/onnx-export/validate_weight_dtypes.py model.gguf=model.onnx:float32 [more.gguf=more.onnx:float16 ...]
```

## Pre-flight op-coverage check

`compatibility_check.py` answers "will my GGUF export cleanly to ONNX?"
*without* producing the multi-GB ONNX artefact. It invokes the existing
`llama-onnx-export-dump` binary into a temp file, walks every node in the
resulting `.gdump`, and cross-references each ggml op (plus the `UNARY` /
`GLU` sub-variant when applicable) against the handler table in
`gguf.gdump_to_onnx._OP_HANDLERS`. The output is a Markdown-friendly table of
per-op counts with an `OK` / `MISSING HANDLER` status column, a final verdict
line, and an exit code (0 = clean, 1 = missing handlers) suitable for CI.
Flags mirror the dumper's: `--include-vision`, `--vision-only`,
`--batch-size`, `--ubatch-size`, `--ctx-size`, plus `--summary-only`,
`--json`, `--keep-dump PATH`, and `--dumper PATH`. No `onnx` /
`onnxruntime` dependency.

```bash
python3 tools/onnx-export/compatibility_check.py model.gguf
```

## Status

End-to-end verified via `tools/onnx-export/compare.py` (wikitext-style
prompt against `llama_decode`):

| family                        | quant   | argmax agree | mean cos | notes |
| ----------------------------- | ------- | ------------ | -------- | ----- |
| Llama 3.2 1B Instruct         | Q4_K_M  | 31/32 (96.9%)| 0.9991   | NORM RoPE |
| Qwen2 0.5B Instruct           | Q4_K_M  | 32/32 (100%) | 0.9994   | clean baseline |
| Qwen3 0.6B                    | Q4_K_M  | 29/32 (90.6%)| 0.9977   | Q/K RMSNorm |
| Gemma 2B IT                   | Q4_K_M  | 31/32 (96.9%)| 0.9996   | GEGLU FFN |
| Phi-3.5-mini Instruct         | Q4_K_M  | 30/32 (93.8%)| 0.9908   | YaRN longrope |
| SmolVLM-500M mmproj (vision)  | n/a     | n/a          | n/a      | loads + runs; numerical reference pending |

Synthetic-GGUF round-trip via `gguf-py/tests/test_onnx_export.py`:
llama, qwen3, qwen3moe, qwen2vl LM, gemma, mamba — all pass. 8 tests, 1
skip (`test_llama_quantised_q4_0` skips when `llama-quantize` isn't on
PATH).

See `docs/development/gguf-to-onnx.md` for the running ggml-op coverage
table and the broader status of vision tracing, recurrent ops, and the
known quantised-weight drift floor (currently `~0.5-0.9` max-abs on
final logits for Q4_K_M models — see the top-level README for details).
