# llama-onnx.cpp

Convert any GGUF model to ONNX by reusing llama.cpp's own graph builders. The
forward pass is defined exactly once, in `src/models/<arch>.cpp`; this fork
asks llama.cpp (via `llama_graph_reserve()`) to build the prefill compute
graph, serialises the resulting `ggml_cgraph`, and translates it to ONNX in
Python. No Python re-implementation of any architecture; new architectures
come for free as soon as llama.cpp supports them and the ops they use are in
the translator.

This is a fork of [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp).
For general llama.cpp usage (running models, `llama-cli`, `llama-server`,
quantisation, backends, bindings, UIs), see the upstream project. This README
covers only the GGUF -> ONNX exporter added in this fork.

## How it works

```
                build/bin/llama-onnx-export-dump
                +--------------------------------+
   model.gguf ->|  load model                    |
                |  llama_graph_reserve()  <----- src/models/<arch>.cpp
                |  walk cgraph, dump tensors     |
                +---------------+----------------+
                                |  model.gdump  (portable binary; see
                                |                tools/onnx-export/README.md)
                                v
                gguf.gdump_to_onnx (Python, in gguf-py/)
                +--------------------------------+
                |  read dump                     |
                |  ggml op -> ONNX op translation|
                |  emit ONNX                     |
                +---------------+----------------+
                                v
                          model.onnx + model.onnx.data
                                |
                                v
                          onnxruntime
```

Three stages: a C++ dumper writes a `.gdump` file, a Python translator reads
the dump and emits ONNX. The C++ binary has no ONNX dependency; the Python
side has no model-architecture knowledge.

## Quick start

Build the C++ dumper:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-onnx-export-dump -j
```

Convert an LM GGUF to ONNX:

```bash
python3 convert_gguf_to_onnx.py path/to/model.gguf \
    --outfile path/to/model.onnx \
    --weight-dtype float32
```

Convert a multimodal (vision-tower) GGUF — adds the `mtmd` `clip_*` graph
on top of the LM graph; use `--vision-only` for a vision-only mmproj GGUF
with no LM:

```bash
build/bin/llama-onnx-export-dump --include-vision \
    -m path/to/mmproj.gguf -o path/to/model.gdump
# or for vision-only mmproj:
build/bin/llama-onnx-export-dump --vision-only \
    -m path/to/mmproj.gguf -o path/to/model.gdump
python3 -c "from gguf.gdump_to_onnx import convert; \
    convert('path/to/model.gdump', 'path/to/model.onnx', weight_dtype='float32')"
```

`--weight-dtype float32` is the safer default. `--weight-dtype float16`
works for Gemma 2B and Phi-3.5-mini (after recent cast fixes for
`Div(inv_freqs, rope_freqs.weight)`); Llama 3.2 may still hit related
fp16 casts on other paths.

Quantised GGUF weights (Q4_0, Q4_K_M, Q8_0, IQ\*, …) are dequantised on
load and cast to the chosen weight dtype. The original ggml type is
preserved in per-initializer `doc_string` + model-level `metadata_props`
(see [Inspecting weight metadata](#inspecting-weight-metadata-in-an-exported-onnx)).

Validate numerically against `llama_decode` on a wikitext-style prompt:

```bash
python3 tools/onnx-export/compare.py path/to/model.gguf
```

The harness runs the C++ dumper with a fixture flag, translates the dump,
runs the ONNX model under onnxruntime with the same inputs, and reports
per-token argmax agreement and cosine similarity against the reference
logits.

### Choosing prefill vs decode shapes at export time

`n_tokens` (new tokens per forward pass), `n_ctx` (KV cache total length),
and `n_seqs` are baked into the exported ONNX file. The dumper inherits
llama.cpp's standard CLI arg parser, so any of these can be overridden:

```bash
# Prefill graph: 256 new tokens, 4096-token KV cache.
build/bin/llama-onnx-export-dump -m model.gguf -o prefill.gdump \
    --batch-size 256 --ubatch-size 256 --ctx-size 4096

# Decode graph (one token at a time): 1 new token, 4096-token KV cache.
build/bin/llama-onnx-export-dump -m model.gguf -o decode.gdump \
    --batch-size 1   --ubatch-size 1   --ctx-size 4096
```

The rule is `n_tokens = min(n_ctx, n_ubatch)`. Defaults if no flag is
given: `n_batch = n_ubatch = 32`, `n_ctx = 512`, `n_seqs = 1`. `n_seqs`
is currently hard-coded to 1 and not exposed via a flag; all six
recurrent op handlers raise `NotImplementedError` for `n_seqs > 1`
anyway.

For a real serving pipeline you typically need two exports of the same
weights: one with `--ubatch-size N` for prefill and one with
`--ubatch-size 1` for decode. The external `.onnx.data` file (multi-GB)
can be shared between them; only the small `.onnx` graph shell differs.

## GGML op coverage

Source of truth: the `_OP_HANDLERS` registry in
`gguf-py/gguf/gdump_to_onnx.py` (re-grep for `@_op(gdump.GgmlOp.` to
re-verify). The full `GgmlOp` enum (94 values) is in
`gguf-py/gguf/gdump.py`.

### Implemented

| op             | notes                                                        |
| -------------- | ------------------------------------------------------------ |
| ADD            |                                                              |
| ADD_ID         | shared bias / MoE indexed-add                                |
| SUB            |                                                              |
| MUL            |                                                              |
| DIV            |                                                              |
| SQR            |                                                              |
| SQRT           |                                                              |
| LOG            |                                                              |
| SIN            |                                                              |
| COS            |                                                              |
| SUM            | `ReduceSum`                                                  |
| SUM_ROWS       | `ReduceSum` over last axis                                   |
| MEAN           | `ReduceMean` over last axis                                  |
| REPEAT         | broadcast via `Expand`                                       |
| CONCAT         |                                                              |
| NORM           | `LayerNorm` without affine                                   |
| RMS_NORM       |                                                              |
| L2_NORM        | Qwen3.5, Qwen3-Next, Kimi-Linear, RWKV7-base                 |
| MUL_MAT        |                                                              |
| MUL_MAT_ID     | TopK + `Gather` + broadcast `Mul` + `ReduceSum` (MoE)        |
| SCALE          |                                                              |
| CPY            | dtype cast                                                   |
| CONT           |                                                              |
| RESHAPE        |                                                              |
| VIEW           | contiguous-slice, slot-squeeze, flat-source-reshape, 0-dim   |
| PERMUTE        | ggml op_params is the *inverse* perm; we invert before mapping |
| TRANSPOSE      |                                                              |
| GET_ROWS       | `Gather` + per-batch `GatherElements`; rank-1 data special-cased |
| SET_ROWS       | `ScatterND`                                                  |
| SOFT_MAX       | with optional additive mask + scale; no ALiBi                |
| ROPE           | NORMAL, NEOX, MROPE; YaRN scaling                            |
| CLAMP          |                                                              |
| ARGSORT        |                                                              |
| TOP_K          | indices cast to int32                                        |
| FLASH_ATTN_EXT | no softcap, no ALiBi                                         |
| SSM_CONV       | depthwise 1-D `Conv(group=d_inner)` (Mamba, hybrids)         |
| SSM_SCAN       | static unroll over n_seq_tokens (Mamba-1 + Mamba-2)          |
| RWKV_WKV6      | static unroll over T (RWKV-6); `(k, v, r, tf, td, state)`    |
| RWKV_WKV7      | static unroll over T (RWKV-7)                                |
| GATED_LINEAR_ATTN | linear-attention outer-product unroll (RWKV-6 hybrid)     |
| GATED_DELTA_NET | delta-rule recurrence; scale `1/sqrt(S)` baked in            |
| CONV_2D        | direct ONNX `Conv`; vision encoders                          |
| CONV_2D_DW     | depthwise `Conv(group=Cin)`                                  |
| POOL_1D        | `MaxPool` / `AveragePool` (rank-1)                           |
| POOL_2D        | `MaxPool` / `AveragePool` (rank-2)                           |
| GROUP_NORM     | opset-17 hand-roll (Reshape + ReduceMean/ReduceVar)          |
| PAD            | `Pad(mode=constant)`; rejects circular padding               |
| IM2COL         | `Gather`+`Concat`+`Transpose`+`Reshape` (raw patch extract)  |
| UPSCALE        | `Resize` with explicit `sizes`                               |
| LEAKY_RELU     | direct `LeakyRelu(alpha=...)`                                |
| UNARY          | see sub-table                                                |
| GLU            | see sub-table                                                |

### UNARY sub-ops (20 of 20 implemented)

| variant     | status |
| ----------- | ------ |
| SILU        | yes    |
| RELU        | yes    |
| SIGMOID     | yes    |
| TANH        | yes    |
| NEG         | yes    |
| EXP         | yes    |
| GELU        | yes    |
| GELU_ERF    | yes    |
| HARDSWISH   | yes    |
| HARDSIGMOID | yes    |
| ABS         | yes    |
| SGN         | yes    |
| STEP        | yes    |
| ELU         | yes    |
| GELU_QUICK  | yes    |
| XIELU       | yes    |
| FLOOR       | yes    |
| CEIL        | yes    |
| ROUND       | yes    |
| TRUNC       | yes    |

### GLU sub-ops (5 of 6 implemented)

| variant     | status |
| ----------- | ------ |
| SWIGLU      | yes    |
| GEGLU       | yes    |
| GEGLU_ERF   | yes    |
| REGLU       | yes    |
| SWIGLU_OAI  | yes (gpt-oss variant)         |
| GEGLU_QUICK | no                            |

### Not implemented, grouped by reason

**Training-only (will not be added):**
DUP, ACC, ADD1, REPEAT_BACK, SILU_BACK, RMS_NORM_BACK, GET_ROWS_BACK,
SOFT_MAX_BACK, ROPE_BACK, IM2COL_BACK, POOL_2D_BACK, FLASH_ATTN_BACK,
CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_BACK, OPT_STEP_ADAMW, OPT_STEP_SGD,
OUT_PROD.

**Vision / convolution (less-common variants):**
CONV_TRANSPOSE_1D, CONV_TRANSPOSE_2D, CONV_3D, IM2COL_3D, PAD_REFLECT_1D,
WIN_PART, WIN_UNPART, GET_REL_POS, ADD_REL_POS.

**Legacy (superseded by masked SOFT_MAX):**
DIAG, DIAG_MASK_INF, DIAG_MASK_ZERO.

**Niche / unused by current LM architectures:**
ARGMAX, COUNT_EQUAL, CUMSUM, TRI, FILL, SOLVE_TRI, ARANGE,
TIMESTEP_EMBEDDING, ROLL, SET, MAP_CUSTOM1/2/3, CUSTOM.

Adding any of these is a localised change: a single `@_op(GgmlOp.X)`
handler in `gguf-py/gguf/gdump_to_onnx.py`.

## LLM families covered today

End-to-end verified on this branch via `tools/onnx-export/compare.py` or
`/tmp/wikitext_compare.py` (wikitext-style prompt, 32 tokens, argmax
agreement and cosine similarity of logits against `llama_decode`):

| model                       | quant   | weight dtype | argmax agree | mean cos | notes |
| --------------------------- | ------- | ------------ | ------------ | -------- | ----- |
| Llama 3.2 1B Instruct       | Q4_K_M  | fp32         | 31/32 (96.9%)| 0.9991   | NORM RoPE |
| Qwen2 0.5B Instruct         | Q4_K_M  | fp32         | 32/32 (100%) | 0.9994   | clean baseline |
| Qwen3 0.6B                  | Q4_K_M  | fp32         | 29/32 (90.6%)| 0.9977   | Q/K RMSNorm |
| Gemma 2B IT                 | Q4_K_M  | fp16         | 31/32 (96.9%)| 0.9996   | GEGLU FFN, fp16 path works |
| Phi-3.5-mini Instruct       | Q4_K_M  | fp16         | 30/32 (93.8%)| 0.9908   | **YaRN longrope** |

Vision encoders verified end-to-end (ONNX graph loads in onnxruntime,
forward pass produces all-finite output; numerical reference vs ggml CPU
is pending — there's no fixture mode for vision graphs yet):

| model                       | weight dtype | output shape | notes |
| --------------------------- | ------------ | ------------ | ----- |
| SmolVLM-500M mmproj         | fp32         | `(64, 960)`  | idefics3 SigLIP base + pixel-shuffle |

Architectures that should work because they only use ops in the
implemented set (no end-to-end wikitext test yet on this branch):

- Llama 1 / 2 / 3.x family generally
- Mistral, Mixtral
- Phi-2, Phi-3 (Phi-3.5 longrope verified above)
- Qwen2, Qwen2.5, Qwen3, Qwen3-MoE
- DeepSeek-V3 longrope (covered by YaRN; same code path verified on Phi-3.5)
- Gemma 1
- gpt-oss (uses the `SWIGLU_OAI` GLU variant)
- MoE variants with shared bias via `ADD_ID`

Architectures with **synthetic-GGUF round-trip verified** in
`gguf-py/tests/test_onnx_export.py` (a tiny in-memory GGUF goes through
dump → translate → onnxruntime forward pass; real-model wikitext compare
is the next step):

- Mamba (`test_mamba`; uses `SSM_CONV` + `SSM_SCAN` + the recently-added
  `_h_view` 0-dim branch and `_h_get_rows` rank-1 special-case).

Architectures with **handlers in place, math-verified against the ggml
CPU reference**, no `tiny_*` fixture yet:

- Mamba-2, LFM2, Kimi-Linear, Qwen3-Next, plamo2 (`SSM_CONV` + `SSM_SCAN`)
- RWKV-6 (`RWKV_WKV6` + `GATED_LINEAR_ATTN`)
- RWKV-7 (`RWKV_WKV7`)
- delta-net-base (`GATED_DELTA_NET`)

Tiers reflect *test coverage*, not exporter functionality — all six
recurrent / state-space handlers are implemented and their math is
verified bit-for-bit against `ggml/src/ggml-cpu/ops.cpp`. All six also
raise `NotImplementedError` for `n_seqs > 1` (multi-sequence batching);
today's llama.cpp prefill always uses `n_seqs = 1`, so this is parked.

## Numerical fidelity and the drift floor

Argmax agreement above is excellent but not perfect: 1-3 token disagreements
out of 32 and a cosine residual of ~10^-3 to 10^-4 against `llama_decode`.
On Llama 3.2 1B Q4_K_M the residual surfaces as a ~0.5-0.9 max-abs gap on
the final logits (with cosine still 0.999). The root cause is a single,
mechanical mismatch between how ggml's CPU backend and onnxruntime perform
the matmul against quantised weights.

**Mechanism.** ggml's CPU `vec_dot` for K-quants (`Q2_K..Q6_K`, `IQ4_NL`)
sets `vec_dot_type = Q8_K`: before doing the dot product against the Q4_K
weight, it **quantises the activation** to int8 with a fp32 scale per
256-element block (see `quantize_row_q8_K_ref` in
`ggml/src/ggml-quants.c`). Q4_0 / Q4_1 do the same with `Q8_0` (32-element
blocks, fp16 scale). The exported ONNX, by contrast, runs `MatMul` on the
raw fp32 activation against the *dequantised* fp32 weight. So every
quantised matmul in the ONNX path is off by ~`max(|x_block|)/127` per
element. Compounded over 16 layers x 7 matmuls/layer, that produces the
observed final-logit gap on Llama 3.2 1B.

**Numerical evidence.** A layer-by-layer comparison (using the `cb_eval`
instrumentation described below) finds the first divergence above 1e-3 in
the layer-0 Q/K/V matmuls; `attn_norm-0` itself matches to 5e-7 between
both paths. Pre-quantising the activation to Q8_K in numpy and then doing
the matmul in fp32 reproduces ggml's runtime `Vcur-0` to within fp32 noise
(max|Δ| 2.9e-4 vs 8.4e-3 without the requantise step) and matches the
captured post-RoPE `Qcur-0` drift signature to four digits.

| probe (Llama 3.2 1B Q4_K_M) | shape         | max\|Δ\|   | cosine    |
|-----------------------------|---------------|------------|-----------|
| `attn_norm-0`               | (32, 2048)    | 4.77e-7    | 1.0000    |
| `Vcur-0`                    | (32, 8, 64)   | 8.43e-3    | 0.9998    |
| `Qcur-0`                    | (32, 32, 64)  | 4.69e-2    | 0.99999   |
| `Kcur-0`                    | (32, 8, 64)   | 4.74e-2    | 0.99999   |
| `attn_out-0`                | (32, 2048)    | 3.52e-3    | 0.9998    |
| `result_output` (logits)    | (32, 128256)  | 8.34e-1    | 0.9993    |

**Proposed fix (single change closes the gap).** Inside `_h_mul_mat` in
`gguf-py/gguf/gdump_to_onnx.py`, when the weight's original ggml type is
in `{Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL}`, pre-process the activation
with the same Q8_K quantise/dequantise round-trip inside the ONNX graph
(`ReduceMax -> Div -> Round -> Clip -> Mul`). Same idea for the non-K
quants with `vec_dot_type = Q8_0` (Q4_0, Q4_1). The original ggml type is
already recorded per-initializer in `_original_dtypes`, so the dispatch
is local.

**Things ruled out.** `attn_norm-0`'s 5e-7 match shows that RMSNorm,
GLU operand order, MatMul reduction order, fp16 KV-cache cast, and the
YaRN refactor are all bit-equivalent (or close enough not to dominate).
Don't chase them.

**Tooling left behind for further investigation.**

- `tools/onnx-export/onnx-export-dump.cpp` — when `--fixture` is given,
  installs a `cb_eval` callback that captures every named intermediate
  (`attn_norm-N`, `Qcur-N`, `Kcur-N`, `Vcur-N`, `attn_out-N`,
  `ffn_inp-N`, `ffn_norm-N`, `ffn_out-N`, `l_out-N`, `result_norm`,
  `result_output`) as fp32 into a v3 fixture block.
- `gguf-py/gguf/gdump_fixture.py` — fixture **v3** reader; back-compatible
  with v2.
- `gguf-py/gguf/gdump_to_onnx.py` — `Translator(extra_output_names=[...])`
  exposes named ggml intermediates as additional `dbg__<safe>` ONNX
  outputs.
- `tools/onnx-export/per_layer_diff.py` — driver that runs the full
  comparison and prints max|Δ| / mean|Δ| / cosine per probe, flagging
  the first divergence above a threshold.

## Inspecting weight metadata in an exported ONNX

Every initializer carries the source GGUF's ggml dtype on its
`doc_string` (`original_ggml_type=<TYPE>`), and the model-level
`metadata_props` carries a histogram. Inspect with:

```bash
python3 tools/onnx-export/inspect_weights.py path/to/model.onnx \
    [--top 20] [--sort name|bytes|dtype] [--summary-only]
```

Programmatic API (`tools/onnx-export/inspect_weights.py`): the
`WeightInspector` class returns per-initializer `Weight` records
(`name`, `onnx_dtype`, `ggml_dtype`, `shape`, `n_elements`,
`n_bytes_onnx`, `is_external`) plus `metadata()` and observed-vs-recorded
histogram helpers.

`tools/onnx-export/validate_weight_dtypes.py` cross-checks (1) every
initializer has a parseable annotation, (2) the per-initializer histogram
matches the model-level one, (3) ONNX dtypes match the chosen
`--weight-dtype`, and (4) every cgraph-referenced GGUF tensor lands in
ONNX.

Sample output for a Qwen2 0.5B Q4_K_M export is checked in at
[`tools/onnx-export/sample-output/qwen2-0.5b-weights.txt`](tools/onnx-export/sample-output/qwen2-0.5b-weights.txt).
The model-level histogram for that model:

```
llama_onnx.original_dtype_counts = F32:121,Q4_K:24,Q5_0:264,Q6_K:24,Q8_0:27
```

Note that `gguf-py/gguf/gdump.py`'s local `GgmlType` enum is intentionally
small (`F32, F16, Q4_0, Q4_1, BF16, F64, I*`). The label resolver in
`_add_weight_initializer` falls back to `gguf.constants.GGMLQuantizationType`
for the full quant set (`Q4_K, Q5_K, Q6_K, Q8_0, IQ*` …) so the labels in
`doc_string` and `original_dtype_counts` come out human-readable.

## Known limitations

- **Static shapes.** `n_tokens` is baked in at export time
  (`min(n_ctx, n_ubatch)`). Re-export for a different prefill length. A
  follow-up will symbolify the time axis.
- **`--weight-dtype float16` works for Gemma 2B IT and Phi-3.5-mini**
  (post-recent fixes to the `Div(inv_freqs, rope_freqs.weight)` cast in
  `_h_rope`). Llama 3.2 may still hit related casts on other paths;
  `--weight-dtype float32` remains the safer default.
- **KV cache I/O is always fp16.** The graph is reserved with llama.cpp's
  default cache type; ONNX `past_key_values.*` inputs and `present.*`
  outputs are f16. Casting on the fly is the caller's responsibility.
- **Vision encoder numerical correctness is unverified.** The vision
  graph traces (`--include-vision` / `--vision-only`) and runs in
  onnxruntime, but there's no fixture mode yet for `clip_*` graphs to
  compare against ggml CPU. Shape inference is clean (0 / 98
  incompatible MatMuls on SmolVLM-500M after the recent `_h_permute` /
  `_h_mul_mat` rank-aware fix); finite-output sanity check passes; bit-
  level comparison against `clip.cpp` requires a `--vision-fixture` flow
  that doesn't exist yet.
- **Recurrent ops force `n_seqs = 1`.** All six handlers (SSM_SCAN,
  SSM_CONV, RWKV_WKV6/7, GATED_LINEAR_ATTN, GATED_DELTA_NET) raise
  `NotImplementedError` for multi-sequence prefill. Today's llama.cpp
  prefill graph always uses 1.
- **Quantised-weight drift floor.** The exported ONNX runs MatMul on the
  raw fp32 activation against the dequantised fp32 weight; ggml's CPU
  backend instead quantises the activation to Q8_K (for K-quants) or
  Q8_0 (for Q4_0 / Q4_1) before the dot product. Net effect: a
  ~`max(|x|)/127` per-element offset that compounds to ~0.5–0.9 max-abs
  on the final logits over 16 layers, while cosine stays ≥ 0.999. Fix is
  scoped (single change in `_h_mul_mat` keyed on `_original_dtypes`).
  See [Numerical fidelity](#numerical-fidelity-and-the-drift-floor).

## Further reading

- `docs/development/gguf-to-onnx.md` — design doc, full pipeline notes,
  ONNX I/O contract, quantised-weight handling.
- `tools/onnx-export/README.md` — `.gdump` binary format, build/run
  details for the C++ dumper.
- `tools/onnx-export/onnx-export-dump.cpp` — C++ dumper source.
- `gguf-py/gguf/gdump.py` — `.gdump` reader and `GgmlOp` enum (source of
  truth for ggml op identifiers).
- `gguf-py/gguf/gdump_to_onnx.py` — Python translator; the
  `_OP_HANDLERS` registry (search for `@_op(gdump.GgmlOp.`) lists every
  implemented op handler.
- `tools/onnx-export/per_layer_diff.py` — layer-by-layer comparison
  harness; reads fixture v3 with `cb_eval`-captured intermediates and
  prints max|Δ| / cosine per probe. Used to identify the Q8_K drift.
- `tools/onnx-export/inspect_weights.py` — `WeightInspector` class + CLI
  for ONNX initializer dtype / original-ggml-type tables.
- `tools/onnx-export/validate_weight_dtypes.py` — cross-checks GGUF →
  ONNX weight-dtype metadata round-trip.
- `convert_gguf_to_onnx.py` — user-facing driver.
- `tools/onnx-export/compare.py` — numerical comparison harness.
- `gguf-py/tests/test_onnx_export.py` and `tiny_models.py` — synthetic
  per-architecture round-trip tests. Currently 8 pass, 1 skip
  (`test_llama_quantised_q4_0` skips when `llama-quantize` isn't on
  PATH).

A phased roadmap exists for extending coverage (vision fixture mode,
real-GGUF tests for the remaining recurrent families, dynamic shapes,
the Q8_K activation-quantisation drift fix); it tracks the work
item-by-item but lives outside this repo.


