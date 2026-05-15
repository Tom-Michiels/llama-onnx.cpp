# Exporting a GGUF model to ONNX

`convert_gguf_to_onnx.py` reads a llama.cpp GGUF file and emits an ONNX graph
that implements the same forward pass. The exported model exposes a
`past_key_values` / `present` KV cache so it can drive autoregressive token
generation from any ONNX runtime (e.g. `onnxruntime`).

This is a first-cut converter; it currently supports:

| feature                       | status                                            |
| ----------------------------- | ------------------------------------------------- |
| architectures                 | `llama` (Llama 1/2/3 family, dense only)          |
| tensor dtypes in the GGUF     | F32, F16, BF16 (and F64). Quantized types refused |
| ONNX compute / weight dtype   | `--dtype float32` (default) or `--dtype float16`  |
| grouped-query attention       | supported (`n_head` != `n_head_kv`)               |
| Llama-3 RoPE frequency factors | supported (reads `rope_freqs.weight`)            |
| tied LM head                  | supported (when `output.weight` is missing)       |
| MoE / experts                 | not supported                                     |
| quantized weights             | not supported                                     |

The exporter targets ONNX **opset 17**. The opset is broad enough that the
model loads under recent `onnxruntime` builds without contrib ops.

## Usage

```bash
# Convert a GGUF file (e.g. Llama-3.2-1B-Instruct, F16) to ONNX in fp32.
python convert_gguf_to_onnx.py \
    models/Llama-3.2-1B-Instruct.F16.gguf \
    --outfile models/llama-3.2-1b.onnx \
    --dtype float32
```

The result is two files:

```
models/llama-3.2-1b.onnx        # graph (small, ~tens of KB)
models/llama-3.2-1b.onnx.data   # weight blob (external data)
```

ONNX runtimes load both automatically as long as they live next to each other.

## ONNX I/O contract

Inputs:

| name                              | dtype  | shape                                  |
| --------------------------------- | ------ | -------------------------------------- |
| `input_ids`                       | int64  | `[batch, seq_len]`                     |
| `position_ids`                    | int64  | `[batch, seq_len]`                     |
| `past_key_values.{i}.key`         | fp32/16 | `[batch, n_head_kv, past_len, head_dim]` |
| `past_key_values.{i}.value`       | fp32/16 | `[batch, n_head_kv, past_len, head_dim]` |

Outputs:

| name                | dtype   | shape                                   |
| ------------------- | ------- | --------------------------------------- |
| `logits`            | fp32/16 | `[batch, seq_len, vocab_size]`          |
| `present.{i}.key`   | fp32/16 | `[batch, n_head_kv, total_len, head_dim]` |
| `present.{i}.value` | fp32/16 | `[batch, n_head_kv, total_len, head_dim]` |

`total_len = past_len + seq_len`. On the first call you pass empty caches
(shape `[batch, n_head_kv, 0, head_dim]`); on subsequent calls you feed the
`present.*` tensors from the previous step back in as `past_key_values.*`.

## Driving generation

Greedy decoding pseudo-code (see also `gguf-py/tests/test_onnx_export.py`):

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("llama-3.2-1b.onnx", providers=["CPUExecutionProvider"])
n_layer, n_head_kv, head_dim = 16, 8, 64  # from the GGUF metadata
empty = np.zeros((1, n_head_kv, 0, head_dim), dtype=np.float32)
past = [(empty, empty) for _ in range(n_layer)]

# Tokenise the prompt with your tokeniser of choice (the converter does not
# emit a tokeniser; pull it from the original HF repo or use the embedded
# tokeniser metadata in the GGUF).
ids = np.array([prompt_token_ids], dtype=np.int64)
pos = np.arange(ids.shape[1], dtype=np.int64)[None, :]

for step in range(max_new_tokens):
    feeds = {"input_ids": ids, "position_ids": pos}
    for i, (k, v) in enumerate(past):
        feeds[f"past_key_values.{i}.key"] = k
        feeds[f"past_key_values.{i}.value"] = v
    outs = sess.run(None, feeds)
    logits, presents = outs[0], outs[1:]
    next_id = int(logits[0, -1].argmax())
    past = [(presents[2*i], presents[2*i+1]) for i in range(n_layer)]
    ids = np.array([[next_id]], dtype=np.int64)
    pos = pos[:, -1:] + 1
    if next_id == eos_token_id:
        break
```

## Restrictions and rationale

* **Floats only.** The first pass deliberately rejects quantised GGUF files.
  Dequantising on the fly is straightforward but ONNX has no native support
  for ggml's block-quantised formats, and we want the graph to be runnable on
  any standard runtime.
* **`llama` only.** Other architectures share most of the building blocks but
  add quirks (Gemma's softcapping, Phi's parallel residual, MoE routing, etc.).
  They will be added incrementally; for now we error out on
  `general.architecture != "llama"`.
* **No tokeniser export.** The ONNX format does not have a tokeniser
  primitive, so the model is the pure transformer/network — the caller is
  expected to provide tokens (and a detokeniser for the output).
* **NORM-style RoPE.** llama.cpp's `LLAMA_ROPE_TYPE_NORM` rotates *consecutive
  pairs* `(x[2i], x[2i+1])`. The HF-side `convert_hf_to_gguf.py` script
  re-permutes the Q/K weights so that consecutive-pair rotation produces the
  same result as HF's halves-split (NEOX) rotation. The exporter applies the
  rotation on the GGUF-stored weights directly, so the resulting ONNX is
  mathematically equivalent to running the model with llama.cpp.

## Tests

```
python gguf-py/tests/test_onnx_export.py
```

The tests build a tiny random llama model, write it to a temp GGUF, convert
to ONNX, then compare the ONNX outputs against a numpy reference forward pass
for several configurations: MHA, GQA, explicit LM head, and Llama-3 RoPE
frequency factors. The same two cases also exercise the prefill/decode
boundary by checking that two-step generation (prefill + 1-token decode) and
single-shot prefill produce the same logits for the new token.
