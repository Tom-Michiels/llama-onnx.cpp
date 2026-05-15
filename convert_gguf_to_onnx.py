#!/usr/bin/env python3
"""Convert a GGUF llama model to an ONNX file.

Example:
    python convert_gguf_to_onnx.py models/Llama-3.2-1B-Instruct-F16.gguf \
        --outfile models/llama-3.2-1b.onnx --dtype float32

Only the LLaMA architecture is currently supported, and only floating-point
coefficients (F32, F16, BF16). The exported model takes ``input_ids``,
``position_ids`` and a per-layer KV cache pair as inputs, and returns logits
together with the updated cache, which is enough to drive autoregressive token
generation from any ONNX runtime.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a GGUF llama model to ONNX")
    parser.add_argument("gguf", type=Path, help="path to the input .gguf file")
    parser.add_argument("--outfile", "-o", type=Path, default=None,
                        help="path to the output .onnx file (default: <input>.onnx next to the input)")
    parser.add_argument("--dtype", default="float32", choices=["float32", "fp32", "f32", "float16", "fp16", "f16"],
                        help="compute / weight dtype in the generated ONNX (default: float32)")
    parser.add_argument("--verbose", "-v", action="store_true", help="enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Allow running directly from a source checkout without `pip install`-ing
    # gguf-py first.
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "gguf-py"))

    from gguf.onnx_export import convert  # noqa: E402

    outfile = args.outfile
    if outfile is None:
        outfile = args.gguf.with_suffix(".onnx")

    convert(args.gguf, outfile, dtype=args.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
