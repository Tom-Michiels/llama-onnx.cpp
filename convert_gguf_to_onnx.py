#!/usr/bin/env python3
"""Convert a GGUF llama-family model to an ONNX file.

This is a thin driver around two pieces:

1. ``llama-onnx-export-dump`` (built from ``tools/onnx-export/``) — a C++
   binary that loads the GGUF, calls ``llama_graph_reserve()`` to obtain the
   ggml compute graph built by llama.cpp's own ``src/models/<arch>.cpp``
   builders, and serialises it (along with the weight tensor data) to a
   portable ``.gdump`` file.

2. ``gguf.gdump_to_onnx`` — a Python module that reads the dump and emits
   an ONNX model that can be run with ``onnxruntime``.

Example:

    cmake --build build --target llama-onnx-export-dump -j
    python convert_gguf_to_onnx.py path/to/model.gguf \
        --outfile path/to/model.onnx \
        --dumper ./build/bin/llama-onnx-export-dump

If a ``.gdump`` already exists, pass ``--from-dump`` to skip the C++ step.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _find_dumper(hint: str | None) -> Path:
    if hint:
        p = Path(hint)
        if not p.is_file():
            raise FileNotFoundError(f"dumper not found at {p}")
        return p
    # Look in the conventional build output directory next to the repo root.
    repo = Path(__file__).resolve().parent
    candidates = [
        repo / "build" / "bin" / "llama-onnx-export-dump",
        repo / "build" / "Release" / "bin" / "llama-onnx-export-dump",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "could not find llama-onnx-export-dump. Build it first:\n"
        "    cmake --build build --target llama-onnx-export-dump -j\n"
        "Or pass its path with --dumper."
    )


def _run_dumper(dumper: Path, gguf: Path, dump_out: Path) -> None:
    # llama.cpp's binaries pick up shared libs from build/bin; make sure that's
    # on LD_LIBRARY_PATH so they can be invoked directly.
    env = os.environ.copy()
    env.setdefault("LD_LIBRARY_PATH", str(dumper.parent))
    cmd = [str(dumper), "-m", str(gguf), "-o", str(dump_out)]
    logging.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a GGUF model to ONNX via the C++ graph dumper")
    parser.add_argument("input", type=Path, help="path to a .gguf file (or a .gdump if --from-dump is set)")
    parser.add_argument("--outfile", "-o", type=Path, default=None, help="output .onnx path")
    parser.add_argument("--dumper", type=Path, default=None, help="path to llama-onnx-export-dump (auto-detected by default)")
    parser.add_argument("--from-dump", action="store_true", help="treat input as a pre-built .gdump file")
    parser.add_argument("--keep-dump", action="store_true", help="don't delete the intermediate .gdump file")
    parser.add_argument("--weight-dtype", default="float16",
                        choices=["float16", "fp16", "f16", "float32", "fp32", "f32"],
                        help="dtype to use for the ONNX weight initializers (default float16). "
                             "Quantised GGUF tensors are always dequantised on load; this controls "
                             "the precision of the resulting initializers.")
    parser.add_argument("--verbose", "-v", action="store_true", help="enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Allow running directly from a source checkout without `pip install`-ing gguf.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "gguf-py"))
    from gguf.gdump_to_onnx import convert as gdump_to_onnx

    onnx_path = args.outfile or args.input.with_suffix(".onnx")

    if args.from_dump:
        gdump_to_onnx(args.input, onnx_path, weight_dtype=args.weight_dtype)
        return 0

    dumper = _find_dumper(str(args.dumper) if args.dumper else None)
    if args.keep_dump:
        dump_path = args.input.with_suffix(".gdump")
        _run_dumper(dumper, args.input, dump_path)
        gdump_to_onnx(dump_path, onnx_path, weight_dtype=args.weight_dtype)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            dump_path = Path(tmp) / (args.input.stem + ".gdump")
            _run_dumper(dumper, args.input, dump_path)
            gdump_to_onnx(dump_path, onnx_path, weight_dtype=args.weight_dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
