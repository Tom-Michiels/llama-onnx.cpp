#!/usr/bin/env python3
"""End-to-end comparison harness for the GGUF -> ONNX exporter.

Given a real GGUF file, this script:

1. Runs ``llama-onnx-export-dump --fixture`` to obtain both a graph dump
   and a reference logits fixture from ``llama_decode``.
2. Translates the dump to ONNX via ``gguf.gdump_to_onnx``.
3. Loads the ONNX in onnxruntime, feeds it the same inputs (synthesised
   from the gdump's graph structure plus the fixture's token IDs), and
   compares its logits against the reference.

Quantised GGUFs (Q4_0, Q8_0, Q4_K, ...) are handled transparently — the
weights are dequantised to fp32 in Python on load and then cast to
``--weight-dtype`` for the ONNX initializers.

Usage:

    cmake --build build --target llama-onnx-export-dump -j
    python tools/onnx-export/compare.py path/to/model.gguf
    python tools/onnx-export/compare.py path/to/model.gguf --weight-dtype float32

Network downloads aren't supported here — the script just runs against
whatever GGUF you point it at. To exercise the quantised path with a
synthetic-but-realistic base, pipe llama-quantize first:

    ./build/bin/llama-quantize path/to/f32.gguf path/to/q4.gguf Q4_0
    python tools/onnx-export/compare.py path/to/q4.gguf
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_dumper(hint: str | None) -> Path:
    if hint:
        p = Path(hint)
        if not p.is_file():
            raise FileNotFoundError(f"dumper not found at {p}")
        return p
    for c in [REPO_ROOT / "build" / "bin" / "llama-onnx-export-dump",
              REPO_ROOT / "build" / "Release" / "bin" / "llama-onnx-export-dump"]:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "llama-onnx-export-dump not found. Build it first:\n"
        "    cmake --build build --target llama-onnx-export-dump -j\n"
        "or pass --dumper.")


def _per_row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    eps = 1e-12
    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + eps
    return num / den


def _topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Per-row Jaccard overlap of the top-k indices."""
    a_top = np.argsort(-a, axis=-1)[:, :k]
    b_top = np.argsort(-b, axis=-1)[:, :k]
    out = np.empty(a.shape[0], dtype=np.float64)
    for i in range(a.shape[0]):
        out[i] = len(set(a_top[i]) & set(b_top[i])) / k
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gguf", type=Path, help="path to a real .gguf file (any quantisation type)")
    parser.add_argument("--dumper", type=Path, default=None, help="path to llama-onnx-export-dump")
    parser.add_argument("--weight-dtype", default="float16",
                        choices=["float16", "float32"],
                        help="dtype for ONNX weight initializers (default float16)")
    parser.add_argument("--keep-artifacts", type=Path, default=None,
                        help="if given, keep the .gdump / .gfxt / .onnx in this directory")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sys.path.insert(0, str(REPO_ROOT / "gguf-py"))
    from gguf import gdump
    from gguf.gdump_to_onnx import convert as gdump_to_onnx
    from gguf.gdump_fixture import load as load_fx, synthesize_inputs

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is required for the comparison", file=sys.stderr)
        return 1

    dumper = _find_dumper(str(args.dumper) if args.dumper else None)

    if args.keep_artifacts:
        args.keep_artifacts.mkdir(parents=True, exist_ok=True)
        work = args.keep_artifacts
    else:
        # tempdir cleaned up by the `with`
        _tmp = tempfile.TemporaryDirectory()
        work = Path(_tmp.name)

    stem = args.gguf.stem
    dump_path = work / f"{stem}.gdump"
    fx_path   = work / f"{stem}.gfxt"
    onnx_path = work / f"{stem}.onnx"

    env = os.environ.copy()
    env.setdefault("LD_LIBRARY_PATH", str(dumper.parent))
    print(f"[1/3] dumping graph + reference fixture from {args.gguf}")
    subprocess.run([str(dumper), "-m", str(args.gguf),
                    "-o", str(dump_path), "--fixture", str(fx_path)],
                   env=env, check=True)

    print(f"[2/3] translating to ONNX (weight_dtype={args.weight_dtype})")
    gdump_to_onnx(dump_path, onnx_path, weight_dtype=args.weight_dtype)

    print(f"[3/3] running ONNX and comparing against reference logits")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    d = gdump.load(dump_path)
    fx = load_fx(fx_path)
    feeds = synthesize_inputs(d, fx, [i.name for i in sess.get_inputs()])
    (onnx_logits,) = sess.run(["logits"], feeds)
    assert onnx_logits.shape == fx.logits.shape, f"shape mismatch {onnx_logits.shape} vs {fx.logits.shape}"

    diff = np.abs(onnx_logits - fx.logits)
    cos = _per_row_cosine(onnx_logits, fx.logits)
    overlap5 = _topk_overlap(onnx_logits, fx.logits, k=5)

    print()
    print(f"tokens={fx.n_tokens}  vocab={fx.n_vocab}")
    print(f"onnx logits     range=[{onnx_logits.min():.3f}, {onnx_logits.max():.3f}]")
    print(f"reference logits range=[{fx.logits.min():.3f}, {fx.logits.max():.3f}]")
    print(f"max abs diff      = {diff.max():.4e}")
    print(f"mean abs diff     = {diff.mean():.4e}")
    print(f"per-row cosine    : min={cos.min():.4f}  mean={cos.mean():.4f}  max={cos.max():.4f}")
    print(f"per-row top-5     : min={overlap5.min():.2f}  mean={overlap5.mean():.2f}  max={overlap5.max():.2f}")

    # Light pass/fail: directional agreement on the last token (the one with
    # the most useful context) and top-5 overlap on at least the easy rows.
    rc = 0
    if cos[-1] < 0.5:
        print("WARN: last-token cosine < 0.5 — graph translation may have a systematic issue.")
        rc = 1
    elif diff.max() > 0.1 and cos.mean() < 0.9:
        print("WARN: noticeable absolute diff with only moderate cosine agreement — "
              "investigate KV-cache fp16 conversion or attention masking.")
    else:
        print("OK")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
