#!/usr/bin/env python3
"""Static inspector for ONNX models produced by ``gguf.gdump_to_onnx``.

For every initializer this surfaces:
  - the in-ONNX dtype the translator cast to (``weight_dtype``),
  - the *original* ggml dtype (parsed from the ``original_ggml_type=...``
    annotation that ``Translator._add_weight_initializer`` stamps on the
    initializer's ``doc_string``),
  - shape, element count, and in-ONNX byte size.

It also exposes the model-level ``metadata_props`` block stamped by the
translator (``llama_onnx.original_dtype_counts``, etc.) and a quick
observed-vs-recorded histogram diff.

Pure ``onnx`` + stdlib — no ``onnxruntime``, so this works on artefacts that
reference external ``.onnx.data`` (we open with ``load_external_data=False``
by default so we don't have to actually read the multi-GB blob).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import TensorProto


# Element size in bytes for the ONNX TensorProto dtypes that the translator
# ever emits as a weight initializer. (gdump_to_onnx casts everything to
# either FLOAT or FLOAT16 today; the integer / int64 entries are here for
# the constant-shape initializers that don't carry an annotation.)
_ELEM_BYTES = {
    TensorProto.FLOAT: 4,
    TensorProto.FLOAT16: 2,
    TensorProto.BFLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.INT8: 1, TensorProto.UINT8: 1,
    TensorProto.INT16: 2, TensorProto.UINT16: 2,
    TensorProto.INT32: 4, TensorProto.UINT32: 4,
    TensorProto.INT64: 8, TensorProto.UINT64: 8,
    TensorProto.BOOL: 1,
}


def _onnx_dtype_name(dt: int) -> str:
    # TensorProto enum-int -> name (e.g. 1 -> "FLOAT"). Fall back to int.
    return TensorProto.DataType.Name(dt) if dt in TensorProto.DataType.values() else f"TYPE_{dt}"


class WeightInspector:
    """Read an ONNX file's initializers and recover the
    ``original_ggml_type=...`` annotation stamped by ``gdump_to_onnx``.
    """

    @dataclass
    class Weight:
        name: str
        onnx_dtype: str
        ggml_dtype: str | None
        shape: tuple[int, ...]
        n_elements: int
        n_bytes_onnx: int
        is_external: bool

    def __init__(self, onnx_path: str | Path, *, load_external_data: bool = False):
        self.path = Path(onnx_path)
        # load_external_data=False keeps us from materialising .onnx.data;
        # we only need shapes/dtypes/doc_strings, which all live in the
        # main .onnx file.
        self.model = onnx.load(str(self.path), load_external_data=load_external_data)
        self._weights: list[WeightInspector.Weight] = []
        for init in self.model.graph.initializer:
            shape = tuple(init.dims)
            n = 1
            for d in shape:
                n *= int(d)
            ds = init.doc_string or ""
            ggml = ds.split("original_ggml_type=", 1)[1] if "original_ggml_type=" in ds else None
            self._weights.append(WeightInspector.Weight(
                name=init.name,
                onnx_dtype=_onnx_dtype_name(init.data_type),
                ggml_dtype=ggml,
                shape=shape,
                n_elements=n,
                n_bytes_onnx=n * _ELEM_BYTES.get(init.data_type, 0),
                is_external=(init.data_location == TensorProto.EXTERNAL),
            ))

    def weights(self) -> list["WeightInspector.Weight"]:
        return list(self._weights)

    def metadata(self) -> dict[str, str]:
        return {e.key: e.value for e in self.model.metadata_props}

    def histogram_observed(self) -> dict[str, int]:
        c: Counter[str] = Counter()
        for w in self._weights:
            if w.ggml_dtype is not None:
                c[w.ggml_dtype] += 1
        return dict(c)

    def histogram_metadata(self) -> dict[str, int]:
        raw = self.metadata().get("llama_onnx.original_dtype_counts", "")
        out: dict[str, int] = {}
        for item in raw.split(",") if raw else []:
            k, _, v = item.partition(":")
            if k:
                try:
                    out[k] = int(v)
                except ValueError:
                    pass
        return out

    def pretty(self, *, top: int | None = None, sort_by: str = "n_bytes_onnx") -> str:
        key = {
            "n_bytes_onnx": lambda w: -w.n_bytes_onnx,
            "bytes": lambda w: -w.n_bytes_onnx,
            "name": lambda w: w.name,
            "dtype": lambda w: (w.ggml_dtype or "", -w.n_bytes_onnx),
        }.get(sort_by, lambda w: -w.n_bytes_onnx)
        ws = sorted(self._weights, key=key)
        if top is not None:
            ws = ws[:top]
        # Markdown table.
        lines = [
            "| name | shape | onnx_dtype | ggml_dtype | n_bytes_onnx |",
            "|------|-------|------------|------------|-------------:|",
        ]
        for w in ws:
            shape = "x".join(str(d) for d in w.shape) if w.shape else "()"
            lines.append(f"| {w.name} | {shape} | {w.onnx_dtype} | {w.ggml_dtype or '-'} | {w.n_bytes_onnx} |")
        return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("onnx", type=Path, help="path to a .onnx file")
    ap.add_argument("--top", type=int, default=20, help="rows of the per-tensor table to print (default 20)")
    ap.add_argument("--sort", choices=["name", "bytes", "dtype"], default="bytes")
    ap.add_argument("--summary-only", action="store_true", help="skip the per-tensor table")
    args = ap.parse_args(argv)

    insp = WeightInspector(args.onnx)
    if not args.summary_only:
        print(f"Top {args.top} initializers by {args.sort}:")
        print(insp.pretty(top=args.top, sort_by=("n_bytes_onnx" if args.sort == "bytes" else args.sort)))
        print()

    print("Model metadata_props:")
    md = insp.metadata()
    for k in sorted(md):
        print(f"  {k} = {md[k]}")
    print()

    obs = insp.histogram_observed()
    rec = insp.histogram_metadata()
    keys = sorted(set(obs) | set(rec))
    print("Histogram (observed vs recorded):")
    mismatch = False
    for k in keys:
        o, r = obs.get(k, 0), rec.get(k, 0)
        tag = "" if o == r else "  <-- MISMATCH"
        if o != r:
            mismatch = True
        print(f"  {k:>16}: {o:5d} (observed) vs {r:5d} (recorded){tag}")
    print()

    annotated = [w for w in insp.weights() if w.ggml_dtype is not None]
    total_bytes = sum(w.n_bytes_onnx for w in annotated)
    print(f"Totals: {len(annotated)} annotated weight initializers, "
          f"{len(insp.weights())} total initializers, "
          f"{total_bytes:,} bytes (annotated, in-ONNX dtype).")

    return 1 if mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
