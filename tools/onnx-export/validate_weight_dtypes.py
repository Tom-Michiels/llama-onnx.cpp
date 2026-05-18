#!/usr/bin/env python3
"""End-to-end validator for the GGUF -> ONNX dtype-metadata path.

For one or more (gguf, onnx) pairs, this checks:

  (1) Every weight initializer carries an ``original_ggml_type=<TYPE>``
      annotation, and ``<TYPE>`` matches the corresponding GGUF tensor's
      actual ggml type.
  (2) The model-level ``llama_onnx.original_dtype_counts`` histogram
      matches the per-initializer histogram.
  (3) Every initializer's ONNX dtype matches the chosen ``--weight-dtype``
      (FLOAT for float32, FLOAT16 for float16).
  (4) Every weight tensor in the source GGUF (modulo tensors the cgraph
      doesn't reference, which is expected) appears as an ONNX initializer.

Each check is reported independently — a single failure does not abort.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# Make the in-tree gguf-py importable when running from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))

from gguf import GGUFReader  # noqa: E402
from gguf.constants import GGMLQuantizationType  # noqa: E402

from inspect_weights import WeightInspector  # noqa: E402


# The translator falls back to ``type_<int>`` when its compact ``GgmlType``
# enum doesn't list a numeric value (see gdump_to_onnx._add_weight_initializer
# and gdump.py:class GgmlType). To compare against the full GGUF type table
# we accept either spelling.
_FALLBACK_RE = re.compile(r"^type_(\d+)$")


def _parse_ggml_label(label: str) -> int | None:
    """Map the translator's per-initializer label to a numeric ggml type id."""
    if label is None:
        return None
    m = _FALLBACK_RE.match(label)
    if m:
        return int(m.group(1))
    try:
        return int(GGMLQuantizationType[label])
    except KeyError:
        return None


def _strip_init_suffix(name: str) -> str:
    """Translator's ``_fresh`` appends ``_<counter>`` to each initializer
    name. The bit before the *final* underscore is (usually) the original
    GGUF tensor name."""
    return name.rsplit("_", 1)[0]


def validate(gguf_path: Path, onnx_path: Path, *, expect_weight_dtype: str) -> bool:
    print(f"\n=== {onnx_path.name} (source: {gguf_path.name}) ===")
    insp = WeightInspector(onnx_path)
    md = insp.metadata()
    print(f"  metadata: arch={md.get('llama_onnx.architecture')}  "
          f"weight_dtype={md.get('llama_onnx.weight_dtype')}  "
          f"counts={md.get('llama_onnx.original_dtype_counts')}")

    reader = GGUFReader(str(gguf_path))
    gguf_types: dict[str, int] = {t.name: int(t.tensor_type) for t in reader.tensors}

    annotated = [w for w in insp.weights() if w.ggml_dtype is not None]
    all_pass = True

    # (1) Per-initializer annotation matches the GGUF tensor's actual type.
    mismatches: list[str] = []
    unknown_label: list[str] = []
    not_in_gguf: list[str] = []
    for w in annotated:
        tid = _parse_ggml_label(w.ggml_dtype)
        if tid is None:
            unknown_label.append(f"{w.name}: {w.ggml_dtype!r}")
            continue
        base = _strip_init_suffix(w.name)
        gt = gguf_types.get(base)
        if gt is None:
            # Some initializers are pre-transposed copies (suffix ``_T``) of
            # the same GGUF weight; trim that too before complaining.
            base2 = base.rsplit("_T", 1)[0] if base.endswith("_T") else base
            gt = gguf_types.get(base2)
        if gt is None:
            not_in_gguf.append(w.name)
            continue
        if gt != tid:
            mismatches.append(f"{w.name}: annotated={w.ggml_dtype} (id={tid}), gguf={GGMLQuantizationType(gt).name} (id={gt})")
    ok1 = not mismatches and not unknown_label
    print(f"  [1] per-initializer annotation matches GGUF:        {'PASS' if ok1 else 'FAIL'}"
          f"  ({len(annotated)} weights checked, "
          f"{len(mismatches)} mismatches, {len(unknown_label)} unparseable, "
          f"{len(not_in_gguf)} not-found-in-gguf)")
    for m in mismatches[:5]:
        print(f"      mismatch: {m}")
    for u in unknown_label[:5]:
        print(f"      unparseable label: {u}")
    all_pass &= ok1

    # (2) Model-level histogram matches per-initializer histogram.
    obs = insp.histogram_observed()
    rec = insp.histogram_metadata()
    ok2 = obs == rec
    print(f"  [2] model histogram matches per-init histogram:    {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        for k in sorted(set(obs) | set(rec)):
            if obs.get(k, 0) != rec.get(k, 0):
                print(f"      {k}: observed={obs.get(k,0)} recorded={rec.get(k,0)}")
    all_pass &= ok2

    # (3) Every initializer's ONNX dtype matches --weight-dtype.
    want = {"float32": "FLOAT", "float16": "FLOAT16"}[expect_weight_dtype]
    bad_dtype = [w for w in annotated if w.onnx_dtype != want]
    ok3 = not bad_dtype
    print(f"  [3] all weight initializers are ONNX {want}:        {'PASS' if ok3 else 'FAIL'}"
          f"  ({len(bad_dtype)} divergent)")
    for w in bad_dtype[:3]:
        print(f"      {w.name}: {w.onnx_dtype}")
    all_pass &= ok3

    # (4) Every weight tensor in the GGUF has at least one initializer.
    # (Modulo tensors the cgraph never references — e.g. some rope_freqs
    # variants — those are reported as info, not failures.)
    annotated_bases: set[str] = set()
    for w in annotated:
        b = _strip_init_suffix(w.name)
        annotated_bases.add(b)
        if b.endswith("_T"):
            annotated_bases.add(b[:-2])
    missing = [name for name in gguf_types if name not in annotated_bases]
    ok4 = not missing
    print(f"  [4] every GGUF tensor present as initializer:      {'PASS' if ok4 else 'INFO'}"
          f"  ({len(gguf_types)} gguf tensors, {len(missing)} not exported)")
    for m in missing[:10]:
        gt = gguf_types[m]
        print(f"      not exported: {m} ({GGMLQuantizationType(gt).name})")
    # Treat (4) as informational rather than fatal: cgraph may legitimately
    # skip some tensors (e.g. rope_freqs.weight is computed at runtime).
    return all_pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pairs", nargs="+",
                    help="one or more GGUF=ONNX[:weight_dtype] pairs, e.g. "
                         "model.gguf=out.onnx:float32 (weight_dtype defaults to float32)")
    args = ap.parse_args(argv)

    overall = True
    for spec in args.pairs:
        gguf, _, rhs = spec.partition("=")
        onnx, _, wd = rhs.partition(":")
        wd = wd or "float32"
        if wd not in ("float32", "float16"):
            print(f"bad weight_dtype {wd!r} in {spec!r}", file=sys.stderr)
            return 2
        ok = validate(Path(gguf), Path(onnx), expect_weight_dtype=wd)
        overall &= ok

    print("\nOVERALL:", "PASS" if overall else "FAIL")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
