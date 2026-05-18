#!/usr/bin/env python3
"""Layer-by-layer drift investigator for the gdump -> ONNX pipeline.

Setup:
  1. Run ``llama-onnx-export-dump`` with ``--fixture`` to capture llama.cpp's
     reference logits *and* a bunch of named intermediate tensors during
     ``llama_decode`` (via a ggml eval-callback).
  2. Re-export the .gdump to ONNX with a list of those same intermediate
     names exposed as additional ONNX outputs (via gdump_to_onnx's new
     ``extra_output_names`` parameter).
  3. Feed the fixture's inputs through onnxruntime and read the extra
     outputs back. Compare against the captured ggml values.

This makes it possible to bisect the small (~0.5 max-abs) drift between
llama.cpp and onnxruntime on Llama 3.2 1B Q4_K_M: we can ask "at which
op in which layer does the divergence first exceed threshold T?".
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path("/Users/tom/Documents/Projects/llama-onnx.cpp")
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from gguf import gdump
from gguf.gdump_to_onnx import convert as gdump_to_onnx
from gguf.gdump_fixture import load as load_fx, synthesize_inputs


WIKITEXT = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
    "in Paris, France. It is named after the engineer Gustave Eiffel, whose "
    "company designed and built the tower from 1887 to 1889. Locally nicknamed "
    "\"La dame de fer\", it was constructed as the centrepiece of the 1889 World's "
    "Fair and to crown the centennial anniversary of the French Revolution."
)


# Names we capture per layer (must match the regex set in the C++ dumper).
PER_LAYER_PROBES = [
    "attn_norm",
    "Qcur", "Kcur", "Vcur",
    "attn_out",
    "ffn_inp",
    "ffn_norm",
    "ffn_out",
    "l_out",
]
GLOBAL_PROBES = ["result_norm", "result_output"]


def collect_probe_names(n_layers: int) -> list[str]:
    """Generate the full list of intermediate names to compare."""
    names: list[str] = []
    for il in range(n_layers):
        for p in PER_LAYER_PROBES:
            names.append(f"{p}-{il}")
    names.extend(GLOBAL_PROBES)
    return names


def n_layers_from_dump(d: gdump.GraphDump) -> int:
    """Infer the layer count by counting attn_norm-<il> tensors."""
    seen = set()
    for t in d.tensors:
        if t.name.startswith("attn_norm-"):
            seen.add(t.name)
    return len(seen)


def metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Return a dict of comparison metrics. Both arrays must broadcast."""
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    diff = np.abs(a - b)
    rel = diff / (np.maximum(np.abs(a), np.abs(b)) + 1e-30)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return {
        "n": int(a.size),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float(rel.max()),
        "mean_rel": float(rel.mean()),
        "cos": cos,
        "rms_a": float(np.sqrt((a * a).mean())),
        "rms_b": float(np.sqrt((b * b).mean())),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("gguf", type=Path)
    p.add_argument("--prompt", type=str, default=WIKITEXT)
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--workdir", type=Path, default=None)
    p.add_argument("--weight-dtype", default="float32", choices=["float16", "float32"])
    p.add_argument("--dumper", type=Path,
                   default=REPO_ROOT / "build" / "bin" / "llama-onnx-export-dump")
    p.add_argument("--skip-dump", action="store_true",
                   help="re-use existing .gdump/.gfxt if present")
    p.add_argument("--threshold", type=float, default=1e-3,
                   help="max-abs threshold for first-divergence reporting")
    p.add_argument("--probes", type=str, default=None,
                   help="comma-separated subset of probe stems "
                        "(e.g. 'attn_norm,Qcur,Kcur,ffn_norm'). Overrides PER_LAYER_PROBES.")
    p.add_argument("--layers", type=str, default=None,
                   help="comma-separated list of layer ids to probe (default: all)")
    args = p.parse_args()

    label = args.label or args.gguf.stem
    if args.workdir is None:
        args.workdir = Path(f"/tmp/llama-onnx-{label}-layerdiff")
    args.workdir.mkdir(parents=True, exist_ok=True)
    stem = label
    dump_path = args.workdir / f"{stem}.gdump"
    fx_path = args.workdir / f"{stem}.gfxt"
    onnx_path = args.workdir / f"{stem}.layerdiff.onnx"

    # ---- 1. dump + fixture (with intermediate capture) ---------------------
    if not args.skip_dump or not dump_path.exists() or not fx_path.exists():
        print(f"[1/4] dumping graph + fixture (with eval-callback captures)")
        env = os.environ.copy()
        env.setdefault("LD_LIBRARY_PATH", str(args.dumper.parent))
        subprocess.run(
            [str(args.dumper),
             "-m", str(args.gguf),
             "-o", str(dump_path),
             "--fixture", str(fx_path),
             "--fixture-prompt", args.prompt],
            env=env, check=True)
    else:
        print(f"[1/4] re-using existing dump/fixture")

    d = gdump.load(dump_path)
    fx = load_fx(fx_path)
    n_layers = n_layers_from_dump(d)
    print(f"      arch={d.arch} n_layers={n_layers} n_tokens={fx.n_tokens} "
          f"captures={len(fx.intermediates)}")

    if args.probes:
        per_layer_probes = [s.strip() for s in args.probes.split(",") if s.strip()]
    else:
        per_layer_probes = PER_LAYER_PROBES

    if args.layers:
        layer_ids = [int(s) for s in args.layers.split(",") if s.strip()]
    else:
        layer_ids = list(range(n_layers))

    want_names: list[str] = []
    for il in layer_ids:
        for p_ in per_layer_probes:
            want_names.append(f"{p_}-{il}")
    want_names.extend(GLOBAL_PROBES)

    # ---- 2. translate to ONNX with extra outputs --------------------------
    print(f"[2/4] translating to ONNX (weight_dtype={args.weight_dtype}, "
          f"extra outputs={len(want_names)})")
    gdump_to_onnx(dump_path, onnx_path,
                  weight_dtype=args.weight_dtype,
                  extra_output_names=want_names)

    # ---- 3. run ONNX -------------------------------------------------------
    print(f"[3/4] loading ONNX and running")
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3  # ERROR
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    feeds = synthesize_inputs(d, fx, [i.name for i in sess.get_inputs()])
    want_onnx_names = [f"dbg__{n.replace('-', '-')}" for n in want_names]  # match the safe-name pattern
    # Filter to the outputs that actually exist (some may not have been resolvable)
    avail_outputs = {o.name for o in sess.get_outputs()}
    want_onnx_names = [n for n in want_onnx_names if n in avail_outputs]
    full_outs = ["logits"] + want_onnx_names
    full_outs = [n for n in full_outs if n in avail_outputs]
    run_outs = sess.run(full_outs, feeds)
    onnx_by_name = dict(zip(full_outs, run_outs))

    # ---- 4. compare per layer / per probe ---------------------------------
    print(f"[4/4] comparing each captured intermediate vs ONNX side\n")

    # Build lookup: ggml name -> intermediate.data
    ref_by_name: dict[str, np.ndarray] = {}
    for inter in fx.intermediates:
        ref_by_name[inter.name] = inter.data

    rows = []
    for name in want_names:
        onnx_name = "dbg__" + "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        if onnx_name not in onnx_by_name:
            continue
        ref = ref_by_name.get(name)
        if ref is None:
            continue
        cur = onnx_by_name[onnx_name]
        # Shapes: the ggml capture is row-major with trailing 1s dropped.
        # The ONNX side does the same (via _ggml_shape_to_logical), so the
        # shapes should already line up. If not, try a flat compare.
        if cur.shape != ref.shape:
            try:
                cur_flat = cur.reshape(ref.shape)
                cur = cur_flat
            except ValueError:
                # Could be a permute/view artifact — fall back to element-wise
                # over the flat arrays of equal element count.
                if cur.size != ref.size:
                    print(f"  SKIP {name}: shape mismatch {cur.shape} vs {ref.shape} "
                          f"(elements {cur.size} vs {ref.size})")
                    continue
        m = metrics(ref, cur)
        rows.append((name, ref.shape, m))

    # Pretty-print
    print(f"{'name':<22}  {'shape':<22}  {'rms_ref':>9} {'rms_onnx':>9}  "
          f"{'max|Δ|':>9}  {'mean|Δ|':>9}  {'cos':>7}  status")

    # Per-layer block ordering
    def layer_of(name: str) -> int:
        if "-" in name and name.split("-")[-1].isdigit():
            return int(name.rsplit("-", 1)[1])
        return 10**6

    def rank_within_layer(name: str) -> int:
        stem = name.rsplit("-", 1)[0]
        ranks = {p: i for i, p in enumerate(PER_LAYER_PROBES + GLOBAL_PROBES)}
        return ranks.get(stem, ranks.get(name, 999))

    rows.sort(key=lambda r: (layer_of(r[0]), rank_within_layer(r[0])))

    first_bad = None
    for name, shape, m in rows:
        flag = ""
        if m["max_abs"] > args.threshold:
            flag = "  ** > threshold"
            if first_bad is None:
                first_bad = (name, m)
        print(f"{name:<22}  {str(list(shape)):<22}  "
              f"{m['rms_a']:>9.4f} {m['rms_b']:>9.4f}  "
              f"{m['max_abs']:>9.3e}  {m['mean_abs']:>9.3e}  "
              f"{m['cos']:>7.4f}{flag}")

    print()
    if first_bad:
        name, m = first_bad
        print(f"FIRST DIVERGENCE > {args.threshold}: {name}")
        print(f"  max|Δ|={m['max_abs']:.3e}  mean|Δ|={m['mean_abs']:.3e}  "
              f"cos={m['cos']:.6f}  rms_ref={m['rms_a']:.4f}")
    else:
        print(f"all probes within {args.threshold}")

    # Final logits comparison too
    if "logits" in onnx_by_name:
        logit_metrics = metrics(fx.logits, onnx_by_name["logits"])
        print()
        print(f"FINAL LOGITS: max|Δ|={logit_metrics['max_abs']:.3e}  "
              f"mean|Δ|={logit_metrics['mean_abs']:.3e}  "
              f"cos={logit_metrics['cos']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
