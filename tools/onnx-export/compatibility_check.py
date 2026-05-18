#!/usr/bin/env python3
"""Pre-flight ggml -> ONNX op-coverage checker.

Answers the question "will my GGUF export cleanly to ONNX?" *without*
producing the multi-GB ONNX artefact. We invoke the existing
``llama-onnx-export-dump`` C++ binary, parse the resulting ``.gdump`` with
``gguf.gdump``, and cross-reference every ggml op in the graph against the
handler table in ``gguf.gdump_to_onnx._OP_HANDLERS`` (plus the UNARY / GLU
sub-variant tables hand-curated below).

Exit code is 0 if every op in the graph has a handler, 1 otherwise (so the
tool drops cleanly into CI / shell pipelines).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

# Make the in-tree gguf-py importable when running from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))

from gguf import gdump  # noqa: E402
from gguf.gdump_to_onnx import _OP_HANDLERS  # noqa: E402


# Sub-variants of UNARY that ``_h_unary`` in gguf-py/gguf/gdump_to_onnx.py
# explicitly handles. Source of truth: the ``if uop == ...`` chain and the
# ``table = {...}`` block at the bottom of that function. UNARY coverage is
# currently 20/20; if a new GgmlUnaryOp ever lands, add it here.
_UNARY_SUPPORTED: frozenset[int] = frozenset({
    gdump.GgmlUnaryOp.SILU.value,
    gdump.GgmlUnaryOp.GELU.value,
    gdump.GgmlUnaryOp.GELU_ERF.value,
    gdump.GgmlUnaryOp.GELU_QUICK.value,
    gdump.GgmlUnaryOp.STEP.value,
    gdump.GgmlUnaryOp.XIELU.value,
    gdump.GgmlUnaryOp.TRUNC.value,
    gdump.GgmlUnaryOp.RELU.value,
    gdump.GgmlUnaryOp.SIGMOID.value,
    gdump.GgmlUnaryOp.TANH.value,
    gdump.GgmlUnaryOp.NEG.value,
    gdump.GgmlUnaryOp.EXP.value,
    gdump.GgmlUnaryOp.HARDSWISH.value,
    gdump.GgmlUnaryOp.HARDSIGMOID.value,
    gdump.GgmlUnaryOp.ABS.value,
    gdump.GgmlUnaryOp.SGN.value,
    gdump.GgmlUnaryOp.ELU.value,
    gdump.GgmlUnaryOp.FLOOR.value,
    gdump.GgmlUnaryOp.CEIL.value,
    gdump.GgmlUnaryOp.ROUND.value,
})

# Sub-variants of GLU that ``_h_glu`` in gguf-py/gguf/gdump_to_onnx.py
# explicitly handles. ``GEGLU_QUICK`` is currently *not* implemented.
_GLU_SUPPORTED: frozenset[int] = frozenset({
    gdump.GgmlGluOp.SWIGLU.value,
    gdump.GgmlGluOp.GEGLU.value,
    gdump.GgmlGluOp.GEGLU_ERF.value,
    gdump.GgmlGluOp.REGLU.value,
    gdump.GgmlGluOp.SWIGLU_OAI.value,
})


def _default_dumper() -> Path:
    return Path(__file__).resolve().parents[2] / "build" / "bin" / "llama-onnx-export-dump"


def _run_dumper(dumper: Path, gguf: Path, out: Path, *,
                include_vision: bool, vision_only: bool,
                batch_size: int, ubatch_size: int, ctx_size: int) -> None:
    if not dumper.exists():
        raise FileNotFoundError(f"dumper binary not found: {dumper}")
    cmd = [
        str(dumper), "-m", str(gguf), "-o", str(out),
        "-b", str(batch_size), "-ub", str(ubatch_size), "-c", str(ctx_size),
    ]
    if include_vision:
        cmd.append("--include-vision")
    if vision_only:
        cmd.append("--vision-only")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        sys.stderr.write(res.stdout.decode("utf-8", errors="replace"))
        raise SystemExit(f"dumper exited with status {res.returncode}")


def _classify(dump: gdump.GraphDump) -> tuple[Counter, Counter, Counter, int, int]:
    """Return (op_counts, unary_counts, glu_counts, n_leaves, n_nodes)."""
    op_counts: Counter = Counter()
    unary_counts: Counter = Counter()
    glu_counts: Counter = Counter()
    n_leaves = 0
    n_nodes = 0
    for t in dump.tensors:
        if t.is_leaf:
            n_leaves += 1
            continue
        n_nodes += 1
        op = t.ggml_op
        op_counts[op] += 1
        if op == gdump.GgmlOp.UNARY:
            uop = t.op_params_i32(1)[0]
            unary_counts[uop] += 1
        elif op == gdump.GgmlOp.GLU:
            gop = t.op_params_i32(1)[0]
            glu_counts[gop] += 1
    return op_counts, unary_counts, glu_counts, n_leaves, n_nodes


def _enum_name(cls, val: int) -> str:
    try:
        return cls(val).name
    except ValueError:
        return f"UNKNOWN_{val}"


def _render_table(dump: gdump.GraphDump, op_counts: Counter,
                  unary_counts: Counter, glu_counts: Counter,
                  n_leaves: int, n_nodes: int) -> tuple[str, int]:
    supported_ops = set(_OP_HANDLERS.keys())
    missing = 0
    lines: list[str] = []
    lines.append(f"Architecture: {dump.arch}")
    lines.append(f"n_tokens: {dump.n_tokens}   n_seqs: {dump.n_seqs}")
    lines.append(f"Tensors: {len(dump.tensors)} ({n_leaves} leaves, {n_nodes} nodes)")
    lines.append("")
    lines.append("Op usage (count, status):")
    for op, count in op_counts.most_common():
        name = op.name if isinstance(op, gdump.GgmlOp) else f"UNKNOWN_{op}"
        if op in supported_ops:
            status = "OK"
        else:
            status = "MISSING HANDLER"
            missing += 1
        lines.append(f"   {name:<18} {count:>5}   {status}")

    if unary_counts:
        lines.append("")
        lines.append("UNARY sub-ops:")
        for uop, count in unary_counts.most_common():
            name = _enum_name(gdump.GgmlUnaryOp, uop)
            if uop in _UNARY_SUPPORTED:
                status = "OK"
            else:
                status = "MISSING HANDLER"
                missing += 1
            lines.append(f"   {name:<18} {count:>5}   {status}")

    if glu_counts:
        lines.append("")
        lines.append("GLU sub-ops:")
        for gop, count in glu_counts.most_common():
            name = _enum_name(gdump.GgmlGluOp, gop)
            if gop in _GLU_SUPPORTED:
                status = "OK"
            else:
                status = "MISSING HANDLER"
                missing += 1
            lines.append(f"   {name:<18} {count:>5}   {status}")

    lines.append("")
    if missing == 0:
        lines.append("Verdict: all ops supported — export should succeed.")
    else:
        lines.append(f"Verdict: {missing} missing handler(s) — export will fail.")
        missing_ops = [op for op in op_counts if op not in supported_ops]
        if missing_ops:
            sample = missing_ops[0]
            sample_name = sample.name if isinstance(sample, gdump.GgmlOp) else str(sample)
            lines.append(f"   Add @_op(GgmlOp.{sample_name}) in gguf-py/gguf/gdump_to_onnx.py.")
    return "\n".join(lines), missing


def _render_json(dump: gdump.GraphDump, op_counts: Counter,
                 unary_counts: Counter, glu_counts: Counter,
                 n_leaves: int, n_nodes: int) -> tuple[str, int]:
    supported_ops = set(_OP_HANDLERS.keys())
    ops_out = []
    missing = 0
    for op, count in op_counts.most_common():
        name = op.name if isinstance(op, gdump.GgmlOp) else f"UNKNOWN_{op}"
        ok = op in supported_ops
        if not ok:
            missing += 1
        ops_out.append({"op": name, "count": count, "supported": ok})
    unary_out = []
    for uop, count in unary_counts.most_common():
        ok = uop in _UNARY_SUPPORTED
        if not ok:
            missing += 1
        unary_out.append({"op": _enum_name(gdump.GgmlUnaryOp, uop), "count": count, "supported": ok})
    glu_out = []
    for gop, count in glu_counts.most_common():
        ok = gop in _GLU_SUPPORTED
        if not ok:
            missing += 1
        glu_out.append({"op": _enum_name(gdump.GgmlGluOp, gop), "count": count, "supported": ok})
    payload = {
        "arch": dump.arch,
        "n_tokens": dump.n_tokens,
        "n_seqs": dump.n_seqs,
        "n_tensors": len(dump.tensors),
        "n_leaves": n_leaves,
        "n_nodes": n_nodes,
        "ops": ops_out,
        "unary": unary_out,
        "glu": glu_out,
        "n_missing": missing,
        "supported": missing == 0,
    }
    return json.dumps(payload, indent=2), missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gguf", type=Path, help="path to the source GGUF model")
    ap.add_argument("--dumper", type=Path, default=_default_dumper(),
                    help="path to llama-onnx-export-dump (default: build/bin/...)")
    ap.add_argument("--include-vision", action="store_true",
                    help="also trace the mtmd vision graph")
    ap.add_argument("--vision-only", action="store_true",
                    help="trace ONLY the vision graph (for mmproj-only GGUFs)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ubatch-size", type=int, default=32)
    ap.add_argument("--ctx-size", type=int, default=512)
    ap.add_argument("--keep-dump", type=Path, default=None,
                    help="save the produced .gdump here for inspection")
    ap.add_argument("--summary-only", action="store_true",
                    help="print only the final verdict line")
    ap.add_argument("--json", action="store_true",
                    help="emit JSON instead of the markdown-style table")
    args = ap.parse_args(argv)

    tmpdir = tempfile.mkdtemp(prefix="onnx-compat-")
    dump_path = Path(tmpdir) / "check.gdump"
    try:
        _run_dumper(args.dumper, args.gguf, dump_path,
                    include_vision=args.include_vision,
                    vision_only=args.vision_only,
                    batch_size=args.batch_size,
                    ubatch_size=args.ubatch_size,
                    ctx_size=args.ctx_size)
        # When --vision-only is set the LM dump is the vision graph itself
        # (the dumper writes the requested -o file with the vision payload).
        # When --include-vision is set without --vision-only there's also a
        # sibling .vision.gdump; we walk both to give a full picture.
        dumps_to_check: list[Path] = [dump_path]
        sibling = dump_path.with_suffix(".vision.gdump")
        if args.include_vision and not args.vision_only and sibling.exists():
            dumps_to_check.append(sibling)

        rendered: list[str] = []
        total_missing = 0
        for dp in dumps_to_check:
            dump = gdump.load(dp)
            op_counts, unary_counts, glu_counts, n_leaves, n_nodes = _classify(dump)
            if args.json:
                text, missing = _render_json(dump, op_counts, unary_counts, glu_counts,
                                             n_leaves, n_nodes)
            else:
                text, missing = _render_table(dump, op_counts, unary_counts, glu_counts,
                                              n_leaves, n_nodes)
            rendered.append(text)
            total_missing += missing

        if args.keep_dump is not None:
            args.keep_dump.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dump_path, args.keep_dump)

        if args.summary_only:
            if total_missing == 0:
                print("Verdict: all ops supported — export should succeed.")
            else:
                print(f"Verdict: {total_missing} missing handler(s) — export will fail.")
        else:
            print("\n\n".join(rendered))

        return 0 if total_missing == 0 else 1
    finally:
        try:
            for child in Path(tmpdir).iterdir():
                child.unlink()
            os.rmdir(tmpdir)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
