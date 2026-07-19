#!/usr/bin/env python3
"""
Stage 4 virtual surgery: edit the airway, compare geometry pre/post (no CFD),
and write the edited label map so the CFD pipeline can re-run on it.

Applies a parameterized surgical edit (turbinate reduction / septoplasty) to a
NasalSeg case's label map, then reports the pre/post change in the Stage-2
geometry metrics — per-side volume, minimal cross-sectional area (MCA), and the
L/R asymmetry ratio. The edited label NRRD is written so
`scripts/process_case.py --label <edited>` (then export -> scaffold -> solve)
produces the post-operative CFD for a full pre/post resistance + cooling
comparison.

Usage:
  # geometry-only pre/post (works today, no Docker)
  py -3.12 scripts/virtual_surgery.py --case P001 --procedure turbinate_reduction --side left --depth-mm 3

  # let the tool pick the obstructed side from the Stage-2 L/R ratio
  py -3.12 scripts/virtual_surgery.py --case P001 --procedure septoplasty --side auto --depth-mm 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sinus_cfd.passage_metrics import analyze_bilateral  # noqa: E402
from sinus_cfd.pipeline import resolve_nasalseg_case  # noqa: E402
from sinus_cfd.virtual_surgery import apply  # noqa: E402


def _fmt_delta(pre: float, post: float, unit: str, improve_up: bool) -> str:
    d = post - pre
    pct = (d / pre * 100.0) if pre else float("nan")
    arrow = "↑" if d > 0 else "↓" if d < 0 else "→"
    better = (d > 0) == improve_up
    tag = "better" if (d != 0 and better) else ("worse" if d != 0 else "")
    return f"{pre:.1f} → {post:.1f} {unit}  ({arrow}{abs(pct):.0f}% {tag})"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="P001")
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--label", type=Path, default=None, help="explicit label NRRD (else NasalSeg case)")
    p.add_argument("--image", type=Path, default=None, help="explicit image NRRD (for spacing)")
    p.add_argument("--procedure", choices=("turbinate_reduction", "septoplasty"), required=True)
    p.add_argument("--side", default="left", help="left | right | auto (auto = obstructed side from L/R MCA)")
    p.add_argument("--depth-mm", type=float, default=3.0)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    if args.label is not None:
        label_path = args.label
        image_path = args.image
    else:
        image_path, label_path = resolve_nasalseg_case(args.data_root, args.case)

    image = sitk.ReadImage(str(image_path)) if image_path else sitk.ReadImage(str(label_path))
    spacing_xyz = tuple(float(v) for v in image.GetSpacing())
    label_img = sitk.ReadImage(str(label_path))
    label = sitk.GetArrayFromImage(label_img)

    pre = analyze_bilateral(label, spacing_xyz, args.case, mask_source="labels")

    # Resolve side.
    side = args.side
    if side == "auto":
        side = pre.more_obstructed_side if pre.more_obstructed_side in ("left", "right") else "left"
        print(f"[{args.case}] auto-selected obstructed side: {side} (L/R MCA ratio {pre.mca_ratio:.2f})")

    result = apply(args.procedure, label, spacing_xyz, side=side, depth_mm=args.depth_mm)
    for n in result.notes:
        print(f"  {n}")

    post = analyze_bilateral(result.edited_label, spacing_xyz, args.case, mask_source="labels")

    out_dir = args.output_dir or (REPO_ROOT / "outputs" / f"{args.case}_postop_{args.procedure}_{side}")
    out_dir.mkdir(parents=True, exist_ok=True)
    edited_img = sitk.GetImageFromArray(result.edited_label.astype(np.uint8))
    edited_img.CopyInformation(label_img)
    edited_path = out_dir / f"{args.case}_postop_label.nrrd"
    sitk.WriteImage(edited_img, str(edited_path))

    # Pre/post comparison table.
    print(f"\n[{args.case}] {args.procedure} on {side}, depth {args.depth_mm:.1f} mm — geometry pre/post")
    print(f"  air added: {result.added_volume_ml:.2f} mL")
    ps, po = pre.__dict__, post.__dict__
    for key, lbl in (("left", "Left "), ("right", "Right")):
        a, b = ps[key], po[key]
        if a["present"] and b["present"]:
            print(f"  {lbl} volume : {_fmt_delta(a['volume_ml'], b['volume_ml'], 'mL', True)}")
            print(f"  {lbl} MCA    : {_fmt_delta(a['mca_mm2'], b['mca_mm2'], 'mm²', True)}")
    if not (isinstance(pre.mca_ratio, float) and np.isnan(pre.mca_ratio)):
        print(f"  L/R MCA ratio: {pre.mca_ratio:.2f} → {post.mca_ratio:.2f}  "
              f"(1.0 = symmetric; {'more' if post.mca_ratio > pre.mca_ratio else 'less'} balanced)")

    report = {
        "case": args.case,
        "procedure": args.procedure,
        "side": side,
        "depth_mm": args.depth_mm,
        "air_added_ml": result.added_volume_ml,
        "pre": pre.to_dict(),
        "post": post.to_dict(),
    }
    report_path = out_dir / f"{args.case}_virtual_surgery.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  wrote {edited_path.name} (feed to process_case --label for post-op CFD)")
    print(f"  wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
