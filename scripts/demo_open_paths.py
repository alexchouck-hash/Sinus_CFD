#!/usr/bin/env python3
"""
Demo-quality dual centerlines: most-open path naris→trachea + inferred open space.

Uses classical EDT-weighted geodesics (prefer wide lumen) and soft L/R symmetry.

Writes / updates:
  - outputs/<case>/<case>_open_paths.json
  - passage.json centerline_left/right_mm (viewer magenta lines)
  - open_space mask + STL
  - optional passage_lumen from open space for demo domain

Usage:
  py -3.12 scripts/demo_open_paths.py --case VisibleHuman_Head
  py -3.12 scripts/demo_open_paths.py --case VisibleHuman_Head --symmetry 0.4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sinus_cfd.open_path import compute_dual_most_open_paths  # noqa: E402
from sinus_cfd.pipeline import _mask_to_mesh  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="VisibleHuman_Head")
    p.add_argument("--outputs-root", type=Path, default=REPO / "outputs")
    p.add_argument("--power", type=float, default=2.0, help="Most-open cost power 1/r^p")
    p.add_argument(
        "--symmetry",
        type=float,
        default=0.35,
        help="Soft L/R symmetry blend 0..1 (0=independent, 1=pure mirrors)",
    )
    p.add_argument("--radius-scale", type=float, default=1.2)
    p.add_argument(
        "--domain",
        choices=("passage", "airway", "cavity_union"),
        default="cavity_union",
        help="Air mask used as navigable domain",
    )
    args = p.parse_args()

    case_dir = args.outputs_root / args.case
    if not case_dir.is_dir():
        print(f"Missing {case_dir}", file=sys.stderr)
        return 1

    # Domain mask
    candidates = {
        "passage": case_dir / f"{args.case}_passage_lumen.nrrd",
        "airway": case_dir / f"{args.case}_airway_mask.nrrd",
        "cavity_union": None,  # left|right if present
    }
    left_p = case_dir / f"{args.case}_cavity_left.nrrd"
    right_p = case_dir / f"{args.case}_cavity_right.nrrd"
    ref = None
    if args.domain == "cavity_union" and left_p.is_file() and right_p.is_file():
        left = sitk.ReadImage(str(left_p))
        right = sitk.ReadImage(str(right_p))
        air = (sitk.GetArrayFromImage(left) > 0) | (sitk.GetArrayFromImage(right) > 0)
        ref = left
        print(f"[{args.case}] domain=L|R cavities ({int(air.sum())} vx)")
    else:
        path = candidates.get(args.domain) or candidates["airway"]
        if not path.is_file():
            path = case_dir / f"{args.case}_airway_mask.nrrd"
        if not path.is_file():
            print("No domain mask found", file=sys.stderr)
            return 1
        ref = sitk.ReadImage(str(path))
        air = sitk.GetArrayFromImage(ref) > 0
        print(f"[{args.case}] domain={path.name} ({int(air.sum())} vx)")

    spacing = tuple(float(v) for v in ref.GetSpacing())
    origin = tuple(float(v) for v in ref.GetOrigin())

    # Ports
    nares = json.loads((case_dir / f"{args.case}_nares.json").read_text(encoding="utf-8"))
    bc = json.loads(
        (case_dir / f"{args.case}_boundary_conditions.json").read_text(encoding="utf-8")
    )
    left_mm = right_mm = None
    for pt in nares.get("naris_points") or []:
        if pt.get("name") == "left_nostril":
            left_mm = pt["center_mm"]
        elif pt.get("name") == "right_nostril":
            right_mm = pt["center_mm"]
    if left_mm is None or right_mm is None:
        for port in bc.get("ports", []):
            if port.get("name") == "left_nostril":
                left_mm = port["center_mm"]
            elif port.get("name") == "right_nostril":
                right_mm = port["center_mm"]
    trachea = None
    for port in bc.get("ports", []):
        if port.get("role") == "outlet" or port.get("name") == "trachea":
            trachea = port["center_mm"]
            break
    if not left_mm or not right_mm or not trachea:
        print("Need left/right naris + trachea in nares.json / BC", file=sys.stderr)
        return 1

    print(f"[{args.case}] most-open dual paths (symmetry={args.symmetry})…")
    result = compute_dual_most_open_paths(
        air=air,
        left_naris_mm=left_mm,
        right_naris_mm=right_mm,
        trachea_mm=trachea,
        spacing_xyz=spacing,
        origin_xyz=origin,
        case_id=args.case,
        power=args.power,
        symmetry_blend=args.symmetry,
        radius_scale=args.radius_scale,
    )
    for n in result.notes:
        print(f"  note: {n}")
    print(
        f"  lengths L={result.length_left_mm:.1f} mm  R={result.length_right_mm:.1f} mm  "
        f"open_space={int(result.open_space.sum())} vx"
    )

    # JSON meta (no huge arrays)
    meta_path = case_dir / f"{args.case}_open_paths.json"
    meta_path.write_text(json.dumps(result.to_meta(), indent=2), encoding="utf-8")
    print(f"  wrote {meta_path.name}")

    # Open space mask + STL
    os_img = sitk.GetImageFromArray(result.open_space.astype(np.uint8))
    os_img.CopyInformation(ref)
    sitk.WriteImage(os_img, str(case_dir / f"{args.case}_open_space.nrrd"))
    try:
        mesh = _mask_to_mesh(result.open_space, spacing, origin)
        if len(mesh.faces) > 25000:
            try:
                mesh = mesh.simplify_quadric_decimation(25000)
            except Exception:
                pass
        mesh.export(case_dir / f"{args.case}_open_space.stl")
        # Also as airway for viewer combined mesh option
        mesh.export(case_dir / f"{args.case}_airway.stl")
        print("  wrote open_space.stl + airway.stl")
    except Exception as exc:
        print(f"  STL warning: {exc}")

    # Update passage.json for dual magenta lines in viewer
    passage_path = case_dir / f"{args.case}_passage.json"
    if passage_path.is_file():
        passage = json.loads(passage_path.read_text(encoding="utf-8"))
    else:
        passage = {"case_id": args.case}
    passage["centerline_left_mm"] = result.centerline_left_mm
    passage["centerline_right_mm"] = result.centerline_right_mm
    passage["centerline_mm"] = result.centerline_mid_mm
    passage["open_path_method"] = result.method
    passage["open_path_notes"] = result.notes
    passage["x_midplane_mm"] = result.x_midplane_mm
    metrics = passage.get("metrics") or {}
    metrics["centerline_length_mm"] = 0.5 * (
        result.length_left_mm + result.length_right_mm
    )
    metrics["centerline_left_mm"] = result.length_left_mm
    metrics["centerline_right_mm"] = result.length_right_mm
    metrics["open_space_voxels"] = int(result.open_space.sum())
    passage["metrics"] = metrics
    passage_path.write_text(json.dumps(passage, indent=2), encoding="utf-8")
    print(f"  updated {passage_path.name} (dual centerlines for viewer)")

    # Optional: use open space as passage lumen for demo domain
    sitk.WriteImage(os_img, str(case_dir / f"{args.case}_passage_lumen.nrrd"))
    sitk.WriteImage(os_img, str(case_dir / f"{args.case}_airway_mask.nrrd"))
    print("  updated passage_lumen + airway_mask from open-space tubes")
    print("Done. Reload Streamlit: Clear cache & reload data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
