#!/usr/bin/env python3
"""
Export solid air body + open-port STL patches for OpenFOAM.

Requires a processed case with passage masks (run analyze_passage.py first).
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

from sinus_cfd.openfoam_export import export_openfoam_geometry  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="VisibleHuman_Head")
    p.add_argument("--outputs-root", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument(
        "--no-sinuses",
        action="store_true",
        help="Solid air body = passage only (exclude connected sinus air)",
    )
    args = p.parse_args()

    case_dir = args.outputs_root / args.case
    lumen_p = case_dir / f"{args.case}_passage_lumen.nrrd"
    if not lumen_p.is_file():
        lumen_p = case_dir / f"{args.case}_airway_mask.nrrd"
    inlet_p = case_dir / f"{args.case}_passage_inlet_open.nrrd"
    outlet_p = case_dir / f"{args.case}_passage_outlet_open.nrrd"
    all_air_p = case_dir / f"{args.case}_all_interior_air.nrrd"
    bc_p = case_dir / f"{args.case}_boundary_conditions.json"

    for req in (lumen_p, inlet_p, outlet_p):
        if not req.is_file():
            raise SystemExit(
                f"Missing {req.name}. Run:\n"
                f"  py -3.12 scripts/analyze_passage.py --case {args.case}"
            )

    lumen_img = sitk.ReadImage(str(lumen_p))
    lumen = sitk.GetArrayFromImage(lumen_img).astype(bool)
    inlet = sitk.GetArrayFromImage(sitk.ReadImage(str(inlet_p))).astype(bool)
    outlet = sitk.GetArrayFromImage(sitk.ReadImage(str(outlet_p))).astype(bool)
    all_air = None
    if all_air_p.is_file() and not args.no_sinuses:
        all_air = sitk.GetArrayFromImage(sitk.ReadImage(str(all_air_p))).astype(bool)

    spacing = tuple(float(v) for v in lumen_img.GetSpacing())
    origin = tuple(float(v) for v in lumen_img.GetOrigin())

    # Optional L/R split using BC centers
    left_m = right_m = None
    if bc_p.is_file():
        bc = json.loads(bc_p.read_text(encoding="utf-8"))
        # split inlet open by x of port centers in index space
        sx, sy, sz = spacing
        ox, oy, oz = origin
        for port in bc.get("ports", []):
            if port.get("role") != "inlet":
                continue
            cx, cy, cz = port["center_mm"]
            # only need x for split later — openfoam_export does median split if None

    result = export_openfoam_geometry(
        case_id=args.case,
        output_dir=case_dir,
        lumen=lumen,
        inlet_open=inlet,
        outlet_open=outlet,
        spacing=spacing,
        origin=origin,
        all_interior_air=all_air,
        include_sinuses=not args.no_sinuses,
        reference_image=lumen_img,
        left_inlet_mask=left_m,
        right_inlet_mask=right_m,
    )
    print(f"[{args.case}] OpenFOAM geometry → {result.out_dir}")
    print(f"  solid air volume: {result.solid_air_volume_ml:.2f} mL")
    for name, fname in result.patches.items():
        print(f"  patch {name}: {fname}")
    for n in result.notes:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
