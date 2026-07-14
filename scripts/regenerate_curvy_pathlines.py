#!/usr/bin/env python3
"""
Regenerate dense *curvy* pathlines: **nostrils → trachea**.

Every path starts at a naris and ends at the trachea. Seeds near both
nostrils (plus optional mid-passage seeds that are reoriented naris→trachea).
Trilinear velocity sampling + small steps bend lines around corners.
Optional light swirl for a turbulent *look* (demo viz, not LES).

Example:
  py -3.12 scripts/regenerate_curvy_pathlines.py --case VisibleHuman_Head
  py -3.12 scripts/regenerate_curvy_pathlines.py --case VisibleHuman_Head --swirl 0.12 --naris-seeds 120
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

from sinus_cfd.flow_field import compute_curvy_volume_pathlines  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="VisibleHuman_Head")
    ap.add_argument("--volume-seeds", type=int, default=200)
    ap.add_argument("--naris-seeds", type=int, default=220)
    ap.add_argument("--max-lines", type=int, default=330)
    ap.add_argument("--step-mm", type=float, default=0.16)
    ap.add_argument(
        "--swirl",
        type=float,
        default=0.20,
        help="0=pure streamlines; 0.15–0.28 curvy/turbulent look toward trachea",
    )
    ap.add_argument("--no-bidirectional", action="store_true")
    args = ap.parse_args()

    case_id = args.case
    out = REPO_ROOT / "outputs" / case_id
    npz_path = out / f"{case_id}_flow.npz"
    if not npz_path.is_file():
        print(f"Missing {npz_path}")
        return 1

    data = np.load(npz_path)
    airway = data["airway"].astype(bool)
    ux = data["ux"].astype(float)
    uy = data["uy"].astype(float)
    uz = data["uz"].astype(float)
    speed = data["speed"].astype(float)
    spacing = tuple(float(v) for v in data["spacing_xyz_mm"])
    origin = tuple(float(v) for v in data["origin_xyz_mm"])
    notes: list[str] = []

    # Domain: L/R cavities + passage + full airway (fills sinuses if present)
    domain = airway.copy()
    for name in ("cavity_left", "cavity_right", "passage_lumen", "all_interior_air"):
        p = out / f"{case_id}_{name}.nrrd"
        if p.is_file():
            m = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(bool)
            if m.shape == airway.shape:
                domain |= m & airway
    # Prefer any air with mapped speed
    domain = domain & (speed > 1e-8)
    if not domain.any():
        domain = airway
    notes.append(f"Curvy pathline domain: {int(domain.sum())} voxels")

    # Naris centers from nares.json / BC
    naris: list[list[float]] = []
    nares_json = out / f"{case_id}_nares.json"
    if nares_json.is_file():
        nj = json.loads(nares_json.read_text(encoding="utf-8"))
        for npnt in nj.get("naris_points") or []:
            if npnt.get("center_mm"):
                naris.append([float(v) for v in npnt["center_mm"]])
    if len(naris) < 2:
        bc_path = out / f"{case_id}_boundary_conditions.json"
        if bc_path.is_file():
            bc = json.loads(bc_path.read_text(encoding="utf-8"))
            for port in bc.get("ports", []):
                if port.get("role") == "inlet" and port.get("center_mm"):
                    naris.append([float(v) for v in port["center_mm"]])

    if not naris:
        print("Need naris centers in nares.json or BC ports (role=inlet).")
        return 1

    outlet = None
    bc_path = out / f"{case_id}_boundary_conditions.json"
    if bc_path.is_file():
        bc = json.loads(bc_path.read_text(encoding="utf-8"))
        for port in bc.get("ports", []):
            if port.get("role") == "outlet" and port.get("center_mm"):
                outlet = [float(v) for v in port["center_mm"]]
    if outlet is None:
        print("Need trachea/outlet center in BC ports.")
        return 1

    centerline_mm = None
    for cl_name in (
        f"{case_id}_passage.json",
        f"{case_id}_open_paths.json",
    ):
        cl_path = out / cl_name
        if not cl_path.is_file():
            continue
        pj = json.loads(cl_path.read_text(encoding="utf-8"))
        for key in (
            "centerline_mm",
            "centerline_mid_mm",
            "centerline_left_mm",
            "centerline_right_mm",
        ):
            cl = pj.get(key) or []
            if len(cl) >= 2:
                centerline_mm = cl
                notes.append(f"Trachea extension via {cl_name}:{key}")
                break
        if centerline_mm is not None:
            break

    lines = compute_curvy_volume_pathlines(
        ux,
        uy,
        uz,
        domain,
        spacing,
        origin,
        naris_centers_mm=naris,
        outlet_center_mm=outlet,
        n_volume_seeds=args.volume_seeds,
        n_naris_seeds=args.naris_seeds,
        max_steps=1800,
        step_mm=args.step_mm,
        swirl=float(args.swirl),
        max_lines=args.max_lines,
        bidirectional=not args.no_bidirectional,
        centerline_mm=centerline_mm,
        naris_start_max_mm=14.0,
        trachea_end_max_mm=18.0,
    )
    notes.append(
        f"Naris→trachea curvy pathlines: {len(lines)} "
        f"(naris_seeds={args.naris_seeds}, volume_seeds={args.volume_seeds}, "
        f"swirl={args.swirl}, step={args.step_mm} mm)"
    )
    notes.append(
        f"Inlets L/R={naris}; outlet trachea={outlet}"
    )

    ox, oy, oz = origin
    sx, sy, sz = spacing
    nz, ny, nx = speed.shape
    lines_xyz: list[list] = []
    speeds_out: list[list] = []
    for line in lines:
        arr = np.asarray(line, dtype=float)
        if len(arr) < 6:
            continue
        ix = np.clip(np.rint((arr[:, 0] - ox) / sx).astype(int), 0, nx - 1)
        iy = np.clip(np.rint((arr[:, 1] - oy) / sy).astype(int), 0, ny - 1)
        iz = np.clip(np.rint((arr[:, 2] - oz) / sz).astype(int), 0, nz - 1)
        sp = speed[iz, iy, ix].astype(float)
        lines_xyz.append(arr.tolist())
        speeds_out.append(sp.tolist())

    sl_path = out / f"{case_id}_streamlines.json"
    with sl_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "case_id": case_id,
                "source": "naris_to_trachea_curvy",
                "n_lines": len(lines_xyz),
                "lines": lines_xyz,
                "speeds_m_s": speeds_out,
                "notes": notes,
                "params": {
                    "volume_seeds": args.volume_seeds,
                    "naris_seeds": args.naris_seeds,
                    "swirl": args.swirl,
                    "step_mm": args.step_mm,
                    "bidirectional": not args.no_bidirectional,
                    "max_lines": args.max_lines,
                    "flow": "naris → trachea",
                },
            },
            f,
        )

    print(f"OK curvy pathlines={len(lines_xyz)} → {sl_path.name}")
    for n in notes:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
