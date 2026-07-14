#!/usr/bin/env python3
"""
Analyze nasal passage boundaries and recompute path-aware airflow.

Reads existing airway mask + BC ports from a processed case, then writes:
  - passage lumen / wall / open-port masks
  - centerline + cross-sections JSON
  - passage surface STL
  - updated flow field seeded along the passage
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sinus_cfd.flow_field import compute_flow_field  # noqa: E402
from sinus_cfd.nasal_passage import analyze_nasal_passage, write_passage_outputs  # noqa: E402
from sinus_cfd.physiology import PatientBreathing  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="VisibleHuman_Head")
    p.add_argument("--outputs-root", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument("--open-radius-mm", type=float, default=6.0)
    p.add_argument("--skip-flow", action="store_true")
    p.add_argument("--flow-iterations", type=int, default=450)
    p.add_argument("--streamline-seeds", type=int, default=120)
    args = p.parse_args()

    case_dir = args.outputs_root / args.case
    lumen_path = case_dir / f"{args.case}_airway_mask.nrrd"
    # Prefer dedicated passage lumen if we refine later
    if (case_dir / f"{args.case}_passage_lumen.nrrd").is_file() and False:
        lumen_path = case_dir / f"{args.case}_passage_lumen.nrrd"
    bc_path = case_dir / f"{args.case}_boundary_conditions.json"
    if not lumen_path.is_file():
        raise SystemExit(f"Missing lumen: {lumen_path}")
    if not bc_path.is_file():
        raise SystemExit(f"Missing BCs: {bc_path}")

    img = sitk.ReadImage(str(lumen_path))
    lumen = sitk.GetArrayFromImage(img).astype(bool)
    spacing = tuple(float(v) for v in img.GetSpacing())
    origin = tuple(float(v) for v in img.GetOrigin())

    bc = json.loads(bc_path.read_text(encoding="utf-8"))
    inlets = [p["center_mm"] for p in bc["ports"] if p.get("role") == "inlet"]
    outlets = [p["center_mm"] for p in bc["ports"] if p.get("role") == "outlet"]
    if not inlets or not outlets:
        raise SystemExit("BC JSON needs inlet and outlet ports with center_mm")

    # External skin nares (face) — required so open-air model includes nostrils
    skin_nares: list[list[float]] = []
    nares_path = case_dir / f"{args.case}_nares.json"
    if nares_path.is_file():
        nj = json.loads(nares_path.read_text(encoding="utf-8"))
        for npnt in nj.get("naris_points") or []:
            if npnt.get("center_mm"):
                skin_nares.append([float(v) for v in npnt["center_mm"]])
    if not skin_nares:
        skin_nares = [list(map(float, c)) for c in inlets]
        print(f"[{args.case}] WARNING: no nares.json — using BC inlet centers as skin nares")
    else:
        print(f"[{args.case}] skin nares from nares.json: {skin_nares}")

    print(f"[{args.case}] analyzing nasal passage domain (extend to external nostrils)…")
    masks, passage, metrics = analyze_nasal_passage(
        lumen=lumen,
        spacing=spacing,
        origin=origin,
        inlet_centers_mm=inlets,
        outlet_center_mm=outlets[0],
        case_id=args.case,
        open_radius_mm=args.open_radius_mm,
        skin_naris_centers_mm=skin_nares,
        tunnel_radius_mm=3.5,
    )
    paths = write_passage_outputs(
        args.case,
        case_dir,
        masks,
        passage,
        spacing,
        origin,
        reference_image=img,
    )
    print(
        f"[{args.case}] lumen={metrics.lumen_volume_ml:.1f} mL  "
        f"centerline={metrics.centerline_length_mm:.1f} mm  "
        f"area min/mean/max="
        f"{metrics.min_cross_section_mm2:.1f}/"
        f"{metrics.mean_cross_section_mm2:.1f}/"
        f"{metrics.max_cross_section_mm2:.1f} mm²"
    )
    print(f"[{args.case}] wall voxels={metrics.wall_voxels:,}  "
          f"inlet_open={metrics.inlet_open_voxels:,}  "
          f"outlet_open={metrics.outlet_open_voxels:,}")
    for k, path in paths.items():
        print(f"  wrote {k}: {path.name}")

    # Keep airway mask + air-space STL in sync with extended passage (includes nares)
    out_img = sitk.GetImageFromArray(masks["lumen"].astype("uint8"))
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, str(case_dir / f"{args.case}_airway_mask.nrrd"))
    # Viewer "Air space" mesh
    try:
        from sinus_cfd.pipeline import _mask_to_mesh  # noqa: E402

        amesh = _mask_to_mesh(masks["lumen"], spacing, origin)
        if len(amesh.faces) > 30000:
            try:
                amesh = amesh.simplify_quadric_decimation(30000)
            except Exception:
                pass
        amesh.export(case_dir / f"{args.case}_airway.stl")
        print(f"[{args.case}] wrote airway.stl (includes external nares)")
    except Exception as exc:
        print(f"[{args.case}] airway.stl export failed: {exc}")

    if not args.skip_flow:
        print(f"[{args.case}] path-aware flow field…")
        breathing = PatientBreathing.typical_resting_adult(patient_id=args.case)
        # Use physiology from BC if present
        b = bc.get("breathing") or {}
        if b.get("tidal_volume_L"):
            breathing = PatientBreathing(
                patient_id=args.case,
                tidal_volume_L=float(b["tidal_volume_L"]),
                respiratory_rate_per_min=float(b.get("respiratory_rate_per_min", 12)),
                inspiratory_fraction=float(b.get("inspiratory_fraction", 1 / 3)),
                left_nostril_flow_fraction=float(b.get("left_nostril_flow_fraction", 0.5)),
                right_nostril_flow_fraction=float(b.get("right_nostril_flow_fraction", 0.5)),
            )
        compute_flow_field(
            airway_mask_path=case_dir / f"{args.case}_airway_mask.nrrd",
            boundary_json_path=bc_path,
            output_dir=case_dir,
            case_id=args.case,
            breathing=breathing,
            pressure_iterations=args.flow_iterations,
            n_streamline_seeds=args.streamline_seeds,
            port_radius_mm=args.open_radius_mm,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
