"""
Virtual surgery edits on the airway mask (research demo).

v1: inferior turbinate (IT) reduction — expand the air lumen into a lateral
inferior soft-tissue band adjacent to the nasal cavity (digital "shave").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from .cfd_metrics import write_cfd_metrics
from .geometry_metrics import (
    compute_geometry_metrics,
    load_lumen_for_geometry,
    side_geometry,
    write_geometry_metrics,
)


def _x_midplane_mm(
    case_dir: Path,
    case_id: str,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    lumen: np.ndarray,
) -> float:
    nares = case_dir / f"{case_id}_nares.json"
    if nares.is_file():
        nj = json.loads(nares.read_text(encoding="utf-8"))
        pts = [
            [float(v) for v in p["center_mm"]]
            for p in (nj.get("naris_points") or [])
            if p.get("center_mm")
        ]
        if len(pts) >= 2:
            return 0.5 * (pts[0][0] + pts[1][0])
    zz, yy, xx = np.where(lumen)
    if len(xx) == 0:
        return float(origin[0] + lumen.shape[2] * spacing[0] * 0.5)
    return float(origin[0] + np.median(xx) * spacing[0])


def virtual_it_reduction_mask(
    lumen: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    x_mid_mm: float,
    shave_mm: float = 2.0,
    lateral_band_mm: tuple[float, float] = (8.0, 18.0),
    inferior_fraction: float = 0.38,
    # Restrict AP extent to nasal cavity (not full pharynx)
    y_anterior_fraction: float = 0.55,
    max_removed_voxels: int = 4000,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Dilate airway into lateral inferior tissue = virtual IT reduction.

    Returns (edited_lumen, tissue_removed_mask, notes).
    """
    notes: list[str] = []
    sx, sy, sz = spacing
    ox, oy, oz = origin
    lumen = lumen.astype(bool)
    if not lumen.any():
        return lumen.copy(), np.zeros_like(lumen), ["Empty lumen — no edit."]

    # Ball radius in voxels for morphological dilation (cap at 3)
    r_vox = max(1, min(3, int(round(shave_mm / min(sx, sy, sz)))))
    struct = ndi.generate_binary_structure(3, 1)
    dilated = ndi.binary_dilation(lumen, structure=struct, iterations=r_vox)

    # Soft-tissue candidates = newly opened voxels only
    candidate = dilated & ~lumen

    zz, yy, xx = np.where(candidate)
    if len(zz) == 0:
        notes.append("Dilation added no new voxels.")
        return lumen.copy(), np.zeros_like(lumen), notes

    wx = ox + xx.astype(float) * sx
    lat = np.abs(wx - x_mid_mm)

    # Inferior + anterior (nasal) band relative to lumen extent
    lz, ly, lx = np.where(lumen)
    z_lo, z_hi = float(lz.min()), float(lz.max())
    y_lo, y_hi = float(ly.min()), float(ly.max())
    z_cut = z_lo + inferior_fraction * (z_hi - z_lo)
    # low y = anterior on VH
    y_cut = y_lo + y_anterior_fraction * (y_hi - y_lo)

    lat_lo, lat_hi = lateral_band_mm
    keep = (
        (lat >= lat_lo)
        & (lat <= lat_hi)
        & (zz.astype(float) <= z_cut)
        & (yy.astype(float) <= y_cut)
    )
    removed = np.zeros_like(lumen)
    if keep.any():
        removed[zz[keep], yy[keep], xx[keep]] = True
    else:
        keep = (
            (lat >= lat_lo)
            & (zz.astype(float) <= z_cut)
            & (yy.astype(float) <= y_cut)
        )
        removed[zz[keep], yy[keep], xx[keep]] = True
        notes.append("Relaxed lateral band for IT candidates.")

    # Cap resection volume (prefer voxels closest to original lumen wall)
    n_rem = int(removed.sum())
    if n_rem > max_removed_voxels:
        # distance to original lumen — keep closest
        dist_out = ndi.distance_transform_edt(~lumen, sampling=(sz, sy, sx))
        rz, ry, rx = np.where(removed)
        d = dist_out[rz, ry, rx]
        order = np.argsort(d)
        keep_n = order[:max_removed_voxels]
        removed = np.zeros_like(lumen)
        removed[rz[keep_n], ry[keep_n], rx[keep_n]] = True
        notes.append(
            f"Capped IT removal at {max_removed_voxels} voxels "
            f"(had {n_rem})."
        )

    # Keep only components attached to original lumen after edit
    edited = lumen | removed
    lab, n = ndi.label(edited)
    if n > 1:
        keep_labs = set(np.unique(lab[lumen])) - {0}
        mask = np.isin(lab, list(keep_labs))
        edited = mask
        removed = removed & edited

    notes.append(
        f"Virtual IT reduction: shave≈{shave_mm} mm, dilated r_vox={r_vox}, "
        f"removed_voxels={int(removed.sum())}, "
        f"lumen {int(lumen.sum())} → {int(edited.sum())}."
    )
    notes.append(
        "Heuristic lateral-inferior expansion — not a surgeon-validated resection plan."
    )
    return edited, removed, notes


def write_mask_nrrd(
    mask: np.ndarray,
    path: Path,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    reference: sitk.Image | None = None,
) -> None:
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    if reference is not None:
        img.SetDirection(reference.GetDirection())
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))


def run_virtual_it_reduction(
    case_dir: Path | str,
    case_id: str,
    shave_mm: float = 2.0,
    variant_name: str = "virtual_IT",
    recompute_pathlines: bool = True,
    pathline_seeds: int = 80,
) -> dict[str, Any]:
    """
    Create a virtual-IT case under outputs/{case_id}_{variant_name}/,
    recompute geometry metrics, and optional potential-flow pathlines.
    """
    case_dir = Path(case_dir)
    out_id = f"{case_id}_{variant_name}"
    out_dir = case_dir.parent / out_id
    out_dir.mkdir(parents=True, exist_ok=True)

    lumen, spacing, origin, lumen_src = load_lumen_for_geometry(case_dir, case_id)
    # Prefer original NRRD metadata
    ref_path = case_dir / lumen_src
    ref_img = sitk.ReadImage(str(ref_path)) if ref_path.is_file() else None

    x_mid = _x_midplane_mm(case_dir, case_id, origin, spacing, lumen)
    edited, removed, edit_notes = virtual_it_reduction_mask(
        lumen, spacing, origin, x_mid_mm=x_mid, shave_mm=shave_mm
    )

    # Baseline geometry (ensure exists)
    base_geo_path = case_dir / f"{case_id}_geometry_metrics.json"
    if not base_geo_path.is_file():
        write_geometry_metrics(case_dir, case_id)
    base_geo = json.loads(base_geo_path.read_text(encoding="utf-8"))

    # Copy nares + BC (recompute centerlines on edited lumen; do not reuse
    # baseline passage centerlines which can cut through newly opened tissue).
    for fname in (
        f"{case_id}_nares.json",
        f"{case_id}_boundary_conditions.json",
    ):
        src = case_dir / fname
        if src.is_file():
            dst_name = fname.replace(case_id, out_id, 1)
            raw = json.loads(src.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "case_id" in raw:
                raw["case_id"] = out_id
            (out_dir / dst_name).write_text(
                json.dumps(raw, indent=2), encoding="utf-8"
            )

    # Write edited masks (named for out_id)
    write_mask_nrrd(
        edited, out_dir / f"{out_id}_airway_mask.nrrd", spacing, origin, ref_img
    )
    write_mask_nrrd(
        edited, out_dir / f"{out_id}_passage_lumen.nrrd", spacing, origin, ref_img
    )
    write_mask_nrrd(
        removed, out_dir / f"{out_id}_tissue_removed.nrrd", spacing, origin, ref_img
    )

    # Geometry on edited lumen — prefer *same* baseline centerlines so MCA
    # compare is not confounded by path re-routing through new air.
    virt_geo = compute_geometry_metrics(out_dir, out_id)
    base_cl = {}
    for s in base_geo.get("sides") or []:
        # re-load centerlines from baseline passage if present
        pass
    passage_base = case_dir / f"{case_id}_passage.json"
    if passage_base.is_file():
        pj = json.loads(passage_base.read_text(encoding="utf-8"))
        if pj.get("centerline_left_mm"):
            base_cl["left"] = np.asarray(pj["centerline_left_mm"], dtype=float)
        if pj.get("centerline_right_mm"):
            base_cl["right"] = np.asarray(pj["centerline_right_mm"], dtype=float)
    if base_cl:
        fixed_sides = []
        for name, cl in base_cl.items():
            if len(cl) >= 2:
                fixed_sides.append(
                    side_geometry(name, edited, cl, spacing, origin, sample_every=2)
                )
        if fixed_sides:
            mcas = [s["mca"] for s in fixed_sides if s.get("mca")]
            global_mca = None
            if mcas:
                global_mca = min(mcas, key=lambda m: float(m["mca_mm2"]))
                for s in fixed_sides:
                    if s.get("mca") is global_mca or (
                        s.get("mca")
                        and s["mca"].get("mca_mm2") == global_mca.get("mca_mm2")
                        and s["mca"].get("mca_path_s_mm")
                        == global_mca.get("mca_path_s_mm")
                    ):
                        global_mca = dict(global_mca)
                        global_mca["side"] = s["name"]
                        break
            virt_geo = {
                "case_id": out_id,
                "kind": "geometry_metrics",
                "lumen_source": f"{out_id}_passage_lumen.nrrd",
                "spacing_xyz_mm": list(spacing),
                "origin_xyz_mm": list(origin),
                "lumen_voxels": int(edited.sum()),
                "lumen_volume_ml": float(
                    edited.sum() * float(np.prod(spacing)) / 1000.0
                ),
                "sides": [
                    {k: v for k, v in s.items() if k != "centerline_mm"}
                    for s in fixed_sides
                ],
                "global_mca": global_mca,
                "mca_markers": [
                    {
                        "side": s["name"],
                        "xyz_mm": (s.get("mca") or {}).get("mca_xyz_mm"),
                        "mca_mm2": (s.get("mca") or {}).get("mca_mm2"),
                        "path_s_mm": (s.get("mca") or {}).get("mca_path_s_mm"),
                    }
                    for s in fixed_sides
                    if s.get("mca") and (s["mca"] or {}).get("mca_xyz_mm")
                ],
                "notes": [
                    "MCA/CSA evaluated on *baseline* dual centerlines (fair pre/post).",
                    "Virtual IT geometric edit only.",
                ],
            }
    write_geometry_metrics(out_dir, out_id, report=virt_geo)

    # Optional: potential-flow field + short pathlines on edited airway
    flow_notes: list[str] = []
    if recompute_pathlines:
        try:
            from .flow_field import compute_flow_field
            from .physiology import PatientBreathing

            bc_src = case_dir / f"{case_id}_boundary_conditions.json"
            bc_dst = out_dir / f"{out_id}_boundary_conditions.json"
            if bc_src.is_file() and not bc_dst.is_file():
                # rewrite case_id inside if present
                bc = json.loads(bc_src.read_text(encoding="utf-8"))
                bc["case_id"] = out_id
                bc_dst.write_text(json.dumps(bc, indent=2), encoding="utf-8")

            breathing = PatientBreathing.typical_resting_adult(patient_id=out_id)
            compute_flow_field(
                airway_mask_path=out_dir / f"{out_id}_airway_mask.nrrd",
                boundary_json_path=out_dir / f"{out_id}_boundary_conditions.json",
                output_dir=out_dir,
                case_id=out_id,
                breathing=breathing,
                pressure_iterations=280,
                n_streamline_seeds=pathline_seeds,
                port_radius_mm=6.0,
            )
            flow_notes.append(
                "Potential-flow preview + pathlines recomputed on edited lumen "
                "(not re-meshed OpenFOAM)."
            )
            try:
                write_cfd_metrics(out_dir, out_id)
            except Exception as exc:
                flow_notes.append(f"CFD metrics on virtual case failed: {exc}")
        except Exception as exc:
            flow_notes.append(f"Pathline recompute skipped: {exc}")

    # Baseline CFD if available
    base_cfd = None
    base_cfd_path = case_dir / f"{case_id}_cfd_metrics.json"
    if not base_cfd_path.is_file():
        try:
            write_cfd_metrics(case_dir, case_id)
            base_cfd_path = case_dir / f"{case_id}_cfd_metrics.json"
        except Exception:
            pass
    if base_cfd_path.is_file():
        base_cfd = json.loads(base_cfd_path.read_text(encoding="utf-8"))

    virt_cfd = None
    virt_cfd_path = out_dir / f"{out_id}_cfd_metrics.json"
    # Only trust virtual CFD metrics when we just recomputed flow this run
    if recompute_pathlines and virt_cfd_path.is_file():
        virt_cfd = json.loads(virt_cfd_path.read_text(encoding="utf-8"))
    elif not recompute_pathlines:
        flow_notes.append(
            "Pathlines/CFD metrics not recomputed (--skip-pathlines); "
            "geometry compare only."
        )

    def _mca_val(geo: dict[str, Any]) -> float | None:
        g = geo.get("global_mca") or {}
        v = g.get("mca_mm2")
        return float(v) if v is not None else None

    def _side_mca(geo: dict[str, Any], side: str) -> float | None:
        for s in geo.get("sides") or []:
            if s.get("name") == side and s.get("mca"):
                return float(s["mca"]["mca_mm2"])
        return None

    def _mean_csa(geo: dict[str, Any], side: str | None = None) -> float | None:
        areas: list[float] = []
        for s in geo.get("sides") or []:
            if side and s.get("name") != side:
                continue
            for c in s.get("cross_sections") or []:
                # nasal search window same as MCA
                sm = float(c.get("path_s_mm", 0))
                if 8.0 <= sm <= 85.0:
                    areas.append(float(c.get("area_mm2", 0)))
        if not areas:
            return None
        return float(np.mean(areas))

    mean_base = _mean_csa(base_geo)
    mean_virt = _mean_csa(virt_geo)

    compare = {
        "case_id_baseline": case_id,
        "case_id_virtual": out_id,
        "procedure": "virtual_inferior_turbinate_reduction",
        "shave_mm": shave_mm,
        "lumen_source_baseline": lumen_src,
        "voxels_removed": int(removed.sum()),
        "volume_removed_ml": float(removed.sum() * float(np.prod(spacing)) / 1000.0),
        "lumen_voxels_baseline": int(lumen.sum()),
        "lumen_voxels_virtual": int(edited.sum()),
        "geometry": {
            "mca_mm2_baseline": _mca_val(base_geo),
            "mca_mm2_virtual": _mca_val(virt_geo),
            "mca_delta_mm2": (
                None
                if _mca_val(base_geo) is None or _mca_val(virt_geo) is None
                else _mca_val(virt_geo) - _mca_val(base_geo)
            ),
            "mean_csa_nasal_mm2_baseline": mean_base,
            "mean_csa_nasal_mm2_virtual": mean_virt,
            "mean_csa_delta_mm2": (
                None
                if mean_base is None or mean_virt is None
                else mean_virt - mean_base
            ),
            "left_mca_baseline": _side_mca(base_geo, "left"),
            "left_mca_virtual": _side_mca(virt_geo, "left"),
            "right_mca_baseline": _side_mca(base_geo, "right"),
            "right_mca_virtual": _side_mca(virt_geo, "right"),
            "note": (
                "MCA may be unchanged if the bottleneck is not at the IT "
                "(e.g. nasal valve). Mean nasal CSA and opened volume still "
                "reflect the IT edit."
            ),
        },
        "cfd": {
            "baseline_delta_p": (base_cfd or {}).get("pressure", {}).get(
                "delta_p_inlet_minus_outlet"
            ),
            "virtual_delta_p": (virt_cfd or {}).get("pressure", {}).get(
                "delta_p_inlet_minus_outlet"
            ),
            "baseline_R": (base_cfd or {}).get("resistance", {}).get(
                "R_delta_p_over_Q"
            ),
            "virtual_R": (virt_cfd or {}).get("resistance", {}).get(
                "R_delta_p_over_Q"
            ),
            "baseline_method": (base_cfd or {}).get("method"),
            "virtual_method": (virt_cfd or {}).get("method"),
            "note": (
                "Virtual case uses potential flow unless OpenFOAM is re-run; "
                "ΔR is qualitative when methods differ."
            ),
        },
        "edit_notes": edit_notes,
        "flow_notes": flow_notes,
        "research_disclaimer": (
            "Not a medical device. Virtual resection is a geometric heuristic."
        ),
    }

    # Also store compare JSON on baseline case for viewer convenience
    cmp_path = case_dir / f"{case_id}_virtual_IT_compare.json"
    cmp_path.write_text(json.dumps(compare, indent=2), encoding="utf-8")
    (out_dir / f"{out_id}_virtual_IT_compare.json").write_text(
        json.dumps(compare, indent=2), encoding="utf-8"
    )

    # Lightweight meta for variant
    meta = {
        "case_id": out_id,
        "parent_case": case_id,
        "procedure": "virtual_IT",
        "shave_mm": shave_mm,
        "x_midplane_mm": x_mid,
        "notes": edit_notes + flow_notes,
    }
    (out_dir / f"{out_id}_virtual_surgery_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    return compare
