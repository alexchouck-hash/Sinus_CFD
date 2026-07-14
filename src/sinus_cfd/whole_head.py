"""
Whole-head CT processing (Visible Human and similar).

Produces:
  - head solid mask + surface (semi-transparent body shell for visualization)
  - airway lumen mask (interior air, mouth excluded when possible)
  - boundary ports: both nostrils (inlet), trachea (outlet)
  - STL meshes + BC JSON ready for flow_field.compute_flow_field
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import trimesh
from scipy import ndimage as ndi
from skimage import measure, morphology

from .boundary_conditions import (
    BoundarySetup,
    Port,
    assign_flow,
    export_port_markers_ply,
    tag_mesh_faces_near_ports,
    write_boundary_setup,
    write_openfoam_bc_notes,
)
from .physiology import PatientBreathing, summary_text
from .pipeline import _mask_to_mesh


@dataclass
class WholeHeadResult:
    case_id: str
    image_path: str
    size_xyz: list[int]
    spacing_xyz_mm: list[float]
    origin_xyz_mm: list[float]
    crop_origin_zyx: list[int]
    head_voxels: int
    airway_voxels: int
    airway_volume_ml: float
    head_mesh_faces: int
    airway_mesh_faces: int
    outlet_is_proxy: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bbox(mask: np.ndarray, margin: int = 4) -> tuple[slice, slice, slice]:
    zz, yy, xx = np.where(mask)
    if len(zz) == 0:
        raise ValueError("Empty mask for bounding box")
    z0 = max(int(zz.min()) - margin, 0)
    z1 = min(int(zz.max()) + margin + 1, mask.shape[0])
    y0 = max(int(yy.min()) - margin, 0)
    y1 = min(int(yy.max()) + margin + 1, mask.shape[1])
    x0 = max(int(xx.min()) - margin, 0)
    x1 = min(int(xx.max()) + margin + 1, mask.shape[2])
    return slice(z0, z1), slice(y0, y1), slice(x0, x1)


def segment_head_body(
    hu: np.ndarray,
    body_hu_min: float = -200.0,
    min_component_voxels: int = 50_000,
) -> np.ndarray:
    """
    Soft-tissue + bone body mask (largest component), holes filled.

    Threshold is intentionally inclusive of soft tissue so the outer contour
    is a solid head/neck rather than a bone-only skull.
    """
    seed = hu > body_hu_min
    # Light open to knock dust, then keep large components
    seed = morphology.opening(seed, footprint=morphology.ball(1))
    labeled, n = ndi.label(seed)
    if n == 0:
        raise ValueError("No body voxels found — check HU scale / threshold")
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = counts[1:] >= min_component_voxels
    if not keep.any():
        keep[int(np.argmax(counts))] = True
    body = keep[labeled]
    body = ndi.binary_fill_holes(body)
    # Smooth silhouette slightly
    body = morphology.closing(body, footprint=morphology.ball(2))
    body = ndi.binary_fill_holes(body)
    return body.astype(bool)


def segment_airway_interior(
    hu: np.ndarray,
    body: np.ndarray,
    air_hu_max: float = -200.0,
    air_hu_min: float = -1024.0,
    min_component_voxels: int = 150,
) -> np.ndarray:
    """
    Air / low-density lumen voxels enclosed by the filled body.

    Default air_hu_max=-200 captures partial-volume airway (important on
    cadaver CT where the pharynx/trachea may not be fully free air).
    """
    air = (hu >= air_hu_min) & (hu <= air_hu_max) & body
    air = morphology.closing(air, footprint=morphology.ball(1))
    labeled, n = ndi.label(air)
    if n == 0:
        raise ValueError("No interior air found")
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = counts >= min_component_voxels
    keep[0] = False
    return keep[labeled].astype(bool)


def _orient_head(body: np.ndarray, air: np.ndarray) -> dict[str, Any]:
    """Infer inferior (neck) and anterior (face) axis directions from geometry."""
    n_z = body.shape[0]
    a0 = float(body[: max(n_z // 8, 1)].mean())
    a1 = float(body[-max(n_z // 8, 1) :].mean())
    z_inferior_is_high = a1 < a0

    if z_inferior_is_high:
        z_face = slice(0, int(n_z * 0.55))
        z_neck = slice(int(n_z * 0.80), n_z)
    else:
        z_face = slice(int(n_z * 0.45), n_z)
        z_neck = slice(0, int(n_z * 0.20))

    air_by_y = air[z_face].sum(axis=(0, 2))
    y_len = air_by_y.shape[0]
    score_low = float(air_by_y[: max(y_len // 5, 1)].sum())
    score_high = float(air_by_y[-max(y_len // 5, 1) :].sum())
    y_anterior_is_low = score_low >= score_high

    if y_anterior_is_low:
        y_ant = slice(0, max(int(y_len * 0.25), 10))
    else:
        y_ant = slice(min(int(y_len * 0.75), y_len - 10), y_len)

    return {
        "z_inferior_is_high": z_inferior_is_high,
        "z_face": z_face,
        "z_neck": z_neck,
        "y_anterior_is_low": y_anterior_is_low,
        "y_ant": y_ant,
        "air_y_scores": {"low": score_low, "high": score_high},
    }


def _geodesic_airway_bridge(
    hu: np.ndarray,
    body: np.ndarray,
    air: np.ndarray,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    tube_radius: int = 3,
) -> np.ndarray:
    """
    Low-HU geodesic from start→end through body; dilate into a thin tube.

    Used when free-air is discontinuous (common on cadaver head CT).
    """
    from skimage.graph import route_through_array

    # Cost: prefer air/low HU; forbid exterior
    # Map HU in body to costs in [1, 500]
    costs = np.full(hu.shape, 1.0e6, dtype=np.float64)
    inside = body.astype(bool)
    # Lower HU → lower cost
    costs[inside] = np.clip((hu[inside] + 1000.0) / 8.0, 1.0, 400.0)
    # Strongly prefer existing air voxels
    costs[air] = 0.5

    try:
        indices, _cost = route_through_array(
            costs,
            start=start,
            end=end,
            fully_connected=True,
            geometric=True,
        )
    except Exception:
        return np.zeros_like(air, dtype=bool)

    path = np.zeros_like(air, dtype=bool)
    for iz, iy, ix in indices:
        path[iz, iy, ix] = True
    if tube_radius > 0:
        path = morphology.dilation(path, footprint=morphology.ball(tube_radius))
    path &= body
    return path


def select_nasal_to_trachea_path(
    air: np.ndarray,
    body: np.ndarray,
    hu: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Prefer continuous airway from the face (nostrils) to inferior trachea.

    1. Components touching both nose and trachea seeds.
    2. Morphological closing bridge.
    3. Low-HU geodesic tube nose→neck (cadaver FOV often lacks free tracheal air).
    4. Fallback: largest nasal component with outlet at its inferior tip.
    """
    info: dict[str, Any] = {}
    zz, yy, xx = np.where(air)
    if len(zz) == 0:
        raise ValueError("Empty air mask")

    orient = _orient_head(body, air)
    info.update(
        {
            "z_inferior_is_high": orient["z_inferior_is_high"],
            "y_anterior_is_low": orient["y_anterior_is_low"],
            "air_y_scores": orient["air_y_scores"],
        }
    )
    z_face, z_neck = orient["z_face"], orient["z_neck"]
    y_ant = orient["y_ant"]
    x_mid = int(np.median(xx))

    nostril_region = np.zeros_like(air)
    nostril_region[z_face, y_ant, :] = True
    nostril_region &= air

    x0, x1 = max(x_mid - 45, 0), min(x_mid + 45, air.shape[2])
    trachea_region = np.zeros_like(air)
    trachea_region[z_neck, :, x0:x1] = True
    if hu is not None:
        # Prefer low-HU in neck (partial-volume airway)
        trachea_region &= ((hu <= -80) & body) | air
    else:
        trachea_region &= air
    if trachea_region.sum() < 20:
        trachea_region = np.zeros_like(air)
        trachea_region[z_neck, :, x0:x1] = True
        trachea_region &= air if air[z_neck].any() else body

    info["nostril_seed_voxels"] = int(nostril_region.sum())
    info["trachea_seed_voxels"] = int(trachea_region.sum())

    labeled, n = ndi.label(air)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0

    both, nose_only, trach_only = [], [], []
    for lab_id in range(1, n + 1):
        if counts[lab_id] < 150:
            continue
        comp = labeled == lab_id
        t_nose = bool((comp & nostril_region).any())
        t_trach = bool((comp & trachea_region).any())
        if t_nose and t_trach:
            both.append(lab_id)
        elif t_nose:
            nose_only.append(lab_id)
        elif t_trach:
            trach_only.append(lab_id)

    if both:
        best = max(both, key=lambda i: counts[i])
        keep = labeled == best
        info["selection"] = "connected_nose_to_trachea"
        return keep.astype(bool), info

    # Closing bridge
    for radius in (2, 3, 4):
        bridged = morphology.closing(air, footprint=morphology.ball(radius))
        labeled2, n2 = ndi.label(bridged)
        counts2 = np.bincount(labeled2.ravel())
        counts2[0] = 0
        both2 = [
            i
            for i in range(1, n2 + 1)
            if ((labeled2 == i) & nostril_region).any()
            and ((labeled2 == i) & trachea_region).any()
        ]
        if both2:
            best = max(both2, key=lambda i: counts2[i])
            keep = (labeled2 == best) & (
                air | morphology.dilation(air, footprint=morphology.ball(2))
            )
            info["selection"] = f"bridged_closing_r{radius}"
            return keep.astype(bool), info

    # Geodesic low-HU tube from nose to neck (cadaver airway often soft-tissue filled)
    if hu is not None and nostril_region.any():
        nose_pts = np.column_stack(np.where(nostril_region))
        start = tuple(int(v) for v in nose_pts[len(nose_pts) // 2])
        # Neck target: midline of body in inferior slices
        if orient["z_inferior_is_high"]:
            z_t = body.shape[0] - 8
        else:
            z_t = 8
        # Find body center at that slice
        ys, xs = np.where(body[z_t])
        if len(ys):
            end = (z_t, int(np.median(ys)), int(np.median(xs)))
            tube = _geodesic_airway_bridge(hu, body, air, start, end, tube_radius=3)
            # Union largest nasal air with tube
            if nose_only:
                nose_comp = labeled == max(nose_only, key=lambda i: counts[i])
            else:
                # largest overall air
                nose_comp = labeled == int(np.argmax(counts))
            keep = nose_comp | tube | air & morphology.dilation(tube, footprint=morphology.ball(2))
            # Keep largest connected piece of the union that includes nose
            labk, nk = ndi.label(keep)
            best_k = None
            best_n = 0
            for i in range(1, nk + 1):
                comp = labk == i
                if (comp & nostril_region).any() and int(comp.sum()) > best_n:
                    best_n = int(comp.sum())
                    best_k = i
            if best_k is not None:
                info["selection"] = "geodesic_nose_to_neck_tube"
                info["note"] = (
                    "Pharynx/trachea free-air incomplete on this CT; "
                    "added low-HU geodesic conduit to inferior neck."
                )
                return (labk == best_k).astype(bool), info

    # Fallback: largest nasal component
    if nose_only:
        best = max(nose_only, key=lambda i: counts[i])
    else:
        best = int(np.argmax(counts))
    keep = labeled == best
    info["selection"] = "fallback_largest_nasal"
    info["warning"] = (
        "Could not fully connect nostrils to trachea; outlet is inferior tip of nasal path."
    )
    return keep.astype(bool), info


def detect_ports_whole_head(
    airway: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    y_anterior_is_low: bool,
    z_inferior_is_high: bool,
    port_radius_mm: float = 8.0,
) -> tuple[list[Port], list[str]]:
    """
    Place left/right nostril inlets and a true trachea outlet from geometry.
    """
    warnings: list[str] = []
    zz, yy, xx = np.where(airway)
    if len(zz) == 0:
        raise ValueError("Empty airway")

    pts = np.column_stack(
        [
            xx * spacing_xyz[0] + origin_xyz[0],
            yy * spacing_xyz[1] + origin_xyz[1],
            zz * spacing_xyz[2] + origin_xyz[2],
        ]
    )

    # ---- Trachea: inferior-most air cluster near midline ----
    if z_inferior_is_high:
        z_thr = np.percentile(zz, 92)
        tip_idx = zz >= z_thr
    else:
        z_thr = np.percentile(zz, 8)
        tip_idx = zz <= z_thr
    trach_pts = pts[tip_idx]
    trach_center = trach_pts.mean(axis=0)
    # Area estimate
    face = float(np.prod(sorted(spacing_xyz)[:2]))
    trach_area = float(tip_idx.sum() ** (2 / 3) * face)
    trach_normal = np.array([0.0, 0.0, 1.0 if z_inferior_is_high else -1.0])

    # ---- Nostrils: anterior-most air, split left/right by x median ----
    if y_anterior_is_low:
        y_thr = np.percentile(yy, 12)
        ant = yy <= y_thr
    else:
        y_thr = np.percentile(yy, 88)
        ant = yy >= y_thr
    ant_pts = pts[ant]
    ant_x = xx[ant]
    if len(ant_pts) < 10:
        warnings.append("Sparse anterior air for nostril detection; using global x-split.")
        ant = np.ones(len(xx), dtype=bool)
        ant_pts = pts
        ant_x = xx

    x_med = float(np.median(ant_x))
    # In LPS, +x is patient left; we still label by spatial x
    left_mask = ant_x >= x_med  # higher x
    right_mask = ant_x < x_med

    ports: list[Port] = []
    for name, m, side in (
        ("left_nostril", left_mask, "left"),
        ("right_nostril", right_mask, "right"),
    ):
        if not m.any():
            warnings.append(f"No voxels for {name}")
            continue
        c = ant_pts[m].mean(axis=0)
        area = float(m.sum() ** (2 / 3) * face)
        # Normal points into domain: from exterior (anterior) toward posterior
        n = np.array([0.0, 1.0 if y_anterior_is_low else -1.0, 0.0])
        ports.append(
            Port(
                name=name,
                role="inlet",
                center_mm=c.tolist(),
                area_mm2=area,
                normal_xyz=n.tolist(),
                method=f"whole_head_anterior_air_{side}",
                notes="Inspiratory flow enters at nostril (geometry-based).",
            )
        )

    ports.append(
        Port(
            name="trachea",
            role="outlet",
            center_mm=trach_center.tolist(),
            area_mm2=trach_area,
            normal_xyz=trach_normal.tolist(),
            method="whole_head_inferior_airway",
            notes="True tracheal / subglottic inferior airway outlet (whole-head FOV).",
        )
    )
    return ports, warnings


def process_whole_head(
    image_path: Path | str,
    output_dir: Path | str | None = None,
    case_id: str = "VisibleHuman_Head",
    breathing: PatientBreathing | None = None,
    body_hu_min: float = -200.0,
    air_hu_max: float = -200.0,
    mesh_decimate_head: int = 25000,
    mesh_decimate_airway: int = 20000,
) -> WholeHeadResult:
    """
    Full pipeline for a whole-head CT volume.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir or Path("outputs") / case_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    breathing = breathing or PatientBreathing.typical_resting_adult()

    image = sitk.ReadImage(str(image_path))
    hu_full = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing = tuple(float(v) for v in image.GetSpacing())
    origin = tuple(float(v) for v in image.GetOrigin())
    notes: list[str] = []

    print(f"[{case_id}] segmenting head body…")
    body_full = segment_head_body(hu_full, body_hu_min=body_hu_min)

    # Crop to body for speed
    bb = _bbox(body_full, margin=6)
    crop_origin_zyx = [bb[0].start, bb[1].start, bb[2].start]
    hu = hu_full[bb]
    body = body_full[bb]
    # Adjust origin for crop (z,y,x index → physical)
    # SimpleITK array is z,y,x; physical: x = ox + i*sx
    origin_crop = (
        origin[0] + crop_origin_zyx[2] * spacing[0],
        origin[1] + crop_origin_zyx[1] * spacing[1],
        origin[2] + crop_origin_zyx[0] * spacing[2],
    )

    print(f"[{case_id}] crop shape zyx={hu.shape}  body voxels={int(body.sum()):,}")
    print(f"[{case_id}] segmenting interior airway…")
    air_all = segment_airway_interior(hu, body, air_hu_max=air_hu_max)
    airway, path_info = select_nasal_to_trachea_path(air_all, body, hu=hu)
    notes.append(f"Airway selection: {path_info.get('selection')}")
    if path_info.get("warning"):
        notes.append(path_info["warning"])

    # Mouth closed note: oral cavity dropped when not on nose–trachea path
    notes.append(
        "Mouth treated as closed: fluid domain is selected nasal→pharynx→trachea path."
    )

    # Save masks (crop geometry)
    def _write_mask(mask: np.ndarray, name: str) -> Path:
        img = sitk.GetImageFromArray(mask.astype(np.uint8))
        img.SetSpacing(spacing)
        img.SetOrigin(origin_crop)
        img.SetDirection(image.GetDirection())
        path = output_dir / name
        sitk.WriteImage(img, str(path))
        return path

    head_mask_path = _write_mask(body, f"{case_id}_head_mask.nrrd")
    airway_mask_path = _write_mask(airway, f"{case_id}_airway_mask.nrrd")
    # Also store all interior air for sinus visualization later
    _write_mask(air_all, f"{case_id}_all_interior_air.nrrd")

    print(f"[{case_id}] meshing head solid + airway…")
    head_mesh = _mask_to_mesh(body, spacing, origin_crop)
    airway_mesh = _mask_to_mesh(airway, spacing, origin_crop)

    # Decimate for viewer
    head_mesh_ds = _decimate(head_mesh, mesh_decimate_head)
    airway_mesh_ds = _decimate(airway_mesh, mesh_decimate_airway)

    head_stl = output_dir / f"{case_id}_head.stl"
    airway_stl = output_dir / f"{case_id}_airway.stl"
    head_mesh_ds.export(head_stl)
    airway_mesh_ds.export(airway_stl)
    # Full-res airway for CFD ports (optional keep decimated for flow viz too)
    airway_mesh.export(output_dir / f"{case_id}_airway_full.stl")

    # Ports
    ports, port_warnings = detect_ports_whole_head(
        airway,
        spacing,
        origin_crop,
        y_anterior_is_low=bool(path_info.get("y_anterior_is_low", True)),
        z_inferior_is_high=bool(path_info.get("z_inferior_is_high", True)),
    )
    notes.extend(port_warnings)
    ports = tag_mesh_faces_near_ports(airway_mesh_ds, ports, radius_mm=14.0)
    # If trachea still has no faces, expand tag radius once
    for p in ports:
        if p.role == "outlet" and p.n_faces == 0:
            ports = tag_mesh_faces_near_ports(airway_mesh_ds, ports, radius_mm=22.0)
            break
    flow = assign_flow(ports, breathing)
    outlet_proxy = path_info.get("selection", "").startswith("fallback")
    setup = BoundarySetup(
        case_id=case_id,
        mouth="closed — oral cavity excluded from selected airway path when separable",
        inlet_names=[p.name for p in ports if p.role == "inlet"],
        outlet_name=next((p.name for p in ports if p.role == "outlet"), "trachea"),
        outlet_is_proxy=outlet_proxy,
        ports=ports,
        breathing=breathing.to_dict(),
        flow_assignment=flow,
        mesh_path=str(airway_stl),
        warnings=port_warnings
        + ([path_info["warning"]] if path_info.get("warning") else [])
        + ([path_info["note"]] if path_info.get("note") else []),
    )
    write_boundary_setup(setup, output_dir / f"{case_id}_boundary_conditions.json")
    write_openfoam_bc_notes(setup, output_dir / f"{case_id}_openfoam_bc_sketch.txt")
    try:
        export_port_markers_ply(ports, output_dir / f"{case_id}_port_markers.ply")
    except ValueError:
        pass

    # Preview figure: CT + head outline + airway
    _save_whole_head_preview(
        hu, body, airway, output_dir / f"{case_id}_preview.png", case_id
    )

    voxel_ml = float(np.prod(spacing)) / 1000.0
    result = WholeHeadResult(
        case_id=case_id,
        image_path=str(image_path),
        size_xyz=list(image.GetSize()),
        spacing_xyz_mm=list(spacing),
        origin_xyz_mm=list(origin_crop),
        crop_origin_zyx=crop_origin_zyx,
        head_voxels=int(body.sum()),
        airway_voxels=int(airway.sum()),
        airway_volume_ml=float(airway.sum() * voxel_ml),
        head_mesh_faces=int(len(head_mesh_ds.faces)),
        airway_mesh_faces=int(len(airway_mesh_ds.faces)),
        outlet_is_proxy=outlet_proxy,
        notes=notes,
    )
    with (output_dir / f"{case_id}_stats.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **result.to_dict(),
                "path_info": {
                    k: v
                    for k, v in path_info.items()
                    if not isinstance(v, slice)
                },
                "boundary_summary": {
                    "inlets": setup.inlet_names,
                    "outlet": setup.outlet_name,
                    "outlet_is_proxy": outlet_proxy,
                    "Q_L_per_min": breathing.mean_inspiratory_flow_L_per_min,
                    "Ti_s": breathing.Ti_s,
                },
            },
            f,
            indent=2,
        )

    print(summary_text(breathing))
    print(
        f"[{case_id}] head voxels={result.head_voxels:,}  "
        f"airway={result.airway_voxels:,} ({result.airway_volume_ml:.1f} mL)"
    )
    print(
        f"[{case_id}] head faces={result.head_mesh_faces:,}  "
        f"airway faces={result.airway_mesh_faces:,}"
    )
    print(
        f"[{case_id}] wrote head STL, airway STL, masks, BCs → {output_dir}"
    )
    return result


def _decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(target_faces)
    except Exception:
        idx = np.linspace(0, len(mesh.faces) - 1, target_faces, dtype=int)
        return trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces[idx], process=False
        )


def _save_whole_head_preview(
    hu: np.ndarray,
    body: np.ndarray,
    airway: np.ndarray,
    path: Path,
    case_id: str,
) -> None:
    z, y, x = hu.shape
    mid = (z // 2, y // 2, x // 2)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    views = [
        ("axial", hu[mid[0]], body[mid[0]], airway[mid[0]]),
        ("coronal", hu[:, mid[1], :], body[:, mid[1], :], airway[:, mid[1], :]),
        ("sagittal", hu[:, :, mid[2]], body[:, :, mid[2]], airway[:, :, mid[2]]),
    ]
    for ax, (title, img, b, a) in zip(axes, views):
        disp = np.clip(img, -200, 400)
        ax.imshow(disp, cmap="gray", origin="lower")
        # Head outline
        ax.contour(b.astype(float), levels=[0.5], colors=["#4fc3f7"], linewidths=0.8)
        # Airway fill
        ov = np.ma.masked_where(~a, a.astype(float))
        ax.imshow(ov, cmap="autumn", alpha=0.45, origin="lower")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{case_id} — head outline (cyan) + airway (red)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
