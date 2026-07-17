"""
Load a NasalSeg CT case, build an airway mask, and export a surface mesh.

Primary mask source: expert labels (labels 1–5 = nasal cavity L/R,
nasopharynx, maxillary sinus L/R). Optional HU air threshold for comparison.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import trimesh
from scipy import ndimage as ndi
from skimage import measure, morphology

from .boundary_conditions import (
    build_boundary_setup,
    export_port_markers_ply,
    write_boundary_setup,
    write_openfoam_bc_notes,
)
from .physiology import PatientBreathing, summary_text

# NasalSeg label map (from dataset documentation / paper)
LABEL_NAMES = {
    0: "background",
    1: "left_nasal_cavity",
    2: "right_nasal_cavity",
    3: "nasopharynx",
    4: "left_maxillary_sinus",
    5: "right_maxillary_sinus",
}

# Default structures for continuous airway CFD (exclude isolated sinus air if desired)
DEFAULT_AIRWAY_LABELS = (1, 2, 3)  # both nasal cavities + nasopharynx
DEFAULT_ALL_AIRWAY_LABELS = (1, 2, 3, 4, 5)


@dataclass
class CaseStats:
    case_id: str
    image_path: str
    label_path: str | None
    size_xyz: list[int]
    spacing_xyz_mm: list[float]
    origin_xyz_mm: list[float]
    hu_min: float
    hu_max: float
    hu_mean: float
    mask_source: str
    mask_voxels: int
    mask_volume_mm3: float
    mask_volume_ml: float
    mesh_faces: int
    mesh_vertices: int
    mesh_watertight: bool
    mesh_volume_mm3: float | None
    label_voxel_counts: dict[str, int]
    boundary: dict[str, Any] = field(default_factory=dict)
    breathing_summary: str = ""


def _read_volume(path: Path) -> tuple[sitk.Image, np.ndarray]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)  # z, y, x
    return image, array


def _labels_to_mask(label_zyx: np.ndarray, keep: Iterable[int]) -> np.ndarray:
    keep_set = set(int(v) for v in keep)
    mask = np.isin(label_zyx, list(keep_set))
    return mask.astype(bool)


def _hu_air_mask(
    hu_zyx: np.ndarray,
    hu_max: float = -400.0,
    hu_min: float = -1024.0,
) -> np.ndarray:
    """Threshold approximate air voxels in Hounsfield units."""
    return (hu_zyx >= hu_min) & (hu_zyx <= hu_max)


def _bridge_through_air(
    seed_mask: np.ndarray,
    hu_zyx: np.ndarray | None,
    hu_max: float = -400.0,
    hu_min: float = -1024.0,
    max_gap_voxels: int = 12,
) -> np.ndarray:
    """
    Re-join nearby label components through HU air without flooding exterior air.

    Growth is allowed only in air voxels within ``max_gap_voxels`` of the original
    seed (distance transform band). That bridges small choanal/meatus gaps while
    blocking unrestricted dilation into free air outside the nose.
    """
    seed = seed_mask.astype(bool)
    if hu_zyx is None:
        return seed

    air = (hu_zyx >= hu_min) & (hu_zyx <= hu_max)
    # Band around labels: prevents growing out the nostrils into exterior FOV air
    dist = ndi.distance_transform_edt(~seed)
    allowed = air & (dist <= float(max_gap_voxels))

    bridged = seed.copy()
    structure = morphology.ball(1)
    for _ in range(int(max_gap_voxels) + 2):
        grown = morphology.dilation(bridged, footprint=structure) & allowed
        grown |= seed  # never lose original labels
        if np.array_equal(grown, bridged):
            break
        bridged = grown
        _, n = ndi.label(bridged)
        if n <= 1:
            break
    return bridged


def _clean_mask(
    mask: np.ndarray,
    min_component_voxels: int = 200,
    closing_radius: int = 1,
    keep_all_large_components: bool = True,
    hu_zyx: np.ndarray | None = None,
    hu_max: float = -400.0,
    hu_min: float = -1024.0,
    max_gap_voxels: int = 28,
) -> np.ndarray:
    """
    Close small holes, drop tiny islands, optionally bridge through HU air.

    Default keeps *all* large components (both nostrils + nasopharynx). Using
    only the single largest blob drops the contralateral nasal cavity when
    NasalSeg labels are not voxel-adjacent.
    """
    cleaned = mask.astype(bool)
    if closing_radius > 0:
        structure = morphology.ball(closing_radius)
        cleaned = morphology.closing(cleaned, footprint=structure)

    # Bridge label gaps through true air so L/R/NP form one flow domain
    cleaned = _bridge_through_air(
        cleaned,
        hu_zyx,
        hu_max=hu_max,
        hu_min=hu_min,
        max_gap_voxels=max_gap_voxels,
    )

    labeled, n = ndi.label(cleaned)
    if n == 0:
        return cleaned

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = np.zeros(n + 1, dtype=bool)
    if keep_all_large_components:
        keep[1:] = counts[1:] >= min_component_voxels
    else:
        keep[int(np.argmax(counts))] = True

    return keep[labeled]


def build_hu_threshold_mask(
    hu_zyx: np.ndarray,
    hu_max: float = -400.0,
    hu_min: float = -1024.0,
    min_component_voxels: int = 200,
    max_gap_voxels: int = 28,
) -> np.ndarray:
    """
    Classical threshold + region-grow airway mask from HU alone (no expert labels).

    Steps: HU threshold -> morphological closing -> bridge small gaps through
    air (region-grow) -> drop components smaller than ``min_component_voxels``.
    Shared by ``process_case(mask_source="hu")`` and the NasalSeg Dice evaluation.
    """
    mask = _hu_air_mask(hu_zyx, hu_max=hu_max, hu_min=hu_min)
    return _clean_mask(
        mask,
        min_component_voxels=min_component_voxels,
        keep_all_large_components=True,
        hu_zyx=hu_zyx,
        hu_max=hu_max,
        hu_min=hu_min,
        max_gap_voxels=max_gap_voxels,
    )


def _mask_to_mesh(
    mask_zyx: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    level: float = 0.5,
) -> trimesh.Trimesh:
    """
    Marching cubes on a binary mask.

    skimage expects spacing in (z, y, x) order matching the array axes.
    Vertices are then shifted by the image origin (x, y, z).
    """
    if mask_zyx.sum() == 0:
        raise ValueError("Empty mask — cannot extract surface.")

    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    verts_zyx, faces, _normals, _values = measure.marching_cubes(
        mask_zyx.astype(np.float32),
        level=level,
        spacing=spacing_zyx,
        allow_degenerate=False,
    )
    # verts are (z, y, x) physical offsets from index 0 → convert to (x, y, z)
    verts_xyz = np.column_stack(
        [
            verts_zyx[:, 2] + origin_xyz[0],
            verts_zyx[:, 1] + origin_xyz[1],
            verts_zyx[:, 0] + origin_xyz[2],
        ]
    )
    mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=True)
    return mesh


def _save_preview(
    hu_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    out_path: Path,
    case_id: str,
) -> None:
    z, y, x = hu_zyx.shape
    mid = (z // 2, y // 2, x // 2)
    slices = [
        ("axial (z)", hu_zyx[mid[0]], mask_zyx[mid[0]]),
        ("coronal (y)", hu_zyx[:, mid[1], :], mask_zyx[:, mid[1], :]),
        ("sagittal (x)", hu_zyx[:, :, mid[2]], mask_zyx[:, :, mid[2]]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, img, msk) in zip(axes, slices):
        # Soft-tissue-ish window for context
        display = np.clip(img, -1000, 400).astype(np.float32)
        ax.imshow(display, cmap="gray", origin="lower")
        overlay = np.ma.masked_where(~msk, msk.astype(float))
        ax.imshow(overlay, cmap="autumn", alpha=0.45, origin="lower")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{case_id} — airway mask overlay")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def process_case(
    image_path: Path | str,
    label_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    case_id: str | None = None,
    mask_source: str = "labels",
    airway_labels: Iterable[int] = DEFAULT_AIRWAY_LABELS,
    hu_max: float = -400.0,
    hu_min: float = -1024.0,
    min_component_voxels: int = 200,
    breathing: PatientBreathing | None = None,
    write_bcs: bool = True,
    face_tag_radius_mm: float = 8.0,
) -> CaseStats:
    """
    Process one CT case into mask + STL surface + preview + BC setup.

    Boundary policy (project intent):
      - Inlets: both nostrils (flow rate from physiology)
      - Outlet: trachea (nasopharynx proxy on NasalSeg crops)
      - Mouth: closed / blocked (excluded from fluid domain)

    mask_source:
      - "labels": use expert NasalSeg labels (recommended)
      - "hu": HU air threshold only
      - "labels_and_hu": intersection of labels with HU air
    """
    image_path = Path(image_path)
    if case_id is None:
        # P001_img.nrrd -> P001
        stem = image_path.stem
        case_id = stem.replace("_img", "").replace("_seg", "")

    output_dir = Path(output_dir or Path("outputs") / case_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    breathing = breathing or PatientBreathing.typical_resting_adult()

    image, hu_zyx = _read_volume(image_path)
    spacing_xyz = tuple(float(v) for v in image.GetSpacing())
    origin_xyz = tuple(float(v) for v in image.GetOrigin())
    size_xyz = [int(v) for v in image.GetSize()]

    label_zyx = None
    label_counts: dict[str, int] = {}
    if label_path is not None:
        label_path = Path(label_path)
        _lab_img, label_zyx = _read_volume(label_path)
        for val, name in LABEL_NAMES.items():
            label_counts[name] = int((label_zyx == val).sum())

    mask_source = mask_source.lower().strip()
    if mask_source == "labels":
        if label_zyx is None:
            raise ValueError("mask_source='labels' requires label_path")
        mask = _labels_to_mask(label_zyx, airway_labels)
    elif mask_source == "hu":
        mask = _hu_air_mask(hu_zyx, hu_max=hu_max, hu_min=hu_min)
    elif mask_source in ("labels_and_hu", "intersection"):
        if label_zyx is None:
            raise ValueError("labels_and_hu requires label_path")
        mask = _labels_to_mask(label_zyx, airway_labels) & _hu_air_mask(
            hu_zyx, hu_max=hu_max, hu_min=hu_min
        )
    else:
        raise ValueError(f"Unknown mask_source: {mask_source}")

    # Mouth closed: domain is nasal cavities + nasopharynx only (labels 1–3 by default).
    # Oral cavity is never added to the mask when using label-based sources.
    # Bridge through HU air so left/right nostrils + nasopharynx form one domain.
    mask = _clean_mask(
        mask,
        min_component_voxels=min_component_voxels,
        keep_all_large_components=True,
        hu_zyx=hu_zyx,
        hu_max=hu_max,
        hu_min=hu_min,
    )

    # Save mask as NRRD (same geometry as CT)
    mask_u8 = mask.astype(np.uint8)
    mask_img = sitk.GetImageFromArray(mask_u8)
    mask_img.CopyInformation(image)
    mask_path = output_dir / f"{case_id}_airway_mask.nrrd"
    sitk.WriteImage(mask_img, str(mask_path))

    mesh = _mask_to_mesh(mask, spacing_xyz, origin_xyz)
    stl_path = output_dir / f"{case_id}_airway.stl"
    mesh.export(stl_path)

    preview_path = output_dir / f"{case_id}_preview.png"
    _save_preview(hu_zyx, mask, preview_path, case_id)

    boundary_dict: dict[str, Any] = {}
    breath_txt = summary_text(breathing)
    if write_bcs and label_zyx is not None:
        setup = build_boundary_setup(
            case_id=case_id,
            label_zyx=label_zyx,
            airway_mask=mask,
            spacing_xyz=spacing_xyz,
            origin_xyz=origin_xyz,
            mesh=mesh,
            breathing=breathing,
            mesh_path=stl_path,
            face_tag_radius_mm=face_tag_radius_mm,
        )
        bc_path = write_boundary_setup(setup, output_dir / f"{case_id}_boundary_conditions.json")
        of_path = write_openfoam_bc_notes(setup, output_dir / f"{case_id}_openfoam_bc_sketch.txt")
        try:
            markers_path = export_port_markers_ply(
                setup.ports, output_dir / f"{case_id}_port_markers.ply"
            )
        except ValueError:
            markers_path = None
        boundary_dict = setup.to_dict()
        print(breath_txt)
        print(f"[{case_id}] BCs → {bc_path.name}, {of_path.name}"
              + (f", {markers_path.name}" if markers_path else ""))
        for w in setup.warnings:
            print(f"[{case_id}] warning: {w}")
    elif write_bcs and label_zyx is None:
        print(f"[{case_id}] skip BCs: labels required for nostril/trachea ports")

    voxel_volume = float(np.prod(spacing_xyz))
    mask_voxels = int(mask.sum())
    mask_vol_mm3 = mask_voxels * voxel_volume

    try:
        mesh_vol = float(mesh.volume) if mesh.is_watertight else None
    except Exception:
        mesh_vol = None

    stats = CaseStats(
        case_id=case_id,
        image_path=str(image_path),
        label_path=str(label_path) if label_path else None,
        size_xyz=size_xyz,
        spacing_xyz_mm=list(spacing_xyz),
        origin_xyz_mm=list(origin_xyz),
        hu_min=float(hu_zyx.min()),
        hu_max=float(hu_zyx.max()),
        hu_mean=float(hu_zyx.mean()),
        mask_source=mask_source,
        mask_voxels=mask_voxels,
        mask_volume_mm3=mask_vol_mm3,
        mask_volume_ml=mask_vol_mm3 / 1000.0,
        mesh_faces=int(len(mesh.faces)),
        mesh_vertices=int(len(mesh.vertices)),
        mesh_watertight=bool(mesh.is_watertight),
        mesh_volume_mm3=mesh_vol,
        label_voxel_counts=label_counts,
        boundary={
            "inlets": boundary_dict.get("inlet_names", ["left_nostril", "right_nostril"]),
            "outlet": boundary_dict.get("outlet_name", "trachea_outlet_proxy"),
            "mouth": "closed",
            "mean_inspiratory_flow_L_per_min": breathing.mean_inspiratory_flow_L_per_min,
            "inspiratory_time_s": breathing.Ti_s,
            "outlet_is_proxy": boundary_dict.get("outlet_is_proxy", True),
            "detail_file": f"{case_id}_boundary_conditions.json" if boundary_dict else None,
        },
        breathing_summary=breath_txt,
    )

    stats_path = output_dir / f"{case_id}_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)

    print(f"[{case_id}] mask voxels={mask_voxels:,}  volume={stats.mask_volume_ml:.2f} mL")
    print(f"[{case_id}] mesh verts={stats.mesh_vertices:,} faces={stats.mesh_faces:,} watertight={stats.mesh_watertight}")
    print(f"[{case_id}] wrote {mask_path.name}, {stl_path.name}, {preview_path.name}, {stats_path.name}")
    return stats


def resolve_nasalseg_case(
    data_root: Path | str,
    case_id: str = "P001",
) -> tuple[Path, Path]:
    """Return (image_path, label_path) for a NasalSeg case id like P001."""
    root = Path(data_root)
    # Support either data/NasalSeg or data/NasalSeg/NasalSeg nesting
    candidates = [
        root,
        root / "NasalSeg",
        root / "images",
    ]
    images_dir = None
    labels_dir = None
    for c in candidates:
        if (c / "images").is_dir() and (c / "labels").is_dir():
            images_dir = c / "images"
            labels_dir = c / "labels"
            break
        if c.name == "images" and c.is_dir():
            images_dir = c
            labels_dir = c.parent / "labels"
            break
    if images_dir is None or labels_dir is None:
        raise FileNotFoundError(f"Could not find NasalSeg images/labels under {root}")

    image_path = images_dir / f"{case_id}_img.nrrd"
    label_path = labels_dir / f"{case_id}_seg.nrrd"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    return image_path, label_path
