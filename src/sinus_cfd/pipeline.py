"""
Load a NasalSeg CT case, build an airway mask, and export a surface mesh.

Primary mask source: expert labels (labels 1–5 = nasal cavity L/R,
nasopharynx, maxillary sinus L/R). Optional HU air threshold for comparison.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import trimesh
from scipy import ndimage as ndi
from skimage import measure, morphology

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


def _clean_mask(
    mask: np.ndarray,
    min_component_voxels: int = 200,
    closing_radius: int = 1,
) -> np.ndarray:
    """Binary close small holes, drop tiny components, keep largest blob."""
    cleaned = mask.astype(bool)
    if closing_radius > 0:
        structure = morphology.ball(closing_radius)
        # skimage >=0.26: use morphology.closing on bool images
        cleaned = morphology.closing(cleaned, footprint=structure)

    labeled, n = ndi.label(cleaned)
    if n == 0:
        return cleaned

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    # Drop tiny islands
    for lab_id, count in enumerate(counts):
        if lab_id == 0:
            continue
        if count < min_component_voxels:
            cleaned[labeled == lab_id] = False

    labeled, n = ndi.label(cleaned)
    if n == 0:
        return cleaned
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = int(np.argmax(counts))
    return labeled == largest


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
) -> CaseStats:
    """
    Process one CT case into mask + STL surface + preview.

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

    mask = _clean_mask(mask, min_component_voxels=min_component_voxels)

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
