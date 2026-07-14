"""
Paranasal sinus localization on head CT (demo heuristics).

Labels large dark-air pockets relative to the nasal cavities:
  - frontal: superior + anterior
  - maxillary L/R: lateral mid-face
  - sphenoid: posterior central

Not a trained segmenter — anatomical ROI + connected components on HU air.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology


@dataclass
class SinusLabel:
    name: str
    center_mm: list[float]
    center_zyx: list[int]
    voxels: int
    bbox_zyx: list[list[int]]  # [[z0,z1],[y0,y1],[x0,x1]]
    notes: str = ""


@dataclass
class SinusAnatomyResult:
    case_id: str
    frontal: np.ndarray
    sphenoid: np.ndarray
    maxillary_left: np.ndarray
    maxillary_right: np.ndarray
    labels: list[SinusLabel] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_meta(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "labels": [asdict(L) for L in self.labels],
            "voxels": {
                "frontal": int(self.frontal.sum()),
                "sphenoid": int(self.sphenoid.sum()),
                "maxillary_left": int(self.maxillary_left.sum()),
                "maxillary_right": int(self.maxillary_right.sum()),
            },
            "notes": self.notes,
        }


def _zyx_to_mm(
    z: float, y: float, x: float, spacing_xyz: tuple[float, float, float], origin_xyz: tuple[float, float, float]
) -> list[float]:
    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    return [float(ox + x * sx), float(oy + y * sy), float(oz + z * sz)]


def _largest_components(mask: np.ndarray, min_voxels: int = 80, top_k: int = 6) -> list[np.ndarray]:
    lab, n = ndi.label(mask.astype(bool))
    if n == 0:
        return []
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    order = np.argsort(counts)[::-1]
    out: list[np.ndarray] = []
    for i in order[:top_k]:
        if counts[i] < min_voxels:
            break
        out.append(lab == int(i))
    return out


def _label_from_mask(
    name: str,
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    note: str = "",
) -> SinusLabel | None:
    zz, yy, xx = np.where(mask)
    if len(zz) < 20:
        return None
    zc, yc, xc = float(zz.mean()), float(yy.mean()), float(xx.mean())
    return SinusLabel(
        name=name,
        center_mm=_zyx_to_mm(zc, yc, xc, spacing_xyz, origin_xyz),
        center_zyx=[int(round(zc)), int(round(yc)), int(round(xc))],
        voxels=int(len(zz)),
        bbox_zyx=[
            [int(zz.min()), int(zz.max())],
            [int(yy.min()), int(yy.max())],
            [int(xx.min()), int(xx.max())],
        ],
        notes=note,
    )


def detect_paranasal_sinuses(
    hu: np.ndarray,
    body: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    case_id: str = "case",
    air_hu_max: float = -300.0,
    superior_is_high_z: bool = True,
    y_anterior_is_low: bool = True,
    nasal_mask: np.ndarray | None = None,
) -> SinusAnatomyResult:
    """
    Heuristic paranasal sinus masks from CT air topology.
    """
    notes: list[str] = []
    body = body.astype(bool)
    air = body & (hu <= air_hu_max) & (hu >= -1024)
    air = morphology.closing(air, footprint=morphology.ball(1))
    notes.append(f"Air HU≤{air_hu_max}: {int(air.sum())} voxels")

    nz, ny, nx = air.shape
    zz_a, yy_a, xx_a = np.where(air)
    if len(zz_a) < 100:
        empty = np.zeros_like(air)
        return SinusAnatomyResult(
            case_id=case_id,
            frontal=empty,
            sphenoid=empty,
            maxillary_left=empty,
            maxillary_right=empty,
            notes=notes + ["Insufficient air for sinus detection."],
        )

    # Percentiles of air extent for adaptive ROIs
    z_lo, z_hi = np.percentile(zz_a, [15, 85])
    y_lo, y_hi = np.percentile(yy_a, [10, 90])
    x_mid = float(np.median(xx_a))

    # ----- Frontal: superior + anterior -----
    frontal_roi = air.copy()
    if superior_is_high_z:
        frontal_roi[: int(np.percentile(zz_a, 62))] = False
    else:
        frontal_roi[int(np.percentile(zz_a, 38)) :] = False
    if y_anterior_is_low:
        frontal_roi[:, int(np.percentile(yy_a, 42)) :, :] = False
    else:
        frontal_roi[:, : int(np.percentile(yy_a, 58)), :] = False
    # mid-face x band (avoid orbits far lateral)
    frontal_roi[:, :, : max(0, int(x_mid - 45))] = False
    frontal_roi[:, :, min(nx, int(x_mid + 45)) :] = False
    comps = _largest_components(frontal_roi, min_voxels=120, top_k=8)
    frontal = np.zeros_like(air)
    for c in comps[:3]:  # merge top frontal pockets
        frontal |= c
    # keep largest 1–2 clusters after merge
    frontal_keep = _largest_components(frontal, min_voxels=80, top_k=2)
    frontal = np.zeros_like(air)
    for c in frontal_keep:
        frontal |= c
    notes.append(f"Frontal ROI voxels={int(frontal.sum())} from {len(comps)} candidates")

    # ----- Maxillary L/R: lateral mid-face -----
    z0, z1 = int(np.percentile(zz_a, 28)), int(np.percentile(zz_a, 72))
    if y_anterior_is_low:
        y0, y1 = int(np.percentile(yy_a, 18)), int(np.percentile(yy_a, 68))
    else:
        y0, y1 = int(np.percentile(yy_a, 32)), int(np.percentile(yy_a, 82))

    max_l_roi = air.copy()
    max_l_roi[:z0] = False
    max_l_roi[z1:] = False
    max_l_roi[:, :y0, :] = False
    max_l_roi[:, y1:, :] = False
    max_l_roi[:, :, : int(x_mid + 12)] = False  # left = high x
    # Prefer components outside pure nasal core if provided
    if nasal_mask is not None:
        core = morphology.binary_erosion(nasal_mask.astype(bool), footprint=morphology.ball(2))
        max_l_roi = max_l_roi & ~core
    max_l_comps = _largest_components(max_l_roi, min_voxels=200, top_k=3)
    maxillary_left = max_l_comps[0] if max_l_comps else np.zeros_like(air)

    max_r_roi = air.copy()
    max_r_roi[:z0] = False
    max_r_roi[z1:] = False
    max_r_roi[:, :y0, :] = False
    max_r_roi[:, y1:, :] = False
    max_r_roi[:, :, int(x_mid - 12) :] = False  # right = low x
    if nasal_mask is not None:
        core = morphology.binary_erosion(nasal_mask.astype(bool), footprint=morphology.ball(2))
        max_r_roi = max_r_roi & ~core
    max_r_comps = _largest_components(max_r_roi, min_voxels=200, top_k=3)
    maxillary_right = max_r_comps[0] if max_r_comps else np.zeros_like(air)
    notes.append(
        f"Maxillary L={int(maxillary_left.sum())} R={int(maxillary_right.sum())}"
    )

    # ----- Sphenoid: posterior + mid-superior + central -----
    sph_roi = air.copy()
    if y_anterior_is_low:
        sph_roi[:, : int(np.percentile(yy_a, 52)), :] = False
    else:
        sph_roi[:, int(np.percentile(yy_a, 48)) :, :] = False
    if superior_is_high_z:
        sph_roi[: int(np.percentile(zz_a, 38))] = False
        sph_roi[int(np.percentile(zz_a, 88)) :] = False
    else:
        sph_roi[int(np.percentile(zz_a, 62)) :] = False
        sph_roi[: int(np.percentile(zz_a, 12))] = False
    sph_roi[:, :, : int(x_mid - 28)] = False
    sph_roi[:, :, int(x_mid + 28) :] = False
    # Exclude maxillary laterals
    sph_roi &= ~maxillary_left & ~maxillary_right
    sph_comps = _largest_components(sph_roi, min_voxels=150, top_k=4)
    sphenoid = np.zeros_like(air)
    for c in sph_comps[:2]:
        sphenoid |= c
    notes.append(f"Sphenoid voxels={int(sphenoid.sum())}")

    labels: list[SinusLabel] = []
    for name, m, note in (
        ("frontal", frontal, "Superior-anterior air (frontal sinus region)"),
        ("sphenoid", sphenoid, "Posterior central air (sphenoid region)"),
        ("maxillary_left", maxillary_left, "Left lateral mid-face air"),
        ("maxillary_right", maxillary_right, "Right lateral mid-face air"),
    ):
        L = _label_from_mask(name, m, spacing_xyz, origin_xyz, note)
        if L is not None:
            labels.append(L)
            notes.append(f"{name} center_mm={L.center_mm} n={L.voxels}")

    return SinusAnatomyResult(
        case_id=case_id,
        frontal=frontal.astype(bool),
        sphenoid=sphenoid.astype(bool),
        maxillary_left=maxillary_left.astype(bool),
        maxillary_right=maxillary_right.astype(bool),
        labels=labels,
        notes=notes,
    )
