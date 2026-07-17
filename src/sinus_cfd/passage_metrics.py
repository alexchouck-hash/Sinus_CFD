"""
Per-side nasal airway geometry metrics — Stage 2 of the roadmap
(docs/architecture_and_roadmap.md §5: "Geometry first, no CFD needed").

The clinically load-bearing number is the **minimal cross-sectional area
(MCA)** and where along the passage it sits — the classic constriction
detector, and what acoustic rhinometry / virtual-surgery CFD papers report.

We compute area as the actual lumen voxel count in each coronal
(anterior-posterior) slice, *not* the π·r² distance-to-wall approximation in
`nasal_passage.cross_sections_along_centerline`. That approximation models the
cross-section as a disk of radius = distance to the nearest wall, which
collapses for slit-shaped passages — and the nasal valve, the usual MCA
location, is precisely a tall thin slit. Slice-area is faithful there.

Left and right nasal cavities are separate NasalSeg labels (1 and 2), and the
trained nnU-Net predicts the same, so per-side analysis needs no extra
splitting — it drops straight out of the label map.

Orientation assumption: NasalSeg volumes are axial head CT, so array axis 1
(y) is the anterior-posterior airflow axis; a coronal slice is a fixed-y
(z, x) plane with pixel area sx·sz. `analyze_bilateral` asserts nothing about
handedness of x — it reports whichever labels you pass as left/right.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi


@dataclass
class SideMetrics:
    side: str
    label_id: int
    present: bool
    volume_ml: float
    ap_extent_mm: float
    n_slices: int
    mca_mm2: float
    mca_ap_position_mm: float  # distance from the anterior-most slice of THIS side
    mca_location: str  # "anterior" | "middle" | "posterior" third of the AP extent
    mca_at_terminal_slice: bool  # min sits at the first/last retained slice (end narrowing)
    mean_area_mm2: float
    max_area_mm2: float
    area_profile: list[dict[str, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BilateralMetrics:
    case_id: str
    mask_source: str
    spacing_xyz_mm: list[float]
    ap_axis: str
    left: dict[str, Any]
    right: dict[str, Any]
    mca_ratio: float  # min(L,R) / max(L,R); 1.0 = symmetric, →0 = very asymmetric
    more_obstructed_side: str
    total_volume_ml: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected component (drops stray predicted islands)."""
    if not mask.any():
        return mask
    labeled, n = ndi.label(mask)
    if n <= 1:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return labeled == int(np.argmax(counts))


def coronal_area_profile(
    side_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    ap_axis: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross-sectional lumen area per anterior-posterior slice.

    Returns (ap_positions_mm, areas_mm2) over slices that contain lumen,
    ordered anterior→posterior. ap_positions are relative to the anterior-most
    lumen slice of this mask (so 0 = the naris end of this cavity).
    """
    sx, sy, sz = spacing_xyz
    # In-plane pixel area for a fixed-y (coronal) slice is sx * sz.
    in_plane = {0: sy * sx, 1: sx * sz, 2: sy * sz}[ap_axis]
    slice_spacing = {0: sz, 1: sy, 2: sx}[ap_axis]

    # Count lumen voxels per AP slice.
    other_axes = tuple(a for a in range(3) if a != ap_axis)
    counts = side_mask.sum(axis=other_axes)  # 1-D over the AP axis
    present = np.where(counts > 0)[0]
    if len(present) == 0:
        return np.array([]), np.array([])
    a0 = int(present.min())
    ap_positions = (present - a0).astype(float) * slice_spacing
    areas = counts[present].astype(float) * in_plane
    return ap_positions, areas


def _smooth(a: np.ndarray, window: int = 3) -> np.ndarray:
    """Odd-window moving average; guards single-voxel area spikes."""
    if len(a) < window or window < 2:
        return a
    if window % 2 == 0:
        window += 1
    kern = np.ones(window) / window
    return np.convolve(a, kern, mode="same")


def analyze_side(
    side_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    side: str,
    label_id: int,
    ap_axis: int = 1,
    area_floor_mm2: float = 5.0,
    keep_largest: bool = True,
) -> SideMetrics:
    """
    Volume + minimal cross-sectional area (MCA) for one nasal cavity.

    The MCA is the minimum of the coronal area profile over the airway *body*
    — slices whose area is at least ``area_floor_mm2`` — which drops only the
    single-voxel partial-volume taper at the very ends (area → 0), not the
    genuine anterior narrowing. NasalSeg's nasal-cavity label tends to begin
    around the nasal-valve region, so that anterior narrowing is usually the
    clinically meaningful MCA rather than an artifact.

    These profiles are typically unimodal (widen from naris to a mid-cavity
    peak, narrow to the choana) with no separate internal notch, so the MCA
    commonly sits at the anterior or posterior end of the retained body — that
    is reported honestly via ``mca_location`` and ``mca_at_terminal_slice``
    rather than hidden by trimming.
    """
    notes: list[str] = []
    mask = side_mask.astype(bool)
    if keep_largest:
        mask = _largest_component(mask)

    sx, sy, sz = spacing_xyz
    voxel_ml = (sx * sy * sz) / 1000.0
    volume_ml = float(mask.sum()) * voxel_ml

    ap_mm, areas = coronal_area_profile(mask, spacing_xyz, ap_axis=ap_axis)
    if len(areas) == 0:
        return SideMetrics(
            side=side, label_id=label_id, present=False, volume_ml=0.0,
            ap_extent_mm=0.0, n_slices=0, mca_mm2=0.0, mca_ap_position_mm=0.0,
            mca_location="none", mca_at_terminal_slice=False,
            mean_area_mm2=0.0, max_area_mm2=0.0,
            notes=[f"{side}: no lumen for label {label_id}."],
        )

    areas_s = _smooth(areas, window=3)
    n = len(areas_s)

    # Airway body = slices above the taper floor. Fall back to all slices if
    # the whole passage is below the floor (very small cavity).
    body = np.where(areas_s >= area_floor_mm2)[0]
    if len(body) == 0:
        body = np.arange(n)
        notes.append(f"All slices below area floor {area_floor_mm2} mm²; MCA over full length.")

    j = int(body[np.argmin(areas_s[body])])
    mca = float(areas_s[j])

    frac = ap_mm[j] / ap_mm[-1] if ap_mm[-1] > 0 else 0.0
    location = "anterior" if frac < 1 / 3 else "posterior" if frac > 2 / 3 else "middle"
    at_terminal = j == int(body[0]) or j == int(body[-1])
    if at_terminal:
        notes.append(
            f"MCA at the {location} end of the airway body (unimodal profile, no "
            "distinct internal constriction) — represents end narrowing, not a "
            "focal internal stenosis."
        )

    return SideMetrics(
        side=side,
        label_id=label_id,
        present=True,
        volume_ml=volume_ml,
        ap_extent_mm=float(ap_mm[-1]),
        n_slices=n,
        mca_mm2=mca,
        mca_ap_position_mm=float(ap_mm[j]),
        mca_location=location,
        mca_at_terminal_slice=at_terminal,
        mean_area_mm2=float(np.mean(areas)),
        max_area_mm2=float(np.max(areas)),
        area_profile=[
            {"ap_mm": round(float(p), 3), "area_mm2": round(float(a), 3)}
            for p, a in zip(ap_mm, areas)
        ],
        notes=notes,
    )


def analyze_bilateral(
    label_zyx: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    case_id: str,
    mask_source: str = "labels",
    left_label: int = 1,
    right_label: int = 2,
    ap_axis: int = 1,
) -> BilateralMetrics:
    """
    Left vs right nasal-cavity geometry from a label map (NasalSeg or nnU-Net).

    Reports each side's MCA and the L/R MCA ratio — the asymmetry that flags a
    unilaterally obstructed airway (deviated septum, turbinate hypertrophy).
    """
    left = analyze_side(
        label_zyx == left_label, spacing_xyz, "left", left_label, ap_axis=ap_axis
    )
    right = analyze_side(
        label_zyx == right_label, spacing_xyz, "right", right_label, ap_axis=ap_axis
    )

    notes: list[str] = []
    if left.present and right.present and max(left.mca_mm2, right.mca_mm2) > 0:
        lo = min(left.mca_mm2, right.mca_mm2)
        hi = max(left.mca_mm2, right.mca_mm2)
        mca_ratio = lo / hi
        more_obstructed = left.side if left.mca_mm2 < right.mca_mm2 else right.side
    else:
        mca_ratio = float("nan")
        more_obstructed = "unknown"
        notes.append("One or both nasal cavities absent — cannot compare MCA.")

    return BilateralMetrics(
        case_id=case_id,
        mask_source=mask_source,
        spacing_xyz_mm=list(spacing_xyz),
        ap_axis={0: "z", 1: "y", 2: "x"}[ap_axis],
        left=left.to_dict(),
        right=right.to_dict(),
        mca_ratio=mca_ratio,
        more_obstructed_side=more_obstructed,
        total_volume_ml=left.volume_ml + right.volume_ml,
        notes=notes,
    )
