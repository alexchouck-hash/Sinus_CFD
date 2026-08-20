"""Hybrid sinonasal segmentation: named NasalSeg classes + leftover-air fill.

Learned 5-class labels (expert NasalSeg or Dataset501 nnU-Net) are copied
verbatim. HU air that those labels did not name is assigned to frontal,
ethmoid, or sphenoid by component centroid relative to the named airway.
Named voxels are never overwritten.

Ostia are not a pixel class: ``ostial_contact_voxels`` reports the dilated
overlap between each sinus group and the passage, i.e. the necks used later
for navigation (Stage D).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .segmentation_labels import (
    BACKGROUND,
    LABEL_NAMES,
    LEFT_ETHMOID,
    LEFT_FRONTAL,
    LEFT_MAXILLARY,
    PASSAGE_IDS,
    RIGHT_ETHMOID,
    RIGHT_FRONTAL,
    RIGHT_MAXILLARY,
    SINUS_GROUPS,
    SINUS_IDS,
    SPHENOID,
    label_info,
)


@dataclass
class SinonasalSegmentation:
    case_id: str
    labels: np.ndarray
    notes: list[str] = field(default_factory=list)
    voxel_counts: dict[str, int] = field(default_factory=dict)
    ostial_contact_voxels: dict[str, int] = field(default_factory=dict)
    unassigned_air_voxels: int = 0

    def passage_mask(self) -> np.ndarray:
        return np.isin(self.labels, PASSAGE_IDS)

    def sinus_mask(self) -> np.ndarray:
        return np.isin(self.labels, SINUS_IDS)

    def to_meta(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "label_names": dict(LABEL_NAMES),
            "voxel_counts": self.voxel_counts,
            "ostial_contact_voxels": self.ostial_contact_voxels,
            "unassigned_air_voxels": int(self.unassigned_air_voxels),
            "notes": list(self.notes),
            "classes": [asdict(label_info(i)) for i in sorted(LABEL_NAMES)],
        }


def _air_mask(
    hu: np.ndarray,
    body: np.ndarray | None,
    hu_max: float,
    hu_min: float,
) -> np.ndarray:
    air = (hu >= hu_min) & (hu <= hu_max)
    if body is not None:
        air = air & body.astype(bool)
    return air


def _copy_named(named: np.ndarray) -> np.ndarray:
    """Keep NasalSeg IDs 1–5; drop anything outside that range."""
    out = np.zeros(named.shape, dtype=np.uint8)
    valid = (named >= 1) & (named <= 5)
    out[valid] = named[valid].astype(np.uint8)
    return out


def _split_lr(
    mask: np.ndarray,
    x_mid: float,
    high_x_is_patient_left: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a bilateral mask on a sagittal plane. High x = patient left by default."""
    xx = np.arange(mask.shape[2], dtype=np.float32)[None, None, :]
    if high_x_is_patient_left:
        left = mask & (xx >= x_mid)
        right = mask & (xx < x_mid)
    else:
        left = mask & (xx < x_mid)
        right = mask & (xx >= x_mid)
    return left, right


def _cavity_frame(
    labels: np.ndarray,
    leftover: np.ndarray,
    y_anterior_is_low: bool,
    superior_is_high_z: bool,
    high_x_is_patient_left: bool,
) -> dict[str, float]:
    """Percentile frame of named passage, falling back to leftover air."""
    ref = np.isin(labels, PASSAGE_IDS)
    if int(ref.sum()) < 30:
        ref = leftover
    zz, yy, xx = np.where(ref)
    if len(zz) < 5:
        zz, yy, xx = np.where(np.ones(labels.shape, dtype=bool))
    frame = {
        "z_lo": float(np.percentile(zz, 15)),
        "z_hi": float(np.percentile(zz, 85)),
        "y_lo": float(np.percentile(yy, 15)),
        "y_hi": float(np.percentile(yy, 85)),
        "x_mid": float(np.median(xx)),
        "y_anterior_is_low": float(y_anterior_is_low),
        "superior_is_high_z": float(superior_is_high_z),
        "high_x_is_patient_left": float(high_x_is_patient_left),
    }
    return frame


def _classify_component(
    zc: float,
    yc: float,
    xc: float,
    frame: dict[str, float],
    need_maxillary: bool,
) -> str:
    """Map a leftover-air centroid onto a sinus name."""
    z_lo, z_hi = frame["z_lo"], frame["z_hi"]
    y_lo, y_hi = frame["y_lo"], frame["y_hi"]
    x_mid = frame["x_mid"]
    y_ant_low = bool(frame["y_anterior_is_low"])
    z_sup_high = bool(frame["superior_is_high_z"])

    z_span = max(z_hi - z_lo, 1.0)
    y_span = max(y_hi - y_lo, 1.0)
    z_rel = (zc - z_lo) / z_span  # 0 at z_lo, 1 at z_hi
    if not z_sup_high:
        z_rel = 1.0 - z_rel  # 1 = superior
    y_frac = (yc - y_lo) / y_span  # 0 at y_lo, 1 at y_hi
    y_rel_ant = (1.0 - y_frac) if y_ant_low else y_frac

    x_off = abs(xc - x_mid)

    # Anterior + superior → frontal.
    if z_rel >= 0.55 and y_rel_ant >= 0.55:
        return "frontal"
    # Posterior + central → sphenoid.
    if y_rel_ant <= 0.40 and x_off <= 12 and z_rel >= 0.25:
        return "sphenoid"
    # Far lateral, mid-face → maxillary (only if NasalSeg didn't already).
    if need_maxillary and x_off >= 8 and 0.20 <= z_rel <= 0.70:
        return "maxillary"
    # Remaining mid-superior leftover → ethmoid cells.
    if z_rel >= 0.35:
        return "ethmoid"
    return "unassigned"


def ostial_contact_voxels(labels: np.ndarray, dilation: int = 2) -> dict[str, int]:
    """Voxels where a dilated sinus group touches the passage (ostial necks)."""
    passage = np.isin(labels, PASSAGE_IDS)
    struct = ndi.generate_binary_structure(3, 1)
    out: dict[str, int] = {}
    for name, ids in SINUS_GROUPS.items():
        sinus = np.isin(labels, ids)
        if not sinus.any():
            out[name] = 0
            continue
        dilated = ndi.binary_dilation(sinus, structure=struct, iterations=int(dilation))
        out[name] = int((dilated & passage).sum())
    return out


def expand_named_airspaces(
    hu_zyx: np.ndarray,
    named_labels: np.ndarray | None,
    *,
    case_id: str = "case",
    body: np.ndarray | None = None,
    hu_max: float = -300.0,
    hu_min: float = -1024.0,
    min_component_voxels: int = 40,
    superior_is_high_z: bool = True,
    y_anterior_is_low: bool = True,
    high_x_is_patient_left: bool = True,
) -> SinonasalSegmentation:
    """
    Build the 11-class map.

    ``named_labels`` is NasalSeg / nnU-Net IDs 0–5 on the same (z, y, x) grid.
    Pass None only for a heuristic-sinus preview; the CFD passage will be empty.
    """
    notes: list[str] = []
    air = _air_mask(hu_zyx, body, hu_max=hu_max, hu_min=hu_min)

    if named_labels is None:
        out = np.zeros(hu_zyx.shape, dtype=np.uint8)
        notes.append("no named labels; passage empty; leftover air classified by centroid")
    else:
        if named_labels.shape != hu_zyx.shape:
            raise ValueError(
                f"named_labels shape {named_labels.shape} != HU shape {hu_zyx.shape}"
            )
        out = _copy_named(named_labels)
        notes.append("seeded from NasalSeg-schema IDs 1–5 (never overwritten)")

    leftover = air & (out == BACKGROUND)
    frame = _cavity_frame(
        out, leftover, y_anterior_is_low, superior_is_high_z, high_x_is_patient_left
    )
    x_mid = frame["x_mid"]
    need_maxillary = int((out == LEFT_MAXILLARY).sum()) == 0 or int(
        (out == RIGHT_MAXILLARY).sum()
    ) == 0

    lab, n = ndi.label(leftover)
    assigned = 0
    unassigned_ids: list[int] = []
    for cid in range(1, n + 1):
        comp = lab == cid
        nvox = int(comp.sum())
        if nvox < min_component_voxels:
            continue
        zz, yy, xx = np.where(comp)
        name = _classify_component(
            float(zz.mean()),
            float(yy.mean()),
            float(xx.mean()),
            frame,
            need_maxillary=need_maxillary,
        )
        if name == "frontal":
            left, right = _split_lr(comp, x_mid, high_x_is_patient_left)
            out[left] = LEFT_FRONTAL
            out[right] = RIGHT_FRONTAL
            assigned += nvox
        elif name == "sphenoid":
            out[comp] = SPHENOID
            assigned += nvox
        elif name == "maxillary":
            left, right = _split_lr(comp, x_mid, high_x_is_patient_left)
            if int((out == LEFT_MAXILLARY).sum()) == 0:
                out[left] = LEFT_MAXILLARY
            if int((out == RIGHT_MAXILLARY).sum()) == 0:
                out[right] = RIGHT_MAXILLARY
            assigned += nvox
        elif name == "ethmoid":
            left, right = _split_lr(comp, x_mid, high_x_is_patient_left)
            out[left] = LEFT_ETHMOID
            out[right] = RIGHT_ETHMOID
            assigned += nvox
        else:
            unassigned_ids.append(cid)

    unassigned = int((air & (out == BACKGROUND)).sum())
    notes.append(
        f"leftover air components={n}; assigned_voxels={assigned}; "
        f"unassigned_air_voxels={unassigned}"
    )

    counts = {LABEL_NAMES[i]: int((out == i).sum()) for i in sorted(LABEL_NAMES) if i}
    contacts = ostial_contact_voxels(out)
    notes.append("ostial_contact_voxels=" + str(contacts))

    return SinonasalSegmentation(
        case_id=case_id,
        labels=out,
        notes=notes,
        voxel_counts=counts,
        ostial_contact_voxels=contacts,
        unassigned_air_voxels=unassigned,
    )


def passage_and_sinus_masks(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.isin(labels, PASSAGE_IDS), np.isin(labels, SINUS_IDS)
