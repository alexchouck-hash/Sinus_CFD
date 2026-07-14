"""
Multi-tissue CT classification for Sinus_CFD.

Long-term classes (label IDs):
  0 = exterior / background (outside body)
  1 = air (fluid domain for airflow)
  2 = soft_tissue
  3 = cartilage (approximate HU band; optional)
  4 = bone

Current heuristics are HU + morphology based (no ML). Cartilage is a
rough HU band between soft tissue and bone and should be treated as
approximate until trained models are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

# Canonical label map
TISSUE_LABELS = {
    0: "exterior",
    1: "air",
    2: "soft_tissue",
    3: "cartilage",
    4: "bone",
}

# Default HU windows (approximate, CT number scale)
DEFAULT_HU = {
    "air_max": -300.0,  # free / partial-volume air
    "air_min": -1024.0,
    "soft_min": -200.0,
    "soft_max": 200.0,
    "cartilage_min": 80.0,  # rough; overlaps soft/bone
    "cartilage_max": 300.0,
    "bone_min": 300.0,
}


@dataclass
class TissueSegResult:
    labels: np.ndarray  # int16 (z,y,x)
    body: np.ndarray  # bool body mask
    air: np.ndarray
    soft_tissue: np.ndarray
    cartilage: np.ndarray
    bone: np.ndarray
    params: dict[str, Any]


def segment_body(
    hu: np.ndarray,
    body_hu_min: float = -200.0,
    min_component_voxels: int = 50_000,
) -> np.ndarray:
    """Largest filled body (soft tissue + bone + enclosed cavities)."""
    seed = hu > body_hu_min
    seed = morphology.opening(seed, footprint=morphology.ball(1))
    labeled, n = ndi.label(seed)
    if n == 0:
        raise ValueError("No body tissue found")
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = counts[1:] >= min_component_voxels
    if not keep.any():
        keep[int(np.argmax(counts))] = True
    body = keep[labeled]
    body = ndi.binary_fill_holes(body)
    body = morphology.closing(body, footprint=morphology.ball(2))
    body = ndi.binary_fill_holes(body)
    return body.astype(bool)


def segment_tissues(
    hu: np.ndarray,
    hu_params: dict[str, float] | None = None,
    body_hu_min: float = -200.0,
) -> TissueSegResult:
    """
    Classify voxels into exterior / air / soft tissue / cartilage / bone.

    Order of assignment (later overwrites earlier on conflicts):
      body → soft default → bone → cartilage → air (air wins for fluid domain)
    """
    p = {**DEFAULT_HU, **(hu_params or {})}
    body = segment_body(hu, body_hu_min=body_hu_min)

    labels = np.zeros(hu.shape, dtype=np.int16)
    labels[body] = 2  # soft tissue default inside body

    bone = body & (hu >= p["bone_min"])
    # Clean bone speckles
    bone = morphology.opening(bone, footprint=morphology.ball(1))
    labels[bone] = 4

    # Cartilage approx: intermediate HU, not bone, inside body
    cart = body & (hu >= p["cartilage_min"]) & (hu < p["bone_min"]) & ~bone
    # Keep larger cartilage-ish blobs only
    lab_c, n_c = ndi.label(cart)
    if n_c:
        cc = np.bincount(lab_c.ravel())
        cc[0] = 0
        cart = cc[lab_c] >= 30
    labels[cart] = 3

    # Air: free / partial-volume air enclosed by body
    air = body & (hu >= p["air_min"]) & (hu <= p["air_max"])
    air = morphology.closing(air, footprint=morphology.ball(1))
    lab_a, n_a = ndi.label(air)
    if n_a:
        ca = np.bincount(lab_a.ravel())
        ca[0] = 0
        air = ca[lab_a] >= 80
    labels[air] = 1

    # Soft tissue = body minus air/bone (cartilage remains 3)
    soft = body & ~air & ~bone
    # restore cartilage labels where set
    soft = soft & (labels != 3)
    labels[soft] = 2
    labels[cart] = 3
    labels[air] = 1
    labels[bone] = 4
    labels[~body] = 0

    return TissueSegResult(
        labels=labels,
        body=body,
        air=air.astype(bool),
        soft_tissue=(labels == 2),
        cartilage=(labels == 3),
        bone=(labels == 4),
        params=p,
    )
