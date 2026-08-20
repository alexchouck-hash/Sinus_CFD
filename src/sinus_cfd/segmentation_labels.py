"""Canonical sinonasal airspace label map for Sinus_CFD.

NasalSeg (Zhang et al., Sci Data 2024) IDs 0–5 are kept unchanged so the
existing Dataset501 nnU-Net and expert NRRD labels drop in without remap.
IDs 6–10 are the structures NasalSeg does not name: frontal, ethmoid, and
sphenoid. Those are filled by the hybrid expander in ``segment_sinonasal``.

See ``docs/segmentation_method.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


BACKGROUND = 0
LEFT_NASAL_CAVITY = 1
RIGHT_NASAL_CAVITY = 2
NASOPHARYNX = 3
LEFT_MAXILLARY = 4
RIGHT_MAXILLARY = 5
LEFT_FRONTAL = 6
RIGHT_FRONTAL = 7
LEFT_ETHMOID = 8
RIGHT_ETHMOID = 9
SPHENOID = 10

# NasalSeg / Dataset501 (Zhang et al. 2024). Do not renumber.
NASALSEG_IDS = {
    "background": BACKGROUND,
    "left_nasal_cavity": LEFT_NASAL_CAVITY,
    "right_nasal_cavity": RIGHT_NASAL_CAVITY,
    "nasopharynx": NASOPHARYNX,
    "left_maxillary_sinus": LEFT_MAXILLARY,
    "right_maxillary_sinus": RIGHT_MAXILLARY,
}

# Full target map (Dataset503 when labeled data exists).
LABEL_NAMES = {
    BACKGROUND: "background",
    LEFT_NASAL_CAVITY: "left_nasal_cavity",
    RIGHT_NASAL_CAVITY: "right_nasal_cavity",
    NASOPHARYNX: "nasopharynx",
    LEFT_MAXILLARY: "left_maxillary_sinus",
    RIGHT_MAXILLARY: "right_maxillary_sinus",
    LEFT_FRONTAL: "left_frontal_sinus",
    RIGHT_FRONTAL: "right_frontal_sinus",
    LEFT_ETHMOID: "left_ethmoid",
    RIGHT_ETHMOID: "right_ethmoid",
    SPHENOID: "sphenoid",
}

# CFD inspiratory domain: cavity + nasopharynx. Sinuses are dead-ends.
PASSAGE_IDS = (LEFT_NASAL_CAVITY, RIGHT_NASAL_CAVITY, NASOPHARYNX)
SINUS_IDS = (
    LEFT_MAXILLARY,
    RIGHT_MAXILLARY,
    LEFT_FRONTAL,
    RIGHT_FRONTAL,
    LEFT_ETHMOID,
    RIGHT_ETHMOID,
    SPHENOID,
)

SINUS_GROUPS = {
    "maxillary": (LEFT_MAXILLARY, RIGHT_MAXILLARY),
    "frontal": (LEFT_FRONTAL, RIGHT_FRONTAL),
    "ethmoid": (LEFT_ETHMOID, RIGHT_ETHMOID),
    "sphenoid": (SPHENOID,),
}


@dataclass(frozen=True)
class LabelInfo:
    id: int
    name: str
    role: str  # "background" | "passage" | "sinus"
    source: str  # "nasalseg" | "expanded"


def label_info(label_id: int) -> LabelInfo:
    name = LABEL_NAMES[int(label_id)]
    if label_id == BACKGROUND:
        role = "background"
    elif label_id in PASSAGE_IDS:
        role = "passage"
    else:
        role = "sinus"
    source = "nasalseg" if label_id <= RIGHT_MAXILLARY else "expanded"
    return LabelInfo(id=int(label_id), name=name, role=role, source=source)


def dataset503_json() -> dict:
    """nnU-Net v2 dataset.json labels block for a future Dataset503 train."""
    labels = {name: int(i) for i, name in LABEL_NAMES.items()}
    return {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": 0,
        "file_ending": ".nii.gz",
    }
