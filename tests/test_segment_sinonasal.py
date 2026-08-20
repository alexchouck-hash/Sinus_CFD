"""Hybrid sinonasal expander: keep NasalSeg IDs, fill leftover sinus air."""

from __future__ import annotations

import numpy as np

from sinus_cfd.segmentation_labels import (
    LEFT_FRONTAL,
    LEFT_NASAL_CAVITY,
    NASALSEG_IDS,
    NASOPHARYNX,
    PASSAGE_IDS,
    RIGHT_NASAL_CAVITY,
    SPHENOID,
)
from sinus_cfd.segment_sinonasal import expand_named_airspaces, ostial_contact_voxels


def _box(vol, z0, z1, y0, y1, x0, x1, value):
    vol[z0:z1, y0:y1, x0:x1] = value


def _synthetic_head() -> tuple[np.ndarray, np.ndarray]:
    """Tiny CT: named NasalSeg passage + leftover frontal and sphenoid air."""
    hu = np.full((32, 32, 32), 80.0, dtype=np.float32)
    labels = np.zeros((32, 32, 32), dtype=np.uint8)
    # Passage (NasalSeg 1–3). y low = anterior, z high = superior, x high = left.
    _box(labels, 10, 18, 8, 16, 18, 23, LEFT_NASAL_CAVITY)
    _box(labels, 10, 18, 8, 16, 9, 14, RIGHT_NASAL_CAVITY)
    _box(labels, 10, 16, 18, 24, 12, 20, NASOPHARYNX)
    _box(hu, 10, 18, 8, 16, 18, 23, -900.0)
    _box(hu, 10, 18, 8, 16, 9, 14, -900.0)
    _box(hu, 10, 16, 18, 24, 12, 20, -900.0)
    # Leftover frontal: superior + anterior, unnamed.
    _box(hu, 24, 30, 2, 7, 13, 19, -950.0)
    # Leftover sphenoid: posterior + central, unnamed.
    _box(hu, 14, 19, 26, 31, 13, 19, -950.0)
    return hu, labels


def test_nasalseg_ids_are_stable():
    assert NASALSEG_IDS["left_nasal_cavity"] == 1
    assert NASALSEG_IDS["right_maxillary_sinus"] == 5
    assert PASSAGE_IDS == (1, 2, 3)


def test_named_labels_never_overwritten():
    hu, labels = _synthetic_head()
    result = expand_named_airspaces(hu, labels, case_id="synth", min_component_voxels=8)
    named = labels > 0
    np.testing.assert_array_equal(result.labels[named], labels[named])


def test_leftover_frontal_and_sphenoid_are_named():
    hu, labels = _synthetic_head()
    result = expand_named_airspaces(hu, labels, case_id="synth", min_component_voxels=8)
    frontal = np.isin(result.labels, (LEFT_FRONTAL, LEFT_FRONTAL + 1))
    assert int(frontal[24:30, 2:7, 13:19].sum()) >= 20
    assert int((result.labels[14:19, 26:31, 13:19] == SPHENOID).sum()) >= 20
    assert result.voxel_counts["left_nasal_cavity"] == int((labels == LEFT_NASAL_CAVITY).sum())


def test_passage_excludes_expanded_sinuses():
    hu, labels = _synthetic_head()
    result = expand_named_airspaces(hu, labels, case_id="synth", min_component_voxels=8)
    passage = result.passage_mask()
    assert not np.any(passage & (result.labels == SPHENOID))
    assert not np.any(passage & (result.labels == LEFT_FRONTAL))
    assert np.all(np.isin(result.labels[passage], PASSAGE_IDS))


def test_ostial_contact_zero_when_sinus_isolated():
    labels = np.zeros((12, 12, 12), dtype=np.uint8)
    labels[2:5, 2:5, 2:5] = LEFT_NASAL_CAVITY
    labels[8:11, 8:11, 8:11] = SPHENOID
    contacts = ostial_contact_voxels(labels, dilation=1)
    assert contacts["sphenoid"] == 0


def test_ostial_contact_positive_when_sinus_touches_passage():
    labels = np.zeros((12, 12, 12), dtype=np.uint8)
    labels[4:8, 4:8, 4:7] = LEFT_NASAL_CAVITY
    labels[4:8, 4:8, 7:10] = SPHENOID
    contacts = ostial_contact_voxels(labels, dilation=1)
    assert contacts["sphenoid"] > 0


def test_expand_without_named_labels_leaves_passage_empty():
    hu = np.full((16, 16, 16), 50.0, dtype=np.float32)
    hu[10:14, 2:5, 6:10] = -900.0
    result = expand_named_airspaces(hu, None, case_id="bare", min_component_voxels=8)
    assert int(result.passage_mask().sum()) == 0
    assert int(result.sinus_mask().sum()) >= 8
