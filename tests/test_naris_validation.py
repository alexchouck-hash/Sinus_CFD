"""Naris-pair validation: two distinct seeds, spacing-aware, or HARD-fail.

Regression cover for the THCA failure (docs/handoff.md §9 item 1), where the CT
anterior-air detector returned the *same voxel twice* and the old repair nudged x
by two voxels, leaving both seeds on the same side of the septum (L=1172 R=14).
"""

from __future__ import annotations

import numpy as np

from sinus_cfd.nasal_airway_ct import (
    MIN_NARIS_SEPARATION_MM,
    detect_nares_from_ct_air,
    extract_ct_nasal_airway,
    validate_naris_pair,
)

# Measured on the real cases (see the case metas under outputs/).
THCA_SPACING = (0.9765625, 0.9765625, 1.5)  # anisotropic living CT
VH_SPACING = (1.0, 1.0, 1.0)  # Visible Human, isotropic cadaver CT


# --------------------------------------------------------------------------
# validate_naris_pair
# --------------------------------------------------------------------------


def test_rejects_the_thca_collapsed_pair():
    """The exact pair the old code produced on THCA: 2 voxels apart in x."""
    ok, why = validate_naris_pair((38, 52, 72), (38, 52, 70), THCA_SPACING)
    assert not ok
    assert "lateral separation" in why


def test_rejects_identical_seeds():
    ok, why = validate_naris_pair((38, 52, 71), (38, 52, 71), THCA_SPACING)
    assert not ok
    assert "0.00 mm" in why


def test_accepts_the_thca_prior_pair():
    """Narrow but real: the whole-head prior, 7.81 mm apart laterally."""
    ok, why = validate_naris_pair((50, 10, 103), (47, 7, 95), THCA_SPACING)
    assert ok, why
    assert "9.4" in why  # 9.48 mm 3-D separation


def test_accepts_visible_human_pairs_unchanged():
    """Both VH cases must keep passing — they are the no-regression anchors."""
    ok_f, why_f = validate_naris_pair((77, 39, 116), (79, 37, 78), VH_SPACING)
    ok_m, why_m = validate_naris_pair((68, 15, 148), (69, 8, 108), VH_SPACING)
    assert ok_f, why_f
    assert ok_m, why_m


def test_rejects_pair_split_across_the_cavity_depth():
    """One seed at the nostril, one deep in the cavity: 39 mm apart in y."""
    ok, why = validate_naris_pair((44, 12, 97), (38, 52, 71), THCA_SPACING)
    assert not ok
    assert "anterior-posterior" in why


def test_rejects_swapped_sides():
    """High x is patient left (AGENTS.md). A swapped pair is not a valid pair."""
    ok, why = validate_naris_pair((79, 37, 78), (77, 39, 116), VH_SPACING)
    assert not ok
    assert "patient left" in why


def test_rejects_absurdly_wide_pair():
    ok, why = validate_naris_pair((70, 20, 140), (70, 20, 40), VH_SPACING)
    assert not ok
    assert "separation" in why


def test_missing_seed_is_rejected():
    assert validate_naris_pair(None, (70, 20, 40), VH_SPACING) == (False, "missing seed")
    assert validate_naris_pair((70, 20, 60), None, VH_SPACING) == (False, "missing seed")


def test_validation_is_spacing_aware_not_voxel_counted():
    """Same voxel offset, different physical meaning."""
    left, right = (40, 20, 56), (40, 20, 50)  # 6 voxels apart in x
    ok_coarse, _ = validate_naris_pair(left, right, (1.0, 1.0, 1.0))  # 6.0 mm
    ok_fine, why_fine = validate_naris_pair(left, right, (0.5, 0.5, 1.0))  # 3.0 mm
    assert ok_coarse
    assert not ok_fine
    assert f"< {MIN_NARIS_SEPARATION_MM:.1f} mm" in why_fine


# --------------------------------------------------------------------------
# detect_nares_from_ct_air — peak confidence bar in mm
# --------------------------------------------------------------------------


def _bimodal_shell(sep_voxels: int):
    """A body with two anterior air columns `sep_voxels` apart in x."""
    shape = (30, 40, 60)
    body = np.zeros(shape, dtype=bool)
    body[5:25, 5:35, 5:55] = True
    air = np.zeros(shape, dtype=bool)
    x_l = 30 + sep_voxels // 2
    x_r = 30 - sep_voxels // 2
    for x in (x_l, x_r):
        air[8:22, 10:30, x - 1 : x + 2] = True
    hu = np.zeros(shape, dtype=np.float32)
    hu[air] = -1000.0
    hu[~body] = -1000.0
    return hu, body, air


def test_peak_separation_bar_is_spacing_aware():
    """20 voxels clears 15 mm at 1.0 mm/vox but not at 0.5 mm/vox (=10 mm)."""
    hu, body, air = _bimodal_shell(20)
    _, _, _, notes_coarse = detect_nares_from_ct_air(
        hu, body, air, spacing_xyz=(1.0, 1.0, 1.0)
    )
    _, _, _, notes_fine = detect_nares_from_ct_air(
        hu, body, air, spacing_xyz=(0.5, 0.5, 1.0)
    )
    assert any("x-peaks from CT air histogram" in n for n in notes_coarse)
    assert any("x-peaks weak" in n for n in notes_fine)


# --------------------------------------------------------------------------
# extract_ct_nasal_airway — fallback to prior, then HARD-fail
# --------------------------------------------------------------------------


def _unimodal_head():
    """One medial air slab: the detector cannot resolve two nostrils from it."""
    shape = (30, 40, 30)
    body = np.zeros(shape, dtype=bool)
    body[5:25, 5:35, 5:25] = True
    air = np.zeros(shape, dtype=bool)
    air[8:22, 10:30, 14:17] = True
    hu = np.zeros(shape, dtype=np.float32)
    hu[air] = -1000.0
    hu[~body] = -1000.0
    return hu, body, air


def test_collapsed_detection_falls_back_to_the_prior():
    hu, body, air = _unimodal_head()
    res = extract_ct_nasal_airway(
        hu=hu,
        body=body,
        interior_air=air,
        soft_tissue=body & ~air,
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_xyz=(0.0, 0.0, 0.0),
        prior_left_mm=[19.0, 11.0, 15.0],  # (z,y,x) = (15, 11, 19)
        prior_right_mm=[11.0, 11.0, 15.0],  # (z,y,x) = (15, 11, 11)
        legacy_midplane_split=True,
    )
    assert res.left_naris_center_zyx == (15, 11, 19)
    assert res.right_naris_center_zyx == (15, 11, 11)
    assert any("naris pair rejected" in n for n in res.notes)
    assert any("whole-head prior landmarks" in n for n in res.notes)
    assert "HARD_fail" not in res.method
    assert res.naris_source == "prior"
    assert res.to_meta()["naris_source"] == "prior"


def test_collapsed_detection_without_a_prior_hard_fails():
    """No inventing a seed: no usable prior means HARD-fail, not a 2-voxel nudge."""
    hu, body, air = _unimodal_head()
    res = extract_ct_nasal_airway(
        hu=hu,
        body=body,
        interior_air=air,
        soft_tissue=body & ~air,
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_xyz=(0.0, 0.0, 0.0),
        legacy_midplane_split=True,
    )
    assert res.method == "naris_detection_HARD_fail"
    assert any("HARD fail" in n for n in res.notes)
    assert int(res.left_cavity.sum()) == 0
    assert int(res.right_cavity.sum()) == 0
    assert res.naris_source == "none"


def test_invalid_prior_does_not_rescue_a_collapsed_detection():
    hu, body, air = _unimodal_head()
    res = extract_ct_nasal_airway(
        hu=hu,
        body=body,
        interior_air=air,
        soft_tissue=body & ~air,
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_xyz=(0.0, 0.0, 0.0),
        prior_left_mm=[16.0, 11.0, 15.0],  # only 2 mm apart -> invalid
        prior_right_mm=[14.0, 11.0, 15.0],
        legacy_midplane_split=True,
    )
    assert res.method == "naris_detection_HARD_fail"
    assert any("prior naris pair also rejected" in n for n in res.notes)
    assert res.naris_source == "none"


def test_prior_spliced_in_for_a_none_seed_is_not_reported_as_ct_detected():
    """Provenance: a seed taken from the prior must never be labelled CT-detected.

    The `None`-seed fallback runs before pair validation, so a prior-sourced pair
    that then validates was previously reported as "from CT air shell accepted".
    """
    shape = (30, 40, 30)
    body = np.zeros(shape, dtype=bool)
    body[5:25, 5:35, 5:25] = True
    air = np.zeros(shape, dtype=bool)
    air[14:16, 12:14, 15] = True  # far too small a shell to cluster
    hu = np.zeros(shape, dtype=np.float32)
    hu[air] = -1000.0
    hu[~body] = -1000.0

    res = extract_ct_nasal_airway(
        hu=hu,
        body=body,
        interior_air=air,
        soft_tissue=body & ~air,
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_xyz=(0.0, 0.0, 0.0),
        prior_left_mm=[19.0, 11.0, 15.0],
        prior_right_mm=[11.0, 11.0, 15.0],
        legacy_midplane_split=True,
    )
    assert res.left_naris_center_zyx == (15, 11, 19)
    assert res.right_naris_center_zyx == (15, 11, 11)
    assert res.naris_source == "prior"
    assert not any("from CT air shell accepted" in n for n in res.notes)


def test_rejects_pair_that_does_not_straddle_the_midline():
    """CQ500CT390: two seeds 16.5 mm apart but BOTH patient-left of the midline.

    Separation alone passed every other predicate; the competing flood then came
    out 12:1 because neither seed was in the right cavity.
    """
    left, right = (73, 32, 257), (67, 29, 215)
    sp = (0.38, 0.38, 0.625)
    ok_no_mid, _ = validate_naris_pair(left, right, sp)
    assert ok_no_mid, "without a midline this pair looks fine -- that was the bug"
    ok, why = validate_naris_pair(left, right, sp, x_midline=189.0)
    assert not ok
    assert "straddle" in why


def test_accepts_a_straddling_pair():
    """CQ500CT105: seeds on opposite sides of the midline."""
    ok, why = validate_naris_pair((61, 48, 260), (66, 32, 161), (0.393, 0.393, 0.625),
                                  x_midline=206.0)
    assert ok, why


def test_midline_tolerance_is_applied_in_mm():
    """The straddle test tolerates a seed slightly past the midline, in mm.

    A right seed 3 mm past the midline is inside NARIS_MIDLINE_TOL_MM and passes;
    7 mm past is outside and fails. Both pairs are otherwise identical and valid,
    so only the straddle predicate can be deciding.
    """
    ok_near, _ = validate_naris_pair(
        (40, 20, 80), (40, 20, 53), VH_SPACING, x_midline=50.0
    )
    ok_far, why_far = validate_naris_pair(
        (40, 20, 80), (40, 20, 57), VH_SPACING, x_midline=50.0
    )
    assert ok_near
    assert not ok_far
    assert "straddle" in why_far
