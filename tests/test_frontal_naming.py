"""Frontal sinuses are named from the antral roof, not from a bounding box.

The rule this replaces scored a body's height as a fraction of the AIRWAY
BOUNDING BOX. That box moves with how much pharynx is in the mask and with where
the naris ports landed, so the same anatomy scores differently case to case. On
CQ500CT390 -- a brain-framed scan whose nostrils sit at the bottom edge of the
FOV, so the ports fell back to the airway's anterior opening -- both frontal
sinuses scored 0.60 and 0.68 against a 0.72 threshold and were reported as
``maxillary R`` and ``ethmoid L``.

Measured on 19 hand-identified bodies across five cases, the fraction of a body
lying above the maxillary antral roof separates the two populations with a wide
gap and no overlap:

    maxillary  0.02 - 0.11      frontal  0.99 - 1.00
    ethmoid    0.00             non-sinus air  0.00
"""

from __future__ import annotations

import numpy as np

from sinus_cfd.patency import (
    FRONTAL_ABOVE_ANTRAL_ROOF_FRAC,
    _antral_roof_z,
    refine_frontal_by_antral_roof,
)


def _scene(sup_hi=True):
    """Two lateral antra low down, one small body high above them, one between.

    Label ids: 1 = antrum L, 2 = antrum R, 3 = high paramedian body,
    4 = body at antral level near the midline.
    """
    lab = np.zeros((60, 30, 60), dtype=np.int32)
    lab[10:24, 8:22, 38:52] = 1     # antrum, patient-left (high x)
    lab[10:24, 8:22, 8:22] = 2      # antrum, patient-right (low x)
    lab[44:54, 6:16, 26:36] = 3     # well above both antra, paramedian
    lab[12:20, 8:18, 24:32] = 4     # same height as the antra, paramedian
    if not sup_hi:
        lab = lab[::-1]
    recs = [
        dict(name="maxillary", side="L", volume_ml=14.0, off_midline_mm=+22.0,
             body_ids=[1]),
        dict(name="maxillary", side="R", volume_ml=12.0, off_midline_mm=-22.0,
             body_ids=[2]),
        dict(name="maxillary", side="R", volume_ml=1.5, off_midline_mm=-1.0,
             body_ids=[3]),
        dict(name="ethmoid", side="R", volume_ml=0.8, off_midline_mm=-4.0,
             body_ids=[4]),
    ]
    return lab, recs


def test_body_above_the_antral_roof_is_renamed_frontal():
    lab, recs = _scene()
    notes = []
    out = refine_frontal_by_antral_roof(recs, lab, True, notes)
    by_id = {r["body_ids"][0]: r for r in out}
    assert by_id[3]["name"] == "frontal", by_id[3]
    assert by_id[3]["frac_above_antral_roof"] >= FRONTAL_ABOVE_ANTRAL_ROOF_FRAC
    assert any("-> frontal" in n for n in notes), notes


def test_the_antra_themselves_are_never_renamed():
    lab, recs = _scene()
    out = refine_frontal_by_antral_roof(recs, lab, True, [])
    by_id = {r["body_ids"][0]: r for r in out}
    assert by_id[1]["name"] == "maxillary"
    assert by_id[2]["name"] == "maxillary"


def test_a_body_at_antral_height_keeps_its_name():
    lab, recs = _scene()
    out = refine_frontal_by_antral_roof(recs, lab, True, [])
    by_id = {r["body_ids"][0]: r for r in out}
    assert by_id[4]["name"] == "ethmoid", by_id[4]
    assert by_id[4]["frac_above_antral_roof"] < FRONTAL_ABOVE_ANTRAL_ROOF_FRAC


def test_inverted_z_gives_the_same_answer():
    """superior_is_high_z=False must not flip which body is 'above'."""
    lab, recs = _scene(sup_hi=False)
    out = refine_frontal_by_antral_roof(recs, lab, False, [])
    by_id = {r["body_ids"][0]: r for r in out}
    assert by_id[3]["name"] == "frontal"
    assert by_id[1]["name"] == "maxillary"
    assert by_id[4]["name"] == "ethmoid"


def test_no_lateral_body_means_no_anchor_and_no_rename():
    """With nothing lateral enough to be an antrum, names are left alone."""
    lab = np.zeros((60, 30, 60), dtype=np.int32)
    lab[44:54, 6:16, 26:36] = 3
    recs = [dict(name="ethmoid", side="R", volume_ml=1.5, off_midline_mm=-4.0,
                 body_ids=[3])]
    roof, anchors = _antral_roof_z(recs, lab, True)
    assert roof is None and anchors == []
    out = refine_frontal_by_antral_roof(recs, lab, True, [])
    assert out[0]["name"] == "ethmoid"


def test_the_anchor_is_picked_on_geometry_not_on_the_provisional_name():
    """A mislabelled antrum must still anchor the roof that corrects the rest."""
    lab, recs = _scene()
    recs[0]["name"] = "frontal"        # antrum L provisionally mislabelled
    roof, anchors = _antral_roof_z(recs, lab, True)
    assert roof is not None
    assert {tuple(a["body_ids"]) for a in anchors} == {(1,), (2,)}
    out = refine_frontal_by_antral_roof(recs, lab, True, [])
    assert {r["body_ids"][0] for r in out if r["name"] == "frontal"} == {1, 3}


def test_air_far_below_the_antral_roof_is_not_a_sinus():
    """VH male's neck air (1.89 mL) and skull-base air (0.50 mL) were named
    maxillary L. They sit 58 and 109 mm below the antral roof; no real sinus
    measured sits more than 20.2 mm below it.
    """
    lab = np.zeros((70, 30, 60), dtype=np.int32)
    lab[40:54, 8:22, 38:52] = 1      # antrum L   (roof ~53)
    lab[40:54, 8:22, 8:22] = 2       # antrum R
    lab[2:7, 8:18, 30:40] = 3        # ~49 mm below the roof at 1 mm: neck air
    lab[32:38, 8:18, 26:34] = 4      # ~18 mm below the roof: a real low body
    recs = [
        dict(name="maxillary", side="L", volume_ml=14.0, off_midline_mm=+22.0, body_ids=[1]),
        dict(name="maxillary", side="R", volume_ml=12.0, off_midline_mm=-22.0, body_ids=[2]),
        dict(name="maxillary", side="L", volume_ml=1.9, off_midline_mm=+17.6, body_ids=[3]),
        dict(name="ethmoid", side="R", volume_ml=0.6, off_midline_mm=-4.0, body_ids=[4]),
    ]
    notes = []
    out = refine_frontal_by_antral_roof(recs, lab, True, notes, spacing_xyz=(1.0, 1.0, 1.0))
    by_id = {r["body_ids"][0]: r for r in out}
    assert by_id[3]["name"] == "unknown", by_id[3]
    assert by_id[3]["mm_above_antral_roof"] < -45.0
    assert any("not a paranasal sinus" in n for n in notes), notes
    assert by_id[4]["name"] == "ethmoid" and by_id[4]["mm_above_antral_roof"] > -45.0
    assert by_id[1]["name"] == "maxillary" and by_id[2]["name"] == "maxillary"


def test_without_spacing_the_gate_is_skipped():
    lab, recs = _scene()
    out = refine_frontal_by_antral_roof(recs, lab, True, [])
    assert all("mm_above_antral_roof" not in r for r in out)
    assert all(r["name"] != "unknown" for r in out)


def test_a_large_high_body_is_a_complex_not_a_frontal_sinus():
    """THCA at 1.5 mm slices: a 35 mL right-sided mass (maxillary + ethmoid +
    sphenoid air, unseparated) sat mostly above the antral roof and was
    renamed frontal. A frontal sinus is not 35 mL."""
    lab, recs = _scene()
    recs[2]["volume_ml"] = 35.0
    notes = []
    out = refine_frontal_by_antral_roof(recs, lab, True, notes)
    by_id = {r["body_ids"][0]: r for r in out}
    assert by_id[3]["name"] == "complex", by_id[3]
    assert any("not a frontal sinus" in n for n in notes), notes


def test_a_small_body_with_antral_level_air_is_not_frontal():
    lab, recs = _scene()
    lab[16:44, 6:16, 26:36] = 3        # extend body 3 down to antral height
    recs[2]["volume_ml"] = 6.0        # ~40% of it now lies at or below the roof
    notes = []
    out = refine_frontal_by_antral_roof(recs, lab, True, notes)
    by_id = {r["body_ids"][0]: r for r in out}
    assert by_id[3]["name"] != "frontal", by_id[3]


def test_a_high_body_behind_the_antra_is_not_renamed_frontal():
    """VH male: a 4.47 mL midline sphenoid, 90% above the antral roof, is
    behind both antra. Height alone called it frontal."""
    lab, recs = _scene()
    for r in recs[:2]:
        r["frac_posterior"] = 0.45
    recs[2]["name"] = "sphenoid"
    recs[2]["frac_posterior"] = 0.85
    notes = []
    out = refine_frontal_by_antral_roof(recs, lab, True, notes)
    by_id = {r["body_ids"][0]: r for r in out}
    assert by_id[3]["name"] == "sphenoid", by_id[3]
    assert any("behind the antra" in n for n in notes), notes


def test_a_high_body_in_front_of_the_antra_is_still_renamed_frontal():
    lab, recs = _scene()
    for r in recs[:2]:
        r["frac_posterior"] = 0.45
    recs[2]["frac_posterior"] = 0.20
    out = refine_frontal_by_antral_roof(recs, lab, True, [])
    by_id = {r["body_ids"][0]: r for r in out}
    assert by_id[3]["name"] == "frontal", by_id[3]
