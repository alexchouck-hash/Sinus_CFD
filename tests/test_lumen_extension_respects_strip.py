"""The nostril-tunnel step must not put stripped sinus air back into the passage.

Every CFD domain ever meshed in this repo had its sinuses inside, and this is
why. autosegment strips each sinus out of the flooded airway, which leaves a
chamber-shaped 3-D HOLE in the passage mask (the strip carves a cavity out of a
region that surrounds it). extend_lumen_to_external_nares then ran a 3-D
binary_fill_holes over the whole lumen to close pits in the painted nostril
tunnels -- and refilled every sinus with it. On CQ500CT390 a clean 29.6 mL
passage came out at 35.5 mL with 93-97% of each of the three stripped bodies
back inside (0.29 -> 5.60 mL of sinus air). The fill is now confined to the
neighbourhood of the painted tunnels, the stripped air is subtracted after
every morphological step, and analyze_nasal_passage refuses to return a
passage that still holds it.
"""

from __future__ import annotations

import numpy as np
import pytest

from sinus_cfd.nasal_passage import (
    analyze_nasal_passage,
    extend_lumen_to_external_nares,
)

ISO = (1.0, 1.0, 1.0)
ORIGIN = (0.0, 0.0, 0.0)


def _scene():
    """A lumen block with a carved cavity inside it, and a skin naris outside.

    Array is (z, y, x); physical mm == index for ISO spacing and zero origin.
    The lumen occupies x 10..40 on y 10..30, z 10..30. The cavity (a stripped
    sinus) is a 6x6x6 pocket fully surrounded by lumen. The skin naris sits
    4 voxels in front of the lumen face at low y.
    """
    lumen = np.zeros((40, 40, 60), dtype=bool)
    lumen[10:30, 10:30, 10:40] = True
    cavity = np.zeros_like(lumen)
    cavity[16:22, 16:22, 24:30] = True
    lumen &= ~cavity                        # the strip removed it
    skin_mm = [[25.0, 6.0, 20.0]]           # (x, y, z) in mm -> index (z=20, y=6, x=25)
    return lumen, cavity, skin_mm


def test_without_exclude_the_old_fill_refills_the_cavity():
    """Documents the mechanism: a carved chamber is a 3-D hole and gets filled."""
    lumen, cavity, skin = _scene()
    ext, _notes, _seeds = extend_lumen_to_external_nares(lumen, skin, ISO, ORIGIN)
    # The fill is now confined to the tunnel neighbourhood, so even without
    # `exclude` a cavity far from the paint must survive.
    assert not (ext & cavity).any(), "a stripped sinus 20 voxels from the paint was refilled"


def test_with_exclude_the_cavity_never_comes_back():
    lumen, cavity, skin = _scene()
    ext, _notes, _seeds = extend_lumen_to_external_nares(lumen, skin, ISO, ORIGIN, exclude=cavity)
    assert not (ext & cavity).any()
    # and the extension still did its job: the tunnel reaches the skin naris
    assert ext[20, 6, 25]


def test_pit_beside_the_tunnel_is_still_filled():
    """The fill exists to close small pits in the painted corridor; keep that."""
    lumen, cavity, skin = _scene()
    ext, _n, _s = extend_lumen_to_external_nares(lumen, skin, ISO, ORIGIN, exclude=cavity)
    # Punch a 1-voxel pit into the tunnel just outside the original lumen face,
    # then re-run: it must be filled because it lies within the paint's reach.
    pit = ext.copy()
    pit[20, 9, 25] = False
    assert (ext & ~pit).sum() == 1
    ext2, _n, _s = extend_lumen_to_external_nares(pit, skin, ISO, ORIGIN, exclude=cavity)
    assert ext2[20, 9, 25], "a pit in the painted tunnel was not filled"
    assert not (ext2 & cavity).any()


def test_analyze_nasal_passage_output_is_free_of_excluded_air():
    lumen, cavity, skin = _scene()
    outlet_mm = [25.0, 29.0, 20.0]          # far end of the block
    masks, passage, metrics = analyze_nasal_passage(
        lumen=lumen, spacing=ISO, origin=ORIGIN, inlet_centers_mm=skin,
        outlet_center_mm=outlet_mm, case_id="synthetic", open_radius_mm=3.0,
        skin_naris_centers_mm=skin, tunnel_radius_mm=2.0, sinus_exclude=cavity)
    assert not (masks["lumen"] & cavity).any()
    assert not (masks["fluid"] & cavity).any()
    assert any("sinus-free domain preserved" in n for n in passage.get("notes", []))


def test_analyze_nasal_passage_refuses_a_lumen_that_contains_excluded_air(monkeypatch):
    """If some future step puts the sinus back, the analysis must raise, not write."""
    import sinus_cfd.nasal_passage as np_mod

    lumen, cavity, skin = _scene()
    contaminated = lumen | cavity

    def _no_op_extend(lum, *a, **k):
        return lum.astype(bool), ["(extension disabled for the test)"], []

    monkeypatch.setattr(np_mod, "extend_lumen_to_external_nares", _no_op_extend)
    with pytest.raises(ValueError, match="refusing to produce a CFD domain"):
        analyze_nasal_passage(
            lumen=contaminated, spacing=ISO, origin=ORIGIN, inlet_centers_mm=skin,
            outlet_center_mm=[25.0, 29.0, 20.0], case_id="synthetic",
            open_radius_mm=3.0, skin_naris_centers_mm=skin, tunnel_radius_mm=2.0,
            sinus_exclude=cavity)
