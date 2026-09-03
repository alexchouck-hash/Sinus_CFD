"""The OpenFOAM export must not merge stripped sinus air back into the solid.

export_openfoam_geometry defaulted to include_sinuses=True, and
build_solid_air_body then merged every interior-air component that touches
the passage -- which is exactly the sinuses the strip removed, at their ostia.
On VH male, autosegment and analyze_passage had both kept the domain clean
(47.6 mL, 0.00 mL sinus inside); the export put 23.55 of 23.5 mL back and
handed a 75.5 mL solid to the mesher. Sinus-free is now the default, and the
export refuses a solid that still holds more than 5% of the stripped air.
"""

from __future__ import annotations

import numpy as np
import pytest

from sinus_cfd.openfoam_export import build_solid_air_body, export_openfoam_geometry

ISO = (1.0, 1.0, 1.0)
ORIGIN = (0.0, 0.0, 0.0)


def _scene():
    """A passage tube along y with a sinus pocket joined to it at its ostium.

    build_solid_air_body merges an interior-air component when it OVERLAPS the
    passage (``comp & solid``), which is what a sinus does at its ostium, so the
    pocket shares one voxel column (x=19) with the tube. ``sinus_only`` is the
    pocket minus that shared column -- the air that must stay out.
    """
    passage = np.zeros((30, 60, 30), dtype=bool)
    passage[10:20, 5:55, 10:20] = True
    sinus = np.zeros_like(passage)
    sinus[12:18, 25:33, 19:27] = True          # x=19 is shared with the tube
    sinus_only = sinus & ~passage
    all_air = passage | sinus
    inlet = np.zeros_like(passage); inlet[10:20, 5:7, 10:20] = True
    outlet = np.zeros_like(passage); outlet[10:20, 53:55, 10:20] = True
    return passage, sinus, sinus_only, all_air, inlet, outlet


def test_build_solid_excludes_sinuses_when_told_to():
    passage, _s, _so, all_air, _i, _o = _scene()
    solid = build_solid_air_body(passage, all_air, include_sinuses=False)
    assert np.array_equal(solid, passage)


def test_build_solid_merges_touching_sinus_only_when_asked():
    passage, _s, sinus_only, all_air, _i, _o = _scene()
    solid = build_solid_air_body(passage, all_air, include_sinuses=True)
    assert solid[sinus_only].all(), "a sinus joined at its ostium is merged when explicitly asked"


def test_export_default_keeps_the_sinus_out(tmp_path):
    passage, _s, sinus_only, all_air, inlet, outlet = _scene()
    res = export_openfoam_geometry(
        case_id="synthetic", output_dir=tmp_path, lumen=passage, inlet_open=inlet,
        outlet_open=outlet, spacing=ISO, origin=ORIGIN, all_interior_air=all_air,
        sinus_exclude=sinus_only)
    assert any("sinus-free domain preserved" in n for n in res.notes), res.notes
    # the voxel solid the export wrote must not contain the sinus
    import SimpleITK as sitk
    solid = sitk.GetArrayFromImage(
        sitk.ReadImage(str(tmp_path / "openfoam_geometry" / "synthetic_solid_air_body.nrrd"))
    ).astype(bool)
    assert not (solid & sinus_only).any()


def test_export_refuses_a_lumen_that_already_holds_the_sinus(tmp_path):
    passage, sinus, sinus_only, _a, inlet, outlet = _scene()
    contaminated = passage | sinus
    with pytest.raises(ValueError, match="refusing to export a CFD domain that contains sinuses"):
        export_openfoam_geometry(
            case_id="synthetic", output_dir=tmp_path, lumen=contaminated,
            inlet_open=inlet, outlet_open=outlet, spacing=ISO, origin=ORIGIN,
            all_interior_air=None, sinus_exclude=sinus_only)
