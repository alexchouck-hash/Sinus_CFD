"""A plot label must never move the CFD outlet.

``ports[].center_mm`` in boundary_conditions.json is the domain definition:
analyze_passage seeds the outlet cap from the lumen voxel nearest to it, and
export, scaffold and the solve descend from that cap. The OpenFOAM import used
to overwrite it with the centroid of the previous cap "for viewer labels". The
lumen clips the cap asymmetrically, so the centroid sits anterior of its own
seed; writing it back moved the seed and the next cap grew from there. On
CQ500CT390 a single import walked the seed 3 voxels (1.1 mm) anteriorly and the
cap from 1,780 to 2,450 voxels. The pre-import seed, (27,232,220), was recovered
by rebuilding the cap from candidate seeds; only it reproduced the solved
geometry byte for byte.
"""

from __future__ import annotations

import json

from sinus_cfd.openfoam_import import persist_outlet_viewer_marker


def _bc(tmp_path):
    bc = {
        "case_id": "T",
        "ports": [
            {"name": "left_nostril", "role": "inlet", "center_mm": [1.0, -100.0, 10.0]},
            {"name": "right_nostril", "role": "inlet", "center_mm": [-20.0, -99.0, 12.0]},
            {"name": "trachea", "role": "outlet", "center_mm": [0.0, -29.6, -2.1]},
        ],
    }
    p = tmp_path / "T_boundary_conditions.json"
    p.write_text(json.dumps(bc), encoding="utf-8")
    return p


def test_center_mm_is_never_rewritten(tmp_path):
    p = _bc(tmp_path)
    persist_outlet_viewer_marker(p, [0.05, -30.62, -2.19])
    bc = json.loads(p.read_text(encoding="utf-8"))
    outlet = next(q for q in bc["ports"] if q["role"] == "outlet")
    assert outlet["center_mm"] == [0.0, -29.6, -2.1]
    assert outlet["viewer_marker_mm"] == [0.05, -30.62, -2.19]
    assert outlet["viewer_marker_method"] == "passage_outlet_open_centroid"


def test_inlets_are_untouched(tmp_path):
    p = _bc(tmp_path)
    persist_outlet_viewer_marker(p, [0.05, -30.62, -2.19])
    bc = json.loads(p.read_text(encoding="utf-8"))
    for q in bc["ports"]:
        if q["role"] == "inlet":
            assert "viewer_marker_mm" not in q
    assert next(q for q in bc["ports"] if q["name"] == "left_nostril")["center_mm"] == [1.0, -100.0, 10.0]


def test_repeated_imports_do_not_ratchet(tmp_path):
    """Calling it N times leaves the domain where it started."""
    p = _bc(tmp_path)
    for k in range(5):
        persist_outlet_viewer_marker(p, [0.05 + 0.4 * k, -30.62 - 0.4 * k, -2.19])
    bc = json.loads(p.read_text(encoding="utf-8"))
    outlet = next(q for q in bc["ports"] if q["role"] == "outlet")
    assert outlet["center_mm"] == [0.0, -29.6, -2.1]
    assert outlet["viewer_marker_mm"] == [0.05 + 1.6, -30.62 - 1.6, -2.19]


def test_returns_what_it_touched(tmp_path):
    p = _bc(tmp_path)
    out = persist_outlet_viewer_marker(p, [1.0, 2.0, 3.0])
    assert out == {"ports": ["trachea"], "marker_mm": [1.0, 2.0, 3.0]}
