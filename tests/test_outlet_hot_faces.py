"""A choked outlet cap must be refused, and it is only visible in the solved field.

Both Visible Human heads solved with their caudal trachea cap gave a pressure
drop 1000x off. On the solved field the trachea patch carried 57x (female) and
8x (male) the mean face speed while the inlets sat at 1.0x -- the whole drop
lived on a few outlet faces. Two pre-solve geometric predictors were measured
and rejected: the cap's PCA axes (every cap is a fat 6 mm ball clipped by the
lumen; the choked 12x6x6 mm cap looks like a clean 19x8x8 one) and whether the
cap touches the image boundary (none does). The solved patch separates the
populations at 15x, so that is the test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sinus_cfd.openfoam_import import (
    OUTLET_HOT_FACE_MAX_RATIO,
    outlet_patch_velocity_stats,
)


def _u_file(path: Path, trachea_vals, inlet_vals=((0.2, 0, 0),) * 4) -> Path:
    def block(vals):
        rows = "\n".join(f"({v[0]} {v[1]} {v[2]})" for v in vals)
        return f"value nonuniform List<vector>\n{len(vals)}\n(\n{rows}\n);"
    path.write_text(
        "internalField nonuniform List<vector>\n2\n(\n(1 0 0)\n(1 0 0)\n);\n"
        "boundaryField\n{\n"
        f"    left_nostril\n    {{\n        type flowRateInletVelocity;\n        {block(inlet_vals)}\n    }}\n"
        "    wall\n    {\n        type noSlip;\n        value uniform (0 0 0);\n    }\n"
        f"    trachea\n    {{\n        type pressureInletOutletVelocity;\n        {block(trachea_vals)}\n    }}\n"
        "}\n",
        encoding="utf-8",
    )
    return path


def test_clean_outlet_is_near_one(tmp_path):
    u = _u_file(tmp_path / "U", [(1.1, 0, 0), (1.0, 0, 0), (0.9, 0, 0), (1.0, 0, 0)])
    s = outlet_patch_velocity_stats(u, "trachea")
    assert s["n_faces"] == 4
    assert s["max_over_mean"] == pytest.approx(1.1, rel=1e-6)
    assert s["hot_faces"] == 0
    assert s["max_over_mean"] < OUTLET_HOT_FACE_MAX_RATIO


def test_one_hot_face_is_caught(tmp_path):
    """One face at 60x the rest -- the VH female trachea cap in miniature."""
    vals = [(1.0, 0, 0)] * 19 + [(60.0, 0, 0)]
    s = outlet_patch_velocity_stats(_u_file(tmp_path / "U", vals), "trachea")
    assert s["hot_faces"] == 1
    assert s["max_over_mean"] > OUTLET_HOT_FACE_MAX_RATIO


def test_inlet_patch_is_read_independently(tmp_path):
    u = _u_file(tmp_path / "U", [(1.0, 0, 0)] * 3)
    s = outlet_patch_velocity_stats(u, "left_nostril")
    assert s["n_faces"] == 4 and s["max_over_mean"] == pytest.approx(1.0)


def test_uniform_or_missing_patch_returns_none(tmp_path):
    u = _u_file(tmp_path / "U", [(1.0, 0, 0)] * 3)
    assert outlet_patch_velocity_stats(u, "wall") is None
    assert outlet_patch_velocity_stats(u, "no_such_patch") is None
