"""A solve is converged when the ANSWER stops moving, not when a residual does.

On CQ500CT390 the first-corrector p residual plateaued at 3.8e-3 against a 1e-3
residualControl and never tripped it, while the inlet pressures were flat to
0.17% from iteration 50 onward. Judging on the residual called 40 extra minutes
necessary; judging on the pressure drop shows they bought nothing. THCA's July
solve, by contrast, hit its 500-iteration cap with the inlet pressure still
moving 2.61% -- a resistance that was reported but was never a measurement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sinus_cfd.openfoam_import import (
    DP_STABLE_MAX_REL,
    RHO_AIR_KG_M3,
    pressure_drop_verdict,
)


def _history(root: Path, name: str, rows):
    d = root / "postProcessing" / name / "0"
    d.mkdir(parents=True, exist_ok=True)
    lines = ["# Time  value"] + [f"{t}\t{v:.8e}" for t, v in rows]
    (d / "surfaceFieldValue.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _case(tmp_path, p_left, p_right, p_out=0.0, q=3.0e-4):
    ts = [50 * (i + 1) for i in range(len(p_left))]
    _history(tmp_path, "p_left_nostril", zip(ts, p_left))
    _history(tmp_path, "p_right_nostril", zip(ts, p_right))
    _history(tmp_path, "p_trachea", [(t, p_out) for t in ts])
    _history(tmp_path, "Q_trachea", [(t, q) for t in ts])
    return tmp_path


def test_settled_inlet_pressure_yields_resistance(tmp_path):
    case = _case(tmp_path, [6.30, 6.18, 6.17, 6.17], [7.00, 6.94, 6.94, 6.94])
    v = pressure_drop_verdict(case)
    assert v is not None and v["stable"]
    dp_kin = 0.5 * (6.17 + 6.94)
    assert v["dp_pa"] == pytest.approx(RHO_AIR_KG_M3 * dp_kin)
    assert v["q_L_min"] == pytest.approx(18.0)
    assert v["resistance_pa_s_per_ml"] == pytest.approx(RHO_AIR_KG_M3 * dp_kin / 300.0)
    assert v["stability_rel_change_last_samples"] <= DP_STABLE_MAX_REL


def test_still_moving_inlet_pressure_raises(tmp_path):
    """The iteration cap is not convergence. A moving answer must fail loudly."""
    case = _case(tmp_path, [9.0, 8.0, 7.2, 6.6], [9.5, 8.6, 7.9, 7.3])
    with pytest.raises(ValueError, match="before the pressure drop settled"):
        pressure_drop_verdict(case)


def test_no_history_is_reported_as_unverified_not_guessed(tmp_path):
    assert pressure_drop_verdict(tmp_path) is None


def test_per_side_pressure_is_kept_not_averaged_away(tmp_path):
    """L/R asymmetry is the clinical signal; the mean dP must not hide it."""
    case = _case(tmp_path, [18.0, 18.0, 18.0], [9.0, 9.0, 9.0])
    v = pressure_drop_verdict(case)
    assert v["dp_left_pa"] == pytest.approx(RHO_AIR_KG_M3 * 18.0)
    assert v["dp_right_pa"] == pytest.approx(RHO_AIR_KG_M3 * 9.0)
    assert v["dp_left_pa"] > v["dp_pa"] > v["dp_right_pa"]
