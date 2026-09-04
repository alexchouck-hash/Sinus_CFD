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


def test_oscillation_around_a_stable_answer_is_settled(tmp_path):
    """THCA at 800: per-sample wobble 1.5%, but the window mean moved 0.08%.

    A rule on the wobble alone called this unconverged after 400 extra
    iterations in which the resistance moved 0.6%. Drift is what 'still
    moving' means; wobble is only the amplitude and gets a looser bound.
    """
    # Period-3 oscillation, so the two 3-sample windows are identical: drift is
    # exactly zero while the wobble is ~1.3%. (A period-2 series splits a
    # 3-sample window 2:1 and shows drift of about a third of the amplitude --
    # still far under the 1% bound, but not the clean case this test is for.)
    left = [6.05, 6.20, 6.12] * 2
    right = [6.85, 7.00, 6.92] * 2
    v = pressure_drop_verdict(_case(tmp_path, left, right))
    assert v["stable"]
    assert v["drift_rel_change"] < 0.005
    assert 0.01 < v["wobble_rel_change"] < 0.03


def test_a_trend_is_not_settled_even_when_each_step_is_small(tmp_path):
    """Steps of 0.9% each look calm sample to sample; the mean is still moving."""
    left = [7.00, 6.94, 6.88, 6.82, 6.76, 6.70]
    right = [7.50, 7.43, 7.36, 7.30, 7.23, 7.17]
    with pytest.raises(ValueError, match="DRIFTING"):
        pressure_drop_verdict(_case(tmp_path, left, right))


def test_large_wobble_is_not_settled(tmp_path):
    left = [6.0, 6.5, 5.9, 6.4, 5.8, 6.5]            # ~12% swings, no trend
    right = [6.8, 7.3, 6.7, 7.2, 6.6, 7.3]
    with pytest.raises(ValueError, match="WOBBLES"):
        pressure_drop_verdict(_case(tmp_path, left, right))


def test_residual_control_stop_is_recorded_and_settles_with_few_samples(tmp_path):
    # CQ500CT390 on the sinus-free domain: simpleFoam tripped residualControl at
    # 250, five samples, too few to judge drift. The solver's own verdict is
    # recorded instead of leaving the reader with a nan.
    flat = [10.0, 10.02, 10.01, 10.0, 10.01]
    root = _case(tmp_path, flat, flat)
    (root / "log.simpleFoam").write_text(
        "Time = 250\n\nSIMPLE solution converged in 250 iterations\n\nEnd\n", encoding="utf-8")
    v = pressure_drop_verdict(root)
    assert v["stable"]
    assert v["converged_by_residual"] is True
    assert v["converged_at_iteration"] == 250
    assert v["drift_rel_change"] != v["drift_rel_change"]  # nan: 5 samples < 2 windows


def test_run_that_hit_its_cap_is_not_marked_converged(tmp_path):
    flat = [10.0, 10.02, 10.01, 10.0, 10.01, 10.0]
    root = _case(tmp_path, flat, flat)
    (root / "log.simpleFoam").write_text("Time = 400\n\nEnd\n", encoding="utf-8")
    v = pressure_drop_verdict(root)
    assert v["converged_by_residual"] is False
    assert v["converged_at_iteration"] is None


def test_time_dir_without_p_is_not_a_solve(tmp_path):
    """A U-only time directory (a function object's output after a solver
    that died at start-up) must not be matched by cell count."""
    from sinus_cfd.openfoam_import import select_time_dir_matching_cells
    n = 5
    body = ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class volVectorField;\n"
            "    object U;\n}\n\ninternalField nonuniform List<vector>\n"
            f"{n}\n(\n" + "\n".join("(1 0 0)" for _ in range(n)) + "\n)\n;\n")
    for t, with_p in (("50", False), ("100", False)):
        d = tmp_path / t
        d.mkdir()
        (d / "U").write_text(body, encoding="utf-8")
        if with_p:
            (d / "p").write_text("x", encoding="utf-8")
    assert select_time_dir_matching_cells(tmp_path, n) is None
    (tmp_path / "50" / "p").write_text("x", encoding="utf-8")
    assert select_time_dir_matching_cells(tmp_path, n) == "50"
