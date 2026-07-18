#!/usr/bin/env python3
"""
Stage 3 validation number: nasal resistance from a converged simpleFoam solve.

Reads the surfaceFieldValue functionObjects written by the scaffolded case
(area-averaged kinematic pressure p and volumetric flow phi on each open
patch), then computes

    R = ρ · ΔP / Q          [Pa·s/m³]   (report also in Pa·s/mL)

where ΔP = mean inlet p − outlet p (kinematic, m²/s²), ρ air density, and
Q the total inspiratory volumetric flow. simpleFoam is incompressible, so the
solved p is kinematic (p/ρ); multiplying by ρ recovers physical pressure.

The number is checked against the published resting-breathing range so a run
is either plausible or visibly wrong.

Usage:
  py -3.12 scripts/compute_nasal_resistance.py --case VisibleHuman_Head
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Published total nasal resistance, quiet breathing (both cavities in parallel).
# ~0.15-0.30 Pa·s/mL is the commonly cited healthy range; obstruction runs higher.
PUBLISHED_MIN_PA_S_PER_ML = 0.10
PUBLISHED_MAX_PA_S_PER_ML = 0.35


def _latest_value(fo_dir: Path) -> float | None:
    """Read the last data row's final column from a surfaceFieldValue output."""
    if not fo_dir.is_dir():
        return None
    # postProcessing/<name>/<startTime>/surfaceFieldValue.dat
    dats = sorted(fo_dir.glob("*/surfaceFieldValue.dat"))
    if not dats:
        return None
    last_val = None
    for line in dats[-1].read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            last_val = float(parts[-1])
        except (ValueError, IndexError):
            continue
    return last_val


def resistance_from_postprocessing(pp: Path, rho: float = 1.14) -> dict | None:
    """
    Compute nasal resistance from a postProcessing/ directory.

    Returns a dict with q_total (m³/s), q_lpm, dp_pa, r_pa_s_ml, or None if the
    functionObject outputs are missing. Shared by the single-case CLI below and
    the flow-rate sweep (scripts/run_flow_sweep.py) so both compute R the same way.
    """
    p_left = _latest_value(pp / "p_left_nostril")
    p_right = _latest_value(pp / "p_right_nostril")
    p_out = _latest_value(pp / "p_trachea")
    q_left = _latest_value(pp / "Q_left_nostril")
    q_right = _latest_value(pp / "Q_right_nostril")
    if None in (p_left, p_right, p_out, q_left, q_right):
        return None

    q_total = abs(q_left) + abs(q_right)  # m³/s (inlet phi is negative)
    if q_total <= 0:
        return None
    dp_kinematic = 0.5 * (p_left + p_right) - p_out  # m²/s²
    dp_pa = rho * dp_kinematic
    return {
        "q_total_m3_s": q_total,
        "q_lpm": q_total * 60_000.0,
        "dp_kinematic": dp_kinematic,
        "dp_pa": dp_pa,
        "r_si": dp_pa / q_total,
        "r_pa_s_ml": dp_pa / q_total * 1e-6,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="VisibleHuman_Head")
    p.add_argument("--foam-root", type=Path, default=None)
    p.add_argument("--rho", type=float, default=1.14, help="air density kg/m³ (~37 °C)")
    args = p.parse_args()

    foam = args.foam_root or (REPO_ROOT / "foam" / args.case)
    pp = foam / "postProcessing"
    if not pp.is_dir():
        print(f"ERROR: no postProcessing/ under {foam} — has the solve run?", file=sys.stderr)
        return 1

    # surfaceFieldValue functionObjects silently skip rewriting data for time
    # values they've already written -- if a case is re-run without cleaning
    # (no Allclean) over the *same* time range, postProcessing/ can go stale
    # while the field files (constant/, e.g. 500/U) correctly reflect the new
    # mesh, giving a false "mesh-independent" result. Warn if any
    # postProcessing .dat file predates the mesh, which the polyMesh rewrite
    # always touches.
    poly_mesh = foam / "constant" / "polyMesh" / "owner"
    if poly_mesh.is_file():
        mesh_time = poly_mesh.stat().st_mtime
        stale = [
            dat for dat in pp.glob("*/*/surfaceFieldValue.dat")
            if dat.stat().st_mtime < mesh_time
        ]
        if stale:
            print(
                "WARNING: postProcessing data predates the current mesh "
                f"(stale: {[str(p.relative_to(pp)) for p in stale]}). "
                "This case was likely re-run without Allclean first, so the "
                "resistance below may be a leftover from a previous mesh. "
                "Re-run with Allclean (or the fixed Allrun.docker) before trusting it.",
                file=sys.stderr,
            )

    res = resistance_from_postprocessing(pp, rho=args.rho)
    if res is None:
        print("ERROR: missing/empty functionObject output, or zero flow.", file=sys.stderr)
        print("Re-scaffold (adds resistance FOs) and re-run the solve.", file=sys.stderr)
        return 1

    q_total = res["q_total_m3_s"]
    q_lpm = res["q_lpm"]
    dp_kinematic = res["dp_kinematic"]
    dp_pa = res["dp_pa"]
    r_pa_s_ml = res["r_pa_s_ml"]

    print(f"[{args.case}] nasal resistance (converged simpleFoam)")
    print(f"  total flow Q      : {q_total:.3e} m³/s  ({q_lpm:.1f} L/min)")
    print(f"  ΔP (kinematic)    : {dp_kinematic:.3f} m²/s²")
    print(f"  ΔP (physical)     : {dp_pa:.2f} Pa   (ρ={args.rho} kg/m³)")
    print(f"  resistance R      : {res['r_si']:.3e} Pa·s/m³")
    print(f"                    : {r_pa_s_ml:.3f} Pa·s/mL")
    lo, hi = PUBLISHED_MIN_PA_S_PER_ML, PUBLISHED_MAX_PA_S_PER_ML
    if lo <= r_pa_s_ml <= hi:
        verdict = f"WITHIN published resting range ({lo}-{hi} Pa·s/mL)"
    elif r_pa_s_ml < lo:
        verdict = f"BELOW published range ({lo}-{hi}) — geometry too open / mesh too coarse?"
    else:
        verdict = f"ABOVE published range ({lo}-{hi}) — obstruction, or under-resolved thin passages?"
    print(f"  validation        : {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
