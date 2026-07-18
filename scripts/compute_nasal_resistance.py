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

    p_left = _latest_value(pp / "p_left_nostril")
    p_right = _latest_value(pp / "p_right_nostril")
    p_out = _latest_value(pp / "p_trachea")
    q_left = _latest_value(pp / "Q_left_nostril")
    q_right = _latest_value(pp / "Q_right_nostril")

    missing = [n for n, v in [
        ("p_left_nostril", p_left), ("p_right_nostril", p_right),
        ("p_trachea", p_out), ("Q_left_nostril", q_left), ("Q_right_nostril", q_right),
    ] if v is None]
    if missing:
        print(f"ERROR: missing functionObject output: {missing}", file=sys.stderr)
        print("Re-scaffold (adds resistance FOs) and re-run the solve.", file=sys.stderr)
        return 1

    # Inlet phi is negative (flow into domain); use magnitude for Q.
    q_total = abs(q_left) + abs(q_right)  # m³/s
    p_in_mean = 0.5 * (p_left + p_right)  # kinematic, m²/s²
    dp_kinematic = p_in_mean - p_out
    dp_pa = args.rho * dp_kinematic

    if q_total <= 0:
        print("ERROR: total flow is zero — check inlet BCs / convergence", file=sys.stderr)
        return 1

    r_si = dp_pa / q_total  # Pa·s/m³
    r_pa_s_ml = r_si * 1e-6  # Pa·s/mL
    q_lpm = q_total * 60_000.0  # m³/s → L/min

    print(f"[{args.case}] nasal resistance (converged simpleFoam)")
    print(f"  total flow Q      : {q_total:.3e} m³/s  ({q_lpm:.1f} L/min)")
    print(f"  ΔP (kinematic)    : {dp_kinematic:.3f} m²/s²")
    print(f"  ΔP (physical)     : {dp_pa:.2f} Pa   (ρ={args.rho} kg/m³)")
    print(f"  resistance R      : {dp_pa / q_total:.3e} Pa·s/m³")
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
