#!/usr/bin/env python3
"""
Mucosal cooling / heat load from a thermal CFD solve.

Mucosal cooling — inspired air pulling heat from the nasal mucosa as it warms
toward body temperature — is the strongest correlate of *perceived* nasal
patency (TRPM8 cold receptors sense airflow via cooling, not pressure). The
scaffolded case co-solves a passive temperature scalar (inspired air at ~20 °C,
mucosa wall at ~37 °C); this reads the flow-weighted outlet temperature the
`scalarTransport` functionObject logged and reports:

  - air warming        ΔT = T_out(bulk) − T_in   [K]   (how much the nose
                       conditions inspired air; a well-functioning nose warms
                       air nearly to body temp)
  - total mucosal      Q_heat = ρ·cp·Q·ΔT          [W]   (heat the mucosa gives
    heat loss          up = enthalpy the air gains)

This is a global energy balance (bulk temperatures), robust and mesh-tolerant.
A spatial wall-heat-flux map (where cooling concentrates) needs the near-wall
temperature gradient and is a follow-up once the prism layer is validated.

Usage:
  py -3.12 scripts/compute_mucosal_cooling.py --case P001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from compute_nasal_resistance import _latest_value  # noqa: E402

# Air at ~body temperature.
RHO_AIR = 1.14        # kg/m³
CP_AIR = 1005.0       # J/(kg·K)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="P001")
    p.add_argument("--foam-root", type=Path, default=None)
    p.add_argument("--inlet-temp-K", type=float, default=293.15)
    p.add_argument("--rho", type=float, default=RHO_AIR)
    p.add_argument("--cp", type=float, default=CP_AIR)
    args = p.parse_args()

    foam = args.foam_root or (REPO_ROOT / "foam" / args.case)
    pp = foam / "postProcessing"
    if not (pp / "T_trachea").is_dir():
        print(
            f"ERROR: no T_trachea/ under {pp} — was the case scaffolded with thermal "
            "(default) and re-solved? (a resistance-only run has no T output)",
            file=sys.stderr,
        )
        return 1

    t_out = _latest_value(pp / "T_trachea")  # flow-weighted (bulk) outlet temp
    q_out = _latest_value(pp / "Q_trachea")  # m³/s (outflow, positive)
    if t_out is None or q_out is None:
        print("ERROR: missing T_trachea / Q_trachea functionObject output.", file=sys.stderr)
        return 1

    q_total = abs(q_out)  # m³/s
    dT = t_out - args.inlet_temp_K
    heat_W = args.rho * args.cp * q_total * dT

    print(f"[{args.case}] mucosal cooling / heat load (passive-scalar thermal solve)")
    print(f"  inspired air T_in : {args.inlet_temp_K:.2f} K ({args.inlet_temp_K - 273.15:.1f} °C)")
    print(f"  bulk outlet T_out : {t_out:.2f} K ({t_out - 273.15:.1f} °C)")
    print(f"  air warming ΔT    : {dT:.2f} K")
    print(f"  total flow Q      : {q_total:.3e} m³/s ({q_total * 60_000:.1f} L/min)")
    print(f"  mucosal heat loss : {heat_W:.3f} W   (ρ={args.rho} kg/m³, cp={args.cp} J/kg·K)")
    frac = dT / (310.15 - args.inlet_temp_K) if (310.15 - args.inlet_temp_K) > 0 else 0.0
    print(f"  air conditioning  : {100 * frac:.0f}% of the way to body temperature")
    if frac < 0.5:
        print("  note: air leaves well below body temp — either high flow, short path,")
        print("        or (for a NasalSeg crop) a truncated posterior airway.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
