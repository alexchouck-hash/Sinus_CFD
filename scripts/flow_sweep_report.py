#!/usr/bin/env python3
"""
Test the nonlinearity hypothesis: does nasal resistance R rise with flow rate?

Reads the per-flow-rate postProcessing folders written by foam/<case>/Allsweep.docker
(sweep/<lpm>/postProcessing) and reports R and ΔP at each flow rate. If R rises
with flow, that supports the explanation that the CFD resting-breathing R being
below the published 0.10-0.35 Pa·s/mL range is a genuine flow-dependence effect
(the published range is typically measured by rhinomanometry at ~150 Pa driving
pressure, far above quiet breathing's ~15 Pa) rather than a simulation error.

A pure linear (Poiseuille) resistor has R constant vs flow; the nose is known to
be nonlinear (inertial/turbulent losses at the valve grow faster than linearly),
so ΔP ~ a·Q + b·Q² and R = ΔP/Q ~ a + b·Q rises with Q.

Usage:
  py -3.12 scripts/flow_sweep_report.py --case P001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from compute_nasal_resistance import resistance_from_postprocessing  # noqa: E402


def _plot(rows: list[dict], out_path: Path, case: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = [r["q_lpm"] for r in rows]
    dp = [r["dp_pa"] for r in rows]
    R = [r["r_pa_s_ml"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(q, dp, "o-", color="tab:blue")
    ax1.set_xlabel("flow rate (L/min)")
    ax1.set_ylabel("ΔP (Pa)")
    ax1.set_title(f"{case}: pressure drop vs flow")
    ax1.axhline(150, ls="--", color="gray", alpha=0.6, label="rhinomanometry ~150 Pa")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(q, R, "o-", color="tab:red")
    ax2.set_xlabel("flow rate (L/min)")
    ax2.set_ylabel("resistance R (Pa·s/mL)")
    ax2.set_title(f"{case}: resistance vs flow")
    ax2.axhspan(0.10, 0.35, color="green", alpha=0.12, label="published resting range")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="P001")
    p.add_argument("--foam-root", type=Path, default=None)
    p.add_argument("--rho", type=float, default=1.14)
    args = p.parse_args()

    foam = args.foam_root or (REPO_ROOT / "foam" / args.case)
    sweep_dir = foam / "sweep"
    if not sweep_dir.is_dir():
        print(f"ERROR: no sweep/ under {foam} — run Allsweep.docker first.", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for lpm_dir in sorted(sweep_dir.iterdir(), key=lambda d: int(d.name) if d.name.isdigit() else 0):
        pp = lpm_dir / "postProcessing"
        if not pp.is_dir():
            continue
        res = resistance_from_postprocessing(pp, rho=args.rho)
        if res is None:
            print(f"  skip {lpm_dir.name}: incomplete postProcessing")
            continue
        res["nominal_lpm"] = int(lpm_dir.name)
        rows.append(res)

    if len(rows) < 2:
        print("ERROR: need at least 2 flow rates to assess nonlinearity.", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r["q_lpm"])

    print(f"[{args.case}] flow-rate sweep (existing 259k-cell mesh, laminar simpleFoam)")
    print(f"  {'Q (L/min)':>10}  {'ΔP (Pa)':>9}  {'R (Pa·s/mL)':>12}")
    for r in rows:
        print(f"  {r['q_lpm']:>10.1f}  {r['dp_pa']:>9.2f}  {r['r_pa_s_ml']:>12.4f}")

    r_lo, r_hi = rows[0]["r_pa_s_ml"], rows[-1]["r_pa_s_ml"]
    q_lo, q_hi = rows[0]["q_lpm"], rows[-1]["q_lpm"]
    rise = r_hi / r_lo if r_lo > 0 else float("inf")
    print(
        f"\n  R at {q_lo:.0f} L/min = {r_lo:.4f}  →  R at {q_hi:.0f} L/min = {r_hi:.4f}  "
        f"({rise:.2f}x)"
    )
    if rise >= 1.15:
        print(
            "  VERDICT: resistance rises with flow → nonlinear, as expected for the nose.\n"
            "  Supports the explanation that resting-breathing R below the published\n"
            "  range reflects the low ΔP regime, not a simulation error — the published\n"
            "  range is measured at higher (rhinomanometry ~150 Pa) driving pressure."
        )
    else:
        print(
            "  VERDICT: resistance ~flat with flow → nearly linear here. The below-range\n"
            "  resting R is NOT explained by flow nonlinearity alone; look to cropped-FOV\n"
            "  geometry (missing vestibule) or mesh resolution next."
        )

    out = foam / "sweep" / f"{args.case}_flow_sweep.json"
    out.write_text(json.dumps({"case": args.case, "rho": args.rho, "rows": rows}, indent=2), encoding="utf-8")
    plot_path = REPO_ROOT / "outputs" / args.case / f"{args.case}_flow_sweep.png"
    _plot(rows, plot_path, args.case)
    print(f"\n  wrote {out} and {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
