#!/usr/bin/env python3
"""Compute ΔP, resistance, and L/R inlet allocation from flow.npz."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sinus_cfd.cfd_metrics import compute_cfd_metrics, write_cfd_metrics  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="VisibleHuman_Head")
    p.add_argument("--outputs-root", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument("--port-radius-mm", type=float, default=10.0)
    args = p.parse_args()

    case_dir = args.outputs_root / args.case
    report = compute_cfd_metrics(
        case_dir, args.case, port_radius_mm=args.port_radius_mm
    )
    path = write_cfd_metrics(case_dir, args.case, report=report)
    print(f"[{args.case}] wrote {path.name}")
    pr = report.get("pressure") or {}
    rs = report.get("resistance") or {}
    al = report.get("inlet_allocation") or {}
    print(f"  method={report.get('method')}")
    print(
        f"  p_in={pr.get('inlet_mean')}  p_out={pr.get('outlet_mean')}  "
        f"ΔP={pr.get('delta_p_inlet_minus_outlet')}"
    )
    print(f"  R=ΔP/Q={rs.get('R_delta_p_over_Q')}  Q={rs.get('Q_used_L_per_min')} L/min")
    print(
        f"  L/R fraction={al.get('left_fraction_from_probe'):.3f}/"
        f"{al.get('right_fraction_from_probe'):.3f}  "
        f"Q_L/Q_R scaled="
        f"{al.get('left_Q_L_per_min_scaled_to_target'):.2f}/"
        f"{al.get('right_Q_L_per_min_scaled_to_target'):.2f} L/min"
    )
    for n in (report.get("notes") or [])[:6]:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
