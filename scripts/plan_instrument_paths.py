"""Can each seeker reach its ostium from a nostril? One verdict per tool per ostium.

Usage:
  py -3.12 scripts/plan_instrument_paths.py --case THCA_HeadNeck [--no-paths]

Reads the sinus-free passage, the stripped sinus air, the naris ports and the
drainage records (via qa_patency.audit), places each instrument rigidly with
its tip at every detected ostium of its target sinus, and searches orientation
for a placement that clears the wall along its whole length and exits through
a naris. Writes outputs/<case>/<case>_instrument_fit.json with the verdicts,
the best placement's polyline in mm (for overlay), and the list of assumed
tool numbers -- a verdict on an assumed diameter is flagged as such.

Research prototype. Not a medical device.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from qa_patency import _read, audit  # noqa: E402
from sinus_cfd.instrument_fit import INSTRUMENTS, fit_instruments_to_ostia  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True)
    ap.add_argument("--outputs", type=Path, default=REPO_ROOT / "outputs")
    ap.add_argument("--no-paths", action="store_true", help="omit the polylines from the JSON")
    args = ap.parse_args()
    case, cd = args.case, args.outputs / args.case

    lumen_p = cd / f"{case}_passage_lumen.nrrd"
    inlet_p = cd / f"{case}_passage_inlet_open.nrrd"
    sinus_p = cd / f"{case}_sinus_detour.nrrd"
    for p in (lumen_p, inlet_p):
        if not p.is_file():
            print(f"ERROR: {p.name} missing; run analyze_passage first", file=sys.stderr)
            return 2
    img = sitk.ReadImage(str(lumen_p))
    if not np.allclose(np.asarray(img.GetDirection()).reshape(3, 3), np.eye(3), atol=1e-6):
        print("ERROR: passage_lumen.nrrd has a non-identity direction; "
              "instrument_fit assumes axis-aligned mm coordinates", file=sys.stderr)
        return 2
    spacing = tuple(float(v) for v in img.GetSpacing())
    origin = tuple(float(v) for v in img.GetOrigin())
    passage = sitk.GetArrayFromImage(img).astype(bool)
    sinus = _read(sinus_p) if sinus_p.is_file() else np.zeros_like(passage)
    inlet = _read(inlet_p)
    if not inlet.any():
        print("ERROR: inlet-open mask is empty; the naris port is undefined", file=sys.stderr)
        return 2

    rec = audit(case, args.outputs)
    if "error" in rec:
        print(f"ERROR: qa_patency.audit: {rec['error']}", file=sys.stderr)
        return 2
    recs = (rec.get("drainage") or {}).get("sinuses") or []
    # The tool's tip sits at the ostium, on the interface between passage and
    # sinus air, so clearance is measured against the wall of BOTH: the
    # tip may protrude into the ostium, the shaft must stay in the passage.
    airway = passage | sinus
    out = fit_instruments_to_ostia(recs, airway, inlet, spacing, origin,
                                   instruments=INSTRUMENTS, keep_points=not args.no_paths)
    out["case_id"] = case
    out["passage_used"] = lumen_p.name
    out["n_ostia_considered"] = sum(1 for r in recs if r.get("ostium_zyx") is not None)
    dst = cd / f"{case}_instrument_fit.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[{case}] instrument fit on {lumen_p.name} + stripped sinus air; "
          f"{len(recs)} sinus records, {out['n_ostia_considered']} with an ostium")
    for f in out["fits"]:
        side = f" {f['side']}" if f.get("side") else ""
        od = f" (ostium {f['ostium_diameter_mm']} mm)" if f.get("ostium_diameter_mm") else ""
        print(f"  {f['instrument']:<24} {f['target']}{side}{od}: {f['verdict'].upper()} -- {f['reason']}")
        if f.get("assumed"):
            print(f"      assumed: {', '.join(f['assumed'])}")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
