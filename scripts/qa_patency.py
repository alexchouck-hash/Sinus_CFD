#!/usr/bin/env python3
"""Patency QA: assure a nares -> pharynx flow path, and sinus drainage pathways.

Answers the two questions that gate CLAUDE.md goals 2-4:

  1. Is there a connected route from EACH naris to the outlet inside the CFD
     domain, and how tight is it at its narrowest?
  2. Does each sinus (maxillary / frontal / sphenoid / ethmoid) reach the nasal
     passage, and how wide is its ostium?

Unlike ``qa_connectivity.py`` (which is import-free of ``src/`` by contract),
this script imports the segmentation modules, because the drainage answer needs
the same widest-path machinery the sinus strip uses.

Usage:
  py -3.12 scripts\\qa_patency.py --case CQ500CT105
  py -3.12 scripts\\qa_patency.py --all
  py -3.12 scripts\\qa_patency.py --case VisibleHuman_Head --json out.json

Exit code: 0 if every audited case has a bilateral flow path, 1 otherwise.
Drainage findings are reported but do not gate: an obstructed ostium is a
clinical finding, not a segmentation failure.
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

from sinus_cfd.patency import drainage, flow_path, naris_territory  # noqa: E402


def _read(path: Path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(bool)


def _orientation(stats: dict) -> tuple[bool, bool]:
    y_ant, sup_hi = True, True
    for note in stats.get("notes") or []:
        s = str(note)
        if "y_anterior_is_low=False" in s:
            y_ant = False
        if "superior_is_high_z=False" in s:
            sup_hi = False
    pi = stats.get("path_info") or {}
    if "y_anterior_is_low" in pi:
        y_ant = bool(pi["y_anterior_is_low"])
    if "superior_is_high_z" in pi:
        sup_hi = bool(pi["superior_is_high_z"])
    return y_ant, sup_hi


def audit(case: str, outputs: Path) -> dict:
    cd = outputs / case
    stats_p = cd / f"{case}_stats.json"
    airway_p = cd / f"{case}_airway_mask.nrrd"
    if not stats_p.is_file() or not airway_p.is_file():
        return {"case": case, "error": f"missing process_whole_head outputs in {cd}"}
    stats = json.loads(stats_p.read_text(encoding="utf-8"))
    img = sitk.ReadImage(str(cd / f"{case}_head_mask.nrrd"))
    spacing = tuple(float(v) for v in img.GetSpacing())
    airway = _read(airway_p)

    passage_p = cd / f"{case}_passage_lumen.nrrd"
    passage = _read(passage_p) if passage_p.is_file() else airway
    used_passage = "passage_lumen" if passage_p.is_file() else "airway_mask (no passage_lumen)"

    bc_p = cd / f"{case}_boundary_conditions.json"
    inlets: dict[str, tuple[int, int, int]] = {}
    outlet = None
    if bc_p.is_file():
        for port in json.loads(bc_p.read_text(encoding="utf-8")).get("ports", []):
            mm = tuple(float(v) for v in port["center_mm"])
            ix, iy, iz = img.TransformPhysicalPointToIndex(mm)
            zyx = (int(iz), int(iy), int(ix))
            if port.get("role") == "inlet":
                inlets[str(port.get("name"))] = zyx
            elif port.get("role") == "outlet":
                outlet = zyx

    rec: dict = {"case": case, "domain": used_passage}
    rec["_ref_img"] = img
    rec["_passage"] = passage
    if not inlets or outlet is None:
        rec["flow"] = {"ok": False, "notes": ["boundary_conditions.json has no inlet/outlet ports"]}
    else:
        rec["flow"] = flow_path(passage, inlets, outlet, spacing)
        # Diagnostic: would the route exist on the un-stripped airway?
        if not rec["flow"].get("ok"):
            rec["flow_on_airway_mask"] = flow_path(airway, inlets, outlet, spacing)

    if inlets and len(inlets) >= 2:
        terr, tmeta = naris_territory(passage, inlets, spacing, outlet_zyx=outlet)
        rec["territory"] = tmeta
        rec["_territory"] = terr
    sinus_p = cd / f"{case}_sinus_detour.nrrd"
    if sinus_p.is_file():
        y_ant, sup_hi = _orientation(stats)
        air_p = cd / f"{case}_all_interior_air.nrrd"
        rec["drainage"] = drainage(
            airway, _read(sinus_p), passage, spacing,
            y_anterior_is_low=y_ant, superior_is_high_z=sup_hi,
            interior_air=_read(air_p) if air_p.is_file() else None,
        )
    else:
        rec["drainage"] = {"sinuses": [], "notes": ["no <case>_sinus_detour.nrrd"]}
    return rec


def render(rec: dict) -> bool:
    case = rec["case"]
    print(f"\n{'=' * 74}\n{case}")
    if "error" in rec:
        print(f"  ERROR: {rec['error']}")
        return False
    print(f"  domain: {rec['domain']}")
    flow = rec.get("flow", {})
    ok = bool(flow.get("ok"))
    print(f"\n  [1] FLOW PATH  nares -> pharynx        {'PASS' if ok else 'FAIL'}")
    if flow.get("outlet_snap_mm") is not None:
        print(f"      outlet resolved at {flow['outlet_snap_mm']} mm from its BC port")
    for name, r in (flow.get("inlets") or {}).items():
        if r.get("connected"):
            mr = r.get("min_radius_mm")
            mr_s = f"{mr} mm" if mr is not None else "n/a"
            print(f"      {name:<15} connected  route {r.get('path_len_mm')} mm  "
                  f"tightest radius {mr_s}  (port snapped {r.get('snap_mm')} mm)")
        else:
            print(f"      {name:<15} NOT CONNECTED -- {r.get('reason')}")
    for n in flow.get("notes") or []:
        print(f"      note: {n}")
    alt = rec.get("flow_on_airway_mask")
    if alt is not None:
        good = [k for k, v in (alt.get("inlets") or {}).items() if v.get("connected")]
        print(f"      diagnostic: on airway_mask (pre-strip) {len(good)}/"
              f"{len(alt.get('inlets') or {})} inlet(s) connect"
              f"{' -> the strip or the box crop removed the vestibule' if good else ''}")

    t = rec.get("territory")
    if t:
        print()
        print("  [3] AIRWAY ID  per naris")
        if t.get("left_ml") is not None:
            print(f"      left-fed   {t['left_ml']:>7.2f} mL   (port snapped {t.get('left_snap_mm')} mm)")
            print(f"      right-fed  {t['right_ml']:>7.2f} mL   (port snapped {t.get('right_snap_mm')} mm)")
            print(f"      convergence{t['convergence_ml']:>7.2f} mL   "
                  f"(streams within {t['mixing_tol_mm']} mm of equal route)")
            if t.get("balance") is not None:
                print(f"      L/R balance {t['balance']:.2f}")
        for n in t.get("notes") or []:
            print(f"      note: {n}")
    dr = rec.get("drainage", {})
    sins = dr.get("sinuses") or []
    print(f"\n  [2] DRAINAGE PATHWAYS                  {len(sins)} sinus body(ies)")
    if sins:
        print(f"      {'sinus':<12}{'side':<8}{'vol mL':>8}{'ostium mm':>11}{'drains':>8}  connection")
        for s in sins:
            ost = f"{s['ostium_diameter_mm']:.2f}" if s["drains"] else "-"
            print(f"      {s['name']:<12}{s['side']:<8}{s['volume_ml']:>8.2f}"
                  f"{ost:>11}{str(s['drains']):>8}  {s.get('connection','')}")
    for n in dr.get("notes") or []:
        print(f"      note: {n}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--case")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "outputs")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--write-territory", action="store_true",
                    help="write <case>_naris_territory.nrrd (1=left-fed, 2=right-fed, "
                         "3=convergence) for the viewer")
    args = ap.parse_args()

    cases = (
        sorted(d.name for d in args.outputs_dir.iterdir()
               if d.is_dir() and (d / f"{d.name}_airway_mask.nrrd").is_file())
        if args.all else [args.case]
    )
    recs, all_ok = [], True
    for c in cases:
        r = audit(c, args.outputs_dir)
        if args.write_territory and r.get("_territory") is not None:
            out = args.outputs_dir / c / f"{c}_naris_territory.nrrd"
            im = sitk.GetImageFromArray(r["_territory"].astype(np.uint8))
            im.CopyInformation(r["_ref_img"])
            sitk.WriteImage(im, str(out), useCompression=True)
            print(f"  wrote {out.name}")
        all_ok &= render(r)
        for k in ("_territory", "_ref_img", "_passage"):
            r.pop(k, None)
        recs.append(r)
    print(f"\n{'=' * 74}")
    for r in recs:
        tag = "ERROR" if "error" in r else ("PASS" if r.get("flow", {}).get("ok") else "FAIL")
        nsin = len(r.get("drainage", {}).get("sinuses") or [])
        print(f"  {tag:<6} {r['case']:<26} flow-path   sinuses={nsin}")
    if args.json:
        args.json.write_text(json.dumps(recs, indent=2), encoding="utf-8")
        print(f"  wrote {args.json}")
    print(f"EXIT {0 if all_ok else 1}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
