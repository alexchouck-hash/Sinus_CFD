#!/usr/bin/env python3
"""
Assess unzipped CQ500 (or any) DICOM scans under data/incoming/ for pipeline viability.

For each scan we pick the thinnest "plain" series and report:
  * effective slice spacing (mm)  — want <= ~1.5 mm, ideally <= 1 mm
  * slice count
  * INTERNAL AIR volume (mL)      — nasal cavity + paranasal sinuses show up as large
    *internal* air pockets; brain-only head CT (vertex -> skull base) has almost none.
    This is the FOV-includes-the-nose test.

VIABLE = thin enough AND internal air above threshold (nose/sinuses in frame).

Re-runnable as more scans finish extracting.

Usage:
  py -3.12 scripts/assess_dicom_incoming.py [--root data/incoming] [--max-mm 1.6] [--min-air-ml 6]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

REPO = Path(__file__).resolve().parents[1]


def series_dirs(root: Path) -> list[Path]:
    """Leaf directories that directly contain .dcm files."""
    out = set()
    for f in root.rglob("*.dcm"):
        out.add(f.parent)
    return sorted(out)


def scan_id(series_dir: Path) -> str:
    """The CQ500CTxxx scan folder name (2 levels above the series: scan/Unknown Study/series)."""
    for p in series_dir.parents:
        if p.name.lower().startswith("cq500") or "CT" in p.name:
            return p.name.split()[0]
    return series_dir.parent.parent.name.split()[0]


def eff_spacing_mm(series_dir: Path) -> tuple[float, int]:
    """(effective z-spacing mm, file count) from one header + file count."""
    files = list(series_dir.glob("*.dcm"))
    if not files:
        return (999.0, 0)
    r = sitk.ImageFileReader()
    r.SetFileName(str(files[0]))
    try:
        r.ReadImageInformation()
    except Exception:
        return (999.0, len(files))
    def tag(t):
        return r.GetMetaData(t).strip() if r.HasMetaDataKey(t) else ""
    # SliceThickness FIRST. SpacingBetweenSlices (0018,0088) is documented as the
    # true z-step, but CQ500 fills it with junk: 10-21 mm on series whose slices
    # really are 0.625 mm apart. Preferring it made this screen reject 18 of the
    # 32 thin series in qct01 alone, which is why only 2 usable cases were ever
    # promoted out of 491 scans -- the gate was throwing away the dataset.
    #
    # SliceThickness was correct on every series checked against a stacked
    # volume, and it is what identifies 'CT 5mm POST CONTRAST' as 0.625 mm.
    # Series names lie here too, so neither name nor 0018,0088 is trusted.
    #
    # Known limit: thickness is not the reconstruction interval, so a 0.625 mm
    # slice reconstructed every 5 mm would be admitted. That is the safe error
    # for a SCREEN -- the stacking step sees the real positions and rejects it,
    # whereas under-admitting loses a scan silently.
    sbs = tag("0018|0088")  # SpacingBetweenSlices -- unreliable on CQ500
    st = tag("0018|0050")   # SliceThickness
    val = st or sbs
    try:
        z = abs(float(val.split("\\")[0]))
    except ValueError:
        z = 999.0
    return (z if z > 0 else 999.0, len(files))


def internal_air_ml(series_dir: Path) -> tuple[float, tuple, tuple]:
    """Stack the series and measure air ENCLOSED BY THE HEAD in mL (nasal cavity + sinuses).

    Method: body-hull air, NOT "air components that don't touch the FOV face."

    The old face-touching test was wrong for the design-target cohort: on an
    open-naris living head the nasal/pharyngeal lumen is 26-connected to the
    ambient FOV air through the nostrils, so the whole airway is one component
    that touches a volume face -> it got labeled "exterior" and dropped, and an
    open-airway head scored ~0 mL and was rejected as "brain-only?". (Sealed
    maxillary sinuses survived only because they are NOT connected to ambient.)

    Instead we measure air that lies inside the body silhouette:
      1. body/tissue mask = HU > -300 (skin, muscle, bone, teeth...)
      2. keep the largest 3D body component (the head; drops table specks etc.)
      3. per axial slice, HOLE-FILL the body silhouette. In-plane the nasal
         lumen, the sinuses and the pharynx are holes in the tissue outline
         (the nostril and the neck cut open along z, not within a slice);
         ambient air reaches the slice border and is never a hole.
      4. internal air = (HU < -400) AND inside the stacked filled silhouette.
    This used to be the per-slice CONVEX HULL, chosen because 3-D fill_holes
    cannot reclaim an airway that vents through the nares. The hull also spans
    the concavity in front of the nose and beside the ear, so it counted room
    air as internal -- inflating the very number this screen uses to decide
    whether a scan holds a nasal airway at all. The per-slice fill recovers the
    airway (measured 100% of it on CQ500CT390 and CQ500CT105) without the room
    air. A genuine brain-only scan (vertex->skull base) still reads low.
    """
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not names:
        return (0.0, (), ())
    reader.SetFileNames(names)
    img = reader.Execute()
    a = sitk.GetArrayFromImage(img)  # HU, (z,y,x)
    sp = img.GetSpacing()
    air = a < -400
    if not air.any():
        return (0.0, img.GetSize(), sp)

    # 1-2. body mask, largest 3D component = the head
    body = a > -300
    if not body.any():
        return (0.0, img.GetSize(), sp)
    lab, nlab = ndi.label(body)
    if nlab > 1:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        body = lab == int(counts.argmax())

    # 3. per-axial-slice HOLE FILL of the body silhouette. In-plane the nasal
    #    lumen, sinuses and pharynx are holes in the tissue outline; ambient air
    #    reaches the slice border and is not. This was the per-slice convex hull,
    #    which also spans the concavity in front of the nose and beside the ear
    #    and so counted room air as internal -- inflating the very number this
    #    screen uses to decide whether a scan contains a nasal airway at all.
    enclosed = np.zeros_like(body)
    for z in range(body.shape[0]):
        sl = body[z]
        if sl.sum() < 50:  # too little tissue in this slice to define a silhouette
            continue
        enclosed[z] = ndi.binary_fill_holes(sl)

    # 4. air enclosed by the head silhouette
    internal = air & enclosed
    ml = float(internal.sum()) * float(np.prod(sp)) / 1000.0
    return (ml, img.GetSize(), tuple(round(float(v), 2) for v in sp))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=REPO / "data" / "incoming")
    p.add_argument("--max-mm", type=float, default=1.6, help="Max effective slice spacing to bother stacking")
    p.add_argument("--min-air-ml", type=float, default=6.0, help="Internal-air threshold for 'nose in FOV'")
    p.add_argument("--limit", type=int, default=0, help="Stack at most N thin candidates (0 = all)")
    p.add_argument("--report", type=Path, default=None,
                   help="Write every verdict to this JSON file, one row per scan, updated "
                        "after each scan so a killed run keeps what it measured. The first "
                        "sweep of this dataset was never persisted, so its verdicts could "
                        "not be audited when the internal-air metric turned out to count "
                        "room air.")
    p.add_argument("--csv", type=Path, default=None, help="Same rows as --report, as CSV")
    args = p.parse_args()

    rows: list[dict] = []

    def _flush() -> None:
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps({
                "root": str(args.root), "max_mm": args.max_mm,
                "min_air_ml": args.min_air_ml,
                "metric": "internal_air_ml = air inside the per-slice hole-filled body silhouette",
                "rows": rows,
            }, indent=1), encoding="utf-8")
        if args.csv and rows:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    # group series by scan, keep the thinnest per scan (prefer names with THIN)
    by_scan: dict[str, list[tuple[float, int, Path]]] = {}
    for sd in series_dirs(args.root):
        z, n = eff_spacing_mm(sd)
        by_scan.setdefault(scan_id(sd), []).append((z, n, sd))
    print(f"scans found: {len(by_scan)}  (root={args.root})\n")

    # pick thin series per scan
    candidates = []
    for sid, lst in sorted(by_scan.items()):
        thinny = [t for t in lst if "thin" in t[2].name.lower()]
        pool = thinny or lst
        z, n, sd = min(pool, key=lambda t: (t[0], -t[1]))  # thinnest, then most slices
        candidates.append((sid, z, n, sd))

    thin = [c for c in candidates if c[1] <= args.max_mm and c[2] >= 80]
    print(f"{'scan':12} {'series':22} {'mm':>5} {'slices':>6} {'air_mL':>7} viable")
    print("-" * 72)
    viable = []
    todo = thin if not args.limit else thin[: args.limit]
    for sid, z, n, sd in sorted(candidates, key=lambda c: c[1]):
        if (sid, z, n, sd) in todo:
            ml, size, sp = internal_air_ml(sd)
            ok = ml >= args.min_air_ml
            flag = "YES" if ok else "no (brain-only?)"
            if ok:
                viable.append((sid, z, n, sd, ml))
            print(f"{sid:12} {sd.name[:22]:22} {z:5.2f} {n:6d} {ml:7.1f} {flag}", flush=True)
            rows.append({"scan": sid, "series": sd.name, "slice_mm": round(z, 3),
                         "slices": n, "internal_air_ml": round(ml, 1),
                         "size_xyz": list(size), "spacing_xyz": list(sp),
                         "viable": bool(ok),
                         "reason": "" if ok else "internal air below threshold (brain-only?)",
                         "dicom_dir": str(sd)})
            _flush()
        else:
            reason = "too thick" if z > args.max_mm else ("few slices" if n < 80 else "skip")
            print(f"{sid:12} {sd.name[:22]:22} {z:5.2f} {n:6d} {'':>7} -- {reason}")
            rows.append({"scan": sid, "series": sd.name, "slice_mm": round(z, 3),
                         "slices": n, "internal_air_ml": None, "size_xyz": None,
                         "spacing_xyz": None, "viable": False, "reason": reason,
                         "dicom_dir": str(sd)})
    _flush()

    print("\n" + "=" * 72)
    if viable:
        print(f"VIABLE ({len(viable)}) — thin + nose/sinuses in FOV:")
        for sid, z, n, sd, ml in sorted(viable, key=lambda v: (v[1], -v[4])):
            print(f"  {sid}: {z:.2f} mm, {n} slices, {ml:.0f} mL internal air")
            print(f"     --dicom-dir \"{sd}\"")
    else:
        print("No viable scans yet (thin + nasal FOV). Assess again as more extract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
