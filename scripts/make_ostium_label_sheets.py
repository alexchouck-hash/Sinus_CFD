#!/usr/bin/env python3
"""Generate ostium labelling sheets for manual measurement.

The automatic ostium calibre (2 x median half-width across the sinus/passage
interface) is currently BOUNDED by the 0.2-6 mm anatomical range but not
VALIDATED against ground truth. This produces, for every sinus body whose ostium
the pipeline resolved, a zoomed tri-planar sheet centred on the detected ostium
so it can be measured by hand, plus a CSV to record the measurement.

Each sheet shows the CT at that location with the segmentation faint underneath,
a crosshair at the detected ostium, and a millimetre scale bar. The pipeline's
own estimate is deliberately printed in the FOOTER rather than the title, so it
is visible for comparison but not the first thing read -- an anchoring number at
the top of a measurement task biases the measurement.

Usage:
  py -3.12 scripts\\make_ostium_label_sheets.py                 # all cases
  py -3.12 scripts\\make_ostium_label_sheets.py --case CQ500CT390
  py -3.12 scripts\\make_ostium_label_sheets.py --zoom-mm 12
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sinus_cfd.patency import drainage  # noqa: E402

SINUS_RGB = (1.00, 0.50, 0.05)
PASSAGE_RGB = (0.10, 0.60, 1.00)
IFACE_RGB = (0.95, 0.95, 0.10)     # the sinus/passage contact -- the ostium is ON this
# Plane that shows each ostium best. Maxillary drains through the superomedial
# wall into the middle meatus, which coronal shows in cross-section; the frontal
# recess runs superoinferiorly and is a sagittal structure; sphenoid opens
# anteriorly into the sphenoethmoidal recess, seen on axial.
BEST_PLANE = {"maxillary": 1, "frontal": 2, "sphenoid": 0, "ethmoid": 1}
PLANE_NAME = {0: "AXIAL", 1: "CORONAL", 2: "SAGITTAL"}
N_STRIP = 5                         # slices either side of centre, per sheet


def _read(p: Path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(bool)


def _orientation(stats: dict) -> tuple[bool, bool]:
    pi = stats.get("path_info") or {}
    return bool(pi.get("y_anterior_is_low", True)), bool(pi.get("superior_is_high_z", True))


def _load(case: str, outputs: Path):
    cd = outputs / case
    need = [f"{case}_airway_mask.nrrd", f"{case}_passage_lumen.nrrd",
            f"{case}_sinus_detour.nrrd", f"{case}_stats.json", f"{case}_head_mask.nrrd"]
    if not all((cd / n).is_file() for n in need):
        return None
    stats = json.loads((cd / f"{case}_stats.json").read_text(encoding="utf-8"))
    ref = sitk.ReadImage(str(cd / f"{case}_head_mask.nrrd"))
    sp = tuple(float(v) for v in ref.GetSpacing())
    airway = _read(cd / f"{case}_airway_mask.nrrd")
    passage = _read(cd / f"{case}_passage_lumen.nrrd")
    sinus = _read(cd / f"{case}_sinus_detour.nrrd")
    air_p = cd / f"{case}_all_interior_air.nrrd"
    interior = _read(air_p) if air_p.is_file() else None
    img_p = Path(stats.get("image_path") or "")
    if not img_p.is_file():
        return None
    full = sitk.GetArrayFromImage(sitk.ReadImage(str(img_p))).astype(np.float32)
    cz, cy, cx = stats.get("crop_origin_zyx") or [0, 0, 0]
    nz, ny, nx = airway.shape
    hu = np.full(airway.shape, -1024.0, np.float32)
    z1 = min(nz, full.shape[0] - cz); y1 = min(ny, full.shape[1] - cy)
    x1 = min(nx, full.shape[2] - cx)
    hu[:z1, :y1, :x1] = full[cz:cz + z1, cy:cy + y1, cx:cx + x1]
    return dict(hu=hu, airway=airway, passage=passage, sinus=sinus,
                interior=interior, spacing=sp, stats=stats)


def _panel(ax, ct2d, sin2d, pas2d, cc, rr, half_r, half_c, vratio, title, mm_per_col):
    ax.imshow(ct2d, cmap="gray", vmin=-1000, vmax=900, interpolation="nearest",
              aspect=vratio)
    ov = np.zeros(ct2d.shape + (4,))
    ov[pas2d] = [*PASSAGE_RGB, 0.22]
    ov[sin2d] = [*SINUS_RGB, 0.22]
    ax.imshow(ov, interpolation="nearest", aspect=vratio)
    # crosshair with a gap so the ostium itself is not covered
    gap = max(half_c * 0.10, 1.5)
    ax.plot([cc - half_c * 0.55, cc - gap], [rr, rr], color="#00ff88", lw=1.2)
    ax.plot([cc + gap, cc + half_c * 0.55], [rr, rr], color="#00ff88", lw=1.2)
    ax.plot([cc, cc], [rr - half_r * 0.55, rr - gap], color="#00ff88", lw=1.2)
    ax.plot([cc, cc], [rr + gap, rr + half_r * 0.55], color="#00ff88", lw=1.2)
    # 5 mm scale bar
    bar = 5.0 / mm_per_col
    x0 = cc - half_c * 0.9
    y0 = rr + half_r * 0.82
    ax.plot([x0, x0 + bar], [y0, y0], color="w", lw=3, solid_capstyle="butt")
    ax.text(x0 + bar / 2, y0 - half_r * 0.06, "5 mm", color="w", fontsize=8,
            ha="center", va="bottom")
    ax.set_xlim(cc - half_c, cc + half_c)
    ax.set_ylim(rr + half_r, rr - half_r)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def sheets_for_case(case: str, outputs: Path, out_dir: Path, zoom_mm: float,
                    all_planes: bool = False) -> list[dict]:
    d = _load(case, outputs)
    if d is None:
        print(f"  {case}: missing outputs, skipped")
        return []
    sp = d["spacing"]; sx, sy, sz = sp[0], sp[1], sp[2]
    y_ant, sup_hi = _orientation(d["stats"])
    res = drainage(d["airway"], d["sinus"], d["passage"], sp,
                   y_anterior_is_low=y_ant, superior_is_high_z=sup_hi,
                   interior_air=d["interior"])
    hu, sin, pas = d["hu"], d["sinus"], d["passage"]
    rows = []
    for rec in res.get("sinuses", []):
        if not rec.get("drains") or not rec.get("ostium_zyx"):
            continue
        z, y, x = rec["ostium_zyx"]
        nz, ny, nx = hu.shape
        if not (0 <= z < nz and 0 <= y < ny and 0 <= x < nx):
            continue
        planes = [0, 1, 2] if all_planes else [BEST_PLANE.get(rec["name"], 1)]
        for plane in planes:
            # a strip of slices: an ostium is a hole in a curved wall and is rarely
            # crisp on exactly one slice, so show several and let the marker choose
            centre = (z, y, x)[plane]
            n_slices = hu.shape[plane]
            step = max(int(round(0.8 / (sz, sy, sx)[2 - plane] if plane else 0.8 / sz)), 1)
            idxs = [centre + k * step for k in range(-(N_STRIP // 2), N_STRIP // 2 + 1)]
            idxs = [i for i in idxs if 0 <= i < n_slices]
            fig, axes = plt.subplots(1, len(idxs), figsize=(3.1 * len(idxs), 3.9))
            if len(idxs) == 1:
                axes = [axes]
            iface = ndi.binary_dilation(sin, np.ones((3, 3, 3), dtype=bool)) & pas
            for ax, si in zip(axes, idxs):
                if plane == 0:
                    c2, s2, p2, i2 = hu[si], sin[si], pas[si], iface[si]
                    cc, rr, hc, hr, vr, mmc = x, y, zoom_mm / sx, zoom_mm / sy, sy / sx, sx
                elif plane == 1:
                    c2, s2, p2, i2 = hu[:, si, :], sin[:, si, :], pas[:, si, :], iface[:, si, :]
                    cc, rr, hc, hr, vr, mmc = x, z, zoom_mm / sx, zoom_mm / sz, sz / sx, sx
                else:
                    c2, s2, p2, i2 = hu[:, :, si], sin[:, :, si], pas[:, :, si], iface[:, :, si]
                    cc, rr, hc, hr, vr, mmc = y, z, zoom_mm / sy, zoom_mm / sz, sz / sy, sy
                # Clamp the window inside the volume. The detected ostium can sit at
                # the very edge (CQ500CT390 maxillary L is at z=1), which put half the
                # window outside the data and rendered as a blank strip.
                nrow, ncol = c2.shape
                ccl = float(np.clip(cc, hc, max(ncol - hc, hc)))
                rrl = float(np.clip(rr, hr, max(nrow - hr, hr)))
                ax.imshow(c2, cmap="gray", vmin=-1000, vmax=900, interpolation="nearest",
                          aspect=vr)
                ov = np.zeros(c2.shape + (4,))
                ov[p2] = [*PASSAGE_RGB, 0.18]
                ov[s2] = [*SINUS_RGB, 0.18]
                ov[i2] = [*IFACE_RGB, 0.55]
                ax.imshow(ov, interpolation="nearest", aspect=vr)
                # Scale bar in AXES fractions: its data-space y depends on the
                # flip, and computing it from ylim put it off-panel. Length is
                # still physical -- 5 mm as a fraction of the 2*zoom_mm window.
                bar_frac = (5.0 / mmc) / (2.0 * hc)
                ax.plot([0.06, 0.06 + bar_frac], [0.07, 0.07], transform=ax.transAxes,
                        color="w", lw=3, solid_capstyle="butt", zorder=5)
                ax.text(0.06 + bar_frac / 2, 0.10, "5 mm", transform=ax.transAxes,
                        color="w", fontsize=8, ha="center", va="bottom", zorder=5)
                ax.set_xlim(ccl - hc, ccl + hc)
                # ORIENTATION. On coronal/sagittal the rows are z, and with
                # superior_is_high_z the LOW z end is inferior -- so the default
                # image order (row 0 at top) renders the head UPSIDE DOWN. That is
                # not cosmetic: marks made on inverted coronal sheets came back on
                # soft tissue (HU +41) 18-25 mm from the real ostium, and were
                # mistaken for a segmentation error. Put superior at the top.
                if plane == 0:
                    # axial: rows are y, anterior should be at the top
                    if y_ant:
                        ax.set_ylim(rrl + hr, rrl - hr)
                    else:
                        ax.set_ylim(rrl - hr, rrl + hr)
                else:
                    if sup_hi:
                        ax.set_ylim(rrl - hr, rrl + hr)   # high z (superior) at top
                    else:
                        ax.set_ylim(rrl + hr, rrl - hr)
                # Set the aspect AFTER the limits and force the box to follow it.
                # Passing aspect= to imshow alone leaves tight_layout free to stretch
                # the axes, which letterboxed a window that is square in mm
                # (2*zoom_mm on both axes) into a 2.5:1 strip.
                ax.set_aspect(vr, adjustable="box")
                ax.set_title(f"{PLANE_NAME[plane]} {si}" + ("  <- centre" if si == centre else ""),
                             fontsize=9)
                up = "ANT" if plane == 0 else "SUP"
                ax.text(0.02, 0.97, up, transform=ax.transAxes, color="#00ff88",
                        fontsize=8, va="top", ha="left")
                ax.axis("off")
            name = f"{rec['name']}_{rec['side']}"
            fig.suptitle(
                f"{case}  -  {rec['name']} {rec['side']}  ({rec['volume_ml']:.2f} mL)"
                "\n"
                "YELLOW = sinus/passage contact. Mark the OSTIUM: the narrow gap "
                "where orange actually opens into blue.",
                fontsize=12)
            fig.text(0.5, 0.015,
                     f"voxel {sx:.3f} x {sy:.3f} x {sz:.3f} mm  ·  strip step {step} slice(s)  ·  "
                     f"pipeline calibre {rec['ostium_diameter_mm']:.2f} mm (location not yet reliable)",
                     ha="center", fontsize=8, color="#666666")
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"__{PLANE_NAME[plane]}" if all_planes else ""
            png = out_dir / f"{case}__{name}{suffix}.png"
            fig.tight_layout(rect=[0, 0.06, 1, 0.86])
            fig.savefig(png, dpi=115, facecolor="white")
            plt.close(fig)
            rows.append({
                "case": case, "sinus": rec["name"], "side": rec["side"],
                "volume_ml": f"{rec['volume_ml']:.2f}",
                "plane": PLANE_NAME[plane], "centre_slice": centre,
                "ostium_z": z, "ostium_y": y, "ostium_x": x,
                "pipeline_diameter_mm": f"{rec['ostium_diameter_mm']:.2f}",
                "measured_diameter_mm": "", "measured_on_plane": "", "notes": "",
                "sheet": png.name,
            })
        print(f"  {png.name}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default=None)
    ap.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "outputs")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "outputs" / "ostium_labeling")
    ap.add_argument("--zoom-mm", type=float, default=15.0,
                    help="half-width of the zoom window in mm (default 15)")
    ap.add_argument("--all-planes", action="store_true",
                    help="emit one sheet per ostium PER PLANE. Marking the same "
                         "ostium from three independent views separates marking "
                         "precision (spread between the views) from real bias "
                         "(their consensus vs the detection).")
    args = ap.parse_args()

    cases = ([args.case] if args.case else
             sorted(p.name for p in args.outputs_dir.iterdir()
                    if p.is_dir() and (p / f"{p.name}_sinus_detour.nrrd").is_file()))
    all_rows = []
    for c in cases:
        print(f"{c}:")
        all_rows.extend(sheets_for_case(c, args.outputs_dir, args.out_dir,
                                        args.zoom_mm, args.all_planes))
    if not all_rows:
        print("No ostia with a resolved calibre found.")
        return 1
    csv_p = args.out_dir / "ostium_labels.csv"
    with csv_p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n{len(all_rows)} sheets + {csv_p}")
    print("Fill in measured_diameter_mm (and which plane you measured on).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
