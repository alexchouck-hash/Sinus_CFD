#!/usr/bin/env python3
"""Render a case's auto-segmentation as figures: slices, per-naris ID, and 3D.

This is the "show me" view of what the pipeline produced. It reads the SAME
masks ``qa_patency.py`` audits, so the picture and the numbers cannot disagree.

Three deliberate choices, each of which was a bug at some point in this repo:

* Orientation comes from the DATA, never from an assumption. Superior/anterior
  from the stats orientation flags; patient-left from the two naris ports, which
  are labelled. Slices are then drawn radiologically (patient-left on the right).
* Every panel carries a millimetre scale bar computed from the spacing, so a
  1-3 mm ostium can be judged by eye rather than taken on trust.
* Sinuses are coloured by NAME using the per-body labels ``drainage`` returns,
  not by re-labelling the mask -- re-labelling silently re-fuses adjacent bodies.

Usage:
  py -3.12 scripts\\render_segmentation_figure.py --case CQ500CT390
  py -3.12 scripts\\render_segmentation_figure.py --case CQ500CT390 --out-dir outputs/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sinus_cfd.patency import drainage, flow_path, naris_territory  # noqa: E402

OUTPUTS = REPO_ROOT / "outputs"

CT_WINDOW = (-1000.0, 400.0)     # wide enough to show both air and bone
ALPHA = 0.55

LEFT_COLOR = "#2b8cbe"
RIGHT_COLOR = "#d6604d"
SHARED_COLOR = "#8cc9e8"
CONVERGE_COLOR = "#9c6ade"
SINUS_COLORS = {
    "maxillary": "#f28e2b",
    "frontal": "#e8c33c",
    "sphenoid": "#59a14f",
    "ethmoid": "#e377c2",
    "unknown": "#999999",
}


# ----------------------------------------------------------------- loading

def _read(p: Path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(bool)


def _orientation(stats: dict) -> tuple[bool, bool]:
    y_ant, sup_hi = True, True
    for note in stats.get("notes") or []:
        s = str(note)
        if "y_anterior_is_low=False" in s:
            y_ant = False
        if "superior_is_high_z=False" in s:
            sup_hi = False
    pi = stats.get("path_info") or {}
    y_ant = bool(pi.get("y_anterior_is_low", y_ant))
    sup_hi = bool(pi.get("superior_is_high_z", sup_hi))
    return y_ant, sup_hi


def load_case(case: str, outputs: Path) -> dict:
    cd = outputs / case
    stats = json.loads((cd / f"{case}_stats.json").read_text(encoding="utf-8"))
    ref = sitk.ReadImage(str(cd / f"{case}_head_mask.nrrd"))
    spacing = tuple(float(v) for v in ref.GetSpacing())
    airway = _read(cd / f"{case}_airway_mask.nrrd")
    shape = airway.shape

    pas_p = cd / f"{case}_passage_lumen.nrrd"
    passage = _read(pas_p) if pas_p.is_file() else airway

    full = sitk.GetArrayFromImage(sitk.ReadImage(stats["image_path"])).astype(np.float32)
    cz, cy, cx = stats.get("crop_origin_zyx") or [0, 0, 0]
    nz, ny, nx = shape
    hu = np.full(shape, -1024.0, np.float32)
    z1 = min(nz, full.shape[0] - cz); y1 = min(ny, full.shape[1] - cy)
    x1 = min(nx, full.shape[2] - cx)
    hu[:z1, :y1, :x1] = full[cz:cz + z1, cy:cy + y1, cx:cx + x1]

    inlets, outlet = {}, None
    for port in json.loads(
            (cd / f"{case}_boundary_conditions.json").read_text(encoding="utf-8")
    ).get("ports", []):
        ix, iy, iz = ref.TransformPhysicalPointToIndex(
            tuple(float(v) for v in port["center_mm"]))
        zyx = (int(iz), int(iy), int(ix))
        if port.get("role") == "inlet":
            inlets[str(port.get("name"))] = zyx
        elif port.get("role") == "outlet":
            outlet = zyx

    y_ant, sup_hi = _orientation(stats)
    sin_p = cd / f"{case}_sinus_detour.nrrd"
    air_p = cd / f"{case}_all_interior_air.nrrd"
    dr = drainage(
        airway, _read(sin_p) if sin_p.is_file() else np.zeros(shape, bool),
        passage, spacing, y_anterior_is_low=y_ant, superior_is_high_z=sup_hi,
        interior_air=_read(air_p) if air_p.is_file() else None, hu=hu)

    read_opt = lambda n: (_read(cd / f"{case}_{n}.nrrd")
                          if (cd / f"{case}_{n}.nrrd").is_file() else None)
    terr, tmeta = None, {}
    if len(inlets) >= 2:
        terr, tmeta = naris_territory(passage, inlets, spacing, outlet_zyx=outlet)

    # Patient-left is read off the LABELLED naris ports, not assumed.
    ln, rn = inlets.get("left_nostril"), inlets.get("right_nostril")
    x_left_is_high = bool(ln[2] > rn[2]) if (ln and rn) else True

    return dict(
        case=case, stats=stats, spacing=spacing, shape=shape, hu=hu,
        airway=airway, passage=passage, left=read_opt("cavity_left"),
        right=read_opt("cavity_right"), merge=read_opt("merge_zone"),
        drain=dr, territory=terr, tmeta=tmeta, inlets=inlets, outlet=outlet,
        y_ant=y_ant, sup_hi=sup_hi, x_left_is_high=x_left_is_high,
        flow=flow_path(passage, inlets, outlet, spacing) if (inlets and outlet) else {},
    )


# ----------------------------------------------------------------- drawing

def _rgb_overlay(ct2d, layers):
    """Greyscale CT with translucent colour layers. layers = [(mask2d, hex)]."""
    lo, hi = CT_WINDOW
    g = np.clip((ct2d - lo) / (hi - lo), 0.0, 1.0)
    img = np.dstack([g, g, g])
    for mask, colour in layers:
        if mask is None or not np.any(mask):
            continue
        c = np.array(to_rgb(colour))
        img[mask] = (1 - ALPHA) * img[mask] + ALPHA * c
    return img


def _scalebar(ax, mm_per_px_x, length_mm=10.0, colour="w"):
    x0, x1 = ax.get_xlim()
    frac = (length_mm / mm_per_px_x) / abs(x1 - x0)
    ax.plot([0.06, 0.06 + frac], [0.055, 0.055], color=colour, lw=2.2,
            solid_capstyle="butt", transform=ax.transAxes, zorder=6)
    ax.text(0.06 + frac / 2, 0.075, f"{length_mm:g} mm", color=colour,
            fontsize=6.5, ha="center", va="bottom", transform=ax.transAxes,
            zorder=6)


def _corner(ax, txt, xy, colour="#ffe08a"):
    ax.text(xy[0], xy[1], txt, color=colour, fontsize=6.5, ha="center",
            va="center", transform=ax.transAxes, zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.45))


def _slice_layers(d, plane, idx, body_colour, use_territory=False):
    """Return (ct2d, [(mask2d, colour)]) for one slice, in ARRAY order."""
    sl = [slice(None)] * 3
    sl[plane] = idx
    sl = tuple(sl)
    ct2d = d["hu"][sl]
    layers = []
    if use_territory and d["territory"] is not None:
        t = d["territory"][sl]
        layers = [(t == 1, LEFT_COLOR), (t == 2, RIGHT_COLOR),
                  (t == 3, CONVERGE_COLOR)]
    else:
        bl = d["drain"].get("body_labels")
        if bl is not None:
            for ids, colour in body_colour:
                layers.append((np.isin(bl[sl], ids), colour))
        pas = d["passage"][sl]
        left = d["left"][sl] if d["left"] is not None else np.zeros_like(pas)
        right = d["right"][sl] if d["right"] is not None else np.zeros_like(pas)
        layers.append((pas & ~left & ~right, SHARED_COLOR))
        layers.append((pas & left, LEFT_COLOR))
        layers.append((pas & right, RIGHT_COLOR))
    return ct2d, layers


def _orient2d(d, plane, ct2d, layers):
    """Array slice -> display orientation. Returns (ct, layers, mmx, mmy, tags)."""
    sx, sy, sz = d["spacing"]
    flip_lr = not d["x_left_is_high"]        # radiological: patient-left on right
    if plane == 0:      # axial: rows=y, cols=x
        mmx, mmy = sx, sy
        flip_ud = not d["y_ant"]             # want anterior at TOP
        tags = [("A", (0.5, 0.955)), ("P", (0.5, 0.045)),
                ("R", (0.045, 0.5)), ("L", (0.955, 0.5))]
    elif plane == 1:    # coronal: rows=z, cols=x
        mmx, mmy = sx, sz
        flip_ud = d["sup_hi"]                # want superior at TOP
        tags = [("S", (0.5, 0.955)), ("I", (0.5, 0.045)),
                ("R", (0.045, 0.5)), ("L", (0.955, 0.5))]
    else:               # sagittal: rows=z, cols=y
        mmx, mmy = sy, sz
        flip_ud = d["sup_hi"]
        flip_lr = not d["y_ant"]             # want anterior on the LEFT
        tags = [("S", (0.5, 0.955)), ("I", (0.5, 0.045)),
                ("A", (0.045, 0.5)), ("P", (0.955, 0.5))]

    def fix(a):
        if a is None:
            return None
        if flip_ud:
            a = a[::-1]
        if flip_lr:
            a = a[:, ::-1]
        return a

    return fix(ct2d), [(fix(m), c) for m, c in layers], mmx, mmy, tags


def _panel(ax, d, plane, idx, body_colour, use_territory=False, title=None):
    ct2d, layers = _slice_layers(d, plane, idx, body_colour, use_territory)
    ct2d, layers, mmx, mmy, tags = _orient2d(d, plane, ct2d, layers)
    img = _rgb_overlay(ct2d, layers)
    ax.imshow(img, interpolation="nearest", aspect=mmy / mmx, origin="upper")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#333"); s.set_linewidth(0.5)
    for t, xy in tags:
        _corner(ax, t, xy)
    _scalebar(ax, mmx)
    if title:
        ax.set_title(title, fontsize=7.5, color="#ddd", pad=3)


def _sinus_body_colours(d):
    out, legend, seen = [], [], {}
    for r in d["drain"]["sinuses"]:
        ids = r.get("body_ids") or ([r["body_id"]] if r.get("body_id") else [])
        if not ids:
            continue
        colour = SINUS_COLORS.get(r["name"], SINUS_COLORS["unknown"])
        out.append((ids, colour))
        if r["name"] not in seen:
            seen[r["name"]] = colour
            legend.append((r["name"].capitalize() + " sinus", colour))
    return out, legend


def _spread(mask, axis, n):
    other = tuple(i for i in range(3) if i != axis)
    idx = np.where(mask.any(axis=other))[0]
    if idx.size == 0:
        return []
    lo, hi = int(idx.min()), int(idx.max())
    span = hi - lo
    return [int(round(v)) for v in np.linspace(lo + span * 0.08, hi - span * 0.08, n)]


def _fig(nrow, ncol, w, h):
    fig, axes = plt.subplots(nrow, ncol, figsize=(w, h), facecolor="#111")
    return fig, np.atleast_1d(axes).ravel()


def _legend(fig, entries, y=0.015):
    handles = [Line2D([], [], marker="s", ls="", ms=7, color=c, label=n)
               for n, c in entries]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(entries), 6),
               frameon=False, fontsize=7.5, labelcolor="#ddd",
               bbox_to_anchor=(0.5, y))


def render(case: str, outputs: Path, out_dir: Path) -> list[Path]:
    d = load_case(case, outputs)
    out_dir.mkdir(parents=True, exist_ok=True)
    body_colour, sin_legend = _sinus_body_colours(d)
    # Only NAMED bodies. body_labels also carries the unnamed leftovers (mastoid,
    # orbital and residual ambient air), and spanning slices over those puts half
    # the panels behind the head where there is nothing to see.
    named = np.isin(d["drain"]["body_labels"],
                    [i for ids, _c in body_colour for i in ids])
    shown = d["passage"] | named
    written = []

    anat_legend = ([("Left cavity", LEFT_COLOR), ("Right cavity", RIGHT_COLOR),
                    ("Shared / nasopharynx", SHARED_COLOR)] + sin_legend)

    for plane, n, name in ((0, 8, "axial"), (1, 8, "coronal"), (2, 4, "sagittal")):
        idxs = _spread(shown, plane, n)
        ncol = n // 2 if n > 4 else n
        nrow = 2 if n > 4 else 1
        fig, axes = _fig(nrow, ncol, 15.5, 4.6 if nrow == 1 else 7.4)
        step = (d["spacing"][2], d["spacing"][1], d["spacing"][0])[plane]
        for ax, i in zip(axes, idxs):
            _panel(ax, d, plane, i, body_colour,
                   title=f"{name[:3]} {i}   ({i * step:.0f} mm)")
        for ax in axes[len(idxs):]:
            ax.axis("off")
        fig.suptitle(f"{case} — auto-segmentation, {name}",
                     color="#eee", fontsize=11, y=0.985)
        _legend(fig, anat_legend)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.10,
                            wspace=0.03, hspace=0.14)
        p = out_dir / f"{case}_seg_{name}.png"
        fig.savefig(p, dpi=170, facecolor=fig.get_facecolor())
        plt.close(fig)
        written.append(p)

    if d["territory"] is not None:
        fig, axes = _fig(2, 5, 13.0, 7.4)
        idxs = _spread(d["passage"], 1, 5) + _spread(d["passage"], 0, 5)
        planes = [1] * 5 + [0] * 5
        for ax, pl, i in zip(axes, planes, idxs):
            _panel(ax, d, pl, i, body_colour, use_territory=True,
                   title=f"{'cor' if pl == 1 else 'ax'} {i}")
        t = d["tmeta"]
        fig.suptitle(
            f"{case} — airway ID per naris   "
            f"L-fed {t.get('left_ml', 0):.1f} mL · R-fed {t.get('right_ml', 0):.1f} mL "
            f"· convergence {t.get('convergence_ml', 0):.1f} mL",
            color="#eee", fontsize=11, y=0.985)
        _legend(fig, [("Left-nostril fed", LEFT_COLOR),
                      ("Right-nostril fed", RIGHT_COLOR),
                      ("Fed by both", CONVERGE_COLOR)])
        fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.10,
                            wspace=0.03, hspace=0.14)
        p = out_dir / f"{case}_seg_territory.png"
        fig.savefig(p, dpi=170, facecolor=fig.get_facecolor())
        plt.close(fig)
        written.append(p)

    written.append(_render_3d(d, out_dir))
    return [p for p in written if p is not None]


def _mesh_from_mask(mask, spacing_xyz, smooth_sigma=0.7, taubin=24):
    """Marching-cubes surface of a voxel mask, in millimetres, lightly smoothed."""
    import pyvista as pv
    from scipy import ndimage as ndi

    if not mask.any():
        return None
    sx, sy, sz = spacing_xyz
    vol = ndi.gaussian_filter(np.pad(mask.astype(np.float32), 1), smooth_sigma)
    # (z,y,x) array -> ImageData wants x fastest, so transpose then flatten F-order.
    grid = pv.ImageData(dimensions=vol.transpose(2, 1, 0).shape,
                        spacing=(sx, sy, sz))
    grid.point_data["m"] = vol.transpose(2, 1, 0).ravel(order="F")
    surf = grid.contour([0.5], scalars="m")
    if surf.n_points == 0:
        return None
    if taubin:
        surf = surf.smooth_taubin(n_iter=taubin, pass_band=0.1)
    return surf.compute_normals(auto_orient_normals=True, split_vertices=False)


def _render_3d(d, out_dir: Path):
    """Lit, depth-peeled VTK render. Four standard views, patient-anatomical.

    The camera directions are built from the case's own orientation flags and
    naris labels, so 'right lateral' means the patient's right on every case
    regardless of how the scanner stored the volume.
    """
    try:
        import pyvista as pv
    except Exception as exc:                              # pragma: no cover
        print(f"  3D skipped: {exc}")
        return None
    pv.OFF_SCREEN = True

    dom = _mesh_from_mask(d["passage"], d["spacing"])
    if dom is None:
        return None
    body_colour, sin_legend = _sinus_body_colours(d)
    sinus_meshes = []
    for ids, colour in body_colour:
        m = _mesh_from_mask(np.isin(d["drain"]["body_labels"], ids), d["spacing"])
        if m is not None:
            sinus_meshes.append((m, colour))

    ant = np.array([0.0, -1.0, 0.0]) if d["y_ant"] else np.array([0.0, 1.0, 0.0])
    sup = np.array([0.0, 0.0, 1.0]) if d["sup_hi"] else np.array([0.0, 0.0, -1.0])
    left = np.array([1.0, 0.0, 0.0]) if d["x_left_is_high"] else np.array([-1.0, 0.0, 0.0])
    views = [("right lateral", -left, sup), ("left lateral", left, sup),
             ("anterior", ant, sup), ("inferior", -sup, ant)]

    ctr = np.array(dom.center)
    span = float(max(dom.bounds[1] - dom.bounds[0], dom.bounds[3] - dom.bounds[2],
                     dom.bounds[5] - dom.bounds[4]))
    axis_tag = [(ant, "A"), (-ant, "P"), (sup, "S"), (-sup, "I"),
                (left, "L"), (-left, "R")]
    tiles = []
    for _name, cam_dir, up in views:
        p = pv.Plotter(off_screen=True, window_size=(700, 820))
        p.set_background("#111111")
        p.add_mesh(dom, color="#4fc3f7", opacity=0.42, smooth_shading=True,
                   specular=0.3, specular_power=18, ambient=0.25, diffuse=0.85)
        for mesh, colour in sinus_meshes:
            p.add_mesh(mesh, color=colour, opacity=1.0, smooth_shading=True,
                       specular=0.4, specular_power=25, ambient=0.3, diffuse=0.9)
        p.enable_depth_peeling(number_of_peels=12)
        p.camera.position = tuple(ctr + cam_dir * span * 2.4)
        p.camera.focal_point = tuple(ctr)
        p.camera.up = tuple(up)
        p.camera.parallel_projection = True
        p.camera.parallel_scale = span * 0.58
        p.reset_camera_clipping_range()
        tiles.append(p.screenshot(return_img=True))
        p.close()

    fig, axes = plt.subplots(1, len(tiles), figsize=(15.5, 5.6), facecolor="#111")
    for ax, img, (name, cam_dir, up) in zip(np.atleast_1d(axes).ravel(), tiles, views):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(name, color="#ddd", fontsize=9, pad=2)
        # Which anatomical direction is up / right ON SCREEN, derived from the
        # camera basis rather than assumed. Lateral-view handedness is exactly
        # the kind of thing that silently flips a figure left-right.
        view = -np.asarray(cam_dir, float)          # camera -> focal point
        right = np.cross(view, np.asarray(up, float))
        for vec, xy in ((right, (0.94, 0.5)), (-right, (0.06, 0.5)),
                        (np.asarray(up, float), (0.5, 0.965)),
                        (-np.asarray(up, float), (0.5, 0.035))):
            tag = max(axis_tag, key=lambda t: float(np.dot(t[0], vec)))[1]
            _corner(ax, tag, xy)
    fig.suptitle(
        f"{d['case']} — CFD flow domain (translucent blue) and the sinus air "
        "held out of it", color="#eee", fontsize=11, y=0.985)
    _legend(fig, [("Flow domain (nares → nasopharynx)", "#4fc3f7")] + sin_legend,
            y=0.02)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.08, wspace=0.02)
    p = out_dir / f"{d['case']}_seg_3d.png"
    fig.savefig(p, dpi=165, facecolor=fig.get_facecolor())
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--outputs-dir", type=Path, default=OUTPUTS)
    ap.add_argument("--out-dir", type=Path, default=OUTPUTS / "figures")
    a = ap.parse_args()
    for p in render(a.case, a.outputs_dir, a.out_dir):
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
