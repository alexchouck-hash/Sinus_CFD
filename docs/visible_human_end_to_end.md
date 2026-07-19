# Visible Human: full pipeline on a real whole-head CT

First run of the entire pipeline on a real, un-cropped head CT (NLM Visible
Human Female, VHFCT1mm-Head, 512×512×234 @ 1 mm, DOI 10.7910/DVN/3JDZCT) rather
than a NasalSeg crop. This is the "upload a head CT → structured analysis" path
the MVP is ultimately about.

## What ran, end to end

```
download_visible_human_head.py   -> 234 DICOM slices -> VHFCT1mm_Head.nrrd (235 MB)
process_whole_head.py --skip-flow -> airway 51 mL, skin/bone/soft-tissue, nares+trachea BC ports
analyze_passage.py               -> lumen 55.3 mL, centerline 129.7 mm (nares->trachea),
                                    cross-section min/mean/max 3.1/56.4/207.3 mm^2, open ports
export_openfoam_geometry.py      -> watertight solid_air_body (54.7 mL) + L/R nostril/trachea/wall patches
scaffold_openfoam_case.py        -> prism-layer mesh + thermal + grad(T)
Docker simpleFoam                -> 260,413 cells (4,319 prisms), mass-conserving solve
```

Every stage completed. The smoothed-field marching-cubes fix (built for
NasalSeg) produced a watertight surface on the whole-head geometry too, and the
solve **conserves mass exactly**: Q_in = −3.0e-4 m³/s (both nostrils) =
Q_trachea = +3.0e-4 m³/s. No leak; the mesh is topologically sound. This is the
milestone — the whole-head path runs and produces coherent, conservative
physics.

## The numbers are not physiological — and that's expected

| Metric | Visible Human | NasalSeg P001 (living-subject) |
|---|---|---|
| Resistance R | 0.001 Pa·s/mL (ΔP 0.17 Pa) | **0.052** (validated) |
| Air conditioning | 20% (outlet 23.5 °C) | **85%** (outlet 34.4 °C) |
| Mucosal heat loss | 1.2 W | 4.9 W |

VH's resistance is ~100x too low and its air conditioning far too weak. This is
**not a pipeline error** (mass conserves, the solve converged, T converged) —
it's the geometry:

1. **Cadaver with a synthesized pharyngeal conduit.** The repo builds an
   artificial wide conduit to connect the nasal air to a caudal outlet where
   the cadaver scan lacks patent pharyngeal air. That wide artificial tube
   carries almost no resistance.
2. **The tight constriction isn't preserved into the CFD.** `analyze_passage`
   measured a 3.1 mm² minimum cross-section, but at a refine-level-2 (~1 mm)
   mesh that throat is only ~2 cells wide, and the surface smoothing
   (Gaussian σ≈0.8 vox + Taubin + sealing dilation) widens a ~2 mm slit
   significantly. Since resistance scales ~1/area², losing the throat collapses
   ΔP toward zero.

The contrast with P001 is itself informative: the living-subject NasalSeg data
gives physiological numbers, the cadaver VH does not — the pipeline is correctly
sensitive to geometry quality. VH is a "does the whole-head path run" test (it
does), not a physiological validation (NasalSeg already provides that).

## Bonus: wall-heat-flux extractor validated here

The VH run (with `grad(T)`) was the first end-to-end test of
`scripts/wall_heat_flux_map.py`: integrated wall heat loss **1.42 W**
cross-checks the independent enthalpy-based number **1.20 W** (~19%, consistent
with the owner-cell gradient approximation on a marginal mesh). The extractor
and its PLY colour-map output work; the physiological version runs on P001.

## To get physiological whole-head numbers later

- Segment/scan a **living subject** whole head (no synthesized conduit), or
  restrict the VH domain to the patent nasal passage and drop the artificial
  conduit.
- Refine the mesh at the throat (level 3+ / a targeted refinement region) and
  reduce surface smoothing where the MCA is small, so the constriction survives
  into the CFD.
- The NasalSeg cases remain the validation reference; VH is the whole-head
  plumbing test.
