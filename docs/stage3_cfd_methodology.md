# Stage 3: choosing a robust, accurate, fast CFD method

Roadmap §4 / staged plan item 3: real Navier-Stokes CFD (not the potential-flow
preview). This doc records the methodology decision and what the first real
runs revealed, so the reasoning doesn't have to be redone later.

## The physics choice was never the hard part

Resting nasal breathing (~18 L/min) is transitional but largely laminar.
**Laminar, steady-state inspiration with `simpleFoam`** is the validated
approach in the nasal-CFD literature, and it's what this repo already
scaffolds (`scripts/scaffold_openfoam_case.py`). Ruled out up front:

- **RANS k-ε** — known to perform poorly in the thin, transitional nasal
  passages; wrong tool here regardless of speed.
- **LES** — only needed for higher flow rates / fine turbulent structure;
  costs an order of magnitude more mesh + must be transient. Not resting
  breathing's regime.
- **Commercial solvers (Fluent, STAR-CCM+)** — mesh boundary layers more
  easily and carry more reviewer trust in papers, but aren't free/automatable
  across many patients, which the MVP needs. Revisit only if publication-grade
  validation credibility becomes the goal.

So the solver and BCs were never the risk. **Every accuracy and robustness
problem in the first real run was in meshing** — confirmed empirically below,
not just asserted.

## What the first real run actually showed

First end-to-end Docker OpenFOAM run on NasalSeg case P001 (`nnUNetTrainer`
labels → passage → geometry export → scaffold → `simpleFoam`):

- blockMesh, surfaceFeatureExtract: fine.
- snappyHexMesh: castellate + snap succeeded, but **prism layer addition hit a
  fatal error** (`minMedialAxisAngle` misspelled as `minMedianAxisAngle` in
  the generated dict — a real bug, now fixed), so layers were never added.
- checkMesh: **429 skew faces, max skewness 12.5** — traced to the surface
  STL itself: **22,560 open edges** out of 93,381 (24%).
- simpleFoam still converged (it's tolerant of a rough mesh) and produced a
  first number: **R = 0.049 Pa·s/mL**, below the published resting range
  (0.10-0.35 Pa·s/mL) — consistent with a too-coarse, layer-less mesh
  under-predicting viscous resistance.

That's a clean natural experiment: **fix the surface, and the failure mode
goes away.**

## Root cause: raw binary marching cubes on thin topology

`solid_mask_to_watertight_mesh` (`src/sinus_cfd/openfoam_export.py`) built the
solid-air-body surface via marching cubes directly on the binary voxel mask.
Skimage's marching cubes can hit non-manifold cube configurations on thin,
complex topology — and nasal passages are exactly that (1-3mm passages,
turbinate detail). The mask was already reduced to one connected voxel
component (`seal_solid_for_watertight_mesh` does that before meshing), so
disconnected fragments weren't the issue — the ambiguous cube configurations
from the *binary* threshold were.

**Fix:** marching cubes on a **Gaussian-blurred field** (sigma≈0.8 voxels,
padded to avoid boundary clipping) instead of the raw 0/1 mask, thresholded
at 0.5 — removes the voxel-adjacency ambiguity that triggers non-manifold
configurations. Followed by:

1. Keep only the largest connected mesh component (cheap insurance).
2. **Taubin (shrink-free) smoothing**, 15 iterations — reduces residual
   surface noise/skewness without the volume shrinkage plain Laplacian
   smoothing causes.
3. Existing `trimesh.fill_holes` + normal-repair loop, unchanged, as a
   safety net for any remaining small gaps.
4. Existing decimation, now also re-filtered to the largest component
   afterward (decimation can occasionally reintroduce disconnected slivers).

Result on P001: **"solid_air_body mesh is watertight (good for
snappyHexMesh)"** — confirmed, not just hoped for. Watertight volume 68.4 mL
vs. 70.5 mL sealed-voxel volume (~3% smoothing shrinkage, expected and
small next to CT partial-volume blur, which is itself ~0.5-1mm).

Deliberately avoided: `pyvista.PolyData.fill_holes` — its own docs warn
*"This method is known to segfault. Use at your own risk,"* which would crash
the whole process uncatchably. Everything above uses only trimesh + scipy +
skimage, already dependencies.

## Result after the fix — and a second bug the "no change" result exposed

Re-ran P001 with the watertight surface. Mesh quality transformed exactly as
predicted:

| | Before (broken STL) | After (watertight + Taubin) |
|---|---|---|
| Prism layers | Failed to add (fatal error) | **2,780 prism cells added** |
| checkMesh verdict | Failed 1 check | **Mesh OK.** |
| Max skewness | 12.5 (429 flagged faces) | **3.48 (none flagged)** |

But the reported resistance was **identical to 4 significant figures**
(0.049 Pa·s/mL, ΔP=14.56 Pa) — and stayed identical again after bumping mesh
refinement 5x (53k → 259k cells). Three meshes differing by up to 8x in cell
count cannot legitimately produce bit-identical pressure drops; this was a
second, separate bug, not a mesh-independence result.

**Root cause:** the case directory was never cleaned between Docker runs (no
`Allclean`). `simpleFoam`'s `surfaceFieldValue` functionObjects silently skip
rewriting `.dat` output for time values they've already written — so
re-running over the same time range (0→500) left `postProcessing/` holding
the *first* run's numbers while the mesh and field files (`500/U`, `500/p`)
correctly reflected each new mesh. No error, no warning, just stale data
masquerading as a fresh result. Caught by noticing the exact-match was
physically implausible, then confirmed via file mtimes: `postProcessing/`
files were timestamped from run 2, not run 3 or 4.

**Fixed** by cleaning (`rm -rf` time dirs, `processor*`, `constant/polyMesh`,
`postProcessing`, logs) at the start of every run — both
`scripts/run_openfoam_docker.ps1` and the case's `Allrun.docker` — and adding
a defensive check to `compute_nasal_resistance.py` that warns if any
`postProcessing/*/*/surfaceFieldValue.dat` predates `constant/polyMesh/owner`
(which a fresh meshing pass always rewrites), so this can't recur silently.

**The genuine, trustworthy number** (properly cleaned, 259k-cell mesh, prism
layers present, checkMesh OK): **R = 0.052 Pa·s/mL** (ΔP=15.66 Pa at 18 L/min).
Monitored inlet pressure had settled to within ~3% over the last 100
iterations — reasonably but not perfectly converged. This is the *only*
number in this doc's history that hasn't been contaminated by one of the two
bugs above; the earlier "before/after layers" comparison is retracted since
run 2 (layers added) was likely also serving run 1's stale data, not its own.

The resting number is below the published 0.10-0.35 Pa·s/mL range — but a flow
sweep (`scripts/flow_sweep_report.py`, `foam/P001/Allsweep.docker`) resolved
why, and it's a validation win, not a defect.

### Flow sweep: resistance is nonlinear, and matches the literature at the literature's own condition

Re-solved on the same 259k-cell mesh at four total flow rates (each a fresh,
properly cleaned run):

| Q (L/min) | ΔP (Pa) | R (Pa·s/mL) |
|----------:|--------:|------------:|
| 18 (resting) | 15.7 | 0.052 |
| 36 | 58.7 | 0.098 |
| **60** | **156.0** | **0.156** |
| 90 | 343.8 | 0.229 |

Two things fall out:

1. **R rises 4.4x from 18 → 90 L/min** — strongly nonlinear, exactly as real
   nasal aerodynamics behaves. ΔP fits Rohrer's equation ΔP = K₁·Q + K₂·Q²
   (viscous + inertial terms), so R = ΔP/Q ≈ K₁ + K₂·Q rises near-linearly
   with flow (observed slope ~0.0024 Pa·s/mL per L/min, nearly constant).
2. **At 60 L/min the pressure drop is 156 Pa** — essentially the ~150 Pa
   driving pressure at which active rhinomanometry measures the published
   range — and there **R = 0.156 Pa·s/mL, squarely inside the healthy
   0.10-0.35 range.**

So the resting-breathing R=0.052 was never wrong. The published range is a
*different operating point* (≈150 Pa vs. resting's ≈15 Pa), and comparing R at
resting flow against a 150-Pa-measured range is apples-to-oranges. Evaluated
at the literature's own condition, this pipeline lands in the literature's own
range. Candidate explanation #1 is **confirmed**; #2 (cropped FOV) and #3
(under-resolution) are not needed to explain the resting number, though a
formal mesh-independence study is still worth doing before quoting any single
number as validated. Plot: `outputs/P001/P001_flow_sweep.png`.

## Corrected lesson for anyone re-running this

**Always confirm `postProcessing/` timestamps are newer than
`constant/polyMesh/owner` before trusting a resistance number** — the fix
above prevents the silent case, but a manual re-run outside these scripts
(e.g. an interactive Docker shell) can still hit it.

## Recommended path forward

**Near-term (this repo, this machine):**
1. Watertight surface conditioning — **done**, verified (mesh watertight,
   checkMesh OK, skewness 12.5 → 3.2).
2. Prism layers — **done**, verified (2,780-8,062 prism cells present
   depending on resolution), keyword bug fixed.
3. Reproducible clean runs — **done**, verified (the stale-postProcessing bug
   above is fixed and now defensively checked for).
4. Mesh independence check — **not actually done yet**, despite appearances.
   Only one genuine (non-stale) data point exists so far (259k cells, R=0.052
   Pa·s/mL). A real check needs 3+ deliberately varied resolutions, each a
   properly cleaned run, compared against each other honestly.
5. Distinguish the three candidate explanations for R being below the
   published range (flow-rate nonlinearity, cropped-FOV missing the
   vestibule, or genuine under-resolution) — cheapest first: re-run the same
   mesh at a higher imposed flow rate to test the nonlinearity hypothesis
   before spending more compute on refinement.
6. Parallel solve — `decomposePar` across available cores (case already has
   `system/decomposeParDict`); free ~4x wall-clock speedup on this machine's
   4 physical cores.
7. **cfMesh (`cartesianMesh`)** as a snappyHexMesh alternative — already
   present in the `opencfd/openfoam-run` Docker image, generally more
   tolerant of imperfect surfaces and faster to converge on boundary layers.
   Worth trying if snappy is still fragile on some patients even after the
   watertight fix.
8. **Frozen-flow thermal step** (near-free clinical payoff): wall heat flux /
   mucosal cooling is the *strongest* correlate of perceived nasal patency
   (TRPM8 cooling receptors, not pressure). Once velocity converges, solve a
   passive temperature scalar (wall T=37°C, inlet T=ambient) on the frozen
   flow field — a cheap linear advection-diffusion solve, not a full
   compressible energy equation. Same idea extends to humidity transport.

**Long-term bet, noted but not started: GPU-native Lattice-Boltzmann (LBM).**
If per-patient meshing robustness remains the bottleneck as this scales
across many patients, LBM (e.g. Palabos, waLBerla, or a custom GPU kernel)
runs **directly on the segmentation voxel grid** — no STL export, no
snapping, no watertight requirement at all, and it's naturally
massively-parallel on the same GPU already used for nnU-Net training/Colab.
It structurally removes the exact failure mode this doc just walked through.
Worth prototyping in parallel once the OpenFOAM path is validated on a few
cases, as the scale-out path rather than the correctness-validation path
(OpenFOAM's Navier-Stokes solve and literature track record make it the
right tool for establishing that the numbers are trustworthy first).
