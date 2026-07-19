# Stage 4: virtual surgery loop

Roadmap §5 (virtual surgery) and staged plan item 4: digitally edit the airway
to mimic an operation, re-run the pipeline, and compare pre/post. The literature
does exactly this for turbinate reduction and septoplasty and shows the
simulated change predicts the *direction* and rough *magnitude* of real
outcomes.

## What it does

`scripts/virtual_surgery.py` applies a parameterized edit to a case's label map,
then reports the pre/post change in the Stage-2 geometry metrics. The edited
label NRRD is written so the Stage-3 CFD pipeline (`process_case --label` →
export → scaffold → solve) produces the post-operative resistance and mucosal
cooling for a full pre/post CFD comparison.

```powershell
# geometry-only pre/post (works today, no CFD)
py -3.12 scripts/virtual_surgery.py --case P001 --procedure turbinate_reduction --side left --depth-mm 3

# let the tool pick the obstructed side from the Stage-2 L/R MCA ratio
py -3.12 scripts/virtual_surgery.py --case P065 --procedure septoplasty --side auto --depth-mm 3
```

## Procedures

| Procedure | Edit | Mimics |
|---|---|---|
| `turbinate_reduction` | grow cavity air **laterally** (away from septum midline, where turbinates protrude) | shaving an enlarged inferior/lateral turbinate |
| `septoplasty` | grow the obstructed side's air **medially** toward midline (clipped at midline) | moving a deviated septal wall back to midline |

Both grow the air lumen only into tissue/background voxels — never overwriting
the contralateral cavity or the nasopharynx. `--side auto` selects the narrower
side from the L/R MCA ratio.

## Validated behaviour (geometry, no CFD needed)

| Case (baseline) | Edit | Result |
|---|---|---|
| P001 (symmetric, L/R MCA ratio 0.96) | turbinate reduction, left, 3 mm | left MCA 19.9 → 40.4 mm² (+103%), left volume +22%, right unchanged. L/R ratio 0.96 → **0.47** — correctly flags that unilaterally widening a *symmetric* nose *creates* asymmetry (a real clinical caution). |
| P065 (obstructed, L/R MCA ratio 0.46) | septoplasty, `--side auto`, 3 mm | auto-picked the obstructed left side; left MCA 9.1 → 19.3 mm² (+113%), right unchanged. L/R ratio 0.46 → **0.99** — asymmetric to balanced, exactly what a successful septoplasty achieves. |

The two cases together show the loop does the right thing in both directions:
it improves an obstructed airway toward symmetry, and it warns when an edit would
*introduce* imbalance.

## Honest scope

- **Geometric mimics, not tissue-accurate surgery.** We grow the air region;
  we do not model the specific turbinate/septal tissue removed, because the
  NasalSeg labels give the air lumen (L/R cavity, nasopharynx) but not turbinate
  or septal cartilage. Directionally faithful, not a per-patient surgical plan.
- **Ranks candidates, does not prescribe.** As the roadmap states, "which tissue
  to remove" is a human decision informed by the sim. The tool quantifies the
  predicted metric change for a *specified* parameterized edit; it does not
  decide the operation.
- **Validation evidence is real but limited** (small cohorts; the nasal cycle
  confounds pre/post; correlations vary). Treat outputs as research, not
  clinical guidance.

## Full pre/post CFD (when the edited label feeds Stage 3)

```powershell
# post-op geometry + CFD from the edited label
py -3.12 scripts/process_case.py --case P065 --label outputs/P065_postop_septoplasty_left/P065_postop_label.nrrd
py -3.12 scripts/export_openfoam_geometry.py --case P065_postop ...
# scaffold + Docker solve, then:
py -3.12 scripts/compute_nasal_resistance.py --case P065_postop
py -3.12 scripts/compute_mucosal_cooling.py --case P065_postop
```

The pre/post resistance drop is the headline clinical number a surgeon would
weigh — the geometry MCA change above is its fast, CFD-free proxy (resistance
scales roughly with 1/MCA, confirmed nonlinear by the Stage-3 flow sweep).
