# Neural net training for nasal air (NasalSeg → nnU-Net)

Goal: **sharper nares + L/R cavities** that generalize across patients, then
feed masks into the existing classical post-process (openings, dual centerlines,
septum, OpenFOAM).

## Data (already downloadable)

| Dataset | Size | Labels |
|---------|------|--------|
| **[NasalSeg](https://zenodo.org/records/13893419)** | 130 CTs, ~224 MB zip | L/R nasal cavity, nasopharynx, L/R maxillary sinus |

Cite: Zhang et al., *Scientific Data* 2024 — NasalSeg.

```powershell
cd C:\Users\houck\Documents\Sinus_CFD
py -3.12 scripts\download_nasalseg.py
py -3.12 scripts\prepare_nnunet_nasalseg.py
```

### Layout after prepare

```text
data/
  NasalSeg/                    # original NRRD (gitignored)
    images/P00x_img.nrrd
    labels/P00x_seg.nrrd
  nnUNet_raw/                  # nnU-Net v2 raw (gitignored)
    Dataset501_NasalSeg/
      dataset.json
      imagesTr/P001_0000.nii.gz
      labelsTr/P001.nii.gz
      splits_sinus_cfd.json
      set_nnunet_env.ps1
  nnUNet_preprocessed/         # created by nnU-Net
  nnUNet_results/              # trained weights
models/                        # optional copy of best checkpoint for inference
```

### Label IDs (`--remap full`, default)

| ID | Structure |
|----|-----------|
| 0 | background |
| 1 | left_nasal_cavity |
| 2 | right_nasal_cavity |
| 3 | nasopharynx |
| 4 | left_maxillary_sinus |
| 5 | right_maxillary_sinus |

Optional CFD-oriented merge:

```powershell
py -3.12 scripts\prepare_nnunet_nasalseg.py --remap airway --dataset-id 502 --dataset-name NasalSegAirway
```

| ID | Structure |
|----|-----------|
| 0 | background |
| 1 | nasal_airway (1+2+3) |
| 2 | maxillary_sinus (4+5) |

## Install nnU-Net (training machine)

GPU strongly recommended (CUDA). CPU works for smoke tests only.

```powershell
py -3.12 -m pip install -r requirements-nn.txt
```

`requirements-nn.txt` pins `nnunetv2` and torch. Install a CUDA build of PyTorch
from https://pytorch.org if the default CPU wheel is too slow.

## Train (5-fold style)

```powershell
# Paths
. .\data\nnUNet_raw\Dataset501_NasalSeg\set_nnunet_env.ps1

# Integrity + plans (once)
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity

# Train fold 0 of 3d_fullres (repeat 0..4 for full CV)
nnUNetv2_train 501 3d_fullres 0
```

Inference on a new CT (NIfTI or convert NRRD first):

```powershell
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 501 -c 3d_fullres -f 0
```

## How this connects to Sinus_CFD

```text
nnU-Net labels (L/R cavity + NP)
        ↓
export binary air / L / R masks
        ↓
existing refine_nasal_ct / passage / dual centerlines / OpenFOAM
```

Next engineering step (after first successful train):  
`scripts/infer_nnunet_nasal.py` → write `*_cavity_left.nrrd` etc. for the viewer.

## Hardware expectations

| Setup | Realistic use |
|-------|----------------|
| NVIDIA GPU ≥8 GB | Full 3d_fullres training |
| CPU only | `prepare` + tiny `--max-cases 2` smoke tests |
| Laptop GPU 4–6 GB | May need 3d_lowres or patch config tweaks |

## License / ethics

- NasalSeg: follow Zenodo + paper terms (research citation required).  
- Do **not** commit `data/NasalSeg`, `nnUNet_*`, or weights to git.  
- Clinical / commercial use: review licenses and add de-identified hospital data only under IRB.

## Citation

```bibtex
@article{zhang2024nasalseg,
  title={NasalSeg: A Dataset for Automatic Segmentation of Nasal Cavity and Paranasal Sinuses from 3D CT Images},
  author={Zhang, Yichi and others},
  journal={Scientific Data},
  year={2024}
}
```
