# Trained models (gitignored weights)

Put exported nnU-Net checkpoints or ONNX/TorchScript bundles here for local inference.

Suggested:

```text
models/
  nnunet_nasalseg_501/
    fold_0/
    dataset.json
    plans.json
```

Do not commit large weight files. Document the Zenodo/GitHub URL and training command in `docs/nnunet_nasal.md`.
