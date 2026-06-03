# STF-KernelSHAP

Repository with code, notebooks, figures, selected data, trained artifacts, and result files for experiments using STF-KernelSHAP on EEG classification tasks.

## Contents

- `Model_utils/`: model definitions, training/evaluation utilities, and STF-KernelSHAP implementation.
- `Notebooks/`: experiment, training, result, and visualization notebooks.
- `Figures/`: figures generated for the paper.
- `Results/`: classification, attribution, and faithfulness result files.
- `Data/TDAH/`: included TDAH dataset files used by the experiments.
- `Models/TDAH/`: included trained TDAH model artifacts and Optuna journals.

## Large Files

This repository uses Git LFS for binary scientific artifacts such as `.npz`, `.mat`, `.h5`, `.pkl`, and `.pdf` files.

The full local project also contains large Motor Imagery artifacts that are intentionally excluded from Git because they add approximately 6.9 GB:

- `Data/MI/` (~5.6 GB)
- `Models/MI/` (~1.3 GB)

For publication, place those large artifacts in a data repository such as Zenodo, OSF, institutional storage, or Google Drive, then add the public DOI/link here.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git lfs install
```

## Notes for Reproducibility

The notebooks assume the directory structure shown above. If the excluded Motor Imagery artifacts are needed, restore them under:

- `Data/MI/`
- `Models/MI/`

before running the corresponding notebooks or scripts.

## Citation

If this repository supports a paper, cite the paper and the external data archive DOI once available.
