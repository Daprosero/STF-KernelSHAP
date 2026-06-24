# STF-KernelSHAP

Repository with code, notebooks, figures, trained artifacts, and result files for experiments using STF-KernelSHAP on EEG classification tasks.

## Contents

- `src/stf_kernelshap/`: model definitions, training/evaluation utilities, STF-KernelSHAP implementation, reporting, and visualization helpers.
- `Notebooks/Train_models.ipynb`: Optuna training runner.
- `Notebooks/TDAH_MI_Results.ipynb`: best-config training and CSV evaluation runner.
- `Notebooks/run_xai_models_2.ipynb`: XAI attribution runner.
- `Notebooks/Classification_results.ipynb`: classification summary figure runner.
- `Notebooks/Faithfulness_summary.ipynb`: faithfulness ranking/table runner.
- `Notebooks/Topoplots.ipynb`: topoplot, selected faithfulness, and spectra figure runner.
- `Figures/`: figures generated for the paper.
- `Results/`: classification, attribution, and faithfulness result files.
- `Models/TDAH/`: included trained TDAH model artifacts and Optuna journals.

## Large Files

This repository uses Git LFS for binary scientific artifacts such as `.npz`, `.mat`, `.h5`, `.pkl`, and `.pdf` files.

The full local project also contains datasets and large Motor Imagery artifacts that are intentionally excluded from Git:

- `Data/`
- `Data/MI/` (~5.6 GB)
- `Models/MI/` (~1.3 GB)

For publication, place those large artifacts in a data repository such as Zenodo, OSF, institutional storage, or Google Drive, then add the public DOI/link here.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
git lfs install
```

## Notes for Reproducibility

The notebooks are thin runners that import reusable code from `src/stf_kernelshap`.
Restore datasets and excluded Motor Imagery artifacts locally under:

- `Data/TDAH/`
- `Data/MI/`
- `Models/MI/`

before running the corresponding notebooks or scripts.

## Citation

If this repository supports a paper, cite the paper and the external data archive DOI once available.
