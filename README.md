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
- `Models/`: included trained MI/TDAH model artifacts and Optuna journals.
- `MI_TDAH_Dataset/`: optional local dataset folder ignored by Git.
- `Temp/`: optional debug output folder ignored by Git.

## Large Files

This repository uses Git LFS for binary scientific artifacts such as `.npz`, `.mat`, `.h5`, `.pkl`, and `.pdf` files.

The full local project also contains datasets and debug outputs that are intentionally excluded from Git:

- `Data/`
- `MI_TDAH_Dataset/`
- `Temp/`
- `Data/MI/` (~5.6 GB)

For publication, place those large artifacts in a data repository such as Zenodo, OSF, institutional storage, or Google Drive, then add the public DOI/link here.

The notebooks can also download the dataset in a cloned repo with:

```python
import kagglehub

path = kagglehub.dataset_download("daprosero/mi-tdah-dataset")
print("Path to dataset files:", path)
```

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
They resolve data in this order:

1. `MI_TDAH_Dataset/` in the repository root.
2. Legacy local `Data/` in the repository root.
3. Kaggle download `daprosero/mi-tdah-dataset` through `kagglehub`.

Set `DEBUG = True` in any notebook setup cell to write outputs under `Temp/` with the same folder layout, for example `Temp/Results/`, `Temp/Figures/`, and `Temp/Models/`. With `DEBUG = False`, outputs go to the normal repository folders.

Each notebook includes an initial bootstrap cell that:

1. Finds the repository root when running locally.
2. Clones `https://github.com/Daprosero/STF-KernelSHAP.git` when running from a loose notebook in Colab or Kaggle.
3. Installs `requirements.txt` automatically in Colab or Kaggle.
4. Adds `src/` to `sys.path`.
5. Starts with `DEBUG = True`, so generated outputs are written under `Temp/`.

In local runs, dependencies are expected to be installed in the active virtual environment unless you manually run the install cell.

Restore datasets locally under:

- `Data/TDAH/`
- `Data/MI/`

before running the corresponding notebooks or scripts.

## Citation

If this repository supports a paper, cite the paper and the external data archive DOI once available.
