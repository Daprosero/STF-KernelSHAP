# STF-KernelSHAP

Reproducible code, notebooks, trained artifacts, figures, and result files for the STF-KernelSHAP EEG explainability experiments on motor imagery (MI) and ADHD/TDAH classification tasks.

The repository is organized so that notebooks are thin runners and the implementation lives in `src/stf_kernelshap/`.

## Repository Contents

- `src/stf_kernelshap/`: reusable package with data loading, model builders, training/evaluation pipelines, XAI methods, STF-KernelSHAP, reporting, and visualization utilities.
- `Notebooks/Train_models.ipynb`: Optuna training runner for MI and ADHD/TDAH models.
- `Notebooks/TDAH_MI_Results.ipynb`: best-configuration retraining and classification CSV generation.
- `Notebooks/run_xai_models_2.ipynb`: explanation generation.
- `Notebooks/Computational_cost_scalability.ipynb`: T-GARNet computational-cost timing logs and figure.
- `Notebooks/Classification_results.ipynb`: classification summary figures.
- `Notebooks/Faithfulness_summary.ipynb`: faithfulness AUC summaries and rankings.
- `Notebooks/Topoplots.ipynb`: topographic, selected relevance, and spectra figures.
- `Models/`: trained model weights and Optuna journals included through Git LFS.
- `Results/`: classification, attribution, faithfulness, and timing result files.
- `Figures/`: generated paper figures.
- `MI_TDAH_Dataset/`: optional local dataset folder ignored by Git.
- `Temp/`: optional debug output folder ignored by Git.

## Large Files and Data

This repository uses Git LFS for binary scientific artifacts such as `.npz`, `.mat`, `.h5`, `.pkl`, and `.pdf` files.

The full local project also contains datasets and debug outputs intentionally excluded from Git:

- `Data/`
- `MI_TDAH_Dataset/`
- `Temp/`
- `Data/MI/` (about 5.6 GB)

The notebooks resolve the dataset automatically in this order:

1. `MI_TDAH_Dataset/` in the repository root.
2. Legacy local `Data/` in the repository root.
3. Kaggle dataset `daprosero/mi-tdah-dataset` through `kagglehub`.

Manual dataset download:

```python
import kagglehub

path = kagglehub.dataset_download("daprosero/mi-tdah-dataset")
print("Path to dataset files:", path)
```

Expected local layouts:

- `MI_TDAH_Dataset/Data/MI/`
- `MI_TDAH_Dataset/Data/TDAH/`
- or legacy `Data/MI/` and `Data/TDAH/`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
git lfs install
```

All notebooks start with a bootstrap cell that finds the repository root, optionally clones the repository in Colab/Kaggle, installs `requirements.txt` in hosted runtimes, adds `src/` to `sys.path`, and calls `setup_notebook_environment`.

Set `DEBUG = True` to write outputs under `Temp/Results`, `Temp/Figures`, and `Temp/Models`. Set `DEBUG = False` to write to the public repository folders.

## Recommended Reproduction Order

1. `Notebooks/Train_models.ipynb`: run Optuna studies or reproduce training artifacts.
2. `Notebooks/TDAH_MI_Results.ipynb`: retrain/evaluate with best Optuna parameters and generate `Results/MI_results.csv` and `Results/TDAH_results.csv`.
3. `Notebooks/run_xai_models_2.ipynb`: generate XAI attribution files.
4. `Notebooks/Computational_cost_scalability.ipynb`: generate `Results/xai_timing_logs.csv` and `Figures/Computational_cost_scalability.pdf`.
5. `Notebooks/Faithfulness_summary.ipynb`: compute MoRF/ROAD faithfulness summaries and rankings.
6. `Notebooks/Classification_results.ipynb`: generate classification-result figures.
7. `Notebooks/Topoplots.ipynb`: generate topographic and STF relevance figures.

## Notebook Guide

`Train_models.ipynb`

- Uses `stf_kernelshap.experiments.training.run_mi_optuna_for_worker` and `run_tdah_optuna`.
- Writes Optuna journals under `Models/.../Optuna` and best fold weights under `Models/.../Models`.
- Main knobs: `model_name`, `window_name`, `worker_id`, `n_trials`, `epochs`, `batch_size`, and output model directory.

`TDAH_MI_Results.ipynb`

- Loads the best Optuna parameters from the journals.
- Repeats/folds the final model evaluation with controlled seeds.
- Writes `Results/MI_results.csv` and `Results/TDAH_results.csv`.
- Main knobs: selected MI windows/subjects, selected models, `seed`, `seed_gap`, number of repeats, and output CSV paths.

`run_xai_models_2.ipynb`

- Builds MI and ADHD/TDAH test cases, loads trained weights, and runs XAI methods.
- Saves attribution files under `Results/attributions_y_test/...` or `Results/attributions_y_pred/...`.
- Main knobs: `mi_subjects_to_extract`, `folds_to_extract`, `xai_methods`, `models_to_run`, `use_y_test`, and output paths.

`Computational_cost_scalability.ipynb`

- Reuses the data/model-loading pattern from `run_xai_models_2.ipynb`, but is dedicated only to timing.
- Runs T-GARNet as the heaviest evaluated classifier.
- Uses one representative subject per MI window: `2.5-5` and `0-7`.
- Uses a comparable ADHD/TDAH fold.
- Runs all evaluated XAI methods over a percentage-based stratified sample from each case's `y_test`: `timing_sample_fraction = 0.05`, `timing_min_trials = 20`, `timing_max_trials = 100`, and `timing_random_state = 42`.
- Writes `Results/xai_timing_logs.csv` and saves `Figures/Computational_cost_scalability.pdf`.
- Main knobs: `timing_mi_subjects`, `timing_tdah_folds`, `timing_sample_fraction`, `timing_min_trials`, `timing_max_trials`, `timing_random_state`, `timing_xai_methods`, `timing_model`, `timing_hardware`, and `save_attributions=False`.

`Faithfulness_summary.ipynb`

- Reads faithfulness result CSVs and computes MoRF/ROAD AUC summaries.
- Writes `Results/faithfulness_selected_relevance/faithfulness_auc_summary_for_paper.csv` and related per-sample/plot-ready files.
- Main knobs: selected windows, methods, models, faithfulness metric, and early AUC percentage cutoff.

`Classification_results.ipynb`

- Reads `Results/MI_results.csv` and `Results/TDAH_results.csv`.
- Generates paper-ready MI and ADHD/TDAH classification plots.
- Main knobs: input CSV paths, save directory, metric order, model order, and selected MI window.

`Topoplots.ipynb`

- Loads attribution `.npz` files and renders method-level scalp maps and STF cell maps.
- Main knobs: MI subjects/folds, ADHD/TDAH fold, model names, XAI methods, frequency bands, time labels, relevance mode, and aggregation mode.

## Random Seeds

The default seed is `42` unless explicitly changed.

- `src/stf_kernelshap/utils/common.py::set_seed(seed=42)` sets `PYTHONHASHSEED`, Python `random`, NumPy, and TensorFlow seeds, with deterministic TensorFlow ops when available.
- Training folds use `seed + fold` or the evaluation notebook's `seed + repeat * 10000 + fold * seed_gap`.
- Optuna uses `GPSampler(seed=42)`.
- XAI background sampling uses `random_state=42` by default.
- STF-KernelSHAP sets `np.random.seed(random_state + sample_position)` per explained sample when `random_state` is provided.

## Main Training Parameters

`run_mi_optuna_for_worker(model_name, window_name, worker_id, data_dir, output_models_dir, n_trials=20, epochs=100, batch_size=16)`

- `model_name`: one of `eegnet`, `shallowconvnet`, or `tgarnet`.
- `window_name`: MI time window, `2.5-5` or `0-7`.
- `worker_id`: 1-indexed subject partition used to distribute MI subjects across workers.
- `data_dir`: dataset root containing `MI/<window>/subject_<id>.npz`.
- `output_models_dir`: output root for journals and weights.
- `n_trials`: Optuna trials per subject/model/window.
- `epochs`: maximum epochs per fold.
- `batch_size`: training batch size.

`run_tdah_optuna(model_name, folds_path, path_adhd, path_control, output_models_dir, n_trials=20, epochs=100, batch_size=16)`

- `model_name`: one of `eegnet`, `shallowconvnet`, or `tgarnet`.
- `folds_path`: pickle file with subject folds.
- `path_adhd`, `path_control`: ADHD and control `.mat` directories.
- `output_models_dir`: output root for journals and weights.
- `n_trials`, `epochs`, `batch_size`: same meaning as MI.

## Main XAI Parameters

The central runner is `compute_xai_all_xtest(...)`, usually called through `run_mi_xai_and_save(...)` or `run_tdah_xai_and_save(...)`.

Shared runner parameters:

- `X_test`: test samples to explain.
- `X_train`, `y_train`: training data used only for stratified backgrounds/reference baselines where required.
- `y_pred`: target labels for explanation; in notebooks this is either model predictions or `y_test`.
- `method`: `KernelSHAP`, `LIME`, `Occlusion`, `IntegratedGradients`, `GradCAM++`, or `STF-KernelSHAP`.
- `sample_indices`: subset of trials to explain; use this for feasible timing experiments.
- `label_source`: `y_pred` or `y_test`, recorded in attribution metadata/logs.
- `output_layer_name`: model output used for logits/probability extraction, default `out_activation`.

Adaptive XAI defaults from `get_adaptive_xai_params(...)`:

- `KernelSHAP`: stratified `background_size = min(100, max(8, 0.05 * n_samples))`, `nsamples=500`, `l1_reg="num_features(200)"`.
- `LIME`: stratified `background_size = min(200, max(30, 0.10 * n_samples))`, `num_features=200`, `num_samples=1000`.
- `Occlusion`: `window_seconds=1.0`, `stride_seconds=0.25`, `baseline_value=None`, `occ_batch_size=256`, background mean reference.
- `IntegratedGradients`: `baseline=None`, stratified mean background reference, `steps=50`, `batch_size=1`.
- `GradCAM++`: last compatible Conv2D layer by default; for ShallowConvNet the notebook uses `layer_name="Conv2D_1"`.
- `STF-KernelSHAP`: `nfft=512`, `nsamples=500`, `l1_reg="num_features(200)"`, `baseline_tf=None`, `silent=True`, finite structured coalitions over time-frequency cells.

STF-KernelSHAP structured features:

- MI windows use frequency bands theta `(4, 8)`, alpha `(8, 13)`, beta `(13, 30)`, and gamma `(30, 40)` Hz.
- ADHD/TDAH uses delta `(0.5, 4)`, theta `(4, 8)`, alpha `(8, 13)`, beta `(13, 30)`, and gamma `(30, 40)` Hz.
- MI `0-7` is segmented into `(0, 2)`, `(2, 2.5)`, `(2.5, 5)`, and `(5, 7)` seconds.
- Other windows use a single full-window segment unless `time_segments_sec` is supplied.
- The structured feature count is `M_breve = C_breve * Q`, where `C_breve` is the number of channel groups/cells and `Q` is the number of time-frequency blocks.

## Minimal MI Example

```python
from stf_kernelshap.visualization.topoplots import build_mi_data_from_npz
from stf_kernelshap.xai.runner import run_mi_xai_and_save

mi_subjects = {"2.5-5": {43: 1}}
mi_data = build_mi_data_from_npz(
    base_path_mi="MI_TDAH_Dataset/Data/MI",
    mi_subjects_to_extract=mi_subjects,
)

run_mi_xai_and_save(
    mi_data=mi_data,
    mi_subjects_to_extract=mi_subjects,
    results_dir="Results",
    models_mi_root="Models/MI",
    models_to_run=("shallowconvnet",),
    xai_methods_to_run=("STF-KernelSHAP",),
    use_y_test=True,
    sample_indices=[0, 1, 2],
    hardware="NVIDIA T4 GPU",
)
```

## Minimal ADHD/TDAH Example

```python
from stf_kernelshap.visualization.topoplots import build_tdah_data_from_segmented
from stf_kernelshap.xai.runner import run_tdah_xai_and_save

tdah_data = {
    4: build_tdah_data_from_segmented(
        fold_to_extract=4,
        folds_path="MI_TDAH_Dataset/Data/TDAH/folds.pkl",
        path_adhd="MI_TDAH_Dataset/Data/TDAH/ieee/ADHD_group",
        path_control="MI_TDAH_Dataset/Data/TDAH/ieee/Control_group",
    )
}

run_tdah_xai_and_save(
    tdah_data_by_fold=tdah_data,
    folds_to_extract=[4],
    results_dir="Results",
    models_tdah_root="Models/TDAH",
    models_to_run=("tgarnet",),
    xai_methods_to_run=("STF-KernelSHAP",),
    use_y_test=True,
    sample_indices=[0, 1, 2],
    hardware="NVIDIA T4 GPU",
)
```

## Computational Cost Figure

`Notebooks/Computational_cost_scalability.ipynb` generates:

- `Results/xai_timing_logs.csv`: one row per explained sampled trial, method, model, paradigm, fold/subject, and runtime.
- `Figures/Computational_cost_scalability.pdf`: 1 x 2 figure with MI and ADHD/TDAH panels for T-GARNet, reporting mean explanation runtime per trial and standard-deviation error bars across the timing logs. The MI panel keeps the `2.5-5` and `0-7` windows separated.

The timing notebook uses `save_attributions=False` so the computational-cost sample does not overwrite full attribution files used by the other analyses.

## Code Availability Statement

This public repository provides the implementation, reproducibility notebooks, configuration parameters, fixed random seeds, trained model artifacts/Optuna journals, scripts for model training and explanation generation, XAI parameter settings, minimal MI and ADHD/TDAH execution examples, timing-log generation, and figure-generation utilities required to reproduce the reported analyses.

## Citation

If this repository supports a paper, cite the paper and the external data archive DOI once available.
