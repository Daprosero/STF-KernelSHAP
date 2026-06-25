# Reviewer Response From Implementation

This document maps each requested revision to the implementation added to the public repository.

## Comment 1: Computational Cost of STF-KernelSHAP

Reviewer concern:

> The manuscript does not provide sufficient information regarding the computational cost of STF-KernelSHAP. A brief discussion of the computational complexity, runtime, and scalability would help readers better understand the practicality of the proposed framework.

Implemented response:

- Added per-trial runtime logging to all evaluated XAI methods: `KernelSHAP`, `LIME`, `Occlusion`, `IntegratedGradients`, `GradCAM++`, and `STF-KernelSHAP`.
- Added timing metadata fields: paradigm, window, subject, fold, model, method, label source, explained sample index, target class, runtime in seconds, hardware, score type, number of timed samples, partition metadata, and method parameters.
- Added a new dedicated notebook, `Notebooks/Computational_cost_scalability.ipynb`, based on the data/model-loading pattern of `Notebooks/run_xai_models_2.ipynb` but separated from the attribution-generation notebook.
- The timing notebook uses T-GARNet as the heaviest evaluated classifier.
- The timing design uses one representative subject per MI window (`2.5-5` and `0-7`) and a comparable ADHD/TDAH fold.
- The timing section uses a representative viable subset selected from each case's `y_test` with percentage-based stratified sampling (`timing_sample_fraction = 0.05`, `timing_min_trials = 20`, `timing_max_trials = 100`, `timing_random_state = 42`) across all evaluated XAI methods.
- The timing run uses `hardware = "NVIDIA T4 GPU"` and `save_attributions=False`, so timing experiments do not overwrite the full attribution files used by topoplots and faithfulness analysis.
- Added the figure generator `src/stf_kernelshap/reporting/computational_cost.py`.
- The generated figure is `Figures/Computational_cost_scalability.pdf`, with one row and two columns: MI and ADHD/TDAH. The MI panel separates the `2.5-5` and `0-7` windows, and both panels report mean explanation runtime per trial with standard-deviation error bars computed from `Results/xai_timing_logs.csv`.

Implementation evidence:

- `src/stf_kernelshap/xai/runner.py`
  - `write_xai_timing_log(...)`
  - `runtime_seconds` recorded for KernelSHAP, LIME, Occlusion, Integrated Gradients, and Grad-CAM++.
  - `run_mi_xai_and_save(..., sample_indices=None, hardware="NVIDIA T4 GPU", save_attributions=True)`
  - `run_tdah_xai_and_save(..., sample_indices=None, hardware="NVIDIA T4 GPU", save_attributions=True)`
- `src/stf_kernelshap/xai/kernelshap.py`
  - `runtime_seconds` recorded for `stf_kernelshap_all_xtest(...)`.
- `src/stf_kernelshap/reporting/computational_cost.py`
  - `load_timing_logs(...)`
  - `summarize_timing_logs(...)`
  - `plot_computational_cost_figure(...)`
  - `make_computational_cost_figure(...)`
- `Notebooks/Computational_cost_scalability.ipynb`
  - New section: `Computational cost and scalability`.
  - Writes `Results/xai_timing_logs.csv`.
  - Saves `Figures/Computational_cost_scalability.pdf`.

Suggested manuscript paragraph:

> Computational cost and scalability. Exact Shapley-value estimation is combinatorial, requiring evaluation over `2^{M-1}` coalitions for each feature when `M` input features are treated independently. STF-KernelSHAP avoids this exhaustive enumeration by using a finite-sampling KernelSHAP approximation over structured time-frequency features. In practice, the runtime is dominated by the number of sampled coalitions `N_s`, the corresponding classifier evaluations, the segmented time-frequency transform and inverse reconstruction required for each coalition, and the weighted least-squares fit over `M_breve = C_breve Q` structured features, where `C_breve` denotes the structured channel/cell dimension and `Q` the number of time-frequency blocks. To quantify the practical cost of the final implementation, we generated timing logs on the reported NVIDIA T4 GPU setup using T-GARNet, the heaviest evaluated classifier, with one representative subject per MI window and a comparable ADHD/TDAH fold. Figure X reports the mean explanation runtime per trial with standard-deviation error bars computed from these logs for each evaluated XAI method.

Important audit note:

- Runtime values should be inserted only after executing `Notebooks/Computational_cost_scalability.ipynb` on the final code and reported T4 GPU setup.
- The authoritative source for runtime numbers is `Results/xai_timing_logs.csv`.

## Comment 2: Public Code Reproducibility and Implementation Details

Reviewer concern:

> The manuscript states that the source code will be released. To facilitate reproducibility, additional implementation details, such as the main parameter settings and the explanation generation procedure, should also be provided with the public code repository.

Implemented response:

- Rewrote `README.md` as a reproducibility guide for the public repository.
- Added a recommended notebook execution order.
- Added a notebook-by-notebook usage guide.
- Added dataset layout and setup instructions.
- Documented random seeds and deterministic controls used by training, Optuna, evaluation, and XAI.
- Documented main training parameters for MI and ADHD/TDAH.
- Documented the explanation generation procedure and main parameters for each XAI method.
- Added minimal MI and ADHD/TDAH execution examples.
- Added a code-availability statement explicitly mentioning implementation, notebooks, configuration parameters, fixed random seeds, trained artifacts/Optuna journals, scripts for model training and explanation generation, XAI parameter settings, minimal execution examples, timing logs, and figure-generation utilities.

Implementation evidence:

- `README.md`
  - `Recommended Reproduction Order`
  - `Notebook Guide`
  - `Random Seeds`
  - `Main Training Parameters`
  - `Main XAI Parameters`
  - `Minimal MI Example`
  - `Minimal ADHD/TDAH Example`
  - `Computational Cost Figure`
  - `Code Availability Statement`

Suggested code-availability statement:

> The source code and reproducibility materials are available in the public STF-KernelSHAP repository. The repository includes the final implementation, reproducibility notebooks, configuration and parameter settings, fixed random seeds, trained model artifacts and Optuna journals, model-training and explanation-generation runners, XAI method settings, minimal MI and ADHD execution examples, computational-cost timing-log generation, and figure-generation utilities.

## Verification Performed

- `python -m compileall src/stf_kernelshap/xai src/stf_kernelshap/reporting`
- Imported the new computational-cost reporting module with `PYTHONPATH=src`.
- Generated a synthetic computational-cost figure in memory.
- Validated `write_xai_timing_log(...)` on a temporary CSV in `/private/tmp`.

No final runtime numbers were added to this response because the timing logs must be generated from the final code on the reported T4 GPU setup.
