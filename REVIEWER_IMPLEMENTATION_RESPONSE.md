# Implementation Response to Reviewer Suggestions

This document is separate from the project README. It maps two reviewer suggestions to the concrete implementation and repository documentation updates.

## 1. Computational Cost, Runtime, and Scalability

Reviewer suggestion:

> The proposed method is well presented and the experimental results are generally convincing. However, the manuscript does not provide sufficient information regarding the computational cost of STF-KernelSHAP. A brief discussion of the computational complexity, runtime, and scalability would help readers better understand the practicality of the proposed framework.

Implemented response:

- Added `Notebooks/Computational_cost_scalability.ipynb` as a dedicated timing workflow.
- Added per-sample timing logs in `Results/xai_timing_logs.csv`.
- Logged paradigm, window, subject/fold, model, XAI method, label source, sample index, target class, runtime in seconds, hardware, score type, number of timed samples, partition metadata, and method parameters.
- Used `timing_model = "tgarnet"` because T-GARNet is the heaviest evaluated classifier.
- Timed representative MI cases from windows `2.5-5` and `0-7`, plus a comparable ADHD/TDAH fold.
- Used stratified timing samples from `y_test` with `timing_sample_fraction = 0.05`, `timing_min_trials = 20`, `timing_max_trials = 100`, and `timing_random_state = 42`.
- Used `save_attributions=False`, so computational-cost runs do not overwrite full attribution files.
- Added `src/stf_kernelshap/reporting/computational_cost.py` to load logs, summarize runtimes, and generate the paper figure.
- Generated `Figures/Computational_cost_scalability.pdf` with methods ordered from highest to lowest mean runtime, vertical XAI labels, no subplot titles, ylabel `Time [s]`, and a vertical color legend in the upper-right area.
- Documented the default behavior in `README.md`: `DEBUG=False` writes outside `Temp/`, and `OVERWRITE_TIMINGS=False` reuses an existing timing log instead of recomputing.

How this addresses the suggestion:

- Computational complexity is clarified by describing the finite KernelSHAP approximation over structured time-frequency coalitions.
- Runtime is made reproducible through `Results/xai_timing_logs.csv`.
- Scalability is assessed through the timing notebook using representative MI and ADHD/TDAH cases and the heaviest classifier.
- The paper figure can be regenerated directly from the public repository.

Suggested manuscript paragraph:

> To assess computational practicality, we added a dedicated timing workflow that records per-trial explanation runtime for all evaluated XAI methods. Exact Shapley-value estimation is combinatorial in the number of features, whereas STF-KernelSHAP uses a finite KernelSHAP approximation over structured time-frequency coalitions. Its practical runtime is mainly determined by the number of sampled coalitions, the number of structured features, the classifier inference cost, and the time-frequency reconstruction required for each coalition. We report mean runtime in seconds with standard-deviation error bars using T-GARNet, the heaviest evaluated classifier, over representative MI and ADHD/TDAH cases.

Implementation evidence:

- `Notebooks/Computational_cost_scalability.ipynb`
- `src/stf_kernelshap/xai/runner.py::write_xai_timing_log(...)`
- `src/stf_kernelshap/xai/runner.py::run_mi_xai_and_save(...)`
- `src/stf_kernelshap/xai/runner.py::run_tdah_xai_and_save(...)`
- `src/stf_kernelshap/xai/kernelshap.py::stf_kernelshap_all_xtest(...)`
- `src/stf_kernelshap/reporting/computational_cost.py`
- `Results/xai_timing_logs.csv`
- `Figures/Computational_cost_scalability.pdf`

## 2. Public Code Reproducibility and Implementation Details

Reviewer suggestion:

> The manuscript states that the source code will be released. To facilitate reproducibility, additional implementation details, such as the main parameter settings and the explanation generation procedure, should also be provided with the public code repository.

Implemented response:

- Rewrote `README.md` as a concrete reproducibility guide for the repository.
- Documented repository structure, setup, data resolution order, and output paths.
- Documented global notebook behavior, including `DEBUG=False` and `DEBUG=True`.
- Documented the recommended reproduction order.
- Added a notebook-by-notebook guide explaining inputs, outputs, defaults, and main configuration variables.
- Documented the XAI explanation-generation procedure through `compute_xai_all_xtest(...)`, `run_mi_xai_and_save(...)`, and `run_tdah_xai_and_save(...)`.
- Documented default XAI parameters from `get_adaptive_xai_params(...)`.
- Documented random seeds and reproducibility controls.
- Added minimal MI and ADHD/TDAH examples.
- Kept implementation-specific reviewer response in this separate file instead of mixing it with the project README.

How this addresses the suggestion:

- The public code now includes the main parameter settings needed to reproduce model training, evaluation, XAI generation, timing analysis, and figures.
- The README explains how each notebook should be run and what each notebook produces.
- The explanation generation procedure is explicit enough to connect data, trained models, target labels, XAI methods, attribution files, timing logs, and figures.

Suggested code-availability statement:

> The public repository includes the final implementation, reproducibility notebooks, configuration and parameter settings, fixed random seeds, trained model artifacts and Optuna journals, model-training runners, explanation-generation runners, XAI method settings, minimal MI and ADHD/TDAH examples, computational-cost timing logs, and figure-generation utilities.

Implementation evidence:

- `README.md`
- `Notebooks/Train_models.ipynb`
- `Notebooks/TDAH_MI_Results.ipynb`
- `Notebooks/run_xai_models_2.ipynb`
- `Notebooks/Computational_cost_scalability.ipynb`
- `Notebooks/Faithfulness_summary.ipynb`
- `Notebooks/Classification_results.ipynb`
- `Notebooks/Topoplots.ipynb`
- `src/stf_kernelshap/xai/runner.py::get_adaptive_xai_params(...)`

## Verification

- Confirmed that the computational-cost notebook is valid JSON and its code cells parse syntactically.
- Compiled `src/stf_kernelshap/reporting/computational_cost.py`.
- Regenerated `Figures/Computational_cost_scalability.pdf` from `Results/xai_timing_logs.csv`.
- Confirmed the current figure has no subplot titles, no X-axis label, vertical XAI labels, ylabel `Time [s]`, descending runtime order, and a vertical upper-right legend.
