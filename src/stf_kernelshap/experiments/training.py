"""Experiment helpers used by the training notebooks."""

from __future__ import annotations

import logging
import os
import pickle
import warnings

import tensorflow as tf

from stf_kernelshap.training.pipeline import run_optuna_experiment
from stf_kernelshap.data import get_segmented_data


MI_VALID_SUBJECTS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 30, 31,
    32, 33, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
]

WORKERS_BY_MODEL = {
    "eegnet": 11,
    "shallowconvnet": 11,
    "tgarnet": 12,
}


def configure_quiet_runtime():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    try:
        from optuna.exceptions import ExperimentalWarning

        warnings.filterwarnings("ignore", category=ExperimentalWarning)
    except Exception:
        pass

    logging.getLogger("tensorflow").setLevel(logging.FATAL)
    tf.get_logger().setLevel("ERROR")


def split_subjects_evenly(subjects, total_workers, worker_id):
    """Return the 1-indexed worker partition for a subject list."""
    n_subjects = len(subjects)
    base = n_subjects // total_workers
    remainder = n_subjects % total_workers

    if worker_id < 1 or worker_id > total_workers:
        raise ValueError(
            f"worker_id must be between 1 and {total_workers}; received {worker_id}"
        )

    if worker_id <= remainder:
        start = (worker_id - 1) * (base + 1)
        end = start + base + 1
    else:
        start = remainder * (base + 1) + (worker_id - remainder - 1) * base
        end = start + base

    return subjects[start:end]


def samples_for_window(window_name, fs=128):
    if window_name == "2.5-5":
        return int(fs * 2.5)
    if window_name == "0-7":
        return int(fs * 7)
    raise ValueError(f"Unsupported MI window: {window_name}")


def load_tdah_training_data(
    folds_path="Data/TDAH/folds.pkl",
    path_adhd="Data/TDAH/ieee/ADHD_group",
    path_control="Data/TDAH/ieee/Control_group",
):
    with open(folds_path, "rb") as f:
        folds = pickle.load(f)

    X, y, sbjs, _ = get_segmented_data(
        path_adhd=path_adhd,
        path_control=path_control,
    )
    return X, y, sbjs, folds


def run_mi_optuna_for_worker(
    model_name,
    window_name,
    worker_id,
    data_dir="Data",
    output_models_dir=".",
    n_trials=20,
    epochs=100,
    batch_size=16,
):
    total_workers = WORKERS_BY_MODEL[model_name]
    subjects = split_subjects_evenly(MI_VALID_SUBJECTS, total_workers, worker_id)
    samples = samples_for_window(window_name)
    studies = {}

    for subject_id in subjects:
        studies[subject_id] = run_optuna_experiment(
            model_name=model_name,
            data_mode="npz_folds",
            study_name=f"study_{model_name}_MI_{window_name}_sj{subject_id}",
            journal_file=f"study_{model_name}_MI_{window_name}_sj{subject_id}.journal",
            base_model_args={
                "nb_classes": 2,
                "Chans": 64,
                "Samples": samples,
            },
            base_compile_args={},
            npz_path=os.path.join(
                data_dir,
                "MI",
                window_name,
                f"subject_{subject_id}.npz",
            ),
            n_trials=n_trials,
            epochs=epochs,
            batch_size=batch_size,
            best_model_dir=os.path.join(
                output_models_dir,
                f"{model_name}_MI_{window_name}_best_models_{subject_id}",
            ),
        )

    return studies


def run_tdah_optuna(
    model_name,
    folds_path="Data/TDAH/folds.pkl",
    path_adhd="Data/TDAH/ieee/ADHD_group",
    path_control="Data/TDAH/ieee/Control_group",
    output_models_dir=".",
    n_trials=20,
    epochs=100,
    batch_size=16,
):
    X, y, sbjs, folds = load_tdah_training_data(
        folds_path=folds_path,
        path_adhd=path_adhd,
        path_control=path_control,
    )
    return run_optuna_experiment(
        model_name=model_name,
        data_mode="subject_folds",
        study_name=f"study_{model_name}_TDAH",
        journal_file=os.path.join(
            output_models_dir,
            f"study_{model_name}_TDAH.journal",
        ),
        base_model_args={
            "nb_classes": 2,
            "Chans": 19,
            "Samples": 512,
            "alpha": 2,
        },
        base_compile_args={},
        X=X,
        y=y,
        sbjs=sbjs,
        folds=folds,
        n_trials=n_trials,
        epochs=epochs,
        batch_size=batch_size,
        best_model_dir=os.path.join(output_models_dir, f"{model_name}_best_models"),
    )
