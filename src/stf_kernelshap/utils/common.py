"""Shared non-domain-specific helpers."""

import os
import random
import re

import numpy as np
import optuna
import tensorflow as tf
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


def set_seed(seed=42, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def ensure_one_hot(y, nb_classes):
    y = np.asarray(y)

    # ya está one-hot
    if y.ndim == 2 and y.shape[1] == nb_classes:
        return y.astype(np.float32)

    # etiquetas enteras -> one-hot
    if y.ndim == 1:
        return tf.keras.utils.to_categorical(y, num_classes=nb_classes).astype(np.float32)

    # caso (N,1)
    if y.ndim == 2 and y.shape[1] == 1:
        return tf.keras.utils.to_categorical(y.squeeze(), num_classes=nb_classes).astype(np.float32)

    raise ValueError(f"Formato de y no soportado: shape={y.shape}")


def get_available_folds_from_npz(data):
    fold_numbers = sorted({
        int(key.split("_")[1])
        for key in data.files
        if key.startswith("fold_") and key.endswith("_test_idx")
    })

    if len(fold_numbers) == 0:
        raise ValueError("No se encontraron llaves fold_k_test_idx en el .npz.")

    return fold_numbers


def infer_chans_samples_from_X(X):
    if X.ndim == 3:
        _, Chans, Samples = X.shape
    elif X.ndim == 4:
        _, Chans, Samples, _ = X.shape
    else:
        raise ValueError(f"Forma de X no soportada: {X.shape}")

    return Chans, Samples


def get_model_folder_name(model_name):
    model_name = model_name.lower()

    folder_map = {
        "eegnet": "EEGNet",
        "shallowconvnet": "Shallowconvnet",
        "tgarnet": "TGARNet",
    }

    if model_name not in folder_map:
        raise ValueError(f"Modelo no soportado: {model_name}")

    return folder_map[model_name]


def load_best_trial_from_journal(journal_file, study_name):
    if not os.path.exists(journal_file):
        raise FileNotFoundError(f"No existe el journal:\n{journal_file}")

    if os.path.getsize(journal_file) == 0:
        raise ValueError(f"El journal está vacío:\n{journal_file}")

    storage = JournalStorage(JournalFileBackend(journal_file))

    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
    )

    if len(study.trials) == 0:
        raise ValueError(f"El estudio {study_name} no tiene trials.")

    return study.best_trial


def extract_probs_from_model_output(preds, model_name):
    model_name = model_name.lower()

    if model_name == "tgarnet":
        if isinstance(preds, dict):
            y_prob = preds["out_activation"]
        elif isinstance(preds, (list, tuple)):
            y_prob = preds[0]
        else:
            y_prob = preds
    else:
        y_prob = preds

    if hasattr(y_prob, "numpy"):
        y_prob = y_prob.numpy()

    return y_prob


def detect_subjects_from_optuna_folder(optuna_dir, model_name, window_name):
    model_name = model_name.lower()

    pattern = re.compile(
        rf"study_{model_name}_MI_{re.escape(window_name)}_sj(\d+)\.journal"
    )

    subjects = []

    if not os.path.exists(optuna_dir):
        raise FileNotFoundError(f"No existe la carpeta Optuna:\n{optuna_dir}")

    for fname in os.listdir(optuna_dir):
        match = pattern.match(fname)
        if match:
            subjects.append(int(match.group(1)))

    return sorted(subjects)
