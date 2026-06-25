from copy import deepcopy
import numpy as np
from stf_kernelshap.modeling.builders import build_eeg_model,build_compile_config,suggest_model_args,suggest_compile_args
import optuna

from stf_kernelshap.data import get_segmented_data
from stf_kernelshap.utils import ensure_one_hot
import tensorflow as tf
import random
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import math
import pickle
from IPython.display import clear_output
import os
import warnings
import logging
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from collections import defaultdict


def train_L24O_cv(
    model_name,
    X,
    y,
    sbjs,
    folds,
    model_args,
    compile_cfg,
    epochs=100,
    batch_size=16,
    seed=42,
):
    """
    Entrenamiento Leave-24-Out CV compatible con EEGNet, ShallowConvNet y TGARNet.

    Parámetros
    ----------
    model_name : str
        'eegnet', 'shallowconvnet' o 'tgarnet'
    X : np.ndarray
        Datos de entrada.
    y : np.ndarray
        Etiquetas one-hot.
    sbjs : list or np.ndarray
        Identificador de sujeto por muestra.
    folds : list
        Lista de tuplas (train_subjects, val_subjects, test_subjects).
    model_args : dict
        Argumentos para build_eeg_model(...).
    compile_cfg : dict
        Argumentos para build_compile_config(...).
    epochs : int
    batch_size : int
    seed : int

    Retorna
    -------
    mean_accuracy : float
    """
    all_fold_metrics = []
    total_histories = []
    models = {}

    model_name = model_name.lower()

    for fold, (train_subjects, val_subjects, _) in enumerate(folds):
        # --------------------------------------------------
        # Índices train/val/test
        # --------------------------------------------------
        train_idx = [i for i, sbj in enumerate(sbjs) if sbj in train_subjects]
        val_idx   = [i for i, sbj in enumerate(sbjs) if sbj in val_subjects]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        sbjs_val = [sbjs[i] for i in val_idx]

        # --------------------------------------------------
        # Reinicio TF + semillas
        # --------------------------------------------------
        tf.keras.backend.clear_session()
        np.random.seed(seed + fold)
        random.seed(seed + fold)
        tf.random.set_seed(seed + fold)

        # --------------------------------------------------
        # Construcción del modelo
        # --------------------------------------------------
        model = build_eeg_model(model_name=model_name, **deepcopy(model_args))

        compile_cfg_local = deepcopy(compile_cfg)
        compile_cfg_local["total_epochs"] = epochs

        compile_args_local, callbacks = build_compile_config(
            model_name=model_name,
            Chans=model_args["Chans"],
            **compile_cfg_local
        )

        model.compile(**compile_args_local)

        # --------------------------------------------------
        # Callbacks comunes
        # --------------------------------------------------
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=0
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0
        )

        callbacks = list(callbacks) + [early_stopping, reduce_lr]

        # --------------------------------------------------
        # Targets según el modelo
        # --------------------------------------------------
        if model_name == "tgarnet":
            y_train_fit = {
                "out_activation": y_train,
                "kernel_entropy": np.zeros((len(y_train), 1), dtype=np.float32),
            }

            y_val_fit = {
                "out_activation": y_val,
                "kernel_entropy": np.zeros((len(y_val), 1), dtype=np.float32),
            }
        else:
            y_train_fit = y_train
            y_val_fit = y_val

        # --------------------------------------------------
        # Datasets
        # --------------------------------------------------
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_fit)).batch(batch_size)
        val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val_fit)).batch(batch_size)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0
        )

        total_histories.append(history)

        # --------------------------------------------------
        # Predicciones en val
        # --------------------------------------------------
        preds = model.predict(X_val, verbose=0)

        if model_name == "tgarnet":
            y_pred_probs = preds["out_activation"]
        else:
            y_pred_probs = preds

        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_val, axis=1)

        # --------------------------------------------------
        # Métricas
        # --------------------------------------------------
        if y_pred_probs.shape[1] == 2:
            auc_value = roc_auc_score(y_true, y_pred_probs[:, 1])
        else:
            auc_value = roc_auc_score(y_val, y_pred_probs, multi_class="ovr")

        fold_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'kappa': cohen_kappa_score(y_true, y_pred),
            'auc': auc_value
        }

        all_fold_metrics.append(fold_metrics)
        models[fold] = model

        # --------------------------------------------------
        # Accuracy por sujeto
        # --------------------------------------------------
        subject_correct = defaultdict(list)
        for yt, yp, sbj in zip(y_true, y_pred, sbjs_val):
            subject_correct[sbj].append(int(yt == yp))

        subject_accuracies = {sbj: np.mean(vals) for sbj, vals in subject_correct.items()}

    # ------------------------------------------------------
    # Resumen global
    # ------------------------------------------------------
    mean_metrics = {}
    for key in all_fold_metrics[0].keys():
        vals = [f[key] for f in all_fold_metrics]
        mean_metrics[f'mean_{key}'] = np.mean(vals)
        mean_metrics[f'std_{key}'] = np.std(vals)

    accs_general = []
    for i, f in enumerate(all_fold_metrics):
        accs_general.append(f['accuracy'])

    return {
        "mean_accuracy": np.mean(accs_general),
        "fold_metrics": all_fold_metrics,
        "mean_metrics": mean_metrics,
        "histories": total_histories,
        "models": models,
    }

def train_npz_folds_cv(
    npz_path,
    model_name,
    model_args,
    compile_cfg,
    epochs=100,
    batch_size=16,
    seed=42,
):
    """
    Entrena un modelo usando folds ya guardados dentro de un archivo .npz.

    El .npz debe contener:
      - X
      - y
      - fold_k_train_idx
      - fold_k_val_idx
      - fold_k_test_idx
    """
    data = np.load(npz_path)

    X = data["X"]
    y = data["y"]

    # asegurar formato compatible con salida softmax (N, nb_classes)
    nb_classes = model_args["nb_classes"]
    y = ensure_one_hot(y, nb_classes)

    # --------------------------------------------------
    # Detectar folds disponibles
    # --------------------------------------------------
    fold_numbers = sorted({
        int(key.split("_")[1])
        for key in data.files
        if key.startswith("fold_") and key.endswith("_train_idx")
    })

    all_fold_metrics = []
    total_histories = []
    models = {}

    model_name = model_name.lower()

    for fold_pos, fold_num in enumerate(fold_numbers):
        train_idx = data[f"fold_{fold_num}_train_idx"]
        val_idx   = data[f"fold_{fold_num}_val_idx"]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # --------------------------------------------------
        # Reinicio TF + semillas
        # --------------------------------------------------
        tf.keras.backend.clear_session()
        np.random.seed(seed + fold_pos)
        random.seed(seed + fold_pos)
        tf.random.set_seed(seed + fold_pos)

        # --------------------------------------------------
        # Construcción del modelo
        # --------------------------------------------------
        model = build_eeg_model(model_name=model_name, **deepcopy(model_args))

        compile_cfg_local = deepcopy(compile_cfg)
        compile_cfg_local["total_epochs"] = epochs

        compile_args_local, callbacks = build_compile_config(
            model_name=model_name,
            Chans=model_args["Chans"],
            **compile_cfg_local
        )

        model.compile(**compile_args_local)

        # --------------------------------------------------
        # Callbacks adicionales
        # --------------------------------------------------
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=25,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=0
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0
        )

        callbacks = list(callbacks) + [early_stopping, reduce_lr]

        # --------------------------------------------------
        # Preparar targets según el modelo
        # --------------------------------------------------
        if model_name == "tgarnet":
            y_train_fit = {
                "out_activation": y_train,
                "kernel_entropy": np.zeros((len(y_train), 1), dtype=np.float32),
            }

            y_val_fit = {
                "out_activation": y_val,
                "kernel_entropy": np.zeros((len(y_val), 1), dtype=np.float32),
            }
        else:
            y_train_fit = y_train
            y_val_fit = y_val

        # --------------------------------------------------
        # Datasets
        # --------------------------------------------------
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_fit)).batch(batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_fit)).batch(batch_size)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0
        )

        total_histories.append(history)

        # --------------------------------------------------
        # Predicciones
        # --------------------------------------------------
        preds = model.predict(X_val, verbose=0)

        if model_name == "tgarnet":
            y_pred_probs = preds["out_activation"]
        else:
            y_pred_probs = preds

        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_val, axis=1)

        # --------------------------------------------------
        # Métricas
        # --------------------------------------------------
        try:
            if y_pred_probs.shape[1] == 2:
                auc_value = roc_auc_score(y_true, y_pred_probs[:, 1])
            else:
                auc_value = roc_auc_score(y_val, y_pred_probs, multi_class="ovr")
        except Exception:
            auc_value = np.nan

        fold_metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(y_true, y_pred),
            "auc": auc_value,
        }

        all_fold_metrics.append(fold_metrics)
        models[fold_num] = model

    # ------------------------------------------------------
    # Promedio global
    # ------------------------------------------------------
    mean_metrics = {}
    for key in all_fold_metrics[0].keys():
        vals = [f[key] for f in all_fold_metrics]
        mean_metrics[f"mean_{key}"] = np.nanmean(vals)
        mean_metrics[f"std_{key}"] = np.nanstd(vals)

    accs_general = [f["accuracy"] for f in all_fold_metrics]

    return {
        "X": X,
        "y": y,
        "mean_accuracy": np.mean(accs_general),
        "fold_metrics": all_fold_metrics,
        "mean_metrics": mean_metrics,
        "histories": total_histories,
        "models": models,
    }


def make_objective(
    model_name,
    data_mode,
    base_model_args,
    base_compile_args,
    epochs=100,
    batch_size=16,
    seed=42,
    X=None,
    y=None,
    sbjs=None,
    folds=None,
    npz_path=None,
    best_model_dir="best_models",
    best_model_name_template=None,
    save_format="weights",   # "weights" o "full"
    direction="maximize",    # "maximize" o "minimize"
):
    """
    Guarda SOLO los modelos del mejor trial actual.
    Al final del estudio, en disco quedarán solo los del mejor trial global.

    Recomendado usar con n_jobs=1.
    """

    def objective(trial):
        # --------------------------------------------------
        # 1. Hiperparámetros
        # --------------------------------------------------
        model_args = suggest_model_args(
            trial=trial,
            model_name=model_name,
            base_model_args=base_model_args
        )

        compile_cfg = suggest_compile_args(
            trial=trial,
            model_name=model_name,
            base_compile_args=base_compile_args
        )

        # --------------------------------------------------
        # 2. Entrenamiento
        # --------------------------------------------------
        if data_mode == "subject_folds":
            results = train_L24O_cv(
                model_name=model_name,
                X=X,
                y=y,
                sbjs=sbjs,
                folds=folds,
                model_args=model_args,
                compile_cfg=compile_cfg,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
            )

        elif data_mode == "npz_folds":
            results = train_npz_folds_cv(
                npz_path=npz_path,
                model_name=model_name,
                model_args=model_args,
                compile_cfg=compile_cfg,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
            )

        else:
            raise ValueError("data_mode debe ser 'subject_folds' o 'npz_folds'")

        accuracy = float(results["mean_accuracy"])

        # --------------------------------------------------
        # 3. Comparar contra el mejor trial previo
        # --------------------------------------------------
        try:
            prev_best = trial.study.best_value
            has_prev_best = True
        except ValueError:
            prev_best = None
            has_prev_best = False

        if direction == "maximize":
            is_new_best = (not has_prev_best) or (accuracy > prev_best)
        elif direction == "minimize":
            is_new_best = (not has_prev_best) or (accuracy < prev_best)
        else:
            raise ValueError("direction debe ser 'maximize' o 'minimize'")

        # --------------------------------------------------
        # 4. Guardar SOLO si este trial es el nuevo mejor
        # --------------------------------------------------
        if is_new_best:
            os.makedirs(best_model_dir, exist_ok=True)

            for fold_id, model in results["models"].items():
                if save_format == "weights":
                    if best_model_name_template is None:
                        model_filename = f"{model_name}_BESTTRIAL_fold{fold_id}.weights.h5"
                    else:
                        model_filename = best_model_name_template.format(
                            model_name=model_name,
                            fold_id=fold_id,
                        )
                    model_path = os.path.join(
                        best_model_dir,
                        model_filename,
                    )
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    model.save_weights(model_path)

                elif save_format == "full":
                    if best_model_name_template is None:
                        model_filename = f"{model_name}_BESTTRIAL_fold{fold_id}.keras"
                    else:
                        model_filename = best_model_name_template.format(
                            model_name=model_name,
                            fold_id=fold_id,
                        )
                    model_path = os.path.join(
                        best_model_dir,
                        model_filename,
                    )
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    model.save(model_path)

                else:
                    raise ValueError("save_format debe ser 'weights' o 'full'")

            trial.set_user_attr("saved_as_best", True)
            trial.set_user_attr("best_model_dir", best_model_dir)

        else:
            trial.set_user_attr("saved_as_best", False)

        # metadata útil
        trial.set_user_attr("model_args", model_args)
        trial.set_user_attr("compile_cfg", compile_cfg)

        return accuracy

    return objective
  

def get_or_create_study_local(study_name, journal_file, direction="maximize"):
    """
    Crea o reutiliza un estudio Optuna con JournalStorage local.

    Parámetros
    ----------
    study_name : str
        Nombre del estudio.
    journal_file : str
        Ruta al archivo .journal donde se guardará el historial.
    direction : str
        'maximize' o 'minimize'.

    Retorna
    -------
    optuna.study.Study
    """
    journal_dir = os.path.dirname(os.path.abspath(journal_file))
    if journal_dir and not os.path.exists(journal_dir):
        os.makedirs(journal_dir, exist_ok=True)

    storage = JournalStorage(JournalFileBackend(journal_file))

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
        sampler=optuna.samplers.GPSampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )

    return study

def run_optuna_experiment(
    model_name,
    data_mode,
    study_name,
    journal_file,
    base_model_args,
    base_compile_args,
    epochs=100,
    batch_size=16,
    seed=42,
    X=None,
    y=None,
    sbjs=None,
    folds=None,
    npz_path=None,
    n_trials=20,
    best_model_dir="best_models",
    best_model_name_template=None,
):
    objective = make_objective(
        model_name=model_name,
        data_mode=data_mode,
        base_model_args=base_model_args,
        base_compile_args=base_compile_args,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        X=X,
        y=y,
        sbjs=sbjs,
        folds=folds,
        npz_path=npz_path,
        best_model_dir=best_model_dir,
        best_model_name_template=best_model_name_template,
    )

    study = get_or_create_study_local(
        study_name=study_name,
        journal_file=journal_file,
        direction="maximize",
    )

    study.optimize(objective, n_trials=n_trials)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    return study
