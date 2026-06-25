import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna

from copy import deepcopy

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    cohen_kappa_score,
    roc_auc_score,
)

from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from stf_kernelshap.modeling.builders import (
    build_eeg_model,
    build_compile_config,
    suggest_compile_args,
    suggest_model_args,
)

from stf_kernelshap.utils import (
    set_seed,
    infer_chans_samples_from_X,
    ensure_one_hot,
    get_available_folds_from_npz,
    load_best_trial_from_journal,
    extract_probs_from_model_output,
    get_model_folder_name,
    detect_subjects_from_optuna_folder,
)


DATA_MI_ROOT = os.path.join("Data", "MI")
MODELS_MI_ROOT = os.path.join("Models", "MI")


def split_best_params_for_training(
    model_name,
    best_params,
    base_model_args,
    base_compile_args=None
):
    model_name = model_name.lower()

    if base_compile_args is None:
        base_compile_args = {}

    fixed_trial = optuna.trial.FixedTrial(best_params)

    model_args = suggest_model_args(
        trial=fixed_trial,
        model_name=model_name,
        base_model_args=deepcopy(base_model_args),
    )

    compile_cfg = suggest_compile_args(
        trial=fixed_trial,
        model_name=model_name,
        base_compile_args=deepcopy(base_compile_args),
    )

    return model_args, compile_cfg


def train_repeated_subject_predictions_cv(
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
    seed_gap=1000,
    n_repeats=10,
):

    total_histories = []
    models = {}
    subject_prediction_rows = []

    model_name = model_name.lower()

    for repeat in range(n_repeats):

        for fold, (train_subjects, val_subjects, test_subjects) in enumerate(folds):
            model_seed = seed + repeat * 10000 + fold * seed_gap

            set_seed(seed=model_seed)
            train_idx = [i for i, sbj in enumerate(sbjs) if sbj in train_subjects]
            val_idx   = [i for i, sbj in enumerate(sbjs) if sbj in val_subjects]
            test_idx  = [i for i, sbj in enumerate(sbjs) if sbj in test_subjects]

            X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
            y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

            sbjs_test = np.array([sbjs[i] for i in test_idx])
            model_args_fold = deepcopy(model_args)
            model_args_fold["seed"] = True
            model_args_fold["num_seed"] = model_seed

            model = build_eeg_model(
                model_name=model_name,
                **model_args_fold
            )
            compile_cfg_local = deepcopy(compile_cfg)
            compile_cfg_local["total_epochs"] = epochs

            compile_args_local, callbacks = build_compile_config(
                model_name=model_name,
                Chans=model_args_fold["Chans"],
                **compile_cfg_local
            )

            model.compile(**compile_args_local)
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
            train_ds = (
                tf.data.Dataset
                .from_tensor_slices((X_train, y_train_fit))
                .batch(batch_size)
            )

            val_ds = (
                tf.data.Dataset
                .from_tensor_slices((X_val, y_val_fit))
                .batch(batch_size)
            )
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=callbacks,
                verbose=0
            )

            total_histories.append(history)
            models[(repeat, fold)] = model
            preds = model.predict(X_test, verbose=0)

            if model_name == "tgarnet":
                y_pred_probs = preds["out_activation"]
            else:
                y_pred_probs = preds

            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(y_test, axis=1)
            for subject in np.unique(sbjs_test):

                subject_idx = sbjs_test == subject

                y_true_s = y_true[subject_idx]
                y_pred_s = y_pred[subject_idx]
                y_prob_s = y_pred_probs[subject_idx]

                true_class = int(y_true_s[0])

                pred_majority_class = int(
                    np.bincount(y_pred_s).argmax()
                )

                prob_mean = y_prob_s.mean(axis=0)

                row = {
                    "model": model_name,
                    "repeat": repeat,
                    "fold": fold,
                    "model_seed": model_seed,
                    "subject": subject,
                    "n_segments": len(y_true_s),
                    "true_class": true_class,
                    "pred_majority_class": pred_majority_class,
                    "segment_accuracy": accuracy_score(y_true_s, y_pred_s),
                }

                for c in range(y_pred_probs.shape[1]):
                    row[f"prob_class_{c}"] = prob_mean[c]

                subject_prediction_rows.append(row)

    subject_predictions_df = pd.DataFrame(subject_prediction_rows)

    return {
        "subject_predictions_df": subject_predictions_df,
        "histories": total_histories,
        "models": models,
    }

def train_tdah_with_best_optuna_config(
    model_name,
    study_name,
    journal_file,
    base_model_args,
    X,
    y,
    sbjs,
    folds,
    epochs=100,
    batch_size=16,
    seed=42,
    seed_gap=1000,
    n_repeats=10,
):
    print(f"DEBUG: Intentando cargar el estudio '{study_name}' del journal '{journal_file}'")

    if not os.path.exists(journal_file):
        raise FileNotFoundError(f"No existe el journal: {journal_file}")

    file_size = os.path.getsize(journal_file)
    print(f"DEBUG: Journal file '{journal_file}' existe. Tamaño: {file_size} bytes.")

    if file_size == 0:
        raise ValueError(f"El archivo journal '{journal_file}' está vacío. No contiene estudios.")

    storage = JournalStorage(JournalFileBackend(journal_file))

    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )

        print(
            f"DEBUG: Estudio '{study_name}' cargado exitosamente. "
            f"Número de trials: {len(study.trials)}"
        )

    except KeyError:
        raise ValueError(
            f"El estudio '{study_name}' NO se encontró dentro del journal '{journal_file}'. "
            "Asegúrate de que el nombre coincide exactamente con un estudio previamente guardado."
        )

    except Exception as e:
        raise RuntimeError(f"Error inesperado al cargar el estudio de Optuna: {e}")

    if len(study.trials) == 0:
        raise ValueError(
            f"El estudio '{study_name}' se cargó pero no tiene trials. "
            "(Esto no debería pasar si fue optimizado correctamente)."
        )

    best_trial = study.best_trial
    best_params = best_trial.params

    print("Best trial:", best_trial.number)
    print("Best value:", best_trial.value)
    print("Best params:", best_params)

    model_args, compile_cfg = split_best_params_for_training(
        model_name=model_name,
        best_params=best_params,
        base_model_args=base_model_args,
    )

    y = ensure_one_hot(y, model_args["nb_classes"])

    results = train_repeated_subject_predictions_cv(
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
        seed_gap=seed_gap,
        n_repeats=n_repeats,
    )

    results["best_trial_number"] = best_trial.number
    results["best_trial_value"] = best_trial.value
    results["best_params"] = best_params
    results["model_args"] = model_args
    results["compile_cfg"] = compile_cfg
    results["seed"] = seed
    results["seed_gap"] = seed_gap
    results["n_repeats"] = n_repeats

    return results


def get_subject_npz_path(window_name, subject_id, data_mi_root=DATA_MI_ROOT):
    return os.path.join(
        data_mi_root,
        window_name,
        f"subject_{subject_id}.npz"
    )


def get_model_paths(window_name, model_name, subject_id, models_mi_root=MODELS_MI_ROOT):
    model_name = model_name.lower()
    model_folder = get_model_folder_name(model_name)

    model_base_dir = os.path.join(
        models_mi_root,
        window_name,
        model_folder
    )

    models_dir = os.path.join(model_base_dir, "Models")
    optuna_dir = os.path.join(model_base_dir, "Optuna")

    journal_file = os.path.join(
        optuna_dir,
        f"study_{model_name}_MI_{window_name}_sj{subject_id}.journal"
    )

    study_name = f"study_{model_name}_MI_{window_name}_sj{subject_id}"

    return models_dir, journal_file, study_name


def get_weights_path(models_dir, model_name, window_name, subject_id, fold_num):
    model_name = model_name.lower()

    return os.path.join(
        models_dir,
        f"{model_name}_{window_name}_sj{subject_id}_fold_{fold_num}.weights.h5"
    )


def compute_fold_metrics(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }

    try:
        if y_prob.shape[1] == 2:
            metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            metrics["auc"] = roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro",
            )
    except Exception:
        metrics["auc"] = np.nan

    return metrics


def evaluate_subject_to_simple_rows(
    window_name,
    model_name,
    subject_id,
    data_mi_root=DATA_MI_ROOT,
    models_mi_root=MODELS_MI_ROOT,
    seed=42,
):
    model_name = model_name.lower()

    npz_path = get_subject_npz_path(
        window_name=window_name,
        subject_id=subject_id,
        data_mi_root=data_mi_root,
    )

    models_dir, journal_file, study_name = get_model_paths(
        window_name=window_name,
        model_name=model_name,
        subject_id=subject_id,
        models_mi_root=models_mi_root,
    )

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"No existe el .npz:\n{npz_path}")

    data = np.load(npz_path)

    X = data["X"]
    y = data["y"]

    Chans, Samples = infer_chans_samples_from_X(X)

    base_model_args = {
        "nb_classes": 2,
        "Chans": Chans,
        "Samples": Samples,
    }

    y = ensure_one_hot(y, base_model_args["nb_classes"])

    best_trial = load_best_trial_from_journal(
        journal_file=journal_file,
        study_name=study_name,
    )

    best_params = best_trial.params

    model_args, _ = split_best_params_for_training(
        model_name=model_name,
        best_params=best_params,
        base_model_args=base_model_args,
    )

    fold_numbers = get_available_folds_from_npz(data)
    fold_metrics = []

    for fold_pos, fold_num in enumerate(fold_numbers):

        test_idx = data[f"fold_{fold_num}_test_idx"]

        if len(test_idx) == 0:
            print(
                f"WARNING: {window_name} | {model_name} | sj{subject_id} "
                f"| fold {fold_num} sin test_idx."
            )
            continue

        X_test = X[test_idx]
        y_test = y[test_idx]

        weights_file = get_weights_path(
            models_dir=models_dir,
            model_name=model_name,
            window_name=window_name,
            subject_id=subject_id,
            fold_num=fold_num,
        )

        if not os.path.exists(weights_file):
            raise FileNotFoundError(f"No existe el archivo de pesos:\n{weights_file}")

        tf.keras.backend.clear_session()

        current_seed = seed + fold_pos

        np.random.seed(current_seed)
        random.seed(current_seed)
        tf.random.set_seed(current_seed)

        model = build_eeg_model(
            model_name=model_name,
            **deepcopy(model_args)
        )

        dummy_input = tf.zeros(
            (1,) + tuple(X_test.shape[1:]),
            dtype=tf.float32
        )

        _ = model(dummy_input, training=False)

        model.load_weights(weights_file)

        X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)

        preds = model(
            X_test_tf,
            training=False
        )

        y_prob = extract_probs_from_model_output(
            preds=preds,
            model_name=model_name,
        )

        y_true = np.argmax(y_test, axis=1)

        metrics = compute_fold_metrics(
            y_true=y_true,
            y_prob=y_prob,
        )

        metrics["fold"] = fold_num
        fold_metrics.append(metrics)

    if len(fold_metrics) == 0:
        raise ValueError(
            f"No se calcularon métricas para {window_name} | {model_name} | sj{subject_id}"
        )

    df_folds = pd.DataFrame(fold_metrics)

    metric_cols = [
        "accuracy",
        "recall",
        "precision",
        "kappa",
        "auc",
    ]

    row = {
        "window": window_name,
        "model": model_name,
        "subject": subject_id,
        "n_folds": len(df_folds),
        "seed": seed,
    }

    for metric in metric_cols:
        row[f"mean_{metric}"] = float(np.nanmean(df_folds[metric].values))
        row[f"std_{metric}"] = float(np.nanstd(df_folds[metric].values, ddof=1))

    return [row]


def evaluate_window_to_single_csv(
    window_name,
    models=("eegnet", "shallowconvnet", "tgarnet"),
    subjects=None,
    data_mi_root=DATA_MI_ROOT,
    models_mi_root=MODELS_MI_ROOT,
    seed=42,
):
    print("\n" + "=" * 80)
    print(f"EVALUANDO VENTANA: {window_name}")
    print("=" * 80)

    all_rows = []

    for model_name in models:

        model_name = model_name.lower()
        model_folder = get_model_folder_name(model_name)

        optuna_dir = os.path.join(
            models_mi_root,
            window_name,
            model_folder,
            "Optuna"
        )

        if subjects is None:
            model_subjects = detect_subjects_from_optuna_folder(
                optuna_dir=optuna_dir,
                model_name=model_name,
                window_name=window_name,
            )
        else:
            model_subjects = list(subjects)

        print(f"\nModelo: {model_name}")
        print(f"Sujetos: {model_subjects}")

        for subject_id in model_subjects:

            try:
                rows = evaluate_subject_to_simple_rows(
                    window_name=window_name,
                    model_name=model_name,
                    subject_id=subject_id,
                    data_mi_root=data_mi_root,
                    models_mi_root=models_mi_root,
                    seed=seed,
                )

                all_rows.extend(rows)

            except Exception as e:
                print(
                    f"ERROR | ventana={window_name} | modelo={model_name} "
                    f"| sujeto={subject_id}: {e}"
                )

    return pd.DataFrame(all_rows)


def evaluate_all_windows_to_single_csv(
    windows=("2.5-5", "0-7"),
    models=("eegnet", "shallowconvnet", "tgarnet"),
    subjects=None,
    data_mi_root=DATA_MI_ROOT,
    models_mi_root=MODELS_MI_ROOT,
    output_csv=None,
    seed=42,
):
    all_dfs = []

    for window_name in windows:

        df_window = evaluate_window_to_single_csv(
            window_name=window_name,
            models=models,
            subjects=subjects,
            data_mi_root=data_mi_root,
            models_mi_root=models_mi_root,
            seed=seed,
        )

        all_dfs.append(df_window)

    if len(all_dfs) == 0:
        return pd.DataFrame()

    df_results = pd.concat(all_dfs, ignore_index=True)

    if output_csv is not None:
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df_results.to_csv(output_csv, index=False)

    return df_results
