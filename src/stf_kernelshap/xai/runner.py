import os
import gc
import csv
import time
import numpy as np
import tensorflow as tf
import shap

from copy import deepcopy
from lime.lime_tabular import LimeTabularExplainer

from stf_kernelshap.modeling.builders import build_eeg_model

from stf_kernelshap.utils import (
    infer_chans_samples_from_X,
    load_best_trial_from_journal,
    extract_probs_from_model_output,
)

from stf_kernelshap.evaluation.pipeline import (
    get_model_paths,
    get_weights_path,
    split_best_params_for_training,
)
from stf_kernelshap.xai.kernelshap import (
    get_classification_output,
    get_sample,
    get_score_model,
    get_target_class_from_labels,
    make_logits_model,
    predict_proba,
    predict_scores,
    predict_scores_from_model,
    stf_kernelshap_all_xtest,
)
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.scores import CategoricalScore


TIMING_LOG_FILENAME = "xai_timing_logs.csv"


def _sync_accelerator():
    """Best-effort synchronization before/after timing GPU-backed calls."""
    try:
        tf.experimental.async_wait()
    except Exception:
        try:
            tf.config.experimental.get_synchronous_execution()
        except Exception:
            pass


def _append_runtime(runtime_seconds, start_time):
    _sync_accelerator()
    runtime_seconds.append(float(time.perf_counter() - start_time))


def write_xai_timing_log(
    results_dir,
    result,
    paradigm,
    model_name,
    label_source,
    hardware="NVIDIA T4 GPU",
    window=None,
    subject=None,
    fold=None,
    method_params=None,
    partition_id=None,
    n_partitions=None,
):
    runtime_seconds = np.asarray(
        result.get("runtime_seconds", []),
        dtype=float,
    )
    sample_indices = np.asarray(result.get("sample_indices", []))
    class_indices = np.asarray(result.get("class_indices", []))

    if runtime_seconds.size == 0:
        return None

    if sample_indices.size != runtime_seconds.size:
        sample_indices = np.arange(runtime_seconds.size)

    if class_indices.size != runtime_seconds.size:
        class_indices = np.full(runtime_seconds.size, np.nan)

    log_path = os.path.join(results_dir, TIMING_LOG_FILENAME)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fieldnames = [
        "paradigm",
        "window",
        "subject",
        "fold",
        "model",
        "method",
        "label_source",
        "sample_idx",
        "class_idx",
        "runtime_seconds",
        "hardware",
        "score_type",
        "n_samples_timed",
        "partition_id",
        "n_partitions",
        "method_params",
    ]

    write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    method_params = method_params or {}

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for sample_idx, class_idx, runtime in zip(
            sample_indices,
            class_indices,
            runtime_seconds,
        ):
            writer.writerow(
                {
                    "paradigm": paradigm,
                    "window": "" if window is None else window,
                    "subject": "" if subject is None else subject,
                    "fold": fold,
                    "model": model_name,
                    "method": result.get("method", ""),
                    "label_source": label_source,
                    "sample_idx": int(sample_idx),
                    "class_idx": "" if np.isnan(class_idx) else int(class_idx),
                    "runtime_seconds": runtime,
                    "hardware": hardware,
                    "score_type": result.get("score_type", ""),
                    "n_samples_timed": runtime_seconds.size,
                    "partition_id": "" if partition_id is None else partition_id,
                    "n_partitions": "" if n_partitions is None else n_partitions,
                    "method_params": repr(method_params),
                }
            )

    return log_path
# ============================================================
# 1. Model loading utilities
# ============================================================

def load_mi_model_and_predict(
    window_name,
    subject_id,
    fold,
    model_name,
    X_test,
    models_mi_root="Models/MI",
):
    model_name = model_name.lower()

    Chans, Samples = infer_chans_samples_from_X(X_test)

    base_model_args = {
        "nb_classes": 2,
        "Chans": Chans,
        "Samples": Samples,
    }

    models_dir, journal_file, study_name = get_model_paths(
        window_name=window_name,
        model_name=model_name,
        subject_id=subject_id,
        models_mi_root=models_mi_root,
    )

    best_trial = load_best_trial_from_journal(
        journal_file=journal_file,
        study_name=study_name,
    )

    model_args, _ = split_best_params_for_training(
        model_name=model_name,
        best_params=best_trial.params,
        base_model_args=base_model_args,
    )

    weights_file = get_weights_path(
        models_dir=models_dir,
        model_name=model_name,
        window_name=window_name,
        subject_id=subject_id,
        fold_num=fold,
    )

    tf.keras.backend.clear_session()

    model = build_eeg_model(
        model_name=model_name,
        **deepcopy(model_args),
    )

    _ = model(
        tf.zeros((1,) + tuple(X_test.shape[1:]), dtype=tf.float32),
        training=False,
    )

    model.load_weights(weights_file)

    preds = model.predict(X_test, verbose=0)

    y_prob = extract_probs_from_model_output(
        preds=preds,
        model_name=model_name,
    )

    y_pred = np.argmax(y_prob, axis=1)

    return model, y_prob, y_pred


def load_tdah_model_and_predict(
    model_name,
    journal_file,
    study_name,
    weights_file,
    X_test,
    base_model_args,
):
    model_name = model_name.lower()

    best_trial = load_best_trial_from_journal(
        journal_file=journal_file,
        study_name=study_name,
    )

    model_args, _ = split_best_params_for_training(
        model_name=model_name,
        best_params=best_trial.params,
        base_model_args=base_model_args,
    )

    tf.keras.backend.clear_session()

    model = build_eeg_model(
        model_name=model_name,
        **deepcopy(model_args),
    )

    _ = model(
        tf.zeros((1,) + tuple(X_test.shape[1:]), dtype=tf.float32),
        training=False,
    )

    model.load_weights(weights_file)

    preds = model.predict(X_test, verbose=0)

    y_prob = extract_probs_from_model_output(
        preds=preds,
        model_name=model_name,
    )

    y_pred = np.argmax(y_prob, axis=1)

    return model, y_prob, y_pred


# ============================================================
# 2. Prediction and tensor utilities
# ============================================================




def make_flat_predict_fn(
    model,
    input_shape,
    predict_batch_size=None,
):

    def predict_fn(X_flat):
        X = X_flat.reshape((-1, *input_shape)).astype(np.float32)

        return predict_proba(
            model=model,
            X=X,
            verbose=0,
            predict_batch_size=predict_batch_size,
        )

    return predict_fn




# ============================================================
# 3. Grad-CAM utilities
# ============================================================

def get_last_gradcam_compatible_layer(model):
    """
    Selecciona explícitamente la última capa Conv2D del modelo.

    Para Grad-CAM++, la capa penúltima debe corresponder a mapas
    de activación convolucionales interpretables, no simplemente a
    cualquier tensor 4D.
    """

    conv_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]

    if len(conv_layers) == 0:
        raise ValueError(
            "No se encontró ninguna capa Conv2D compatible con Grad-CAM++."
        )

    return conv_layers[-1].name

def make_classification_model_for_gradcam(
    model,
    use_logits=True,
    output_layer_name="out_activation",
):
    if use_logits:

        return make_logits_model(
            model=model,
            output_layer_name=output_layer_name,
        )

    model_output = model.output

    if isinstance(model_output, dict):

        class_output = model_output[
            "out_activation"
        ]

    elif isinstance(
        model_output,
        (list, tuple),
    ):

        output_names = model.output_names

        if "out_activation" in output_names:

            idx = output_names.index(
                "out_activation"
            )

            class_output = model_output[idx]

        else:

            class_output = model_output[0]

    else:

        class_output = model_output

    gradcam_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=class_output,
    )

    return gradcam_model


def resize_cam_to_input(cam, input_shape):
    if len(input_shape) == 2:
        target_h, target_w = input_shape
        cam_4d = cam[None, :, :, None]
        cam_resized = tf.image.resize(cam_4d, (target_h, target_w))
        return cam_resized.numpy()[0, :, :, 0]

    elif len(input_shape) == 3:
        target_h, target_w = input_shape[0], input_shape[1]
        cam_4d = cam[None, :, :, None]
        cam_resized = tf.image.resize(cam_4d, (target_h, target_w))
        cam_resized = cam_resized.numpy()[0, :, :, 0]
        return cam_resized[:, :, None]

    else:
        raise ValueError(f"Forma no soportada para resize CAM: {input_shape}")




def sample_stratified_background_indices(
    y_train,
    background_size,
    random_state=42,
):
    y_train = np.asarray(y_train)

    # Cambio necesario: asegurar etiquetas 1D
    if y_train.ndim > 1:
        y_train = np.argmax(y_train, axis=1)

    y_train = y_train.astype(int)

    rng = np.random.default_rng(random_state)

    classes = np.unique(y_train)
    selected_indices = []

    n_classes = len(classes)
    per_class = max(1, background_size // n_classes)

    for cls in classes:
        cls_indices = np.where(y_train == cls)[0]

        n_take = min(per_class, len(cls_indices))

        selected_cls = rng.choice(
            cls_indices,
            size=n_take,
            replace=False,
        )

        selected_indices.extend(selected_cls.tolist())

    selected_indices = np.asarray(selected_indices, dtype=int)

    if len(selected_indices) < background_size:
        remaining_indices = np.setdiff1d(
            np.arange(len(y_train)),
            selected_indices,
        )

        n_extra = min(
            background_size - len(selected_indices),
            len(remaining_indices),
        )

        if n_extra > 0:
            extra_indices = rng.choice(
                remaining_indices,
                size=n_extra,
                replace=False,
            )

            selected_indices = np.concatenate(
                [selected_indices, extra_indices]
            )

    return selected_indices

  
def make_flat_score_fn(
    model,
    input_shape,
    use_logits=False,
    output_layer_name="out_activation",
    predict_batch_size=None,
):
    """
    Crea una función compatible con métodos que reciben datos aplanados,
    como KernelSHAP.

    Entrada:
        X_flat: matriz de forma (N, features)

    Salida:
        scores: matriz de forma (N, n_classes)

    Si use_logits=False:
        devuelve probabilidades.

    Si use_logits=True:
        devuelve logits.
    """

    def predict_fn(X_flat):

        X = X_flat.reshape(
            (-1, *input_shape)
        ).astype(np.float32)

        scores = predict_scores(
            model=model,
            X=X,
            use_logits=use_logits,
            output_layer_name=output_layer_name,
            verbose=0,
            predict_batch_size=predict_batch_size,
        )

        return scores

    return predict_fn



def make_flat_score_fn_from_model(
    score_model,
    input_shape,
    predict_batch_size=None,
):
    """
    Crea una función compatible con métodos que reciben datos aplanados,
    como KernelSHAP.

    Entrada:
        X_flat: matriz de forma (N, features)

    Salida:
        scores: matriz de forma (N, n_classes)

    score_model ya debe ser:
        - el modelo original, si se usan probabilidades
        - el modelo de logits, si se usan logits
    """

    def predict_fn(X_flat):

        X = X_flat.reshape(
            (-1, *input_shape)
        ).astype(np.float32)

        scores = predict_scores_from_model(
            score_model=score_model,
            X=X,
            verbose=0,
            predict_batch_size=predict_batch_size,
        )

        return scores

    return predict_fn
def kernelshap_all_xtest(
    model,
    X_test,
    X_train=None,
    y_train=None,
    y_pred=None,
    sample_indices=None,
    background_size=50,
    nsamples=100,
    l1_reg="num_features(100)",
    predict_batch_size=None,
    random_state=42,
    use_logits=True,
    output_layer_name="out_activation",
):
    X_test = X_test.astype(np.float32)

    if X_train is not None:
        X_train = X_train.astype(np.float32)
        X_background_source = X_train
    else:
        X_background_source = X_test

    if sample_indices is None:
        sample_indices = np.arange(X_test.shape[0])

    sample_indices = np.asarray(sample_indices)

    input_shape = X_test.shape[1:]

    background_size = min(
        background_size,
        X_background_source.shape[0],
    )

    rng = np.random.default_rng(random_state)

    if X_train is not None and y_train is not None:
        background_indices = sample_stratified_background_indices(
            y_train=y_train,
            background_size=background_size,
            random_state=random_state,
        )
    else:
        background_indices = rng.choice(
            X_background_source.shape[0],
            size=background_size,
            replace=False,
        )

    X_background = X_background_source[background_indices]

    X_background_flat = X_background.reshape(
        (X_background.shape[0], -1)
    )

    # Se crea una sola vez.
    score_model = get_score_model(
        model=model,
        use_logits=use_logits,
        output_layer_name=output_layer_name,
    )

    predict_fn = make_flat_score_fn_from_model(
        score_model=score_model,
        input_shape=input_shape,
        predict_batch_size=predict_batch_size,
    )

    explainer = shap.KernelExplainer(
        predict_fn,
        X_background_flat,
    )

    relevance_maps = []
    class_indices = []
    runtime_seconds = []

    for i, sample_idx in enumerate(sample_indices):
        _sync_accelerator()
        start_time = time.perf_counter()

        x = get_sample(
            X_test,
            sample_idx,
        )

        x_flat = x.reshape((1, -1))

        if y_pred is None:
            y_prob = predict_proba(
                model=model,
                X=x,
                verbose=0,
                predict_batch_size=predict_batch_size,
            )

            class_idx = int(
                np.argmax(y_prob[0])
            )

        else:

            class_idx = get_target_class_from_labels(
                labels=y_pred,
                sample_idx=sample_idx,
            )

        shap_values = explainer.shap_values(
            x_flat,
            nsamples=nsamples,
            l1_reg=l1_reg,
        )

        if isinstance(shap_values, list):

            shap_class = shap_values[class_idx]

        else:

            shap_values = np.asarray(shap_values)

            if shap_values.ndim == 3:

                shap_class = shap_values[:, :, class_idx]

            else:

                shap_class = shap_values

        relevance = shap_class.reshape(
            input_shape
        )

        relevance_maps.append(
            relevance.astype(np.float32)
        )

        class_indices.append(
            class_idx
        )
        _append_runtime(
            runtime_seconds,
            start_time,
        )

        print(
            f"[{i + 1}/{len(sample_indices)}] "
            f"KernelSHAP | "
            f"sample_idx={sample_idx} | "
            f"class={class_idx} | "
            f"{'logits' if use_logits else 'probabilities'}"
        )

    relevance_maps = np.asarray(
        relevance_maps,
        dtype=np.float32,
    )

    class_indices = np.asarray(
        class_indices,
        dtype=np.int64,
    )

    return {
        "method": "KernelSHAP",
        "score_type": "logits" if use_logits else "probabilities",
        "sample_indices": sample_indices,
        "class_indices": class_indices,
        "runtime_seconds": np.asarray(runtime_seconds, dtype=np.float64),
        "relevance_maps": relevance_maps,
        "mean_relevance": np.mean(relevance_maps, axis=0),
    }

def lime_all_xtest(
    model,
    X_test,
    X_train=None,
    y_train=None,
    y_pred=None,
    sample_indices=None,
    background_size=100,
    num_features=200,
    num_samples=1000,
    predict_batch_size=None,
    random_state=42,
):

    X_test = X_test.astype(np.float32)

    if X_train is not None:
        X_train = X_train.astype(np.float32)
        X_background_source = X_train
    else:
        X_background_source = X_test

    if sample_indices is None:
        sample_indices = np.arange(X_test.shape[0])

    input_shape = X_test.shape[1:]

    background_size = min(background_size, X_background_source.shape[0])

    rng = np.random.default_rng(random_state)

    if X_train is not None and y_train is not None:
        background_indices = sample_stratified_background_indices(
            y_train=y_train,
            background_size=background_size,
            random_state=random_state,
        )
    else:
        background_indices = rng.choice(
            X_background_source.shape[0],
            size=background_size,
            replace=False,
        )

    X_background = X_background_source[background_indices]
    X_background_flat = X_background.reshape((X_background.shape[0], -1))

    predict_fn = make_flat_predict_fn(
        model=model,
        input_shape=input_shape,
        predict_batch_size=predict_batch_size,
    )

    explainer = LimeTabularExplainer(
        training_data=X_background_flat,
        mode="classification",
        discretize_continuous=False,
        verbose=False,
        random_state=random_state,
    )

    relevance_maps = []
    class_indices = []
    runtime_seconds = []

    for i, sample_idx in enumerate(sample_indices):
        _sync_accelerator()
        start_time = time.perf_counter()

        x = get_sample(X_test, sample_idx)
        x_flat = x.reshape((1, -1))

        if y_pred is None:
            y_prob = predict_proba(
                model=model,
                X=x,
                verbose=0,
                predict_batch_size=predict_batch_size,
            )
            class_idx = int(np.argmax(y_prob[0]))
        else:
            class_idx = get_target_class_from_labels(
                labels=y_pred,
                sample_idx=sample_idx,
            )

        explanation = explainer.explain_instance(
            data_row=x_flat[0],
            predict_fn=predict_fn,
            labels=[class_idx],
            num_features=num_features,
            num_samples=num_samples,
        )

        lime_vector = np.zeros(x_flat.shape[1], dtype=np.float32)

        for feature_idx, weight in explanation.as_map()[class_idx]:
            lime_vector[feature_idx] = weight

        relevance = lime_vector.reshape(input_shape)

        relevance_maps.append(relevance)
        class_indices.append(class_idx)
        _append_runtime(
            runtime_seconds,
            start_time,
        )

        print(
            f"[{i + 1}/{len(sample_indices)}] "
            f"LIME | sample_idx={sample_idx} | class={class_idx}"
        )

    relevance_maps = np.asarray(relevance_maps, dtype=np.float32)
    class_indices = np.asarray(class_indices, dtype=np.int64)

    return {
        "method": "LIME",
        "sample_indices": np.asarray(sample_indices),
        "class_indices": class_indices,
        "runtime_seconds": np.asarray(runtime_seconds, dtype=np.float64),
        "relevance_maps": relevance_maps,
        "mean_relevance": np.mean(relevance_maps, axis=0),
    }
def integrated_gradients_all_xtest(
    model,
    X_test,
    X_train=None,
    y_train=None,
    y_pred=None,
    sample_indices=None,
    baseline=None,
    background_size=100,
    steps=50,
    batch_size=1,
    random_state=42,
    use_logits=True,
    output_layer_name="out_activation",
):
    X_test = X_test.astype(np.float32)

    if X_train is not None:
        X_train = X_train.astype(np.float32)
        X_background_source = X_train
    else:
        X_background_source = X_test

    if sample_indices is None:
        sample_indices = np.arange(X_test.shape[0])

    sample_indices = np.asarray(sample_indices)

    if baseline is None:
        background_size = min(
            background_size,
            X_background_source.shape[0]
        )

        rng = np.random.default_rng(random_state)

        if X_train is not None and y_train is not None:
            background_indices = sample_stratified_background_indices(
                y_train=y_train,
                background_size=background_size,
                random_state=random_state,
            )
        else:
            background_indices = rng.choice(
                X_background_source.shape[0],
                size=background_size,
                replace=False,
            )

        X_background = X_background_source[
            background_indices
        ]

        baseline_reference = np.mean(
            X_background,
            axis=0,
            keepdims=True,
        ).astype(np.float32)

    if use_logits:
        explanation_model = make_logits_model(
            model=model,
            output_layer_name=output_layer_name,
        )
    else:
        explanation_model = model

    relevance_all = []
    class_all = []
    runtime_all = []

    for start in range(
        0,
        len(sample_indices),
        batch_size,
    ):
        _sync_accelerator()
        start_time = time.perf_counter()

        batch_indices = sample_indices[
            start:start + batch_size
        ]

        x_batch = X_test[
            batch_indices
        ].astype(np.float32)

        y_prob = predict_proba(
            model=model,
            X=x_batch,
            verbose=0,
            predict_batch_size=batch_size,
        )

        if y_pred is None:
            class_idx = np.argmax(
                y_prob,
                axis=1
            ).astype(int)
        else:
            class_idx = np.asarray([
                get_target_class_from_labels(
                    labels=y_pred,
                    sample_idx=idx,
                )
                for idx in batch_indices
            ]).astype(int)

        if baseline is None:

            baseline_batch = np.repeat(
                baseline_reference,
                repeats=x_batch.shape[0],
                axis=0,
            )

        else:

            baseline_batch = baseline.astype(
                np.float32
            )

            if baseline_batch.shape[0] == 1:

                baseline_batch = np.repeat(
                    baseline_batch,
                    repeats=x_batch.shape[0],
                    axis=0,
                )

        x_tf = tf.convert_to_tensor(
            x_batch,
            dtype=tf.float32,
        )

        baseline_tf = tf.convert_to_tensor(
            baseline_batch,
            dtype=tf.float32,
        )

        alphas = tf.linspace(
            0.0,
            1.0,
            steps,
        )

        interpolated = []

        for alpha in alphas:

            interpolated.append(
                baseline_tf
                + alpha * (x_tf - baseline_tf)
            )

        interpolated = tf.concat(
            interpolated,
            axis=0,
        )

        class_idx_tiled = np.tile(
            class_idx,
            steps,
        )

        with tf.GradientTape() as tape:

            tape.watch(interpolated)

            y = explanation_model(
                interpolated,
                training=False,
            )

            y_class = get_classification_output(y)

            indices = tf.stack(
                [
                    tf.range(
                        tf.shape(y_class)[0]
                    ),
                    tf.convert_to_tensor(
                        class_idx_tiled,
                        dtype=tf.int32,
                    ),
                ],
                axis=1,
            )

            target_scores = tf.gather_nd(
                y_class,
                indices,
            )

        gradients = tape.gradient(
            target_scores,
            interpolated,
        )

        gradients = tf.reshape(
            gradients,
            (
                steps,
                x_batch.shape[0],
            ) + x_batch.shape[1:]
        )

        grads = (
            gradients[:-1]
            + gradients[1:]
        ) / 2.0

        avg_gradients = tf.reduce_mean(
            grads,
            axis=0,
        )

        relevance = (
            x_tf - baseline_tf
        ) * avg_gradients

        relevance_all.append(
            relevance.numpy().astype(np.float32)
        )

        class_all.append(class_idx)
        _sync_accelerator()
        elapsed_seconds = time.perf_counter() - start_time
        runtime_all.append(
            np.full(
                len(batch_indices),
                elapsed_seconds / max(len(batch_indices), 1),
                dtype=np.float64,
            )
        )

        print(
            f"[{min(start + batch_size, len(sample_indices))}/{len(sample_indices)}] "
            f"IntegratedGradients | "
            f"{'logits' if use_logits else 'probabilities'}"
        )

    relevance_maps = np.concatenate(
        relevance_all,
        axis=0,
    ).astype(np.float32)

    class_indices = np.concatenate(
        class_all,
        axis=0,
    ).astype(np.int64)
    runtime_seconds = np.concatenate(
        runtime_all,
        axis=0,
    ).astype(np.float64)

    return {
        "method": "IntegratedGradients",
        "sample_indices": sample_indices,
        "class_indices": class_indices,
        "runtime_seconds": runtime_seconds,
        "relevance_maps": relevance_maps,
        "mean_relevance": np.mean(relevance_maps, axis=0),
    }


def occlusion_all_xtest(
    model,
    X_test,
    X_train=None,
    y_train=None,
    y_pred=None,
    sample_indices=None,
    fs=128,
    window_seconds=1.0,
    stride_seconds=0.25,
    baseline_value=None,
    background_size=100,
    occ_batch_size=128,
    predict_batch_size=None,
    random_state=42,
    use_logits=True,
    output_layer_name="out_activation",
):
    X_test = X_test.astype(np.float32)

    if X_train is not None:
        X_train = X_train.astype(np.float32)
        X_background_source = X_train
    else:
        X_background_source = X_test

    if sample_indices is None:
        sample_indices = np.arange(X_test.shape[0])

    sample_indices = np.asarray(sample_indices)

    input_shape = X_test.shape[1:]

    window_size = int(window_seconds * fs)
    stride = int(stride_seconds * fs)

    if baseline_value is None:
        background_size = min(
            background_size,
            X_background_source.shape[0],
        )

        rng = np.random.default_rng(random_state)

        if X_train is not None and y_train is not None:
            background_indices = sample_stratified_background_indices(
                y_train=y_train,
                background_size=background_size,
                random_state=random_state,
            )
        else:
            background_indices = rng.choice(
                X_background_source.shape[0],
                size=background_size,
                replace=False,
            )

        X_background = X_background_source[background_indices]

        baseline_reference = np.mean(
            X_background,
            axis=0,
            keepdims=True,
        ).astype(np.float32)

    # Se crea una sola vez.
    score_model = get_score_model(
        model=model,
        use_logits=use_logits,
        output_layer_name=output_layer_name,
    )

    relevance_maps = []
    class_indices = []
    runtime_seconds = []

    for i, sample_idx in enumerate(sample_indices):
        _sync_accelerator()
        start_time = time.perf_counter()

        x = get_sample(
            X_test,
            sample_idx,
        )

        y_prob = predict_proba(
            model=model,
            X=x,
            verbose=0,
            predict_batch_size=predict_batch_size,
        )

        if y_pred is None:
            class_idx = int(
                np.argmax(y_prob[0])
            )
        else:
            class_idx = get_target_class_from_labels(
                labels=y_pred,
                sample_idx=sample_idx,
            )

        original_scores = predict_scores_from_model(
            score_model=score_model,
            X=x,
            verbose=0,
            predict_batch_size=predict_batch_size,
        )

        original_score = original_scores[
            0,
            class_idx,
        ]

        relevance = np.zeros(
            input_shape,
            dtype=np.float32,
        )

        counts = np.zeros(
            input_shape,
            dtype=np.float32,
        )

        occluded_samples = []
        positions = []

        if len(input_shape) == 2:

            n_channels = input_shape[0]
            n_times = input_shape[1]

            for ch in range(n_channels):

                for start in range(
                    0,
                    n_times - window_size + 1,
                    stride,
                ):

                    end = start + window_size

                    x_occ = x.copy()

                    if baseline_value is None:
                        x_occ[:, ch, start:end] = baseline_reference[
                            :,
                            ch,
                            start:end,
                        ]
                    else:
                        x_occ[:, ch, start:end] = baseline_value

                    occluded_samples.append(
                        x_occ[0]
                    )

                    positions.append(
                        (ch, start, end)
                    )

        elif len(input_shape) == 3:

            n_channels = input_shape[0]
            n_times = input_shape[1]

            for ch in range(n_channels):

                for start in range(
                    0,
                    n_times - window_size + 1,
                    stride,
                ):

                    end = start + window_size

                    x_occ = x.copy()

                    if baseline_value is None:
                        x_occ[:, ch, start:end, :] = baseline_reference[
                            :,
                            ch,
                            start:end,
                            :,
                        ]
                    else:
                        x_occ[:, ch, start:end, :] = baseline_value

                    occluded_samples.append(
                        x_occ[0]
                    )

                    positions.append(
                        (ch, start, end)
                    )

        else:
            raise ValueError(
                f"Forma no soportada para Occlusion: {input_shape}"
            )

        occluded_samples = np.asarray(
            occluded_samples,
            dtype=np.float32,
        )

        for batch_start in range(
            0,
            len(occluded_samples),
            occ_batch_size,
        ):

            batch_end = batch_start + occ_batch_size

            x_occ_batch = occluded_samples[
                batch_start:batch_end
            ]

            y_occ_scores = predict_scores_from_model(
                score_model=score_model,
                X=x_occ_batch,
                verbose=0,
                predict_batch_size=predict_batch_size,
            )

            occ_scores = y_occ_scores[
                :,
                class_idx,
            ]

            for local_i, occ_score in enumerate(occ_scores):

                pos_i = batch_start + local_i

                ch, start, end = positions[pos_i]

                drop = original_score - occ_score

                if len(input_shape) == 2:

                    relevance[ch, start:end] += drop
                    counts[ch, start:end] += 1.0

                else:

                    relevance[ch, start:end, :] += drop
                    counts[ch, start:end, :] += 1.0

        relevance = relevance / np.maximum(
            counts,
            1.0,
        )

        relevance_maps.append(
            relevance.astype(np.float32)
        )

        class_indices.append(
            class_idx
        )
        _append_runtime(
            runtime_seconds,
            start_time,
        )

        print(
            f"[{i + 1}/{len(sample_indices)}] "
            f"Occlusion | "
            f"sample_idx={sample_idx} | "
            f"class={class_idx} | "
            f"{'logits' if use_logits else 'probabilities'}"
        )

    relevance_maps = np.asarray(
        relevance_maps,
        dtype=np.float32,
    )

    class_indices = np.asarray(
        class_indices,
        dtype=np.int64,
    )

    return {
        "method": "Occlusion",
        "score_type": "logits" if use_logits else "probabilities",
        "sample_indices": sample_indices,
        "class_indices": class_indices,
        "runtime_seconds": np.asarray(runtime_seconds, dtype=np.float64),
        "relevance_maps": relevance_maps,
        "mean_relevance": np.mean(relevance_maps, axis=0),
    }



def gradcam_plus_plus_all_xtest(
    model,
    X_test,
    y_pred=None,
    sample_indices=None,
    layer_name=None,
    eps=1e-8,
    use_logits=True,
    output_layer_name="out_activation",
):

    X_test = X_test.astype(np.float32)

    if sample_indices is None:
        sample_indices = np.arange(
            X_test.shape[0]
        )

    sample_indices = np.asarray(
        sample_indices
    )

    input_shape = X_test.shape[1:]

    if layer_name is None:

        used_layer_name = (
            get_last_gradcam_compatible_layer(
                model
            )
        )

    else:

        used_layer_name = layer_name

    gradcam_model = (
        make_classification_model_for_gradcam(
            model=model,
            use_logits=use_logits,
            output_layer_name=output_layer_name,
        )
    )

    gradcam_pp = GradcamPlusPlus(
        gradcam_model,
        clone=False,
    )

    relevance_maps = []
    class_indices = []
    runtime_seconds = []

    for i, sample_idx in enumerate(
        sample_indices
    ):
        _sync_accelerator()
        start_time = time.perf_counter()

        x = get_sample(
            X_test,
            sample_idx,
        )

        if y_pred is None:

            y_prob = predict_proba(
                model=model,
                X=x,
                verbose=0,
            )

            class_idx = int(
                np.argmax(y_prob[0])
            )

        else:
            class_idx = get_target_class_from_labels(
                labels=y_pred,
                sample_idx=sample_idx,
            )

        score = CategoricalScore(
            [class_idx]
        )

        cam = gradcam_pp(
            score,
            x,
            penultimate_layer=used_layer_name,
            seek_penultimate_conv_layer=False,
        )

        cam = np.asarray(cam)

        if cam.ndim == 4:

            cam = cam[0, :, :, 0]

        elif cam.ndim == 3:

            cam = cam[0]

        else:

            raise ValueError(
                f"Salida inesperada de GradCAM++: "
                f"cam.shape={cam.shape}"
            )

        relevance = resize_cam_to_input(
            cam=cam,
            input_shape=input_shape,
        )

        relevance_maps.append(
            relevance.astype(np.float32)
        )

        class_indices.append(
            class_idx
        )
        _append_runtime(
            runtime_seconds,
            start_time,
        )

        print(
            f"[{i + 1}/{len(sample_indices)}] "
            f"GradCAM++ | "
            f"sample_idx={sample_idx} | "
            f"class={class_idx} | "
            f"layer={used_layer_name} | "
            f"{'logits' if use_logits else 'probabilities'}"
        )

    relevance_maps = np.asarray(
        relevance_maps,
        dtype=np.float32,
    )

    class_indices = np.asarray(
        class_indices,
        dtype=np.int64,
    )
    return {
        "method": "GradCAM++",
        "sample_indices": sample_indices,
        "class_indices": class_indices,
        "runtime_seconds": np.asarray(runtime_seconds, dtype=np.float64),
        "relevance_maps": relevance_maps,
        "mean_relevance": np.mean(relevance_maps, axis=0),
        "layer_name": used_layer_name,
    }

def compute_xai_all_xtest(
    model,
    X_test,
    X_train=None,
    y_train=None,
    y_pred=None,
    method="IntegratedGradients",
    sample_indices=None,
    batch_size=16,
    label_source=None,
    output_layer_name="out_activation",
    **method_kwargs
):
    """
    Ejecuta el método XAI seleccionado sobre X_test.

    Reglas para el uso de logits:

    label_source == "y_pred":
        Caso orientado a Deletion/ROAD.
        Se usan probabilidades para métodos perturbativos y SHAP:
            KernelSHAP, STF-KernelSHAP, Occlusion, LIME.
        Se usan logits solo en métodos basados en gradiente:
            IntegratedGradients, GradCAM++.

    label_source == "y_test":
        Caso orientado a explicación respecto a la clase real.
        Se usan logits en todos los métodos compatibles:
            KernelSHAP, STF-KernelSHAP, Occlusion,
            IntegratedGradients, GradCAM++.
        LIME se mantiene con probabilidades por su formulación
        como método de clasificación.
    """

    X_test = X_test.astype(np.float32)

    method_key = (
        method.lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )

    if label_source is None:
        raise ValueError(
            "Debes proporcionar label_source='y_pred' o label_source='y_test'."
        )

    if label_source not in ["y_pred", "y_test"]:
        raise ValueError(
            "label_source debe ser 'y_pred' o 'y_test'. "
            f"Valor recibido: {label_source}"
        )

    # ============================================================
    # Decisión centralizada sobre logits
    # ============================================================

    gradient_methods = [
        "integratedgradients",
        "ig",
        "gradcam++",
        "gradcamplusplus",
    ]

    logits_compatible_methods = [
        "kernelshap",
        "kernelshapflat",
        "stfkernelshap",
        "occlusion",
        "occlusionsensitivity",
        "integratedgradients",
        "ig",
        "gradcam++",
        "gradcamplusplus",
    ]

    lime_methods = [
        "lime",
        "limeflat",
    ]

    if method_key in lime_methods:
        use_logits_for_method = False

    elif label_source == "y_pred":
        use_logits_for_method = method_key in gradient_methods

    elif label_source == "y_test":
        use_logits_for_method = method_key in logits_compatible_methods

    else:
        use_logits_for_method = False

    # ============================================================
    # Métodos que requieren background/referencia
    # ============================================================

    methods_requiring_background = [
        "kernelshap",
        "kernelshapflat",
        "lime",
        "limeflat",
        "occlusion",
        "occlusionsensitivity",
        "integratedgradients",
        "ig",
    ]

    if method_key in methods_requiring_background:

        if X_train is None:
            raise ValueError(
                "X_train debe proporcionarse como background/referencia. "
                "No se permite usar X_test como background para evitar fuga de información."
            )

        X_train = X_train.astype(np.float32)

        if y_train is None:
            raise ValueError(
                "y_train debe proporcionarse para construir un background "
                "estratificado por clase."
            )

    # ============================================================
    # Inyectar use_logits solo a métodos compatibles
    # ============================================================

    if method_key in logits_compatible_methods:
        method_kwargs["use_logits"] = use_logits_for_method
        method_kwargs["output_layer_name"] = output_layer_name

    # ============================================================
    # Ejecución del método XAI
    # ============================================================

    if method_key in ["kernelshap", "kernelshapflat"]:

        result = kernelshap_all_xtest(
            model=model,
            X_test=X_test,
            X_train=X_train,
            y_train=y_train,
            y_pred=y_pred,
            sample_indices=sample_indices,
            **method_kwargs,
        )

    elif method_key == "stfkernelshap":

        result = stf_kernelshap_all_xtest(
            model=model,
            X_test=X_test,
            y_pred=y_pred,
            sample_indices=sample_indices,
            **method_kwargs,
        )

    elif method_key in ["lime", "limeflat"]:

        result = lime_all_xtest(
            model=model,
            X_test=X_test,
            X_train=X_train,
            y_train=y_train,
            y_pred=y_pred,
            sample_indices=sample_indices,
            **method_kwargs,
        )

    elif method_key in ["occlusion", "occlusionsensitivity"]:

        result = occlusion_all_xtest(
            model=model,
            X_test=X_test,
            X_train=X_train,
            y_train=y_train,
            y_pred=y_pred,
            sample_indices=sample_indices,
            **method_kwargs,
        )

    elif method_key in ["integratedgradients", "ig"]:

        result = integrated_gradients_all_xtest(
            model=model,
            X_test=X_test,
            X_train=X_train,
            y_train=y_train,
            y_pred=y_pred,
            sample_indices=sample_indices,
            **method_kwargs,
        )

    elif method_key in ["gradcam++", "gradcamplusplus"]:

        result = gradcam_plus_plus_all_xtest(
            model=model,
            X_test=X_test,
            y_pred=y_pred,
            sample_indices=sample_indices,
            **method_kwargs,
        )

    else:
        raise ValueError(
            "Método no reconocido. Usa: "
            "'KernelSHAP', 'KernelSHAP-Flat', "
            "'STF-KernelSHAP', 'LIME', 'LIME-Flat', "
            "'Occlusion', 'IntegratedGradients' o 'GradCAM++'."
        )

    # ============================================================
    # Metadata metodológica
    # ============================================================

    if isinstance(result, dict):
        result["label_source"] = label_source
        result["use_logits"] = use_logits_for_method
        result["score_type"] = (
            "logits" if use_logits_for_method else "probabilities"
        )

    return result

def extract_importance_matrix(xai_result):
    """
    Extrae únicamente la matriz de relevancia generada por compute_xai_all_xtest.

    En compute_xai_all_xtest, la salida principal está en:
        xai_result["relevance_maps"]

    donde relevance_maps tiene forma:
        (n_samples, ...)
    """

    if isinstance(xai_result, np.ndarray):
        return xai_result

    if isinstance(xai_result, dict):

        if "relevance_maps" in xai_result:
            return np.asarray(xai_result["relevance_maps"])

        if "importance" in xai_result:
            return np.asarray(xai_result["importance"])

        possible_keys = [
            "importance_matrix",
            "attributions",
            "saliency",
            "scores",
            "values",
            "xai_values",
        ]

        for key in possible_keys:
            if key in xai_result:
                return np.asarray(xai_result[key])

    raise ValueError(
        "No fue posible identificar la matriz de relevancia. "
        "Se esperaba la clave 'relevance_maps' en la salida de compute_xai_all_xtest."
    )

def make_xai_filename(
    results_dir,
    window_name,
    subject_id,
    fold,
    model_name,
    method_name,
    aux,
):
    safe_method = (
        method_name
        .replace("+", "plus")
        .replace(" ", "_")
    )

    output_dir = os.path.join(
        results_dir,
        f"attributions_{aux}",
        "MI",
        str(window_name),
    )

    filename = (
        f"sj{subject_id}"
        f"_fold{fold}"
        f"_{model_name}"
        f"_{safe_method}.npz"
    )

    return os.path.join(output_dir, filename)

def make_tdah_xai_filename(
    results_dir,
    fold,
    model_name,
    method_name,
    aux,
    partition_id=None,
):
    safe_method = (
        method_name
        .replace("+", "plus")
        .replace(" ", "_")
    )

    output_dir = os.path.join(
        results_dir,
       f"attributions_{aux}",
        "TDAH",
    )

    if partition_id is None:
        filename = (
            f"fold{fold}"
            f"_{model_name}"
            f"_{safe_method}.npz"
        )
    else:
        filename = (
            f"fold{fold}"
            f"_part{partition_id}"
            f"_{model_name}"
            f"_{safe_method}.npz"
        )

    return os.path.join(output_dir, filename)

def get_background_size(method_name, n_samples):
    if method_name == "KernelSHAP":
        background_size = min(100, max(8, int(0.05 * n_samples)))

    elif method_name == "LIME":
        background_size = min(200, max(30, int(0.10 * n_samples)))

    else:
        return None

    return min(background_size, n_samples)

def get_stf_kernelshap_time_segments(
    window_name=None,
):
    if window_name == "0-7":
        return [
            (0.0, 2.0),
            (2.0, 2.5),
            (2.5, 5.0),
            (5.0, 7.0),
        ]

    return None

def get_stf_kernelshap_freq_bands(
    window_name=None,
):
    if window_name is None:
        return [
            (0.5, 4.0),
            (4.0, 8.0),
            (8.0, 13.0),
            (13.0, 30.0),
            (30.0, 40.0),
        ]

    return [
        (4.0, 8.0),
        (8.0, 13.0),
        (13.0, 30.0),
        (30.0, 40.0),
    ]

def get_adaptive_xai_params(
    method_name,
    n_samples,
    fs,
    model_name=None,
    window_seconds=1.0,
    stride_seconds=0.25,
    predict_batch_size=512,
    window_name=None,
    random_state=42,
):

    if method_name == "KernelSHAP":
        return {
            "background_size": get_background_size(method_name, n_samples),
            "nsamples": 500,
            "l1_reg": "num_features(200)",
            "predict_batch_size": predict_batch_size,
            "random_state": random_state,
        }

    elif method_name == "STF-KernelSHAP":
        return {
            "fs": fs,
            "time_segments_sec": get_stf_kernelshap_time_segments(
                window_name=window_name,
            ),
            "freq_bands_hz": get_stf_kernelshap_freq_bands(
                window_name=window_name,
            ),
            "nfft": 512,
            "nsamples": 500,
            "l1_reg": "num_features(200)",
            "baseline_tf": None,
            "predict_batch_size": predict_batch_size,
            "random_state": random_state,
            "silent": True,
        }

    elif method_name == "LIME":
        return {
            "background_size": get_background_size(method_name, n_samples),
            "num_features": 200,
            "num_samples": 1000,
            "predict_batch_size": predict_batch_size,
            "random_state": random_state,
        }

    elif method_name == "Occlusion":
        return {
            "fs": fs,
            "window_seconds": window_seconds,
            "stride_seconds": stride_seconds,
            "baseline_value": None,
            "background_size": get_background_size("LIME", n_samples),
            "occ_batch_size": 256,
            "predict_batch_size": predict_batch_size,
            "random_state": random_state,
        }

    elif method_name == "IntegratedGradients":
        return {
            "baseline": None,
            "background_size": get_background_size("LIME", n_samples),
            "steps": 50,
            "batch_size": 1,
            "random_state": random_state,
        }
        
    elif method_name == "GradCAM++":

        if model_name is not None and model_name.lower() == "shallowconvnet":
            return {
                "layer_name": "Conv2D_1",
            }

        return {
            "layer_name": None,
        }

    else:
        raise ValueError(f"Método XAI no reconocido: {method_name}")

def run_mi_xai_and_save(
    mi_data,
    mi_subjects_to_extract,
    results_dir="Results",
    models_mi_root="Models/MI",
    models_to_run=None,
    xai_methods_to_run=None,
    selected_windows=None,
    selected_subjects=None,
    overwrite=False,
    use_y_test=False,
    predict_batch_size=512,
    sample_indices=None,
    hardware="NVIDIA T4 GPU",
    save_attributions=True,
):
    """
    Ejecuta XAI para MI y guarda únicamente la matriz de importancia
    por cada combinación:

    ventana - sujeto - fold - modelo - método XAI.

    use_y_test=False:
        Usa las predicciones del modelo para calcular la atribución.

    use_y_test=True:
        Usa las etiquetas reales y_test para calcular la atribución.
    """

    os.makedirs(results_dir, exist_ok=True)

    for window_name, subjects_folds in mi_subjects_to_extract.items():

        if selected_windows is not None and window_name not in selected_windows:
            continue

        for subject_id, fold_to_extract in subjects_folds.items():

            if selected_subjects is not None:

                if isinstance(selected_subjects, dict):
                    valid_subjects = selected_subjects.get(window_name, [])
                    if subject_id not in valid_subjects:
                        continue

                elif isinstance(selected_subjects, list):
                    if subject_id not in selected_subjects:
                        continue

                else:
                    raise ValueError(
                        "selected_subjects debe ser None, list o dict."
                    )

            print("\n" + "#" * 90)
            print(
                f"Ventana: {window_name} | "
                f"Sujeto: {subject_id} | "
                f"Fold: {fold_to_extract}"
            )
            print("#" * 90)

            mi_data_case = mi_data[window_name][subject_id]

            X_train_mi = mi_data_case["X_train"].astype(np.float32)

            X_test_mi = mi_data_case["X_test"].astype(np.float32)
            y_test_mi = mi_data_case["y_test"]
            y_train_mi = mi_data_case["y_train"]

            n_samples = X_test_mi.shape[0]
            if sample_indices is None:
                sample_indices_case = None
            else:
                sample_indices_case = np.asarray(sample_indices, dtype=int)
                sample_indices_case = sample_indices_case[
                    sample_indices_case < n_samples
                ]
                if sample_indices_case.size == 0:
                    raise ValueError(
                        "sample_indices no contiene índices válidos "
                        f"para X_test con {n_samples} muestras."
                    )

            print(f"X_test shape: {X_test_mi.shape}")
            print(f"N muestras: {n_samples}")

            for mi_model_name in models_to_run:

                print("\n" + "=" * 80)
                print(f"Cargando modelo: {mi_model_name}")
                print("=" * 80)

                mi_model, _, mi_y_pred = load_mi_model_and_predict(
                    window_name=window_name,
                    subject_id=subject_id,
                    fold=fold_to_extract,
                    model_name=mi_model_name,
                    X_test=X_test_mi,
                    models_mi_root=models_mi_root,
                )

                xai_labels = y_test_mi if use_y_test else mi_y_pred
                label_source = "y_test" if use_y_test else "y_pred"

                print(f"Etiquetas usadas para XAI: {label_source}")

                for method_name in xai_methods_to_run:
                    output_path = make_xai_filename(
                        results_dir=results_dir,
                        window_name=window_name,
                        subject_id=subject_id,
                        fold=fold_to_extract,
                        model_name=mi_model_name,
                        method_name=method_name,
                        aux=label_source,
                    )

                    if save_attributions and os.path.exists(output_path) and not overwrite:
                        print(f"[SKIP] Ya existe: {output_path}")
                        continue

                    print("\n" + "-" * 80)
                    print(
                        f"Modelo: {mi_model_name} | "
                        f"Método: {method_name} | "
                        f"Ventana: {window_name} | "
                        f"Sujeto: {subject_id}"
                    )
                    print("-" * 80)

                    xai_params = get_adaptive_xai_params(
                        method_name=method_name,
                        n_samples=n_samples,
                        fs=128,
                        model_name=mi_model_name,
                        window_name=window_name,
                        predict_batch_size=predict_batch_size,
                    )

                    print(f"Parámetros XAI: {xai_params}")

                    try:
                        result = compute_xai_all_xtest(
                            model=mi_model,
                            X_test=X_test_mi,
                            X_train=X_train_mi,
                            y_train=y_train_mi,
                            y_pred=xai_labels,
                            method=method_name,
                            sample_indices=sample_indices_case,
                            label_source=label_source,
                            **xai_params,
                        )

                        importance_matrix = extract_importance_matrix(result)

                        if save_attributions:
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            np.savez_compressed(
                                output_path,
                                importance=importance_matrix.astype(np.float32),
                            )
                            print(f"[OK] Guardado en: {output_path}")
                        else:
                            print("[OK] Atribuciones calculadas solo para timing.")
                        print(f"Importance shape: {importance_matrix.shape}")
                        timing_log_path = write_xai_timing_log(
                            results_dir=results_dir,
                            result=result,
                            paradigm="MI",
                            window=window_name,
                            subject=subject_id,
                            fold=fold_to_extract,
                            model_name=mi_model_name,
                            label_source=label_source,
                            hardware=hardware,
                            method_params=xai_params,
                        )
                        if timing_log_path is not None:
                            print(f"[OK] Timing log: {timing_log_path}")

                    except Exception as e:
                        print(
                            f"[ERROR] Falló XAI | ventana={window_name} | "
                            f"sujeto={subject_id} | fold={fold_to_extract} | "
                            f"modelo={mi_model_name} | método={method_name}"
                        )
                        print(f"Detalle: {e}")

                    finally:
                        if "result" in locals():
                            del result

                        if "importance_matrix" in locals():
                            del importance_matrix

                        gc.collect()

                del mi_model
                tf.keras.backend.clear_session()
                gc.collect()

            del X_train_mi
            del X_test_mi
            del y_test_mi
            gc.collect()


base_model_args_tdah = {
    "nb_classes": 2,
    "Chans": 19,
    "Samples": 512,
}


def run_tdah_xai_and_save(
    tdah_data_by_fold,
    folds_to_extract,
    results_dir="Results",
    models_tdah_root="Models/TDAH",
    models_to_run=None,
    xai_methods_to_run=None,
    selected_folds=None,
    overwrite=False,
    use_y_test=False,
    predict_batch_size=512,
    n_partitions=None,
    partition_id=None,
    sample_indices=None,
    hardware="NVIDIA T4 GPU",
    save_attributions=True,
):
    """
    Ejecuta XAI para TDAH y guarda únicamente la matriz de relevancia
    por combinación:

    fold - modelo - método XAI.

    Si n_partitions=None:
        Usa todo X_test, igual que la versión original.

    Si n_partitions es un entero:
        Divide X_test en n_partitions partes consecutivas, NO aleatorias.

    use_y_test=False:
        Usa las predicciones del modelo para calcular la atribución.

    use_y_test=True:
        Usa las etiquetas reales y_test para calcular la atribución.
    """

    os.makedirs(results_dir, exist_ok=True)

    if n_partitions is not None:
        if partition_id is None:
            raise ValueError("Si usas n_partitions, debes indicar partition_id.")

        if not isinstance(n_partitions, int) or n_partitions <= 0:
            raise ValueError("n_partitions debe ser un entero positivo.")

        if not isinstance(partition_id, int):
            raise ValueError("partition_id debe ser un entero.")

        if partition_id < 1 or partition_id > n_partitions:
            raise ValueError(
                f"partition_id debe estar entre 1 y {n_partitions}."
            )

    for fold_to_extract in folds_to_extract:

        if selected_folds is not None and fold_to_extract not in selected_folds:
            continue

        print("\n" + "#" * 90)
        print(f"TDAH | Fold: {fold_to_extract}")
        print("#" * 90)

        tdah_data = tdah_data_by_fold[fold_to_extract]

        X_train_tdah = tdah_data["X_train"].astype(np.float32)

        X_test_tdah_full = tdah_data["X_test"].astype(np.float32)
        y_test_tdah_full = tdah_data["y_test"]
        y_train_tdah = tdah_data["y_train"]

        n_samples_full = X_test_tdah_full.shape[0]

        print(f"X_test original shape: {X_test_tdah_full.shape}")
        print(f"N muestras originales: {n_samples_full}")

        if n_partitions is not None:
            sample_indices_parts = np.array_split(
                np.arange(n_samples_full),
                n_partitions,
            )

            selected_indices = sample_indices_parts[partition_id - 1]

            X_test_tdah = X_test_tdah_full[selected_indices]
            y_test_tdah = y_test_tdah_full[selected_indices]

            print(
                f"Partición usada: {partition_id}/{n_partitions} | "
                f"Índices: {selected_indices[0]}-{selected_indices[-1]} | "
                f"N partición: {len(selected_indices)}"
            )

        else:
            X_test_tdah = X_test_tdah_full
            y_test_tdah = y_test_tdah_full

            print("Partición usada: ninguna, se usa todo X_test.")

        n_samples = X_test_tdah.shape[0]
        if sample_indices is None:
            sample_indices_case = None
        else:
            sample_indices_case = np.asarray(sample_indices, dtype=int)
            sample_indices_case = sample_indices_case[
                sample_indices_case < n_samples
            ]
            if sample_indices_case.size == 0:
                raise ValueError(
                    "sample_indices no contiene índices válidos "
                    f"para X_test con {n_samples} muestras."
                )

        print(f"X_test usado shape: {X_test_tdah.shape}")
        print(f"N muestras usadas: {n_samples}")

        for tdah_model_name in models_to_run:

            print("\n" + "=" * 80)
            print(f"Cargando modelo TDAH: {tdah_model_name}")
            print("=" * 80)

            journal_file = (
                f"{models_tdah_root}/Optuna/"
                f"study_{tdah_model_name}_TDAH.journal"
            )
            study_name = f"study_{tdah_model_name}_TDAH"
            weights_file = (
                f"{models_tdah_root}/Models/"
                f"{tdah_model_name}_fold_{fold_to_extract}.weights.h5"
            )

            tdah_model, _, tdah_y_pred = load_tdah_model_and_predict(
                model_name=tdah_model_name,
                journal_file=journal_file,
                study_name=study_name,
                weights_file=weights_file,
                X_test=X_test_tdah,
                base_model_args=base_model_args_tdah,
            )

            xai_labels = y_test_tdah if use_y_test else tdah_y_pred
            label_source = "y_test" if use_y_test else "y_pred"

            print(f"Etiquetas usadas para XAI: {label_source}")

            for method_name in xai_methods_to_run:

                output_path = make_tdah_xai_filename(
                    results_dir=results_dir,
                    fold=fold_to_extract,
                    model_name=tdah_model_name,
                    method_name=method_name,
                    partition_id=partition_id if n_partitions is not None else None,
                    aux=label_source,
                )

                if save_attributions and os.path.exists(output_path) and not overwrite:
                    print(f"[SKIP] Ya existe: {output_path}")
                    continue

                print("\n" + "-" * 80)
                print(
                    f"TDAH | Fold: {fold_to_extract} | "
                    f"Modelo: {tdah_model_name} | "
                    f"Método: {method_name}"
                )

                if n_partitions is not None:
                    print(
                        f"Partición: {partition_id}/{n_partitions} | "
                        f"N muestras: {n_samples}"
                    )

                print("-" * 80)
                xai_params = get_adaptive_xai_params(
                      method_name=method_name,
                      n_samples=n_samples,
                      fs=128,
                      model_name=tdah_model_name,
                      window_name=None,
                      predict_batch_size=predict_batch_size,
                  )

                print(f"Parámetros XAI: {xai_params}")

                try:
                    result = compute_xai_all_xtest(
                        model=tdah_model,
                        X_test=X_test_tdah,
                        X_train=X_train_tdah,          
                        y_train=y_train_tdah,
                        y_pred=xai_labels,
                        method=method_name,
                        sample_indices=sample_indices_case,
                        label_source=label_source,
                        **xai_params,
                    )

                    importance_matrix = extract_importance_matrix(result)

                    if save_attributions:
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        np.savez_compressed(
                            output_path,
                            importance=importance_matrix.astype(np.float32),
                        )
                        print(f"[OK] Guardado en: {output_path}")
                    else:
                        print("[OK] Atribuciones calculadas solo para timing.")
                    print(f"Importance shape: {importance_matrix.shape}")
                    timing_log_path = write_xai_timing_log(
                        results_dir=results_dir,
                        result=result,
                        paradigm="ADHD",
                        window="TDAH",
                        subject=None,
                        fold=fold_to_extract,
                        model_name=tdah_model_name,
                        label_source=label_source,
                        hardware=hardware,
                        method_params=xai_params,
                        partition_id=partition_id if n_partitions is not None else None,
                        n_partitions=n_partitions,
                    )
                    if timing_log_path is not None:
                        print(f"[OK] Timing log: {timing_log_path}")

                except Exception as e:
                    print(
                        f"[ERROR] Falló XAI | TDAH | "
                        f"fold={fold_to_extract} | "
                        f"modelo={tdah_model_name} | método={method_name}"
                    )
                    print(f"Detalle: {e}")

                finally:
                    if "result" in locals():
                        del result

                    if "importance_matrix" in locals():
                        del importance_matrix

                    gc.collect()

            del tdah_model
            tf.keras.backend.clear_session()
            gc.collect()

        del X_train_tdah
        del X_test_tdah
        del y_test_tdah
        del X_test_tdah_full
        del y_test_tdah_full
        gc.collect()
