
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable
import time

import numpy as np
import shap
import tensorflow as tf

try:
    from shap.explainers._kernel import KernelExplainer as OriginalKernelExplainer
except Exception:
    OriginalKernelExplainer = shap.KernelExplainer


def _sync_accelerator():
    try:
        tf.experimental.async_wait()
    except Exception:
        pass

def get_classification_output(
    y,
    output_key="out_activation",
):
    return extract_model_output(
        y,
        output_key=output_key,
    )


def predict_proba(
    model,
    X,
    verbose=0,
    predict_batch_size=None,
):
    predict_kwargs = {
        "verbose": verbose,
    }

    if predict_batch_size is not None:
        predict_kwargs["batch_size"] = predict_batch_size

    y = model.predict(
        X,
        **predict_kwargs,
    )

    if isinstance(y, dict):
        return y["out_activation"]

    if isinstance(y, (list, tuple)):
        return y[0]

    return y


def get_sample(X_test, sample_idx):
    return X_test[sample_idx:sample_idx + 1].astype(np.float32)


def prepare_eeg_sample_for_transform(x):
    x = np.asarray(x, dtype=np.float32)
    input_ndim = x.ndim

    if x.ndim == 3:
        return x[0], input_ndim

    if x.ndim == 4 and x.shape[-1] == 1:
        return x[0, :, :, 0], input_ndim

    raise ValueError(
        "La muestra debe tener forma [1, C, T] o [1, C, T, 1]. "
        f"Forma recibida: {x.shape}"
    )


def restore_batch_shape_from_ct(X_ct_batch, input_ndim):
    X_ct_batch = np.asarray(X_ct_batch, dtype=np.float32)

    if input_ndim == 3:
        return X_ct_batch

    if input_ndim == 4:
        return X_ct_batch[..., None]

    raise ValueError(f"input_ndim no soportado: {input_ndim}")

@dataclass
class SegmentFFTConfig:
    fs: float = 128.0
    time_segments_sec: List[Tuple[float, float]] = field(default_factory=list)
    freq_bands_hz: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (4.0, 8.0),
            (8.0, 13.0),
            (13.0, 30.0),
            (30.0, 40.0),
        ]
    )
    nfft: int = 64


class SegmentFFTTransform:

    def __init__(self, config: SegmentFFTConfig):
        self.config = config

        self.last_freqs_ = np.fft.rfftfreq(
            self.config.nfft,
            d=1.0 / self.config.fs,
        )

        self.last_segment_indices_ = None
        self.last_original_length_ = None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)

        if x.ndim != 2:
            raise ValueError(f"x debe ser [C, T], recibido {x.shape}")

        C, T = x.shape
        self.last_original_length_ = T

        duration_sec = T / self.config.fs
        t = np.arange(T) / self.config.fs

        segment_info = self._build_time_segment_indices(
            t=t,
            duration_sec=duration_sec,
        )

        self.last_segment_indices_ = segment_info

        S = len(segment_info)
        K = len(self.last_freqs_)

        out = np.zeros((C, S, K), dtype=np.complex128)

        for s, seg in enumerate(segment_info):
            idx = seg["sample_indices"]
            x_seg = x[:, idx]

            X_seg = np.fft.rfft(
                x_seg,
                n=self.config.nfft,
                axis=1,
            )

            out[:, s, :] = X_seg

        return out

    def inverse(
        self,
        x_tf,
        original_length: Optional[int] = None,
    ):
        x_tf = np.asarray(x_tf)

        if x_tf.ndim != 3:
            raise ValueError(f"x_tf debe ser [C, S, K], recibido {x_tf.shape}")

        if self.last_segment_indices_ is None:
            raise RuntimeError("Primero debes ejecutar forward(...).")

        C, S, K = x_tf.shape

        if K != len(self.last_freqs_):
            raise ValueError(
                f"K={K} no coincide con el eje frecuencial esperado "
                f"K={len(self.last_freqs_)}."
            )

        if S != len(self.last_segment_indices_):
            raise ValueError(
                f"S={S} no coincide con los segmentos esperados "
                f"S={len(self.last_segment_indices_)}."
            )

        T = original_length if original_length is not None else self.last_original_length_

        if T is None:
            raise RuntimeError("No se conoce original_length.")

        rec = np.zeros((C, T), dtype=np.float32)

        for s, seg in enumerate(self.last_segment_indices_):
            idx = seg["sample_indices"]
            L = len(idx)

            x_seg_rec = np.fft.irfft(
                x_tf[:, s, :],
                n=self.config.nfft,
                axis=1,
            )[:, :L]

            rec[:, idx] = np.real(x_seg_rec).astype(np.float32)

        return rec

    def _build_time_segment_indices(self, t, duration_sec):
        segment_info = []

        for seg_id, (t_start, t_end) in enumerate(
            self.config.time_segments_sec
        ):
            if t_start >= t_end:
                raise ValueError(
                    f"Segmento temporal inválido: ({t_start}, {t_end})"
                )

            if t_start >= duration_sec:
                continue

            t_end_eff = min(t_end, duration_sec)

            if t_start >= t_end_eff:
                continue

            idx = np.where((t >= t_start) & (t < t_end_eff))[0]

            if len(idx) == 0:
                continue

            segment_info.append(
                {
                    "segment_id": len(segment_info),
                    "t_start_sec": float(t_start),
                    "t_end_sec": float(t_end_eff),
                    "sample_start": int(idx[0]),
                    "sample_end": int(idx[-1] + 1),
                    "sample_indices": idx.copy(),
                    "num_samples": int(len(idx)),
                }
            )

        if len(segment_info) == 0:
            raise ValueError(
                "Ningún segmento temporal contiene muestras válidas."
            )

        return segment_info

    def get_frequency_axis_hz(self):
        return self.last_freqs_

    def get_time_segments_info(self):
        if self.last_segment_indices_ is None:
            raise RuntimeError("Primero ejecuta forward(...).")
        return self.last_segment_indices_

@dataclass
class STFCell:
    cell_id: int
    time_label: str
    freq_label: str
    segment_idx: int
    freq_idx_start: int
    freq_idx_end: int


class STFCellPartition:

    def __init__(
        self,
        time_intervals_sec,
        freq_bands_hz,
        time_labels=None,
        freq_labels=None,
    ):
        self.time_intervals_sec = time_intervals_sec
        self.freq_bands_hz = freq_bands_hz
        self.time_labels = time_labels
        self.freq_labels = freq_labels
        self.cells_ = None

    def build_cells(
        self,
        segment_info,
        freqs_hz,
    ):
        if self.time_labels is None:
            self.time_labels = [
                f"T{i}" for i in range(len(self.time_intervals_sec))
            ]

        if self.freq_labels is None:
            self.freq_labels = [
                f"F{i}" for i in range(len(self.freq_bands_hz))
            ]

        if len(self.time_labels) != len(self.time_intervals_sec):
            raise ValueError(
                "time_labels y time_intervals_sec deben tener la misma longitud."
            )

        if len(self.freq_labels) != len(self.freq_bands_hz):
            raise ValueError(
                "freq_labels y freq_bands_hz deben tener la misma longitud."
            )

        cells = []
        cell_id = 0

        for seg in segment_info:
            t0 = float(seg["t_start_sec"])
            t1 = float(seg["t_end_sec"])
            segment_idx = int(seg["segment_id"])

            try:
                time_idx = self.time_intervals_sec.index((t0, t1))
            except ValueError:
                time_idx = segment_idx

            for j, (f0, f1) in enumerate(self.freq_bands_hz):
                if f0 >= f1:
                    raise ValueError(f"Banda inválida: ({f0}, {f1})")

                fidx = np.where((freqs_hz >= f0) & (freqs_hz <= f1))[0]

                if len(fidx) == 0:
                    continue

                cells.append(
                    STFCell(
                        cell_id=cell_id,
                        time_label=self.time_labels[time_idx],
                        freq_label=self.freq_labels[j],
                        segment_idx=segment_idx,
                        freq_idx_start=int(fidx[0]),
                        freq_idx_end=int(fidx[-1] + 1),
                    )
                )

                cell_id += 1

        if len(cells) == 0:
            raise ValueError(
                "No se construyó ninguna celda STF válida."
            )

        self.cells_ = cells

        return cells

    @property
    def cells(self):
        if self.cells_ is None:
            raise RuntimeError("Primero debes llamar build_cells(...).")
        return self.cells_

class STFKernelSHAPExplainer(OriginalKernelExplainer):

    def __init__(
        self,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        x_n,
        transform,
        partition,
        baseline_tf=None,
        link="identity",
        feature_name_prefix="stf",
    ):
        x_n = np.asarray(x_n, dtype=np.float32)

        if x_n.ndim != 2:
            raise ValueError(f"x_n debe ser [C, T], recibido {x_n.shape}")

        self.model_predict_fn = model_predict_fn
        self.x_n = x_n
        self.transform = transform
        self.partition = partition

        self.x_tf = self.transform.forward(self.x_n)

        self.C, self.S, self.K = self.x_tf.shape
        self.T = self.x_n.shape[1]

        self.partition.build_cells(
            segment_info=self.transform.get_time_segments_info(),
            freqs_hz=self.transform.get_frequency_axis_hz(),
        )

        self.Q = len(self.partition.cells)
        self.M_tilde = self.C * self.Q

        if baseline_tf is None:
            self.baseline_tf = np.zeros_like(self.x_tf)
        else:
            baseline_tf = np.asarray(baseline_tf)

            if baseline_tf.shape != self.x_tf.shape:
                raise ValueError(
                    f"baseline_tf debe tener forma {self.x_tf.shape}, "
                    f"recibido {baseline_tf.shape}."
                )

            self.baseline_tf = baseline_tf

        self.background_z = np.zeros((1, self.M_tilde), dtype=float)

        self.feature_names = [
            f"{feature_name_prefix}_{c}_{q}"
            for c in range(self.C)
            for q in range(self.Q)
        ]

        super().__init__(
            model=self._coalition_model,
            data=self.background_z,
            feature_names=self.feature_names,
            link=link,
        )

    def _h_q(self, z_tilde):
        z_tilde = np.asarray(z_tilde, dtype=float)

        if z_tilde.ndim != 1 or z_tilde.size != self.M_tilde:
            raise ValueError(
                f"z_tilde debe tener tamaño {self.M_tilde}, "
                f"recibido {z_tilde.shape}."
            )

        return z_tilde.reshape(self.C, self.Q)

    def _h_tf(self, z_cq):
        mask_tf = np.zeros((self.C, self.S, self.K), dtype=float)

        for q, cell in enumerate(self.partition.cells):
            mask_tf[
                :,
                cell.segment_idx,
                cell.freq_idx_start:cell.freq_idx_end,
            ] = z_cq[:, q][:, None]

        return mask_tf

    def coalition_to_signal(self, z_tilde):
        z_cq = self._h_q(z_tilde)
        mask_tf = self._h_tf(z_cq)

        x_tf_masked = (
            mask_tf * self.x_tf
            + (1.0 - mask_tf) * self.baseline_tf
        )

        x_rec = self.transform.inverse(
            x_tf_masked,
            original_length=self.T,
        )

        return x_rec.astype(np.float32)

    def _coalition_model(self, z_batch):
        z_batch = np.asarray(z_batch, dtype=float)

        if z_batch.ndim == 1:
            z_batch = z_batch[None, :]

        x_batch = np.stack(
            [
                self.coalition_to_signal(z_batch[i])
                for i in range(z_batch.shape[0])
            ],
            axis=0,
        )
        #self.z_bath1=z_batch

        scores = self.model_predict_fn(x_batch)
        scores = np.asarray(scores, dtype=float)

        if scores.ndim == 1:
            scores = scores[:, None]

        if scores.ndim != 2:
            raise ValueError(
                f"El modelo debe devolver [N, num_outputs], recibido {scores.shape}"
            )

        return scores

    def explain_stf(
        self,
        nsamples="auto",
        l1_reg="num_features(10)",
        silent=True,
    ):
        x_full = np.ones((1, self.M_tilde), dtype=float)

        shap_values = self.shap_values(
            x_full,
            nsamples=nsamples,
            l1_reg=l1_reg,
            silent=silent,
        )

        if isinstance(shap_values, list):
            phi = np.stack(
                [sv[0] for sv in shap_values],
                axis=0,
            )

        else:
            shap_values = np.asarray(shap_values)

            if shap_values.ndim == 3:
                phi = np.transpose(shap_values[0], (1, 0))

            elif shap_values.ndim == 2:
                phi = shap_values[0][None, :]

            else:
                raise ValueError(
                    f"Formato inesperado de shap_values: {shap_values.shape}"
                )

        attribution_cq = phi.reshape(phi.shape[0], self.C, self.Q)

        attribution_tf = np.zeros(
            (phi.shape[0], self.C, self.S, self.K),
            dtype=np.float32,
        )

        for c_out in range(phi.shape[0]):
            for q, cell in enumerate(self.partition.cells):
                attribution_tf[
                    c_out,
                    :,
                    cell.segment_idx,
                    cell.freq_idx_start:cell.freq_idx_end,
                ] = attribution_cq[c_out, :, q][:, None]

        return attribution_tf.astype(np.float32)
def extract_model_output(
    y,
    output_key="out_activation",
):
    """
    Extrae una salida numérica del resultado de model.predict(...).

    Soporta:
        - np.ndarray
        - list / tuple
        - dict, como en TGARNet: {"out_activation": ...}
    """

    if isinstance(y, dict):
        if output_key in y:
            return y[output_key]

        for value in y.values():
            if hasattr(value, "shape") and len(value.shape) >= 2:
                return value

        return list(y.values())[0]

    if isinstance(y, (list, tuple)):
        for value in y:
            if hasattr(value, "shape") and len(value.shape) >= 2:
                return value

        return y[0]

    return y

def get_score_model(
    model,
    use_logits=False,
    output_layer_name="out_activation",
):
    """
    Devuelve el modelo que se usará para calcular scores.

    Si use_logits=False:
        devuelve el modelo original.

    Si use_logits=True:
        devuelve un modelo auxiliar cuya salida son los logits pre-softmax.
    """

    if use_logits:
        return make_logits_model(
            model=model,
            output_layer_name=output_layer_name,
        )

    return model

def predict_scores_from_model(
    score_model,
    X,
    verbose=0,
    predict_batch_size=None,
    output_key="out_activation",
):
    """
    Ejecuta predicción usando un modelo de scores ya construido.

    Esta función NO construye el modelo de logits.
    Solo usa score_model.predict(...).
    """

    predict_kwargs = {
        "verbose": verbose,
    }

    if predict_batch_size is not None:
        predict_kwargs["batch_size"] = predict_batch_size

    y = score_model.predict(
        X,
        **predict_kwargs,
    )

    y = get_classification_output(
        y,
        output_key=output_key,
    )

    return np.asarray(
        y,
        dtype=np.float32,
    )

def make_logits_model(
    model,
    output_layer_name="out_activation",
    logits_layer_name=None,
):
    """
    Construye un modelo cuya salida corresponde a los logits pre-softmax.

    Soporta:
        Caso 1:
            Dense(..., activation="softmax", name="out_activation")

        Caso 2:
            Dense(..., activation=None, name="output")
            Activation("softmax", name="out_activation")
    """

    output_layer = model.get_layer(output_layer_name)

    if logits_layer_name is None:
        logits_layer_name = f"{output_layer_name}_logits"

    # ============================================================
    # Caso 1: Dense con softmax
    # ============================================================
    if isinstance(output_layer, tf.keras.layers.Dense):

        if output_layer.activation != tf.keras.activations.softmax:
            raise ValueError(
                f"La capa Dense '{output_layer_name}' no usa softmax."
            )

        pre_output = output_layer.input

        logits = tf.keras.layers.Dense(
            units=output_layer.units,
            activation=None,
            use_bias=output_layer.use_bias,
            kernel_constraint=output_layer.kernel_constraint,
            bias_constraint=output_layer.bias_constraint,
            kernel_regularizer=output_layer.kernel_regularizer,
            bias_regularizer=output_layer.bias_regularizer,
            name=logits_layer_name,
        )(pre_output)

        logits_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=logits,
            name=f"{model.name}_logits",
        )

        logits_layer = logits_model.get_layer(
            logits_layer_name
        )

        logits_layer.set_weights(
            output_layer.get_weights()
        )

        return logits_model

    # ============================================================
    # Caso 2: Activation softmax separada
    # ============================================================
    if isinstance(output_layer, tf.keras.layers.Activation):

        if output_layer.activation != tf.keras.activations.softmax:
            raise ValueError(
                f"La capa Activation '{output_layer_name}' no usa softmax."
            )

        logits = output_layer.input

        logits_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=logits,
            name=f"{model.name}_logits",
        )

        return logits_model

    raise ValueError(
        f"La capa '{output_layer_name}' debe ser Dense con softmax "
        f"o Activation('softmax'). Tipo recibido: {type(output_layer)}"
    )


def predict_scores(
    model,
    X,
    use_logits=False,
    output_layer_name="out_activation",
    verbose=0,
    predict_batch_size=None,
):
    """
    Devuelve la salida usada como score para STF-KernelSHAP.

    Esta función se mantiene como wrapper por compatibilidad.

    Para procesos intensivos, es preferible usar:
        get_score_model(...)
        predict_scores_from_model(...)
    """

    score_model = get_score_model(
        model=model,
        use_logits=use_logits,
        output_layer_name=output_layer_name,
    )

    scores = predict_scores_from_model(
        score_model=score_model,
        X=X,
        verbose=verbose,
        predict_batch_size=predict_batch_size,
    )

    return scores
def get_target_class_from_labels(
    labels,
    sample_idx,
):
    """
    Obtiene la clase objetivo desde etiquetas escalares,
    one-hot, probabilidades o logits.

    Soporta:
        labels.shape == (N,)
        labels.shape == (N, C)
    """

    labels = np.asarray(labels)
    label = np.asarray(labels[sample_idx])

    if label.ndim == 0:
        return int(label)

    if label.ndim == 1:
        return int(np.argmax(label))

    raise ValueError(
        f"Etiqueta no soportada para sample_idx={sample_idx}. "
        f"Forma recibida: {label.shape}"
    )
def stf_kernelshap_all_xtest(
    model,
    X_test,
    y_pred=None,
    sample_indices=None,
    fs=128.0,
    time_segments_sec=None,
    freq_bands_hz=None,
    time_labels=None,
    freq_labels=None,
    nfft=64,
    nsamples=400,
    l1_reg="num_features(10)",
    baseline_tf=None,
    predict_batch_size=None,
    random_state=None,
    silent=True,
    use_logits=True,
    output_layer_name="out_activation",
):
    X_test = np.asarray(
        X_test,
        dtype=np.float32,
    )

    if X_test.ndim not in [3, 4]:
        raise ValueError(
            "X_test debe tener forma [N, C, T] o [N, C, T, 1]. "
            f"Forma recibida: {X_test.shape}"
        )

    if X_test.ndim == 4 and X_test.shape[-1] != 1:
        raise ValueError(
            "Si X_test tiene 4 dimensiones, la última debe ser 1. "
            f"Forma recibida: {X_test.shape}"
        )

    if sample_indices is None:
        sample_indices = np.arange(
            X_test.shape[0]
        )

    sample_indices = np.asarray(
        sample_indices,
        dtype=np.int64,
    )

    if time_segments_sec is None:
        signal_duration_sec = X_test.shape[2] / fs

        time_segments_sec = [
            (0.0, float(signal_duration_sec))
        ]

    if freq_bands_hz is None:
        freq_bands_hz = [
            (4.0, 8.0),
            (8.0, 13.0),
            (13.0, 30.0),
            (30.0, 40.0),
        ]

    input_ndim = X_test.ndim

    config = SegmentFFTConfig(
        fs=fs,
        time_segments_sec=time_segments_sec,
        freq_bands_hz=freq_bands_hz,
        nfft=nfft,
    )

    score_model = get_score_model(
        model=model,
        use_logits=use_logits,
        output_layer_name=output_layer_name,
    )

    def model_predict_fn(X_ct_batch):

        X_model = restore_batch_shape_from_ct(
            X_ct_batch,
            input_ndim=input_ndim,
        )

        scores = predict_scores_from_model(
            score_model=score_model,
            X=X_model,
            verbose=0,
            predict_batch_size=predict_batch_size,
            output_key=output_layer_name,
        )

        return scores

    relevance_maps = []
    class_indices = []
    runtime_seconds = []

    for i, sample_idx in enumerate(sample_indices):
        _sync_accelerator()
        start_time = time.perf_counter()

        if random_state is not None:
            np.random.seed(
                random_state + i
            )

        x = get_sample(
            X_test,
            sample_idx,
        )

        x_ct, _ = prepare_eeg_sample_for_transform(
            x
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

        transform = SegmentFFTTransform(
            config
        )

        partition = STFCellPartition(
            time_intervals_sec=time_segments_sec,
            freq_bands_hz=freq_bands_hz,
            time_labels=time_labels,
            freq_labels=freq_labels,
        )

        explainer = STFKernelSHAPExplainer(
            model_predict_fn=model_predict_fn,
            x_n=x_ct,
            transform=transform,
            partition=partition,
            baseline_tf=baseline_tf,
            link="identity",
        )

        attribution_tf = explainer.explain_stf(
            nsamples=nsamples,
            l1_reg=l1_reg,
            silent=silent,
        )

        relevance = attribution_tf[
            class_idx
        ]

        relevance_maps.append(
            relevance.astype(np.float32)
        )

        class_indices.append(
            class_idx
        )
        _sync_accelerator()
        runtime_seconds.append(
            float(time.perf_counter() - start_time)
        )

        print(
            f"[{i + 1}/{len(sample_indices)}] "
            f"STF-KernelSHAP | "
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
        "method": "STF-KernelSHAP",
        "score_type": "logits" if use_logits else "probabilities",
        "sample_indices": sample_indices,
        "class_indices": class_indices,
        "runtime_seconds": np.asarray(runtime_seconds, dtype=np.float64),
        "relevance_maps": relevance_maps,
        "mean_relevance": np.mean(relevance_maps, axis=0),
    }
