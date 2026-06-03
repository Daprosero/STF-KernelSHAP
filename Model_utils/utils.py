import os
import random
import numpy as np
import scipy.io
import tensorflow as tf
import re
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import backend as K, layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import Loss
from keras_nlp.layers import TransformerEncoder
import optuna
from copy import deepcopy

from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import os
import numpy as np
import scipy.io
from scipy.signal import iirnotch, filtfilt
from sklearn.preprocessing import OneHotEncoder


def remove_powerline_50hz(
    senal,
    fs=128,
    freq=50.0,
    Q=30.0,
    axis=-1
):
    """
    Elimina la componente de red eléctrica en 50 Hz usando un filtro notch.

    Args:
        senal: matriz EEG de forma C x T.
        fs: frecuencia de muestreo.
        freq: frecuencia a remover, normalmente 50 Hz.
        Q: factor de calidad del notch. Mayor Q = filtro más estrecho.
        axis: eje temporal.

    Returns:
        senal_filtrada: matriz EEG filtrada de forma C x T.
    """

    nyquist = fs / 2

    if freq >= nyquist:
        raise ValueError(
            f"No se puede filtrar {freq} Hz con fs={fs}, "
            f"porque Nyquist es {nyquist} Hz."
        )

    b, a = iirnotch(
        w0=freq,
        Q=Q,
        fs=fs
    )

    senal_filtrada = filtfilt(
        b,
        a,
        senal,
        axis=axis
    )

    return senal_filtrada


def segmentar_senales(db, labels):
    """
    Divide las señales EEG en segmentos de 512 instantes con un traslape del 50%.

    Args:
        db: diccionario con señales C x T_i.
        labels: etiquetas por sujeto.

    Returns:
        segmentos, y, sbjs, window_ids
    """

    segmento_tamano = 512
    paso = int(segmento_tamano * 0.5)

    i = 0

    segmentos = []
    y = []
    sbjs = []
    window_ids = []

    for sujeto, senal in db.items():
        C, T = senal.shape
        window_count = 1

        for inicio in range(0, T - segmento_tamano + 1, paso):
            segmento = senal[:, inicio:inicio + segmento_tamano]

            segmentos.append(segmento)
            y.append(labels[i])
            sbjs.append(sujeto)
            window_ids.append(f"Window {window_count}")

            window_count += 1

        i += 1

    return np.array(segmentos), np.array(y), sbjs, window_ids


def get_segmented_data(
    path_adhd,
    path_control,
    fs=128,
    apply_notch=True,
    notch_freq=50.0,
    notch_Q=30.0,
    
):
    """
    Carga la base TDAH, elimina opcionalmente la frecuencia de red en 50 Hz,
    segmenta las señales y codifica las etiquetas.
    """

    ruta_carpeta_TDAH = path_adhd

    ruta_carpeta_control = path_control

    sujetos_TDAH = [
        archivo[:-4]
        for archivo in os.listdir(ruta_carpeta_TDAH)
        if archivo.endswith(".mat")
    ]

    sujetos_TDAH.pop()

    sujetos_control = [
        archivo[:-4]
        for archivo in os.listdir(ruta_carpeta_control)
        if archivo.endswith(".mat")
    ]

    diagnostico = {}

    for sbj in sujetos_TDAH:
        diagnostico[sbj] = 1

    for sbj in sujetos_control:
        diagnostico[sbj] = 0

    eeg_tdah = {}

    for sbj in sujetos_TDAH:
        mat_file_path = ruta_carpeta_TDAH + "/" + sbj + ".mat"

        data = scipy.io.loadmat(mat_file_path)
        columna = list(data.keys())[-1]

        senal = data[columna].T  # C x T

        if apply_notch:
            senal = remove_powerline_50hz(
                senal,
                fs=fs,
                freq=notch_freq,
                Q=notch_Q,
                axis=-1
            )

        eeg_tdah[sbj] = senal

    eeg_control = {}

    for sbj in sujetos_control:
        mat_file_path = ruta_carpeta_control + "/" + sbj + ".mat"

        data = scipy.io.loadmat(mat_file_path)
        columna = list(data.keys())[-1]

        senal = data[columna].T  # C x T

        if apply_notch:
            senal = remove_powerline_50hz(
                senal,
                fs=fs,
                freq=notch_freq,
                Q=notch_Q,
                axis=-1
            )

        eeg_control[sbj] = senal

    db = eeg_control | eeg_tdah

    zeros = np.zeros(len(eeg_control))
    ones = np.ones(len(eeg_tdah))
    labels = np.hstack((zeros, ones))

    X, y, sbjs, window_ids = segmentar_senales(db, labels)

    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y.reshape(-1, 1))

    return X, y, sbjs, window_ids


@tf.keras.utils.register_keras_serializable()
class GaussianKernelLayer(Layer):
    def __init__(self, **kwargs):
        super(GaussianKernelLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GaussianKernelLayer, self).build(input_shape)

    def call(self, inputs):
        """
        inputs: [x, sigma]
        x shape: (N, C, T, F)
        sigma: escalar o tensor compatible
        """
        x, sigma = inputs

        N = tf.shape(x)[0]
        C = tf.shape(x)[1]
        T = tf.shape(x)[2]
        F = tf.shape(x)[3]

        # (N, C, T, F) -> (N, F, C, T)
        x = tf.transpose(x, perm=(0, 3, 1, 2))

        # (N, F, C, T) -> (N*F, C, T)
        x_reshaped = tf.reshape(x, (N * F, C, T))

        # Distancias cuadradas por pares entre canales
        squared_differences = (
            tf.expand_dims(x_reshaped, axis=2) - tf.expand_dims(x_reshaped, axis=1)
        )  # (N*F, C, C, T)
        squared_differences = tf.square(squared_differences)
        pairwise_distances_squared = tf.reduce_sum(squared_differences, axis=-1)  # (N*F, C, C)

        # (N*F, C, C) -> (N, F, C, C) -> (N, C, C, F)
        pairwise_distances_squared = tf.reshape(pairwise_distances_squared, (N, F, C, C))
        pairwise_distances_squared = tf.transpose(pairwise_distances_squared, perm=(0, 2, 3, 1))

        gaussian_kernel = tf.exp(-pairwise_distances_squared / (2.0 * tf.square(sigma)))
        return gaussian_kernel
@tf.keras.utils.register_keras_serializable()
class RenyiEntropyLayer(tf.keras.layers.Layer):
    def __init__(self, alpha=2, eps=1e-8, normalize_by_logc=False, **kwargs):
        super(RenyiEntropyLayer, self).__init__(**kwargs)
        self.alpha = alpha
        self.eps = eps
        self.normalize_by_logc = normalize_by_logc

    def call(self, K):
        """
        K: tensor de forma (B, F, C, C)
           En tu caso con un solo kernel: (B, 1, C, C)

        retorna:
           H: tensor de forma (B, F)
        """
        K = tf.cast(K, tf.float32)

        # Normalización por traza
        trace = tf.linalg.trace(K)                          # (B, F)
        trace = tf.maximum(trace, self.eps)
        A = K / trace[..., None, None]                     # (B, F, C, C)

        if self.alpha == 2:
            AAT = tf.matmul(A, A, transpose_b=True)        # (B, F, C, C)
            H = -tf.math.log(tf.maximum(tf.linalg.trace(AAT), self.eps))  # (B, F)
        else:
            eigvals, _ = tf.linalg.eigh(A)                 # (B, F, C)
            eigvals = tf.maximum(tf.math.real(eigvals), self.eps)
            H = tf.math.log(tf.reduce_sum(tf.pow(eigvals, self.alpha), axis=-1)) / (1.0 - self.alpha)

        if self.normalize_by_logc:
            C = tf.cast(tf.shape(K)[-1], tf.float32)
            H = H / tf.math.log(tf.maximum(C, 2.0))

        return H

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "eps": self.eps,
            "normalize_by_logc": self.normalize_by_logc,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
@tf.keras.utils.register_keras_serializable()
class RenyiEntropyRegularizer(tf.keras.losses.Loss):
    def __init__(self, mode="min", name="renyi_entropy_regularizer", **kwargs):
        """
        mode = "min"  -> minimiza entropía
        mode = "max"  -> maximiza entropía
        """
        super().__init__(name=name, **kwargs)
        if mode not in ["min", "max"]:
            raise ValueError("mode debe ser 'min' o 'max'")
        self.mode = mode

    def call(self, y_true, y_pred):
        """
        y_pred: salida de entropía, típicamente (B, 1) o (B, F)
        """
        H = tf.cast(y_pred, tf.float32)
        H = tf.reduce_mean(H, axis=-1)   # (B,)

        if self.mode == "min":
            return H
        else:
            return -H

    def get_config(self):
        config = super().get_config()
        config.update({"mode": self.mode})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def inception_block(x, kernel_sigma):
    """
    Versión de un solo kernel.
    """
    kernel = GaussianKernelLayer(name="gaussian_layer")(
        [x, tf.convert_to_tensor(kernel_sigma, dtype=tf.float32)]
    )


    return kernel

@tf.keras.utils.register_keras_serializable()
class NormalizedBinaryCrossentropy(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]

        cce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        left = tf.tile(tf.expand_dims([1.0, 0.0], axis=0), [batch_size, 1])
        right = tf.tile(tf.expand_dims([0.0, 1.0], axis=0), [batch_size, 1])

        cce_left = tf.keras.losses.binary_crossentropy(left, y_pred)
        cce_right = tf.keras.losses.binary_crossentropy(right, y_pred)

        cce_norm = tf.divide(cce, (cce_left + cce_right))
        return cce_norm


class InspectableMultiHeadAttention(layers.MultiHeadAttention):
    def get_projection_weights(self):
        if not self.built:
            raise ValueError("La capa no ha sido construida.")

        weights_dict = {
            "query": self._query_dense.kernel.numpy(),
            "key": self._key_dense.kernel.numpy(),
            "value": self._value_dense.kernel.numpy(),
            "output": self._output_dense.kernel.numpy(),
        }
        return weights_dict


class InspectableTransformerEncoder(TransformerEncoder):
    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        key_dim = int(hidden_dim // self.num_heads)

        self._self_attention_layer = InspectableMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            value_dim=key_dim,
            dropout=self.dropout,
            name="self_attention_inspectable",
        )

        self._self_attention_layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self._self_attention_dropout = layers.Dropout(rate=self.dropout)
        self._feedforward_intermediate_dense = layers.Dense(
            self.intermediate_dim, activation=self.activation
        )
        self._feedforward_output_dense = layers.Dense(hidden_dim)
        self._feedforward_layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self._feedforward_dropout = layers.Dropout(rate=self.dropout)

        self._last_attention_scores = None
        self.built = True

    def call(self, inputs, padding_mask=None, training=False):
        attention_output, attention_scores = self._self_attention_layer(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=padding_mask,
            training=training,
            return_attention_scores=True,
        )

        self._last_attention_scores = attention_scores

        attention_output = self._self_attention_dropout(attention_output, training=training)
        attention_output = self._self_attention_layer_norm(inputs + attention_output)

        ff_output = self._feedforward_intermediate_dense(attention_output)
        ff_output = self._feedforward_output_dense(ff_output)
        ff_output = self._feedforward_dropout(ff_output, training=training)
        output = self._feedforward_layer_norm(attention_output + ff_output)

        return output

    def get_attention_scores(self):
        if self._last_attention_scores is None:
            raise ValueError(
                "No se han calculado aún los attention scores. Haz un forward pass primero."
            )
        return self._last_attention_scores

    def get_attention_weights(self):
        if not self.built:
            raise ValueError("La capa Encoder no ha sido construida.")
        return self._self_attention_layer.get_projection_weights()

class DynamicSchedule(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, delta=10):
        super().__init__()
        self.total_epochs = total_epochs
        self.delta = delta
        self.lambda_val = 0.0

    def get_lambda(self, epoch):
        p = epoch / self.total_epochs
        return 2*(1 - np.exp(-self.delta * p)) / (1 + np.exp(-self.delta * p))

    def on_epoch_begin(self, epoch, logs=None):
        self.lambda_val = self.get_lambda(epoch)

        if hasattr(self.model, "loss_weights") and isinstance(self.model.loss_weights, dict):
            self.model.loss_weights["out_activation"] = 1.0
            self.model.loss_weights["kernel_entropy"] = self.lambda_val
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
