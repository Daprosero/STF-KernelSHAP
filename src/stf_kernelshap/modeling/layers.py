"""Custom Keras layers, losses, and callbacks used by STF-KernelSHAP models."""

import numpy as np
import tensorflow as tf
from keras_nlp.layers import TransformerEncoder
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import Loss


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
