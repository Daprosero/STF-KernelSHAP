import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from stf_kernelshap.modeling.layers import DynamicSchedule, NormalizedBinaryCrossentropy, RenyiEntropyRegularizer
from stf_kernelshap.modeling.models import EEGNet, ShallowConvNet, TGARNet

def suggest_model_args(trial, model_name, base_model_args):
    model_name = model_name.lower()

    if model_name == "tgarnet":
        return {
            **base_model_args,
            "filters": trial.suggest_int("filters", 2, 8),
            "kernel_sigma": trial.suggest_float("kernel_sigma", 1.0, 20.0),
            "num_heads": trial.suggest_int("num_heads", 1, 5),
            "intermediate_dim": trial.suggest_categorical("intermediate_dim", [16, 32, 64, 128]),
            "dropoutRate": trial.suggest_float("dropoutRate", 0.1, 0.75, step=0.05),
            "norm_rate": trial.suggest_float("norm_rate", 0.1, 0.75, step=0.05),
        }

    elif model_name == "eegnet":
        F1 = trial.suggest_int("F1", 4, 32, step=4)
        D = trial.suggest_int("D", 1, 4)

        return {
            **base_model_args,
            "kernLength": trial.suggest_categorical("kernLength", [16, 32, 64, 96, 128]),
            "F1": F1,
            "D": D,
            "F2": F1 * D,
            "dropoutRate": trial.suggest_float("dropoutRate", 0.1, 0.75, step=0.05),
            "norm_rate": trial.suggest_float("norm_rate", 0.1, 0.75, step=0.05),
        }
    elif model_name == "shallowconvnet":
        pool_size = trial.suggest_int("pool_size", 16, 64, step=8)
        pool_stride = trial.suggest_int("pool_stride", 2, 32, step=2)
        pool_stride = min(pool_stride, pool_size)
    
        return {
            **base_model_args,
            "n_filters": trial.suggest_int("n_filters", 8, 64, step=8),
            "kernel_length": trial.suggest_int("kernel_length", 8, 128, step=8),
            "pool_size": pool_size,
            "pool_stride": pool_stride,
            "dropoutRate": trial.suggest_float("dropoutRate", 0.1, 0.75, step=0.05),
            "norm_rate": trial.suggest_float("norm_rate", 0.1, 0.75, step=0.05),
        }

    else:
        raise ValueError(
            "model_name must be one of: ['eegnet', 'shallowconvnet', 'tgarnet']"
        )

def build_eeg_model(model_name, **kwargs):
    model_name = model_name.lower()

    nb_classes = kwargs.pop("nb_classes")
    Chans = kwargs.pop("Chans")
    Samples = kwargs.pop("Samples")

    if model_name == "eegnet":
        return EEGNet(
            nb_classes=nb_classes,
            Chans=Chans,
            Samples=Samples,
            dropoutRate=kwargs["dropoutRate"],
            kernLength=kwargs["kernLength"],
            F1=kwargs["F1"],
            D=kwargs["D"],
            F2=kwargs["F2"],
            norm_rate=kwargs["norm_rate"],
        )

    elif model_name == "shallowconvnet":
        return ShallowConvNet(
            nb_classes=nb_classes,
            Chans=Chans,
            Samples=Samples,
            dropoutRate=kwargs["dropoutRate"],
            n_filters=kwargs["n_filters"],
            kernel_length=kwargs["kernel_length"],
            pool_size=kwargs["pool_size"],
            pool_stride=kwargs["pool_stride"],
            norm_rate=kwargs["norm_rate"],
        )

    elif model_name == "tgarnet":
        return TGARNet(
            nb_classes=nb_classes,
            Chans=Chans,
            Samples=Samples,
            norm_rate=kwargs["norm_rate"],
            num_heads=kwargs["num_heads"],
            intermediate_dim=kwargs["intermediate_dim"],
            filters=kwargs["filters"],
            dropoutRate=kwargs["dropoutRate"],
            kernel_sigma=kwargs["kernel_sigma"],
        )

    else:
        raise ValueError(
            "model_name must be one of: ['eegnet', 'shallowconvnet', 'tgarnet']"
        )

def suggest_compile_args(trial, model_name, base_compile_args=None):
    if base_compile_args is None:
        base_compile_args = {}

    model_name = model_name.lower()

    if model_name == "tgarnet":
        return {
            **base_compile_args,
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "activation_loss": trial.suggest_float("activation_loss", 0.1, 0.9),
            
        }

    elif model_name in ["eegnet", "shallowconvnet"]:
        return {
            **base_compile_args,
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        }

    else:
        raise ValueError(
            "model_name must be one of: ['eegnet', 'shallowconvnet', 'tgarnet']"
        )

def build_compile_config(model_name, **kwargs):
    model_name = model_name.lower()
    optimizer = Adam(learning_rate=kwargs["learning_rate"])

    if model_name == "tgarnet":
        compile_args = {
            "optimizer": optimizer,
            "loss": {
                "out_activation": NormalizedBinaryCrossentropy(
                    name="NormalizedBinaryCrossentropy"
                ),
                "kernel_entropy": RenyiEntropyRegularizer(
                    name="RenyiEntropyRegularizer"
                ),
            },
            "loss_weights": {
                "out_activation": kwargs['activation_loss'],
                "kernel_entropy": 1-kwargs['activation_loss'],
            },
            "metrics": {
                "out_activation": [
                    "binary_accuracy",
                    tf.keras.metrics.AUC(name="AUC")
                ]
            }
        }
        callbacks = []
        return compile_args, callbacks

    elif model_name in ["eegnet", "shallowconvnet"]:
        compile_args = {
            "optimizer": optimizer,
            "loss": NormalizedBinaryCrossentropy(
                name="NormalizedBinaryCrossentropy"
            ),
            "metrics": [
                "binary_accuracy",
                tf.keras.metrics.AUC(name="AUC")
            ]
        }

        callbacks = []
        return compile_args, callbacks

    else:
        raise ValueError(
            "model_name must be one of: ['eegnet', 'shallowconvnet', 'tgarnet']"
        )