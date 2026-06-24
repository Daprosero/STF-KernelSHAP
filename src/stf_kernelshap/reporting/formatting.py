"""Shared reporting labels and formatting helpers."""

MODEL_LABELS = {
    "shallowconvnet": "ShallowConvNet",
    "shallow": "ShallowConvNet",
    "eegnet": "EEGNet",
    "tgarnet": "T-GARNet",
    "t-garnet": "T-GARNet",
}


def pretty_model_name(model_name):
    return MODEL_LABELS.get(str(model_name).lower(), str(model_name))
