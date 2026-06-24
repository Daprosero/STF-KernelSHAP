"""Model architectures, builders, custom layers, and losses."""

from stf_kernelshap.modeling.builders import (
    build_compile_config,
    build_eeg_model,
    suggest_compile_args,
    suggest_model_args,
)
from stf_kernelshap.modeling.models import EEGNet, ShallowConvNet, TGARNet

__all__ = [
    "EEGNet",
    "ShallowConvNet",
    "TGARNet",
    "build_compile_config",
    "build_eeg_model",
    "suggest_compile_args",
    "suggest_model_args",
]
