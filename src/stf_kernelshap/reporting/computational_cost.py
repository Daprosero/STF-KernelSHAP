"""Computational-cost summaries and figures for XAI timing logs."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stf_kernelshap.reporting.formatting import pretty_model_name


TIMING_LOG_PATH = "Results/xai_timing_logs.csv"
DEFAULT_METHOD_ORDER = [
    "KernelSHAP",
    "LIME",
    "Occlusion",
    "IntegratedGradients",
    "GradCAM++",
    "STF-KernelSHAP",
]
METHOD_LABELS = {
    "KernelSHAP": "KernelSHAP",
    "LIME": "LIME",
    "Occlusion": "Occlusion",
    "IntegratedGradients": "Integrated\nGradients",
    "GradCAM++": "Grad-CAM++",
    "STF-KernelSHAP": "STF-\nKernelSHAP",
}
MODEL_ORDER = ["eegnet", "shallowconvnet", "tgarnet"]
MODEL_COLORS = {
    "eegnet": "#4477AA",
    "shallowconvnet": "#228833",
    "tgarnet": "#CC6677",
}
WINDOW_COLORS = {
    "2.5-5": "#4477AA",
    "0-7": "#228833",
    "TDAH": "#CC6677",
    "ADHD": "#CC6677",
}

caption_fontsize_pt = 22
figure_fontsize_pt = 0.7 * caption_fontsize_pt
tick_fontsize_pt = 0.6 * caption_fontsize_pt


def load_timing_logs(timing_log_path=TIMING_LOG_PATH):
    """Load timing logs written by the XAI runners."""
    if not os.path.exists(timing_log_path):
        raise FileNotFoundError(
            f"No timing log found at {timing_log_path}. "
            "Run Notebooks/Computational_cost_scalability.ipynb first."
        )

    df = pd.read_csv(timing_log_path)
    required = ["paradigm", "model", "method", "runtime_seconds"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing timing columns: {missing}")

    df = df.copy()
    df["runtime_seconds"] = pd.to_numeric(
        df["runtime_seconds"],
        errors="coerce",
    )
    df = df.dropna(subset=["runtime_seconds"])
    df["paradigm"] = df["paradigm"].replace({"TDAH": "ADHD"})
    df["model"] = df["model"].astype(str).str.lower()
    return df


def summarize_timing_logs(df_timing):
    """Mean and standard deviation of explanation runtime per trial."""
    group_cols = ["paradigm", "model", "method"]
    return (
        df_timing.groupby(group_cols, as_index=False)
        .agg(
            runtime_mean_s=("runtime_seconds", "mean"),
            runtime_std_s=("runtime_seconds", "std"),
            n_trials=("runtime_seconds", "count"),
        )
        .sort_values(group_cols)
    )


def summarize_timing_logs_by_window(df_timing):
    """Mean and standard deviation by paradigm, window, model, and method."""
    df = df_timing.copy()
    if "window" not in df.columns:
        df["window"] = ""
    df["window"] = df["window"].fillna("").astype(str)
    group_cols = ["paradigm", "window", "model", "method"]
    return (
        df.groupby(group_cols, as_index=False)
        .agg(
            runtime_mean_s=("runtime_seconds", "mean"),
            runtime_std_s=("runtime_seconds", "std"),
            n_trials=("runtime_seconds", "count"),
        )
        .sort_values(group_cols)
    )


def _ordered_present(values, preferred_order):
    present = list(dict.fromkeys(values))
    ordered = [value for value in preferred_order if value in present]
    ordered.extend([value for value in present if value not in ordered])
    return ordered


def plot_computational_cost_figure(
    df_summary,
    save_path=None,
    method_order=None,
    model_order=None,
):
    """Create the 1 x 2 computational-cost figure for MI and ADHD."""
    method_order = method_order or DEFAULT_METHOD_ORDER
    model_order = model_order or MODEL_ORDER

    method_order = _ordered_present(df_summary["method"], method_order)
    model_order = _ordered_present(df_summary["model"], model_order)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6), sharey=True)
    paradigms = [("MI", "MI"), ("ADHD", "ADHD")]
    x = np.arange(len(method_order), dtype=float)
    width = 0.72 / max(len(model_order), 1)

    for ax, (paradigm, title) in zip(axes, paradigms):
        data = df_summary[df_summary["paradigm"] == paradigm].copy()

        for model_i, model_name in enumerate(model_order):
            offset = (model_i - (len(model_order) - 1) / 2.0) * width
            model_data = (
                data[data["model"] == model_name]
                .set_index("method")
                .reindex(method_order)
            )
            y = model_data["runtime_mean_s"].to_numpy(dtype=float)
            yerr = model_data["runtime_std_s"].fillna(0.0).to_numpy(dtype=float)

            ax.bar(
                x + offset,
                y,
                width=width,
                yerr=yerr,
                capsize=3,
                linewidth=0.8,
                edgecolor="black",
                color=MODEL_COLORS.get(model_name, None),
                label=pretty_model_name(model_name),
                zorder=3,
            )

        ax.set_title(title, fontsize=figure_fontsize_pt)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [METHOD_LABELS.get(method, method) for method in method_order],
            rotation=0,
            fontsize=tick_fontsize_pt,
        )
        ax.set_xlabel("XAI method", fontsize=figure_fontsize_pt)
        ax.grid(axis="y", alpha=0.35, zorder=1)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=figure_fontsize_pt)

    axes[0].set_ylabel(
        "Mean explanation runtime per trial [s]",
        fontsize=figure_fontsize_pt,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 3),
        fontsize=figure_fontsize_pt,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.subplots_adjust(bottom=0.23, top=0.92, wspace=0.12)

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def make_computational_cost_figure(
    timing_log_path=TIMING_LOG_PATH,
    save_path="Figures/Computational_cost_scalability.pdf",
):
    """Load logs, summarize them, and save the paper-ready figure."""
    df_timing = load_timing_logs(timing_log_path)
    df_summary = summarize_timing_logs(df_timing)
    fig, axes = plot_computational_cost_figure(
        df_summary=df_summary,
        save_path=save_path,
    )
    return df_timing, df_summary, fig, axes


def plot_tgarnet_computational_cost_figure(
    df_summary,
    save_path=None,
    method_order=None,
    mi_window_order=None,
    model_name="tgarnet",
):
    """Create a 1 x 2 figure for T-GARNet timing in MI windows and ADHD."""
    method_order = method_order or DEFAULT_METHOD_ORDER
    mi_window_order = mi_window_order or ["2.5-5", "0-7"]
    method_order = _ordered_present(df_summary["method"], method_order)

    data = df_summary[df_summary["model"].astype(str).str.lower() == model_name].copy()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6), sharey=True)
    x = np.arange(len(method_order), dtype=float)

    mi_data = data[data["paradigm"] == "MI"].copy()
    mi_windows = _ordered_present(mi_data["window"], mi_window_order)
    width = 0.72 / max(len(mi_windows), 1)

    for window_i, window_name in enumerate(mi_windows):
        offset = (window_i - (len(mi_windows) - 1) / 2.0) * width
        window_data = (
            mi_data[mi_data["window"] == window_name]
            .set_index("method")
            .reindex(method_order)
        )
        y = window_data["runtime_mean_s"].to_numpy(dtype=float)
        yerr = window_data["runtime_std_s"].fillna(0.0).to_numpy(dtype=float)
        axes[0].bar(
            x + offset,
            y,
            width=width,
            yerr=yerr,
            capsize=3,
            linewidth=0.8,
            edgecolor="black",
            color=WINDOW_COLORS.get(window_name, None),
            label=f"MI {window_name} s",
            zorder=3,
        )

    adhd_data = data[data["paradigm"] == "ADHD"].copy()
    if adhd_data.empty:
        adhd_data = data[data["paradigm"] == "TDAH"].copy()
    adhd_data = (
        adhd_data.set_index("method")
        .reindex(method_order)
    )
    y = adhd_data["runtime_mean_s"].to_numpy(dtype=float)
    yerr = adhd_data["runtime_std_s"].fillna(0.0).to_numpy(dtype=float)
    axes[1].bar(
        x,
        y,
        width=0.55,
        yerr=yerr,
        capsize=3,
        linewidth=0.8,
        edgecolor="black",
        color=WINDOW_COLORS["ADHD"],
        label="ADHD/TDAH",
        zorder=3,
    )

    for ax, title in zip(axes, ["MI", "ADHD/TDAH"]):
        ax.set_title(f"{title} - {pretty_model_name(model_name)}", fontsize=figure_fontsize_pt)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [METHOD_LABELS.get(method, method) for method in method_order],
            rotation=0,
            fontsize=tick_fontsize_pt,
        )
        ax.set_xlabel("XAI method", fontsize=figure_fontsize_pt)
        ax.grid(axis="y", alpha=0.35, zorder=1)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=figure_fontsize_pt)

    axes[0].set_ylabel(
        "Mean explanation runtime per trial [s]",
        fontsize=figure_fontsize_pt,
    )

    handles = []
    labels = []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 3),
        fontsize=figure_fontsize_pt,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.subplots_adjust(bottom=0.23, top=0.92, wspace=0.12)

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def make_tgarnet_computational_cost_figure(
    timing_log_path=TIMING_LOG_PATH,
    save_path="Figures/Computational_cost_scalability.pdf",
):
    """Load T-GARNet logs, summarize by MI window/ADHD, and save the figure."""
    df_timing = load_timing_logs(timing_log_path)
    df_summary = summarize_timing_logs_by_window(df_timing)
    fig, axes = plot_tgarnet_computational_cost_figure(
        df_summary=df_summary,
        save_path=save_path,
    )
    return df_timing, df_summary, fig, axes
