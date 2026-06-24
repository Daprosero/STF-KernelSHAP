"""Utilities extracted from project notebooks.

This module is intentionally import-safe: it contains reusable functions and
constants only, not notebook execution code.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
from stf_kernelshap.reporting.formatting import pretty_model_name

mi_csv_path = 'Results/MI_results.csv'
tdah_csv_path = 'Results/TDAH_results.csv'
save_dir = 'Figures'
metrics = ['accuracy', 'auc', 'kappa']
metric_labels = {'accuracy': 'Accuracy', 'auc': 'AUC', 'kappa': 'Kappa'}
caption_fontsize_pt = 22
figure_fontsize_pt = 0.7 * caption_fontsize_pt
subject_id_fontsize_pt = 0.6 * caption_fontsize_pt

def to_percent(x):
    """
    Converts values from 0-1 to percentages.
    If values already seem to be in 0-100, they are left unchanged.
    """
    x = pd.to_numeric(x, errors='coerce')
    if x.dropna().empty:
        return x
    if x.dropna().abs().max() <= 1.5:
        return 100.0 * x
    return x

def clean_subject_column(df, subject_col='subject'):
    """
    Ensures that the subject column is sortable.
    Numeric subjects are converted to numbers; otherwise, they remain strings.
    """
    df = df.copy()
    try:
        df[subject_col] = pd.to_numeric(df[subject_col])
    except Exception:
        df[subject_col] = df[subject_col].astype(str)
    return df

def safe_auc(y_true, y_score):
    """
    Computes AUC safely.
    Returns NaN if only one class is present.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan

def safe_metric_group(g):
    """
    Computes subject-level metrics for ADHD/TDAH.
    """
    y_true = g['true_class'].astype(int).values
    y_pred = g['pred_majority_class'].astype(int).values
    if 'prob_class_1' in g.columns:
        y_score = pd.to_numeric(g['prob_class_1'], errors='coerce').values
    else:
        y_score = y_pred
    return pd.Series({'accuracy': accuracy_score(y_true, y_pred), 'kappa': cohen_kappa_score(y_true, y_pred), 'auc': safe_auc(y_true, y_score)})

def summarize_mi_global(mi_csv_path):
    """
    Summarizes MI metrics by window and model.

    Expected columns:
    window, model,
    mean_accuracy, std_accuracy,
    mean_auc, std_auc,
    mean_kappa, std_kappa
    """
    df_mi = pd.read_csv(mi_csv_path)
    required_cols = ['window', 'model']
    missing = [c for c in required_cols if c not in df_mi.columns]
    if missing:
        raise ValueError(f'Missing columns in MI_results.csv: {missing}')
    rows = []
    for metric in metrics:
        mean_col = f'mean_{metric}'
        std_col = f'std_{metric}'
        if mean_col not in df_mi.columns:
            raise ValueError(f"Missing column '{mean_col}' in MI_results.csv")
        if std_col not in df_mi.columns:
            raise ValueError(f"Missing column '{std_col}' in MI_results.csv")
        aux = df_mi.groupby(['window', 'model'], as_index=False).agg(mean_value=(mean_col, 'mean'), std_value=(std_col, 'mean'))
        aux['metric'] = metric
        rows.append(aux)
    summary = pd.concat(rows, ignore_index=True)
    summary['mean_value'] = to_percent(summary['mean_value'])
    summary['std_value'] = to_percent(summary['std_value'])
    return summary

def summarize_mi_subject_level(mi_csv_path):
    """
    Summarizes MI accuracy by window, model, and subject.

    Expected columns:
    window, model, subject, mean_accuracy, std_accuracy
    """
    df_mi = pd.read_csv(mi_csv_path)
    df_mi = clean_subject_column(df_mi, 'subject')
    required_cols = ['window', 'model', 'subject', 'mean_accuracy', 'std_accuracy']
    missing = [c for c in required_cols if c not in df_mi.columns]
    if missing:
        raise ValueError(f'Missing columns in MI_results.csv: {missing}')
    summary = df_mi.groupby(['window', 'model', 'subject'], as_index=False).agg(acc_mean=('mean_accuracy', 'mean'), acc_std=('std_accuracy', 'mean'))
    summary['acc_mean'] = to_percent(summary['acc_mean'])
    summary['acc_std'] = to_percent(summary['acc_std'])
    return summary

def get_subject_order_by_eegnet(data, reference_model='EEGNet', subject_col='subject', acc_col='acc_mean', ascending=False):
    """
    Orders subjects according to EEGNet accuracy.

    ascending=False:
        Highest EEGNet accuracy first.
    """
    data = data.copy()
    ref_mask = data['model'].astype(str).str.lower().eq(reference_model.lower())
    ref_data = data[ref_mask].copy()
    if ref_data.empty:
        raise ValueError(f"Reference model '{reference_model}' was not found. Available models: {sorted(data['model'].dropna().unique())}")
    ref_order = ref_data.groupby(subject_col, as_index=False).agg(ref_acc=(acc_col, 'mean')).sort_values('ref_acc', ascending=ascending)
    subject_order = ref_order[subject_col].tolist()
    remaining_subjects = [s for s in data[subject_col].unique() if s not in subject_order]
    return subject_order + remaining_subjects

def summarize_tdah_global(tdah_csv_path):
    """
    Computes ADHD/TDAH metrics using one prediction per subject.

    Expected columns:
    model, repeat, fold, model_seed, subject,
    true_class, pred_majority_class, prob_class_1
    """
    df_tdah = pd.read_csv(tdah_csv_path)
    required_cols = ['model', 'repeat', 'fold', 'model_seed', 'subject', 'true_class', 'pred_majority_class', 'prob_class_1']
    missing = [c for c in required_cols if c not in df_tdah.columns]
    if missing:
        raise ValueError(f'Missing columns in TDAH_results.csv: {missing}')
    df_subject = df_tdah.sort_values(['model', 'repeat', 'fold', 'model_seed', 'subject']).drop_duplicates(subset=['model', 'repeat', 'fold', 'model_seed', 'subject'], keep='last')
    tdah_fold_metrics = df_subject.groupby(['model', 'repeat', 'fold', 'model_seed']).apply(safe_metric_group).reset_index()
    rows = []
    for metric in metrics:
        aux = tdah_fold_metrics.groupby('model', as_index=False).agg(mean_value=(metric, 'mean'), std_value=(metric, 'std'))
        aux['metric'] = metric
        rows.append(aux)
    summary = pd.concat(rows, ignore_index=True)
    summary['window'] = 'TDAH'
    summary['mean_value'] = to_percent(summary['mean_value'])
    summary['std_value'] = to_percent(summary['std_value'])
    return (summary, tdah_fold_metrics)

def plot_bar_panel(ax, data, model_order=None, metric_order=None, show_yticks=True, show_xlabel=True):
    """
    Horizontal bar plot for metrics by model.
    """
    if metric_order is None:
        metric_order = metrics
    if model_order is None:
        model_order = list(data['model'].dropna().unique())
    n_models = len(model_order)
    n_metrics = len(metric_order)
    y = np.arange(n_models)
    bar_height = 0.14
    offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2) * bar_height
    for j, metric in enumerate(metric_order):
        sub = data[data['metric'] == metric].copy()
        means = []
        stds = []
        for model in model_order:
            row = sub[sub['model'] == model]
            if row.empty:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(row['mean_value'].values[0])
                stds.append(row['std_value'].values[0])
        ax.barh(y + offsets[j], means, height=bar_height, xerr=stds, capsize=2, label=metric_labels.get(metric, metric))
    if show_yticks:
        ax.set_yticks(y)
        ax.set_yticklabels([pretty_model_name(m) for m in model_order])
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
    if show_xlabel:
        ax.set_xlabel('Metric [%]')
    else:
        ax.set_xlabel('')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.35)
    ax.invert_yaxis()
    ax.tick_params(axis='both', labelsize=figure_fontsize_pt)

def plot_subject_acc_panel(ax, data, subject_order=None, model_order=None, show_std=False, xtick_fontsize=None, show_yticks=True, show_ylabel=True, show_xlabel=True):
    """
    Plots subject-level accuracy for MI.
    """
    data = data.copy()
    if xtick_fontsize is None:
        xtick_fontsize = subject_id_fontsize_pt
    if subject_order is None:
        subject_order = get_subject_order_by_eegnet(data)
    if model_order is None:
        model_order = sorted(data['model'].dropna().unique())
    x = np.arange(len(subject_order))
    ax.axhspan(70, 100, facecolor='green', alpha=0.12, zorder=0)
    ax.axhspan(0, 70, facecolor='red', alpha=0.12, zorder=0)
    for model in model_order:
        sub = data[data['model'] == model].copy()
        sub = sub.set_index('subject').reindex(subject_order).reset_index()
        y = sub['acc_mean'].values
        yerr = sub['acc_std'].values
        ax.plot(x, y, marker='o', linewidth=1.0, markersize=2.2, label=pretty_model_name(model), zorder=3)
        if show_std:
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, zorder=2)
    if show_xlabel:
        ax.set_xlabel('Subject ID')
    else:
        ax.set_xlabel('')
    if show_ylabel:
        ax.set_ylabel('Accuracy [%]')
    else:
        ax.set_ylabel('')
    ax.set_ylim(40, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(subject_order, rotation=90, fontsize=xtick_fontsize)
    if len(x) > 0:
        ax.set_xlim(x[0], x[-1])
    if show_yticks:
        ax.tick_params(axis='y', labelleft=True, labelsize=figure_fontsize_pt)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', labelleft=False)
    ax.tick_params(axis='x', labelsize=xtick_fontsize)
    ax.grid(True, alpha=0.35, zorder=1)
    ax.set_axisbelow(True)

def plot_mi_window_figure(window_name, mi_global_summary, mi_subject_summary, save_path=None, reference_model='EEGNet'):
    """
    Generates one MI figure with:
    column 1: global metric bars
    column 2: subject-level accuracy.
    """
    data_bar = mi_global_summary[mi_global_summary['window'].astype(str) == str(window_name)].copy()
    data_subject = mi_subject_summary[mi_subject_summary['window'].astype(str) == str(window_name)].copy()
    if data_bar.empty:
        raise ValueError(f"No global MI data found for window '{window_name}'.")
    if data_subject.empty:
        raise ValueError(f"No subject-level MI data found for window '{window_name}'.")
    model_order = list(data_bar['model'].dropna().unique())
    subject_order = get_subject_order_by_eegnet(data=data_subject, reference_model=reference_model, ascending=False)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6), gridspec_kw={'width_ratios': [1.0, 1.7]})
    plot_bar_panel(ax=axes[0], data=data_bar, model_order=model_order, metric_order=metrics, show_yticks=True, show_xlabel=True)
    plot_subject_acc_panel(ax=axes[1], data=data_subject, subject_order=subject_order, model_order=model_order, show_std=False, xtick_fontsize=subject_id_fontsize_pt, show_yticks=True, show_ylabel=True, show_xlabel=True)
    all_handles = []
    all_labels = []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in all_labels:
                all_handles.append(h)
                all_labels.append(l)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.legend(all_handles, all_labels, loc='lower center', ncol=min(len(all_labels), 6), fontsize=figure_fontsize_pt, frameon=True, bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(bottom=0.2, top=0.95, wspace=0.25)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_tdah_bar_figure(tdah_summary, save_path=None):
    """
    Generates the ADHD/TDAH figure with only the global bar diagram.
    """
    data_bar = tdah_summary.copy()
    if data_bar.empty:
        raise ValueError('No ADHD/TDAH data found.')
    model_order = list(data_bar['model'].dropna().unique())
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    plot_bar_panel(ax=ax, data=data_bar, model_order=model_order, metric_order=metrics, show_yticks=True, show_xlabel=True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=figure_fontsize_pt, frameon=True, bbox_to_anchor=(0.5, -0.28))
    fig.subplots_adjust(bottom=0.08, top=0.95)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
