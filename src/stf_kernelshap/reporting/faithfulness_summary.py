"""Utilities extracted from project notebooks.

This module is intentionally import-safe: it contains reusable functions and
constants only, not notebook execution code.
"""
try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = print


import os
import numpy as np
import pandas as pd
from stf_kernelshap.reporting.formatting import pretty_model_name

def _trapezoid(y, x):
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x)
    return np.trapz(y, x)

def _resolve_metric_column(df, candidates, metric_name):
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(
        f"No se encontró la columna para {metric_name}. "
        f"Se esperaba una de {candidates}. "
        f"Columnas disponibles: {list(df.columns)}"
    )

def compute_faithfulness_auc_summary(df_plot_ready, early_max_percentage=None):
    """
    Calcula AUC de MoRF y ROAD por:
        paradigm, window, subject, fold, model, method

    Si early_max_percentage=None:
        usa todo el rango disponible, por ejemplo 0-100.

    Si early_max_percentage=30:
        usa solo 0-30%.
    """
    df = df_plot_ready.copy()
    if early_max_percentage is not None:
        df = df[df['percentage_removed'] <= early_max_percentage].copy()
    morf_col = _resolve_metric_column(
        df,
        ['MoRF_accuracy', 'MoRF_score_mean'],
        'MoRF',
    )
    road_col = _resolve_metric_column(
        df,
        ['ROAD_accuracy_gap', 'ROAD_score_gap_mean'],
        'ROAD',
    )
    group_cols = ['paradigm', 'window', 'subject', 'fold', 'model', 'method']
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        g = g.sort_values('percentage_removed')
        x = g['percentage_removed'].values / 100.0
        morf_auc = _trapezoid(g[morf_col].values, x)
        road_auc = _trapezoid(g[road_col].values, x)
        row = dict(zip(group_cols, keys))
        row.update({'MoRF_accuracy_AUC': morf_auc, 'ROAD_accuracy_gap_AUC': road_auc})
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate_auc_by_window_model_method(df_auc):
    """
    Agrega los AUC por:
        paradigm, window, model, method

    Para MI:
        media ± std entre sujetos.

    Para TDAH:
        si hay un solo fold, std queda NaN.
    """
    df_summary = df_auc.groupby(['paradigm', 'window', 'model', 'method'], as_index=False).agg(MoRF_accuracy_AUC_mean=('MoRF_accuracy_AUC', 'mean'), MoRF_accuracy_AUC_std=('MoRF_accuracy_AUC', 'std'), ROAD_accuracy_gap_AUC_mean=('ROAD_accuracy_gap_AUC', 'mean'), ROAD_accuracy_gap_AUC_std=('ROAD_accuracy_gap_AUC', 'std'), n_cases=('MoRF_accuracy_AUC', 'count'))
    return df_summary

def get_ranked_morf_by_window(df_auc_summary, window_name):
    """
    Para una ventana específica, devuelve el ranking MoRF AUC
    por cada modelo.

    Criterio:
        Menor MoRF_accuracy_AUC_mean = mejor.
    """
    df_w = df_auc_summary[df_auc_summary['window'].astype(str) == str(window_name)].copy()
    if df_w.empty:
        raise ValueError(f'No hay datos para window={window_name}')
    df_w['model_pretty'] = df_w['model'].apply(pretty_model_name)
    df_w = df_w.sort_values(['model', 'MoRF_accuracy_AUC_mean'], ascending=[True, True])
    df_w['rank_MoRF'] = df_w.groupby('model')['MoRF_accuracy_AUC_mean'].rank(method='dense', ascending=True).astype(int)
    return df_w[['window', 'model_pretty', 'method', 'rank_MoRF', 'MoRF_accuracy_AUC_mean', 'MoRF_accuracy_AUC_std', 'n_cases']].rename(columns={'model_pretty': 'model', 'method': 'strategy'})

def get_ranked_road_by_window(df_auc_summary, window_name):
    """
    Para una ventana específica, devuelve el ranking ROAD AUC
    por cada modelo.

    Criterio:
        Mayor ROAD_accuracy_gap_AUC_mean = mejor.
    """
    df_w = df_auc_summary[df_auc_summary['window'].astype(str) == str(window_name)].copy()
    if df_w.empty:
        raise ValueError(f'No hay datos para window={window_name}')
    df_w['model_pretty'] = df_w['model'].apply(pretty_model_name)
    df_w = df_w.sort_values(['model', 'ROAD_accuracy_gap_AUC_mean'], ascending=[True, False])
    df_w['rank_ROAD'] = df_w.groupby('model')['ROAD_accuracy_gap_AUC_mean'].rank(method='dense', ascending=False).astype(int)
    return df_w[['window', 'model_pretty', 'method', 'rank_ROAD', 'ROAD_accuracy_gap_AUC_mean', 'ROAD_accuracy_gap_AUC_std', 'n_cases']].rename(columns={'model_pretty': 'model', 'method': 'strategy'})

def print_rankings_for_window(df_auc_summary, window_name):
    """
    Imprime dos tablas para una ventana:

        1. Ranking MoRF AUC
        2. Ranking ROAD AUC
    """
    print('=' * 90)
    print(f'WINDOW: {window_name}')
    print('=' * 90)
    print('\nMoRF AUC ranking')
    print('Criterio: menor MoRF_accuracy_AUC_mean = mejor\n')
    df_morf = get_ranked_morf_by_window(df_auc_summary=df_auc_summary, window_name=window_name)
    display(df_morf)
    print('\nROAD AUC ranking')
    print('Criterio: mayor ROAD_accuracy_gap_AUC_mean = mejor\n')
    df_road = get_ranked_road_by_window(df_auc_summary=df_auc_summary, window_name=window_name)
    display(df_road)
    return (df_morf, df_road)

def get_best_only_by_window(df_auc_summary, window_name):
    """
    Devuelve solo el mejor método por modelo para una ventana,
    separado para MoRF y ROAD.
    """
    df_morf = get_ranked_morf_by_window(df_auc_summary=df_auc_summary, window_name=window_name)
    df_road = get_ranked_road_by_window(df_auc_summary=df_auc_summary, window_name=window_name)
    best_morf = df_morf[df_morf['rank_MoRF'] == 1].copy().sort_values('model')
    best_road = df_road[df_road['rank_ROAD'] == 1].copy().sort_values('model')
    return (best_morf, best_road)

def compute_average_rank_by_window_metric(df_auc_summary, windows=('2.5-5', '0-7', 'TDAH')):
    """
    Calcula la posición promedio de cada estrategia XAI por ventana
    y por métrica.

    Usa df_auc_summary con columnas:
        window
        model
        method
        MoRF_accuracy_AUC_mean
        ROAD_accuracy_gap_AUC_mean

    Criterios:
        MoRF:
            menor MoRF_accuracy_AUC_mean = mejor posición

        ROAD:
            mayor ROAD_accuracy_gap_AUC_mean = mejor posición
    """
    rows = []
    for window_name in windows:
        df_w = df_auc_summary[df_auc_summary['window'].astype(str) == str(window_name)].copy()
        if df_w.empty:
            print(f'[WARNING] No hay datos para window={window_name}')
            continue
        df_morf = df_w.copy()
        df_morf['rank'] = df_morf.groupby('model')['MoRF_accuracy_AUC_mean'].rank(method='dense', ascending=True)
        for method_name, g in df_morf.groupby('method'):
            rows.append({'window': window_name, 'metric': 'MoRF', 'method': method_name, 'mean_rank': g['rank'].mean(), 'std_rank': g['rank'].std(), 'min_rank': g['rank'].min(), 'max_rank': g['rank'].max(), 'n_models': g['model'].nunique(), 'mean_AUC': g['MoRF_accuracy_AUC_mean'].mean(), 'std_AUC': g['MoRF_accuracy_AUC_mean'].std()})
        df_road = df_w.copy()
        df_road['rank'] = df_road.groupby('model')['ROAD_accuracy_gap_AUC_mean'].rank(method='dense', ascending=False)
        for method_name, g in df_road.groupby('method'):
            rows.append({'window': window_name, 'metric': 'ROAD', 'method': method_name, 'mean_rank': g['rank'].mean(), 'std_rank': g['rank'].std(), 'min_rank': g['rank'].min(), 'max_rank': g['rank'].max(), 'n_models': g['model'].nunique(), 'mean_AUC': g['ROAD_accuracy_gap_AUC_mean'].mean(), 'std_AUC': g['ROAD_accuracy_gap_AUC_mean'].std()})
    df_rank = pd.DataFrame(rows)
    df_rank = df_rank.sort_values(['window', 'metric', 'mean_rank', 'mean_AUC'], ascending=[True, True, True, False]).reset_index(drop=True)
    return df_rank
