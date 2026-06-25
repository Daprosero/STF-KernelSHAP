"""Data loading and preprocessing helpers."""

import os

import numpy as np
import scipy.io
from scipy.signal import filtfilt, iirnotch
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
