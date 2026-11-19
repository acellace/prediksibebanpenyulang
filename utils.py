# utils.py
# Semua fungsi loading data & model dipusatkan di sini.
# Di-import oleh app.py — tidak ada training ulang di sini.
#
# Asumsi nama kolom CSV:
#   data_aktual.csv         → datetime, beban_aktual
#   data_forecast.csv       → datetime, beban_prediksi
#   data_gabungan.csv      → datetime, beban_aktual, beban_prediksi, jenis
#   ringkasan_forecast.csv → ringkasan forecast Agustus 2026
#   evaluasi_model.csv     → Metric, Value
#   training_history.csv   → loss, val_loss
#   test_predictions.csv   → datetime, actual, predicted

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense


# ============================================================
# PATH
# ============================================================

BASE = os.path.dirname(__file__)

WEIGHTS_PATH = os.path.join(
    BASE,
    'model',
    'model_lstm_weights.weights.h5'
)

SCALER_PATH = os.path.join(
    BASE,
    'data',
    'scaler_og_psp.pkl'
)

DATA_DIR = os.path.join(
    BASE,
    'data'
)


# ============================================================
# PARAMETER MODEL
# ============================================================

WINDOW_SIZE = 48   # 24 jam × 2 (interval 30 menit)


# ============================================================
# CACHE MODEL & SCALER
# ============================================================

_model = None
_scaler = None


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global _model

    if _model is None:

        _model = Sequential([
            LSTM(
                64,
                activation='tanh',
                return_sequences=True,
                input_shape=(48, 1)
            ),

            Dropout(0.2),

            LSTM(
                32,
                activation='tanh'
            ),

            Dropout(0.2),

            Dense(1)
        ])

        _model.load_weights(
            WEIGHTS_PATH
        )

    return _model


# ============================================================
# LOAD SCALER
# ============================================================

def load_scaler():

    global _scaler

    if _scaler is None:
        _scaler = joblib.load(
            SCALER_PATH
        )

    return _scaler


# ============================================================
# LOAD DATA AKTUAL
# ============================================================

def load_aktual() -> pd.DataFrame:

    df = pd.read_csv(
        os.path.join(
            DATA_DIR,
            'data_aktual.csv'
        ),
        parse_dates=['datetime']
    )

    return (
        df
        .sort_values('datetime')
        .reset_index(drop=True)
    )


# ============================================================
# LOAD DATA FORECAST
# ============================================================

def load_forecast() -> pd.DataFrame:

    df = pd.read_csv(
        os.path.join(
            DATA_DIR,
            'data_forecast.csv'
        ),
        parse_dates=['datetime']
    )

    return (
        df
        .sort_values('datetime')
        .reset_index(drop=True)
    )


# ============================================================
# LOAD DATA GABUNGAN
# ============================================================

def load_gabungan() -> pd.DataFrame:

    df = pd.read_csv(
        os.path.join(
            DATA_DIR,
            'data_gabungan.csv'
        ),
        parse_dates=['datetime']
    )

    return (
        df
        .sort_values('datetime')
        .reset_index(drop=True)
    )


# ============================================================
# LOAD RINGKASAN FORECAST
# ============================================================

def load_ringkasan_forecast() -> dict:

    path = os.path.join(
        DATA_DIR,
        'ringkasan_forecast.csv'
    )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            'ringkasan_forecast.csv kosong.'
        )

    return df.iloc[0].to_dict()


# ============================================================
# LOAD EVALUASI MODEL
# ============================================================

def load_evaluasi() -> dict:

    df = pd.read_csv(
        os.path.join(
            DATA_DIR,
            'evaluasi_model.csv'
        )
    )

    # Menyesuaikan kapitalisasi nama kolom
    # agar tetap aman jika CSV menggunakan
    # Metric/Value atau metric/value.
    kolom = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    metric_col = kolom.get('metric')
    value_col = kolom.get('value')

    if metric_col is None or value_col is None:
        raise ValueError(
            'evaluasi_model.csv harus memiliki '
            'kolom Metric dan Value.'
        )

    evaluasi = {}

    for _, row in df.iterrows():

        metric = str(
            row[metric_col]
        ).strip()

        value = row[value_col]


        # ====================================================
        # RMSE NORMALISASI
        # ====================================================

        if metric in [
            'RMSE Normalized',
            'RMSE Normalisasi'
        ]:

            evaluasi[
                'rmse_normalized'
            ] = float(value)


        # ====================================================
        # RMSE SATUAN ASLI
        # ====================================================

        elif metric in [
            'RMSE Ampere',
            'RMSE Satuan Asli'
        ]:

            evaluasi[
                'rmse_ampere'
            ] = float(value)


        # ====================================================
        # MAPE
        # ====================================================

        elif metric == 'MAPE':

            evaluasi[
                'mape'
            ] = float(value)


        # ====================================================
        # WINDOW SIZE
        # ====================================================

        elif metric == 'Window Size':

            evaluasi[
                'window_size'
            ] = int(float(value))


        # ====================================================
        # BATCH SIZE
        # ====================================================

        elif metric == 'Batch Size':

            evaluasi[
                'batch_size'
            ] = int(float(value))


    return evaluasi


# ============================================================
# LOAD TRAINING HISTORY
# ============================================================

def load_history() -> pd.DataFrame:

    return pd.read_csv(
        os.path.join(
            DATA_DIR,
            'training_history.csv'
        )
    )


# ============================================================
# LOAD TEST PREDICTIONS
# ============================================================

def load_test_predictions() -> pd.DataFrame:

    path = os.path.join(
        DATA_DIR,
        'test_predictions.csv'
    )

    df = pd.read_csv(path)

    # Normalisasi nama kolom
    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    # ========================================================
    # SESUAIKAN NAMA KOLOM DENGAN YANG DIPAKAI app.py
    # ========================================================

    rename_map = {}

    if 'aktual_a' in df.columns:
        rename_map['aktual_a'] = 'actual'

    if 'prediksi_a' in df.columns:
        rename_map['prediksi_a'] = 'predicted'

    df = df.rename(
        columns=rename_map
    )

    # ========================================================
    # CEK KOLOM WAJIB
    # ========================================================

    kolom_wajib = [
        'datetime',
        'actual',
        'predicted'
    ]

    kolom_hilang = [
        kolom
        for kolom in kolom_wajib
        if kolom not in df.columns
    ]

    if kolom_hilang:

        raise ValueError(
            "Kolom berikut tidak ditemukan "
            "di test_predictions.csv: "
            + ", ".join(kolom_hilang)
            + ". "
            + "Kolom yang tersedia: "
            + ", ".join(df.columns)
        )

    # ========================================================
    # FORMAT DATETIME
    # ========================================================

    df['datetime'] = pd.to_datetime(
        df['datetime'],
        errors='coerce'
    )

    # ========================================================
    # URUTKAN DATA
    # ========================================================

    df = (
        df[
            [
                'datetime',
                'actual',
                'predicted'
            ]
        ]
        .sort_values('datetime')
        .reset_index(drop=True)
    )

    return df


# ============================================================
# DAFTAR JAM KELIPATAN 30 MENIT
# ============================================================

def daftar_jam():

    return [
        f"{h:02d}:{m:02d}"
        for h in range(24)
        for m in (0, 30)
    ]


# ============================================================
# PREDIKSI SATU TITIK WAKTU
# ============================================================

def prediksi_satu_titik(target_dt: pd.Timestamp) -> float:
    """
    Prediksi satu titik waktu menggunakan recursive one-step LSTM.

    Prosedur recursive forecast dibuat sama dengan notebook:
    1. Mengambil WINDOW_SIZE data terakhir.
    2. Menggunakan model untuk memprediksi satu timestep.
    3. Memasukkan hasil prediksi ke ujung sequence.
    4. Menggeser sequence satu timestep.
    5. Mengulangi sampai mencapai target waktu.
    """

    model = load_model()
    scaler = load_scaler()
    df_aktual = load_aktual()

    last_dt = df_aktual['datetime'].iloc[-1]

    # ========================================================
    # TARGET MASIH DALAM DATA AKTUAL
    # ========================================================

    if target_dt <= last_dt:

        row = df_aktual[
            df_aktual['datetime'] == target_dt
        ]

        if not row.empty:

            return float(
                row['beban_aktual'].values[0]
            )

        # Jika target tidak tepat berada pada
        # timestep 30 menit, interpolasi berdasarkan waktu.

        s = (
            df_aktual
            .set_index('datetime')
            ['beban_aktual']
        )

        s = (
            s
            .reindex(
                s.index.union([target_dt])
            )
            .interpolate(
                method='time'
            )
        )

        return float(
            s.loc[target_dt]
        )

    # ========================================================
    # DATA UNTUK WINDOW AWAL
    # ========================================================

    beban_vals = (
        df_aktual['beban_aktual']
        .values[-WINDOW_SIZE:]
    )

    # Normalisasi menggunakan scaler yang sama
    # dengan saat training.

    beban_scaled = scaler.transform(
        beban_vals.reshape(-1, 1)
    )

    # ========================================================
    # FORECAST INDEX
    # SAMA DENGAN NOTEBOOK
    # ========================================================

    forecast_index = pd.date_range(
        start=last_dt + pd.Timedelta(minutes=30),
        end=target_dt,
        freq='30min'
    )

    # ========================================================
    # CURRENT SEQUENCE
    # SAMA DENGAN NOTEBOOK
    # ========================================================

    current_sequence = (
        beban_scaled[-WINDOW_SIZE:].copy()
    )

    forecast_scaled = []

    # ========================================================
    # RECURSIVE FORECAST
    # ========================================================

    for _ in range(
        len(forecast_index)
    ):

        next_value = model.predict(
            current_sequence.reshape(
                1,
                WINDOW_SIZE,
                1
            ),
            verbose=0
        )[0, 0]

        forecast_scaled.append(
            next_value
        )

        current_sequence = np.append(
            current_sequence[1:],
            [[next_value]],
            axis=0
        )

    # ========================================================
    # INVERSE TRANSFORM
    # SAMA DENGAN NOTEBOOK
    # ========================================================

    forecast_values = scaler.inverse_transform(
        np.array(
            forecast_scaled
        ).reshape(-1, 1)
    ).flatten()

    return float(
        forecast_values[-1]
    )