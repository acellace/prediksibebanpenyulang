---
title: Dashboard Prediksi Beban Penyulang OG PSP
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
short_description: Dashboard forecast beban OG PSP dengan LSTM
---

# Dashboard Prediksi Beban Penyulang OG PSP

Dashboard Gradio untuk prediksi beban Penyulang OG PSP menggunakan LSTM.

## Struktur file

```text
.
├── app.py
├── utils.py
├── requirements.txt
├── README.md
├── model/
│   └── model_lstm_weights.weights.h5
└── data/
    ├── data_aktual.csv
    ├── data_forecast.csv
    ├── data_gabungan.csv
    ├── evaluasi_model.csv
    ├── ringkasan_forecast.csv
    ├── scaler_og_psp.pkl
    ├── test_predictions.csv
```

## Model

Arsitektur yang digunakan untuk inference:

- LSTM 64 unit, tanh, return_sequences=True
- Dropout 0.2
- LSTM 32 unit, tanh
- Dropout 0.2
- Dense(1)
- Window size: 48 timestep
- Interval data: 30 menit

Dashboard tidak melakukan training ulang. Model dan scaler dimuat dari folder `model/` dan `data/`.

## Forecast

Data aktual tersedia sampai 31 Juli 2026. Forecast utama dashboard adalah 1–31 Agustus 2026 dengan interval 30 menit dan metode recursive one-step LSTM.

## File evaluasi

`evaluasi_model.csv` menggunakan format dua kolom:

```text
Metric,Value
RMSE Normalized,...
RMSE Ampere,...
MAPE,...
```

## Menjalankan

```bash
python app.py
```

Untuk Hugging Face Spaces, gunakan Gradio SDK dan unggah seluruh struktur folder di atas. Jika menggunakan ZeroGPU, aktifkan hardware ZeroGPU pada pengaturan Space.
