# ============================================================
# app.py
# Dashboard Prediksi Beban Penyulang OG PSP — Streamlit
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils import (
    load_aktual,
    load_forecast,
    load_ringkasan_forecast,
    load_evaluasi,
    load_test_predictions,
    prediksi_satu_titik,
    daftar_jam,
)

# ============================================================
# PAGE CONFIG & STYLE
# ============================================================

st.set_page_config(
    page_title="Dashboard Prediksi Beban OG PSP",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        text-align: center;
        margin-bottom: 0;
        padding-bottom: 0;
    }

    .main-subtitle {
        text-align: center;
        color: #667085;
        margin-top: 0;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }

    .section-title {
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
    }

    .info-card {
        background: #ffffff;
        border: 1px solid #e3e8ef;
        border-radius: 12px;
        padding: 17px 18px;
        min-height: 112px;
        height: 100%;
        box-sizing: border-box;
    }

    .card-label {
        display: inline-block;
        background: #eaf2ff;
        color: #2876d8;
        border-radius: 6px;
        padding: 4px 9px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .card-value {
        color: #172b4d;
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.45;
        overflow-wrap: anywhere;
    }

    .description-card {
        background: #ffffff;
        border: 1px solid #e3e8ef;
        border-radius: 12px;
        padding: 18px 20px;
        min-height: 140px;
        height: 100%;
    }

    .description-card h4 {
        margin: 0 0 9px 0;
        color: #172b4d;
        font-size: 1rem;
    }

    .description-card p {
        margin: 0;
        color: #667085;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .forecast-note {
        margin: 0.8rem 0 1rem 0;
        padding: 12px 16px;
        border-radius: 10px;
        background: #f4f8fd;
        border-left: 4px solid #357abd;
        color: #475467;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .result-value {
        background: #f0f4ff;
        color: #172b4d;
        border-radius: 10px;
        padding: 18px;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
    }

    div[data-testid="stMetric"] {
        background: #f0f4ff;
        border: 1px solid #e3e8ef;
        border-radius: 10px;
        padding: 14px;
    }

    @media (max-width: 640px) {
        .info-card { min-height: 100px; padding: 14px; }
        .card-value { font-size: 0.95rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def buat_kartu(label: str, value: str):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def buat_grafik_layout(fig):
    fig.update_layout(
        autosize=True,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)


def fn_tren(dari_str: str, sampai_str: str):
    df = load_aktual()

    dari = pd.Timestamp(dari_str)
    sampai = (
        pd.Timestamp(sampai_str)
        + pd.Timedelta(hours=23, minutes=30)
    )

    df_f = df[
        (df["datetime"] >= dari)
        & (df["datetime"] <= sampai)
    ]

    fig = go.Figure()

    if df_f.empty:
        fig.add_annotation(
            text="Tidak ada data pada rentang tanggal ini.",
            showarrow=False,
            font_size=14,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df_f["datetime"],
                y=df_f["beban_aktual"],
                mode="lines",
                name="Beban Aktual",
                line=dict(color="steelblue", width=0.9),
            )
        )

    fig.update_layout(
        title=dict(
            text=(
                "Tren Beban Penyulang OG PSP<br>"
                f"<sup>{dari.strftime('%d %b %Y')} — "
                f"{pd.Timestamp(sampai_str).strftime('%d %b %Y')}</sup>"
            ),
            font_size=15,
        ),
        xaxis_title="Waktu",
        yaxis_title="Beban — Arus (A)",
        hovermode="x unified",
        height=460,
        margin=dict(l=55, r=30, t=75, b=55),
        xaxis=dict(showgrid=True, gridcolor="#eeeeee"),
        yaxis=dict(showgrid=True, gridcolor="#eeeeee"),
    )
    return fig


def fn_forecast():
    df_aktual = load_aktual()
    df_forecast = load_forecast()
    ringkasan = load_ringkasan_forecast()

    konteks = df_aktual.iloc[-7 * 48:]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=konteks["datetime"],
            y=konteks["beban_aktual"],
            mode="lines",
            name="Aktual (7 hari terakhir)",
            line=dict(color="steelblue", width=1.0),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_forecast["datetime"],
            y=df_forecast["beban_prediksi"],
            mode="lines",
            name="Forecast Agustus 2026",
            line=dict(color="tomato", width=1.0),
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                "Forecast Beban Penyulang OG PSP — Agustus 2026<br>"
                "<sup>Garis biru = Aktual | Garis merah = Forecast</sup>"
            ),
            font_size=14,
        ),
        xaxis_title="Waktu",
        yaxis_title="Beban — Arus (A)",
        hovermode="x unified",
        height=460,
        margin=dict(l=55, r=30, t=80, b=55),
        legend=dict(orientation="h", y=1.08),
        xaxis=dict(showgrid=True, gridcolor="#eeeeee"),
        yaxis=dict(showgrid=True, gridcolor="#eeeeee"),
    )

    ringkasan_display = pd.DataFrame(
        {
            "Informasi": [
                "Periode Forecast",
                "Interval",
                "Jumlah Titik Prediksi",
                "Rata-rata Beban Prediksi",
                "Minimum Beban Prediksi",
                "Maksimum Beban Prediksi",
            ],
            "Nilai": [
                (
                    f"{pd.to_datetime(ringkasan['forecast_start']).strftime('%d %B %Y')}"
                    f" — "
                    f"{pd.to_datetime(ringkasan['forecast_end']).strftime('%d %B %Y')}"
                ),
                "30 menit",
                f"{int(ringkasan['jumlah_titik_prediksi']):,} titik",
                f"{float(ringkasan['rata_rata_beban_A']):.2f} A",
                f"{float(ringkasan['minimum_beban_A']):.2f} A",
                f"{float(ringkasan['maksimum_beban_A']):.2f} A",
            ],
        }
    )

    tabel_forecast = df_forecast.copy()
    tabel_forecast["datetime"] = tabel_forecast["datetime"].dt.strftime(
        "%d/%m/%Y %H:%M"
    )
    tabel_forecast["beban_prediksi"] = (
        tabel_forecast["beban_prediksi"].round(2)
    )
    tabel_forecast.columns = ["Waktu", "Beban Prediksi (A)"]

    return fig, ringkasan_display, tabel_forecast


def fn_evaluasi():
    ev = load_evaluasi()
    test = load_test_predictions()

    df_t = test[
        (test["datetime"] >= "2026-01-01")
        & (test["datetime"] < "2027-01-01")
    ]

    fig_test = go.Figure()

    fig_test.add_trace(
        go.Scatter(
            x=df_t["datetime"],
            y=df_t["actual"],
            mode="lines",
            name="Aktual",
            line=dict(color="steelblue", width=1.1),
        )
    )

    fig_test.add_trace(
        go.Scatter(
            x=df_t["datetime"],
            y=df_t["predicted"],
            mode="lines",
            name="Prediksi",
            line=dict(color="darkorange", width=1.1),
        )
    )

    fig_test.update_layout(
        title="Aktual vs Prediksi — Data Test 2026",
        xaxis_title="Waktu",
        yaxis_title="Beban (A)",
        hovermode="x unified",
        height=450,
        margin=dict(l=60, r=30, t=60, b=55),
        legend=dict(orientation="h", y=1.08),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#eeeeee"),
        yaxis=dict(showgrid=True, gridcolor="#eeeeee"),
    )

    tabel_test = test[["datetime", "actual", "predicted"]].copy()
    tabel_test = tabel_test.sort_values("datetime").reset_index(drop=True)
    tabel_test["Selisih (A)"] = (
        tabel_test["actual"] - tabel_test["predicted"]
    ).abs()
    tabel_test["Perbedaan (%)"] = (
        tabel_test["Selisih (A)"]
        .div(tabel_test["actual"].abs().replace(0, np.nan))
        * 100
    )
    tabel_test["datetime"] = pd.to_datetime(
        tabel_test["datetime"]
    ).dt.strftime("%Y-%m-%d %H:%M")
    tabel_test = tabel_test.rename(
        columns={
            "actual": "aktual_A",
            "predicted": "prediksi_A",
        }
    ).round(2)

    return (
        float(ev.get("rmse_normalized", 0)),
        float(ev.get("rmse_ampere", 0)),
        float(ev.get("mape", 0)),
        fig_test,
        tabel_test,
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<h1 class="main-title">⚡ Dashboard Prediksi Beban Penyulang OG PSP</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="main-subtitle">
        PT PLN (Persero) UP2D Kalimantan Barat — Kerja Praktik 2026<br>
        Metode LSTM | Interval Data: 30 Menit | Satuan: Ampere (A)
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# TABS
# ============================================================

tab_beranda, tab_realtime, tab_tren, tab_forecast, tab_evaluasi = st.tabs(
    [
        "🏠 Beranda",
        "⚡ Prediksi Realtime",
        "📈 Tren Historis",
        "🔮 Forecast Agustus 2026",
        "📊 Evaluasi Model",
    ]
)

# ============================================================
# TAB 0 — BERANDA
# ============================================================

with tab_beranda:
    st.markdown("### 📋 Informasi Dataset")

    kolom = st.columns(3)
    kartu_dataset = [
        ("⚡ Penyulang", "OG PSP"),
        ("📅 Periode Data", "Januari 2024 – Juli 2026"),
        ("⏱️ Interval Data", "30 menit"),
        ("📊 Jumlah Data", "45.264 titik"),
        ("📐 Satuan", "Ampere (A)"),
        ("🧠 Metode", "Long Short-Term Memory (LSTM)"),
    ]

    for i, (label, value) in enumerate(kartu_dataset):
        with kolom[i % 3]:
            buat_kartu(label, value)

    st.markdown("### 🔮 Informasi Forecast")
    kolom_fc = st.columns(3)
    kartu_forecast = [
        ("📅 Periode Forecast", "01 – 31 Agustus 2026"),
        ("⏱️ Interval Forecast", "30 menit"),
        ("📊 Jumlah Prediksi", "1.488 titik"),
    ]

    for i, (label, value) in enumerate(kartu_forecast):
        with kolom_fc[i]:
            buat_kartu(label, value)

    st.markdown(
        """
        <div class="forecast-note">
            ℹ️ Forecast Agustus 2026 merupakan hasil prediksi menggunakan
            model LSTM berdasarkan pola data historis beban Penyulang OG PSP.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📝 Keterangan Data")
    deskripsi = st.columns(3)

    isi_deskripsi = [
        (
            "📊 Data Historis",
            "Data beban Penyulang OG PSP yang dicatat setiap 30 menit "
            "selama periode Januari 2024 hingga Juli 2026.",
        ),
        (
            "🔮 Data Forecast",
            "Hasil prediksi beban untuk periode 01 hingga 31 Agustus 2026 "
            "dengan interval waktu 30 menit.",
        ),
        (
            "🧠 Model Prediksi",
            "Model LSTM menggunakan 48 timestep atau data 24 jam sebelumnya "
            "untuk memprediksi beban pada timestep berikutnya.",
        ),
    ]

    for col, (judul, isi) in zip(deskripsi, isi_deskripsi):
        with col:
            st.markdown(
                f"""
                <div class="description-card">
                    <h4>{judul}</h4>
                    <p>{isi}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 🧭 Fitur Dashboard")
    st.markdown(
        """
        | Menu | Fungsi |
        |---|---|
        | ⚡ **Prediksi Realtime** | Menampilkan prediksi beban berdasarkan tanggal dan jam yang dipilih. |
        | 📈 **Tren Historis** | Menampilkan perkembangan beban berdasarkan data historis. |
        | 🔮 **Forecast Agustus 2026** | Menampilkan hasil forecast beban untuk periode Agustus 2026. |
        | 📊 **Evaluasi Model** | Menampilkan performa model dan perbandingan aktual dengan prediksi. |
        """
    )

# ============================================================
# TAB 1 — PREDIKSI REALTIME
# ============================================================

with tab_realtime:
    st.markdown("### Prediksi Beban pada Waktu Tertentu")
    st.write(
        "Masukkan tanggal dan jam (kelipatan 30 menit). "
        "Model akan mengembalikan prediksi beban dalam Ampere."
    )

    col_input, col_output = st.columns(2)

    with col_input:
        tanggal = st.date_input(
            "Tanggal",
            value=None,
            format="DD/MM/YYYY",
        )
        jam = st.selectbox(
            "Jam",
            options=daftar_jam(),
            index=None,
            placeholder="Pilih jam",
        )
        tombol_prediksi = st.button(
            "🔍 Prediksi",
            type="primary",
            use_container_width=True,
        )

    with col_output:
        if tombol_prediksi:
            if tanggal is None or jam is None:
                st.warning("Pilih tanggal dan jam terlebih dahulu.")
            else:
                try:
                    target_dt = pd.Timestamp(f"{tanggal} {jam}")
                    with st.spinner("Menghitung prediksi beban..."):
                        hasil = prediksi_satu_titik(target_dt)

                    st.caption(
                        f"📅 {target_dt.strftime('%d %B %Y, %H:%M')}"
                    )
                    st.markdown(
                        f'<div class="result-value">⚡ {hasil:.2f} A</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

    st.info(
        "Data aktual tersedia hingga Juli 2026. Untuk periode Agustus 2026, "
        "dashboard menggunakan hasil forecast yang telah dihitung dan "
        "disimpan pada data forecast."
    )

# ============================================================
# TAB 2 — TREN HISTORIS
# ============================================================

with tab_tren:
    st.markdown("### Tren Beban Historis Januari 2024 – Juli 2026")
    st.write("Filter rentang tanggal untuk memperbesar periode tertentu.")

    try:
        df_aktual = load_aktual()
        tgl_min = df_aktual["datetime"].min().date()
        tgl_max = df_aktual["datetime"].max().date()

        col_dari, col_sampai = st.columns(2)
        with col_dari:
            tanggal_dari = st.date_input(
                "Dari",
                value=tgl_min,
                min_value=tgl_min,
                max_value=tgl_max,
                key="tanggal_dari",
                format="DD/MM/YYYY",
            )
        with col_sampai:
            tanggal_sampai = st.date_input(
                "Sampai",
                value=tgl_max,
                min_value=tgl_min,
                max_value=tgl_max,
                key="tanggal_sampai",
                format="DD/MM/YYYY",
            )

        if tanggal_dari > tanggal_sampai:
            st.warning("Tanggal awal tidak boleh melebihi tanggal akhir.")
        else:
            fig_tren = fn_tren(
                tanggal_dari.strftime("%Y-%m-%d"),
                tanggal_sampai.strftime("%Y-%m-%d"),
            )
            buat_grafik_layout(fig_tren)

    except Exception as e:
        st.error(f"Gagal menampilkan tren historis: {e}")

# ============================================================
# TAB 3 — FORECAST AGUSTUS 2026
# ============================================================

with tab_forecast:
    st.markdown("### Forecast Beban Penyulang OG PSP — Agustus 2026")
    st.write(
        "Periode 01 Agustus 2026 – 31 Agustus 2026, interval 30 menit. "
        "Garis biru menunjukkan konteks data aktual dan garis merah "
        "menunjukkan forecast."
    )

    try:
        fig_fc, ringkasan_fc, tabel_fc = fn_forecast()

        buat_grafik_layout(fig_fc)

        st.markdown("#### Ringkasan Forecast")
        st.dataframe(
            ringkasan_fc,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Tabel Forecast")
        st.dataframe(
            tabel_fc,
            use_container_width=True,
            hide_index=True,
            height=420,
        )

        st.warning(
            "RMSE/MAPE untuk forecast Agustus 2026 belum dapat dihitung "
            "karena data aktual Agustus 2026 belum tersedia. Metrik pada "
            "tab Evaluasi Model merupakan evaluasi one-step-ahead pada "
            "data testing historis."
        )

    except Exception as e:
        st.error(f"Gagal menampilkan forecast: {e}")

# ============================================================
# TAB 4 — EVALUASI MODEL
# ============================================================

with tab_evaluasi:
    st.markdown("### Evaluasi Model LSTM")

    try:
        rmse_n, rmse_a, mape, fig_test, tabel_test = fn_evaluasi()

        metrik = st.columns(3)
        with metrik[0]:
            st.metric(
                "RMSE (Ternormalisasi 0–1)",
                f"{rmse_n:.4f}",
            )
        with metrik[1]:
            st.metric(
                "RMSE (Ampere)",
                f"{rmse_a:.4f} A",
            )
        with metrik[2]:
            st.metric(
                "MAPE",
                f"{mape:.2f}%",
            )

        st.info(
            "Evaluasi di atas merupakan evaluasi one-step-ahead pada data "
            "testing historis, menggunakan 20% akhir data historis "
            "Januari 2024 – Juli 2026."
        )

        st.markdown("#### Perbandingan Aktual dan Prediksi")
        buat_grafik_layout(fig_test)

        st.markdown("#### Tabel Perbandingan Aktual dan Prediksi")
        st.dataframe(
            tabel_test,
            use_container_width=True,
            hide_index=True,
            height=420,
        )

    except Exception as e:
        st.error(f"Gagal menampilkan evaluasi model: {e}")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Penyulang: OG PSP | Satuan: Ampere (A) | "
    "Periode data: Jan 2024 – Jul 2026 | "
    "WINDOW_SIZE: 48 timestep (24 jam) | Interval: 30 menit"
)
