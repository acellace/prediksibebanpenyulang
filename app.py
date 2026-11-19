# ============================================================
# app.py
# Dashboard Prediksi Beban Penyulang OG PSP — Gradio
# ============================================================
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import spaces
# ============================================================
# ZERO GPU PROBE
# ============================================================
@spaces.GPU(duration=1)
def _zerogpu_probe():
    pass
# ============================================================
# IMPORT UTILITIES
# ============================================================
from utils import (
    load_aktual,
    load_forecast,
    load_gabungan,
    load_ringkasan_forecast,
    load_evaluasi,
    load_test_predictions,
    prediksi_satu_titik,
    daftar_jam
)
# ============================================================
# FUNCTION — PREDIKSI REALTIME
# ============================================================
def fn_prediksi(tanggal_str: str, jam_str: str):
    if not tanggal_str or not jam_str:
        return (
            "⚠️ Pilih tanggal dan jam terlebih dahulu.",
            ""
        )
    try:
        target_dt = pd.Timestamp(
            f"{tanggal_str} {jam_str}"
        )
    except Exception:
        return (
            "⚠️ Format tanggal tidak valid "
            "(gunakan YYYY-MM-DD).",
            ""
        )
    try:
        hasil = prediksi_satu_titik(
            target_dt
        )
        label = (
            f"📅 {target_dt.strftime('%d %B %Y, %H:%M')}"
        )
        nilai = (
            f"### ⚡ {hasil:.2f} A"
        )
        return label, nilai
    except Exception as e:
        return (
            f"⚠️ Error: {e}",
            ""
        )
# ============================================================
# FUNCTION — TREN HISTORIS
# ============================================================
def fn_tren(dari_str: str, sampai_str: str):
    df = load_aktual()
    try:
        dari = pd.Timestamp(
            dari_str
        )
        sampai = (
            pd.Timestamp(sampai_str)
            + pd.Timedelta(
                hours=23,
                minutes=30
            )
        )
    except Exception:
        dari = df["datetime"].min()
        sampai = df["datetime"].max()
    mask = (
        (df["datetime"] >= dari)
        &
        (df["datetime"] <= sampai)
    )
    df_f = df[mask]
    fig = go.Figure()
    if df_f.empty:
        fig.add_annotation(
            text=(
                "Tidak ada data pada rentang tanggal ini."
            ),
            showarrow=False,
            font_size=14
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df_f["datetime"],
                y=df_f["beban_aktual"],
                mode="lines",
                name="Beban Aktual",
                line=dict(
                    color="steelblue",
                    width=0.9
                )
            )
        )
    fig.update_layout(
        title=dict(
            text=(
                "Tren Beban Penyulang OG PSP<br>"
                "<sup>"
                f"{dari.strftime('%d %b %Y')}"
                " — "
                f"{sampai.strftime('%d %b %Y')}"
                "</sup>"
            ),
            font_size=15
        ),
        xaxis_title="Waktu",
        yaxis_title="Beban — Arus (A)",
        hovermode="x unified",
        height=460,
        margin=dict(
            l=55,
            r=30,
            t=75,
            b=55
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=True,
            gridcolor="#eeeeee"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#eeeeee"
        )
    )
    return fig
# ============================================================
# FUNCTION — FORECAST AGUSTUS 2026
# ============================================================
def fn_forecast():
    df_aktual = load_aktual()
    df_forecast = load_forecast()
    ringkasan = load_ringkasan_forecast()
    # ========================================================
    # KONTEKS 7 HARI TERAKHIR
    # ========================================================
    konteks = df_aktual.iloc[
        -7 * 48:
    ]
    # ========================================================
    # FIGURE
    # ========================================================
    fig = go.Figure()
    # --------------------------------------------------------
    # AKTUAL
    # --------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=konteks["datetime"],
            y=konteks["beban_aktual"],
            mode="lines",
            name="Aktual (7 hari terakhir)",
            line=dict(
                color="steelblue",
                width=1.0
            )
        )
    )
    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df_forecast["datetime"],
            y=df_forecast["beban_prediksi"],
            mode="lines",
            name="Forecast Agustus 2026",
            line=dict(
                color="tomato",
                width=1.0
            )
        )
    )
    # ========================================================
    # LAYOUT
    # ========================================================
    fig.update_layout(
        title=dict(
            text=(
                "Forecast Beban Penyulang OG PSP — Agustus 2026"
                "<br>"
                "<sup>"
                "Garis biru = Aktual | "
                "Garis merah = Forecast"
                "</sup>"
            ),
            font_size=14
        ),
        xaxis_title="Waktu",
        yaxis_title="Beban — Arus (A)",
        hovermode="x unified",
        height=460,
        margin=dict(
            l=55,
            r=30,
            t=80,
            b=55
        ),
        legend=dict(
            orientation="h",
            y=1.08
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=True,
            gridcolor="#eeeeee"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#eeeeee"
        )
    )
    # ========================================================
    # RINGKASAN FORECAST
    # ========================================================
    ringkasan_display = pd.DataFrame({
        "Informasi": [
            "Periode Forecast",
            "Interval",
            "Jumlah Titik Prediksi",
            "Rata-rata Beban Prediksi",
            "Minimum Beban Prediksi",
            "Maksimum Beban Prediksi"
        ],
        "Nilai": [
            (
                f"{pd.to_datetime(ringkasan['forecast_start']).strftime('%d %B %Y')}"
                f" — "
                f"{pd.to_datetime(ringkasan['forecast_end']).strftime('%d %B %Y')}"
            ),
            "30 menit",
            (
                f"{int(ringkasan['jumlah_titik_prediksi']):,} titik"
            ),
            (
                f"{float(ringkasan['rata_rata_beban_A']):.2f} A"
            ),
            (
                f"{float(ringkasan['minimum_beban_A']):.2f} A"
            ),
            (
                f"{float(ringkasan['maksimum_beban_A']):.2f} A"
            )
        ]
    })
    # ========================================================
    # TABEL FORECAST
    # ========================================================
    tabel_forecast = df_forecast.copy()
    tabel_forecast["datetime"] = (
        tabel_forecast["datetime"]
        .dt.strftime(
            "%d/%m/%Y %H:%M"
        )
    )
    tabel_forecast["beban_prediksi"] = (
        tabel_forecast["beban_prediksi"]
        .round(2)
    )
    tabel_forecast.columns = [
        "Waktu",
        "Beban Prediksi (A)"
    ]
    return (
        fig,
        ringkasan_display,
        tabel_forecast
    )
# ============================================================
# FUNCTION — EVALUASI MODEL
# ============================================================
def fn_evaluasi():
    ev = load_evaluasi()
    test = load_test_predictions()
    # ========================================================
    # AKTUAL VS PREDIKSI
    # ========================================================
    df_t = test[
        (test["datetime"] >= "2026-01-01")
        &
        (test["datetime"] < "2027-01-01")
    ]
    fig_test = go.Figure()
    # --------------------------------------------------------
    # AKTUAL
    # --------------------------------------------------------
    fig_test.add_trace(
        go.Scatter(
            x=df_t["datetime"],
            y=df_t["actual"],
            mode="lines",
            name="Aktual",
            line=dict(
                color="steelblue",
                width=1.1
            )
        )
    )
    # --------------------------------------------------------
    # PREDIKSI
    # --------------------------------------------------------
    fig_test.add_trace(
        go.Scatter(
            x=df_t["datetime"],
            y=df_t["predicted"],
            mode="lines",
            name="Prediksi",
            line=dict(
                color="darkorange",
                width=1.1
            )
        )
    )
    fig_test.update_layout(
        title="Aktual vs Prediksi — Data Test 2026",
        xaxis_title="Waktu",
        yaxis_title="Beban (A)",
        hovermode="x unified",
        height=450,
        margin=dict(
            l=60,
            r=30,
            t=60,
            b=55
        ),
        legend=dict(
            orientation="h",
            y=1.08
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=True,
            gridcolor="#eeeeee"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#eeeeee"
        )
    )
    tabel_test = test[['datetime', 'actual', 'predicted']].copy()
    tabel_test = tabel_test.sort_values('datetime').reset_index(drop=True)
    tabel_test['Selisih (A)'] = (tabel_test['actual'] - tabel_test['predicted']).abs()
    tabel_test['Perbedaan (%)'] = tabel_test['Selisih (A)'].div(tabel_test['actual'].abs().replace(0, np.nan)) * 100
    tabel_test['datetime'] = pd.to_datetime(tabel_test['datetime']).dt.strftime('%Y-%m-%d %H:%M')
    tabel_test = tabel_test.rename(columns={'actual': 'aktual_A', 'predicted': 'prediksi_A'}).round(2)
    # ========================================================
    # RETURN
    # ========================================================
    return (
        f"{float(ev.get('rmse_normalized', 0)):.4f}",
        f"{float(ev.get('rmse_ampere', 0)):.4f} A",
        f"{float(ev.get('mape', 0)):.2f}%",
        fig_test,
        tabel_test
    )
# ============================================================
# CSS
# ============================================================
CSS = """



/* =========================================================

   FONT

   ========================================================= */



@import url(

    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'

);



body,

.gradio-container,

.gradio-container * {

    font-family: 'Inter', sans-serif !important;

}



/* =========================================================

   HEADER UTAMA — RAPAT

   ========================================================= */



.header-title {

    margin: 0 !important;

    padding: 0 !important;

}



.header-title h1 {

    margin: 0 0 2px 0 !important;

    padding: 0 !important;

    line-height: 1.15 !important;

}



.header-subtitle {

    margin: 0 !important;

    padding: 0 !important;

}



.header-subtitle p {

    margin: 0 !important;

    padding: 0 !important;

    line-height: 1.25 !important;

}



/* =========================================================

   GENERAL

   ========================================================= */



h1 {

    text-align: center;

}



.sub-info {

    text-align: center;

    color: #555;

    font-size: 0.9em;

    margin-top: 5px;

    margin-bottom: 15px;

}



.metric-box textarea {

    font-size: 1.25em !important;

    font-weight: bold;

    text-align: center;

    background: #f0f4ff !important;

}



.hasil-nilai {

    font-size: 2em;

    font-weight: bold;

    text-align: center;

    padding: 20px;

    background: #f0f4ff;

    border-radius: 10px;

}





/* =========================================================

   TAB NAVIGATION

   ========================================================= */



.tabs > .tab-nav {

    width: 100% !important;

    display: flex !important;

    justify-content: flex-start !important;

    align-items: stretch !important;

    flex-wrap: nowrap !important;

    overflow-x: auto !important;

    overflow-y: hidden !important;

    gap: 0 !important;

    margin: 0 !important;

    padding: 0 !important;

}



.tabs > .tab-nav > button {

    flex: 0 0 auto !important;

    box-sizing: border-box !important;

    margin: 0 !important;

    white-space: nowrap !important;

    text-align: center !important;

}



.tabs > .tab-nav > button:nth-child(1) {

    width: 125px !important;

    min-width: 125px !important;

}



.tabs > .tab-nav > button:nth-child(2) {

    width: 165px !important;

    min-width: 165px !important;

}



.tabs > .tab-nav > button:nth-child(3) {

    width: 145px !important;

    min-width: 145px !important;

}



.tabs > .tab-nav > button:nth-child(4) {

    width: 205px !important;

    min-width: 205px !important;

}



.tabs > .tab-nav > button:nth-child(5) {

    width: 165px !important;

    min-width: 165px !important;

}



.tabs > .tab-nav > button.selected,

.tabs > .tab-nav > button[aria-selected="true"] {

    flex-grow: 0 !important;

    flex-shrink: 0 !important;

}



/* =========================================================

   JARAK TAB → ISI BERANDA

   ========================================================= */



.tabs > .tabitem {

    padding-top: 0 !important;

}



.tabs > .tabitem > div {

    margin-top: 0 !important;

}



/* =========================================================

   BERANDA — SPACING

   ========================================================= */



.home-section-title {

    margin-top: 2px !important;

    margin-bottom: 0 !important;

    padding: 0 !important;

}



.home-section-title h3 {

    margin-top: 0 !important;

    margin-bottom: 2px !important;

    padding: 0 !important;

    line-height: 1.15 !important;

}





/* =========================================================

   CARD ROW — RAPAT

   ========================================================= */



.home-card-row {

    gap: 10px !important;

    margin-top: 0 !important;

    margin-bottom: 0 !important;

    padding-top: 0 !important;

    padding-bottom: 0 !important;

}



.home-card-row > div {

    margin-top: 0 !important;

    margin-bottom: 0 !important;

    padding-top: 0 !important;

    padding-bottom: 0 !important;

}





/* =========================================================

   INFORMATION CARD

   ========================================================= */



.home-info-card {

    background: #ffffff;

    border: 1px solid #e3e8ef;

    border-radius: 12px;

    padding: 10px 16px;

    height: 72px;

    min-height: 72px;

    box-sizing: border-box;

    transition: all 0.2s ease;

}



.home-info-card:hover {

    border-color: #b9cce3;

    transform: translateY(-1px);

}



.home-info-card .card-label {

    display: inline-block;

    background: #eaf2ff;

    color: #2876d8;

    border-radius: 6px;

    padding: 3px 8px;

    font-size: 0.74em;

    font-weight: 600;

    margin-bottom: 6px;

}



.home-info-card .card-value {

    color: #172b4d;

    font-size: 0.98em;

    font-weight: 700;

    line-height: 1.2;

}





/* =========================================================

   FORECAST NOTE

   ========================================================= */



.home-forecast-note {

    margin-top: 4px !important;

    margin-bottom: 4px !important;

    padding: 8px 16px;

    border-radius: 10px;

    background: #f4f8fd;

    border-left: 4px solid #357abd;

    color: #475467;

    font-size: 0.84em;

}





/* =========================================================

   DESCRIPTION CARD

   ========================================================= */



.home-description-card {

    background: #ffffff;

    border: 1px solid #e3e8ef;

    border-radius: 12px;

    padding: 14px 18px;

    min-height: 110px;

    box-sizing: border-box;

    transition: all 0.2s ease;

}



.home-description-card:hover {

    border-color: #b9cce3;

    transform: translateY(-1px);

}



.home-description-card h3 {

    margin: 0 0 7px 0;

    font-size: 0.96em;

    font-weight: 700;

    color: #172b4d;

}



.home-description-card p {

    margin: 0;

    color: #667085;

    font-size: 0.84em;

    line-height: 1.5;

}





/* =========================================================

   FEATURE TABLE

   ========================================================= */



.home-feature-table {

    margin-top: 0 !important;

    padding-top: 0 !important;

}



/* =========================================================

   BERANDA — FORECAST NOTE

   ========================================================= */



.home-forecast-note {

    margin-top: 12px;

    padding: 12px 16px;

    border-radius: 10px;

    background: #f4f8fd;

    border-left: 4px solid #357abd;

    color: #475467;

    font-size: 0.86em;

}





/* =========================================================

   BERANDA — FEATURE TABLE

   ========================================================= */



.home-feature-table {

    margin-top: 0 !important;

}




/* Ukuran Beranda diselaraskan dengan komponen tab lainnya. */
.home-card-row { gap: 14px !important; }
.home-info-card {
    height: auto;
    min-height: 112px;
    padding: 18px 20px;
}
.home-info-card .card-label {
    font-size: 14px;
    padding: 4px 9px;
    margin-bottom: 10px;
    line-height: 1.4;
}
.home-info-card .card-value {
    font-size: 18px;
    line-height: 1.45;
    overflow-wrap: anywhere;
}
.home-description-card {
    padding: 18px 20px;
    min-height: 140px;
}
.home-description-card h3 {
    font-size: 16px;
    line-height: 1.4;
    margin-bottom: 9px;
}
.home-description-card p {
    font-size: 14px;
    line-height: 1.6;
}
.home-forecast-note { font-size: 14px; line-height: 1.6; }
.home-feature-table { font-size: 14px; }
.home-section-title h3 { line-height: 1.4 !important; margin-bottom: 8px !important; }
@media (max-width: 640px) {
    .home-info-card { min-height: 104px; padding: 16px; }
    .home-description-card { padding: 16px; min-height: 0; }
    .home-info-card .card-value { font-size: 17px; }
}
"""
# ============================================================
# GRADIO BLOCKS
# ============================================================
with gr.Blocks(
    title="Dashboard Prediksi Beban OG PSP",
    theme=gr.themes.Soft(
        primary_hue="blue"
    ),
    css=CSS
) as demo:
    # ========================================================
    # HEADER UTAMA
    # ========================================================
    gr.Markdown(
        "# ⚡ Dashboard Prediksi Beban Penyulang OG PSP",
        elem_classes="header-title"
    )
    gr.Markdown(
        "PT PLN (Persero) UP2D Kalimantan Barat — Kerja Praktik 2026  \n"
        "Metode LSTM | Interval Data: 30 Menit | Satuan: Ampere (A)",
        elem_classes="sub-info header-subtitle"
    )
    # ========================================================
    # TABS
    # ========================================================
    with gr.Tabs():
        # ====================================================
        # TAB 0 — BERANDA
        # ====================================================
        with gr.Tab("🏠 Beranda"):
            # =================================================
            # INFORMASI DATASET
            # =================================================
            gr.Markdown(
                "### 📋 Informasi Dataset",
                elem_classes="home-section-title"
            )
            # -------------------------------------------------
            # DATASET BARIS 1
            # -------------------------------------------------
            with gr.Row(
                elem_classes="home-card-row"
            ):
                gr.HTML(
                    """

                    <div class="home-info-card">

                        <div class="card-label">

                            ⚡ Penyulang

                        </div>



                        <div class="card-value">

                            OG PSP

                        </div>

                    </div>

                    """
                )
                gr.HTML(
                    """

                    <div class="home-info-card">

                        <div class="card-label">

                            📅 Periode Data

                        </div>



                        <div class="card-value">

                            Januari 2024 – Juli 2026

                        </div>

                    </div>

                    """
                )
                gr.HTML(
                    """

                    <div class="home-info-card">

                        <div class="card-label">

                            ⏱️ Interval Data

                        </div>



                        <div class="card-value">

                            30 menit

                        </div>

                    </div>

                    """
                )
            # -------------------------------------------------
            # DATASET BARIS 2
            # -------------------------------------------------
            with gr.Row(
                elem_classes="home-card-row"
            ):
                gr.HTML(
                    """

                    <div class="home-info-card">

                        <div class="card-label">

                            📊 Jumlah Data

                        </div>



                        <div class="card-value">

                            45.264 titik

                        </div>

                    </div>

                    """
                )
                gr.HTML(
                    """

                    <div class="home-info-card">

                        <div class="card-label">

                            📐 Satuan

                        </div>



                        <div class="card-value">

                            Ampere (A)

                        </div>

                    </div>

                    """
                )
                gr.HTML(
                    """

                    <div class="home-info-card">

                        <div class="card-label">

                            🧠 Metode

                        </div>



                        <div class="card-value">

                            Long Short-Term Memory (LSTM)

                        </div>

                    </div>

                    """
                )
            # =================================================
            # INFORMASI FORECAST
            # =================================================
            gr.Markdown(
                "### 🔮 Informasi Forecast",
                elem_classes="home-section-title"
            )
            with gr.Row(
                elem_classes="home-card-row"
            ):
                gr.HTML(
                    """

                    <div class="home-info-card">

                        <div class="card-label">

                            📅 Periode Forecast

                        </div>



                        <div class="card-value">

                            01 – 31 Agustus 2026

                        </div>

                    </div>

                    """
                )
                gr.HTML(
                    """

                    <div class="home-info-card">

                        <div class="card-label">

                            ⏱️ Interval Forecast

                        </div>



                        <div class="card-value">

                            30 menit

                        </div>

                    </div>

                    """
                )
                gr.HTML(
                    """

                    <div class="home-info-card">

                        <div class="card-label">

                            📊 Jumlah Prediksi

                        </div>



                        <div class="card-value">

                            1.488 titik

                        </div>

                    </div>

                    """
                )
            # =================================================
            # CATATAN FORECAST
            # =================================================
            gr.HTML(
                """

                <div class="home-forecast-note">



                    ℹ️ Forecast Agustus 2026 merupakan hasil

                    prediksi menggunakan model LSTM berdasarkan

                    pola data historis beban Penyulang OG PSP.



                </div>

                """
            )
            # =================================================
            # KETERANGAN DATA
            # =================================================
            gr.Markdown(
                "### 📝 Keterangan Data",
                elem_classes="home-section-title"
            )
            with gr.Row(
                elem_classes="home-card-row"
            ):
                # ------------------------------------------------
                # DATA HISTORIS
                # ------------------------------------------------
                gr.HTML(
                    """

                    <div class="home-description-card">



                        <h3>📊 Data Historis</h3>



                        <p>

                            Data beban Penyulang OG PSP yang dicatat

                            setiap 30 menit selama periode Januari 2024

                            hingga Juli 2026.

                        </p>



                    </div>

                    """
                )
                # ------------------------------------------------
                # DATA FORECAST
                # ------------------------------------------------
                gr.HTML(
                    """

                    <div class="home-description-card">



                        <h3>🔮 Data Forecast</h3>



                        <p>

                            Hasil prediksi beban untuk periode

                            01 hingga 31 Agustus 2026 dengan

                            interval waktu 30 menit.

                        </p>



                    </div>

                    """
                )
                # ------------------------------------------------
                # MODEL
                # ------------------------------------------------
                gr.HTML(
                    """

                    <div class="home-description-card">



                        <h3>🧠 Model Prediksi</h3>



                        <p>

                            Model LSTM menggunakan 48 timestep atau

                            data 24 jam sebelumnya untuk memprediksi

                            beban pada timestep berikutnya.

                        </p>



                    </div>

                    """
                )
            # =================================================
            # FITUR DASHBOARD
            # =================================================
            gr.Markdown(
                "### 🧭 Fitur Dashboard",
                elem_classes="home-section-title"
            )
            gr.Markdown(
                "| Menu | Fungsi |\n"
                "|---|---|\n"
                "| ⚡ **Prediksi Realtime** | Menampilkan prediksi beban berdasarkan tanggal dan jam yang dipilih. |\n"
                "| 📈 **Tren Historis** | Menampilkan perkembangan beban berdasarkan data historis. |\n"
                "| 🔮 **Forecast Agustus 2026** | Menampilkan hasil forecast beban untuk periode Agustus 2026. |\n"
                "| 📊 **Evaluasi Model** | Menampilkan performa model dan perbandingan aktual dengan prediksi. |",
                elem_classes="home-feature-table"
            )
        # ====================================================
        # TAB 1 — PREDIKSI REALTIME
        # ====================================================
        with gr.Tab("⚡ Prediksi Realtime"):
            gr.Markdown(
                "### Prediksi Beban pada Waktu Tertentu\n"
                "Masukkan tanggal dan jam (kelipatan 30 menit). "
                "Model akan mengembalikan prediksi beban dalam Ampere."
            )
            with gr.Row():
                with gr.Column(
                    scale=1,
                    min_width=280
                ):
                    inp_tgl = gr.DateTime(
                        label="Tanggal",
                        value=None,
                        include_time=False,
                        type="string"
                    )
                    inp_jam = gr.Dropdown(
                        label="Jam",
                        choices=daftar_jam(),
                        value=None,
                        allow_custom_value=False
                    )
                    btn_pred = gr.Button(
                        "🔍 Prediksi",
                        variant="primary",
                        size="lg"
                    )
                with gr.Column(
                    scale=1,
                    min_width=280
                ):
                    out_label = gr.Markdown(
                        ""
                    )
                    out_nilai = gr.Markdown(
                        "",
                        elem_classes="hasil-nilai"
                    )
            gr.Markdown(
                "ℹ️ Data aktual tersedia hingga Juli 2026.  \n"
                "Untuk periode Agustus 2026, dashboard menggunakan "
                "hasil forecast yang telah dihitung pada notebook "
                "dan disimpan pada data forecast."
            )
            btn_pred.click(
                fn=fn_prediksi,
                inputs=[
                    inp_tgl,
                    inp_jam
                ],
                outputs=[
                    out_label,
                    out_nilai
                ]
            )
        # ====================================================
        # TAB 2 — TREN HISTORIS
        # ====================================================
        with gr.Tab("📈 Tren Historis"):
            gr.Markdown(
                "### Tren Beban Historis Januari 2024 – Juli 2026\n"
                "Filter rentang tanggal untuk memperbesar periode tertentu."
            )
            _df_aktual = load_aktual()
            _tgl_min = (
                _df_aktual["datetime"]
                .min()
                .strftime("%Y-%m-%d")
            )
            _tgl_max = (
                _df_aktual["datetime"]
                .max()
                .strftime("%Y-%m-%d")
            )
            with gr.Row():
                inp_dari = gr.Textbox(
                    label="Dari (YYYY-MM-DD)",
                    value=_tgl_min,
                    scale=1
                )
                inp_sampai = gr.Textbox(
                    label="Sampai (YYYY-MM-DD)",
                    value=_tgl_max,
                    scale=1
                )
                btn_tren = gr.Button(
                    "Tampilkan",
                    variant="primary",
                    scale=0,
                    min_width=120
                )
            out_tren = gr.Plot(
                value=fn_tren(
                    _tgl_min,
                    _tgl_max
                )
            )
            btn_tren.click(
                fn=fn_tren,
                inputs=[
                    inp_dari,
                    inp_sampai
                ],
                outputs=out_tren
            )
        # ====================================================
        # TAB 3 — FORECAST AGUSTUS 2026
        # ====================================================
        with gr.Tab("🔮 Forecast Agustus 2026"):
            gr.Markdown(
                "### Forecast Beban Penyulang OG PSP — Agustus 2026\n"
                "Periode 01 Agustus 2026 – 31 Agustus 2026, "
                "interval 30 menit. "
                "Garis biru menunjukkan konteks data aktual "
                "dan garis merah menunjukkan forecast."
            )
            (
                _fig_fc,
                _ringkasan_fc,
                _tabel_fc
            ) = fn_forecast()
            out_fc = gr.Plot(
                value=_fig_fc
            )
            gr.Markdown(
                "#### Ringkasan Forecast"
            )
            gr.Dataframe(
                value=_ringkasan_fc,
                headers=[
                    "Informasi",
                    "Nilai"
                ],
                datatype=[
                    "str",
                    "str"
                ],
                interactive=False,
                wrap=True
            )
            gr.Markdown(
                "#### Tabel Forecast"
            )
            gr.Dataframe(
                value=_tabel_fc,
                interactive=False,
                wrap=True,
                max_height=420
            )
            gr.Markdown(
                "⚠️ RMSE/MAPE untuk forecast Agustus 2026 "
                "belum dapat dihitung karena data aktual "
                "Agustus 2026 belum tersedia. "
                "Metrik pada tab Evaluasi Model merupakan "
                "evaluasi one-step-ahead pada data testing historis."
            )
        # ====================================================
        # TAB 4 — EVALUASI MODEL
        # ====================================================
        with gr.Tab("📊 Evaluasi Model"):
            gr.Markdown(
                "### Evaluasi Model LSTM"
            )
            (
                _rmse_n,
                _rmse_a,
                _mape,
                _fig_test,
                _tabel_test
            ) = fn_evaluasi()
            with gr.Row():
                gr.Textbox(
                    label="RMSE (Ternormalisasi 0-1)",
                    value=_rmse_n,
                    interactive=False,
                    elem_classes="metric-box"
                )
                gr.Textbox(
                    label="RMSE (Ampere)",
                    value=_rmse_a,
                    interactive=False,
                    elem_classes="metric-box"
                )
                gr.Textbox(
                    label="MAPE",
                    value=_mape,
                    interactive=False,
                    elem_classes="metric-box"
                )
            gr.Markdown(
                "ℹ️ Evaluasi di atas merupakan evaluasi "
                "one-step-ahead pada data testing historis, "
                "menggunakan 20% akhir data historis "
                "Januari 2024 – Juli 2026."
            )
            gr.Markdown(
                "#### Perbandingan Aktual dan Prediksi"
            )
            gr.Plot(
                value=_fig_test
            )
            gr.Markdown("#### Tabel Perbandingan Aktual dan Prediksi")
            gr.Dataframe(
                value=_tabel_test,
                headers=['datetime', 'aktual_A', 'prediksi_A', 'Selisih (A)', 'Perbedaan (%)'],
                datatype=['str', 'number', 'number', 'number', 'number'],
                interactive=False,
                wrap=True,
                max_height=420
            )
    # ========================================================
    # FOOTER
    # ========================================================
    gr.Markdown(
        "---\n"
        "Penyulang: OG PSP &nbsp;|&nbsp; "
        "Satuan: Ampere (A) &nbsp;|&nbsp; "
        "Periode data: Jan 2024 – Jul 2026 &nbsp;|&nbsp; "
        "WINDOW_SIZE: 48 timestep (24 jam) &nbsp;|&nbsp; "
        "Interval: 30 menit"
    )
# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False
    )