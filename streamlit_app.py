# -*- coding: utf-8 -*-
"""
SABDA Apps — Dashboard Analisis Engagement & Prediksi DAU
Streamlit version converted from Colab notebook
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import os
import io
import tempfile
from datetime import date, datetime

# Keep Matplotlib/Prophet caches in a location the application can write to.
# This also avoids failures on managed Windows profiles with a restricted AppData.
APP_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
os.makedirs(APP_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(APP_CACHE_DIR, "matplotlib"))

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="SABDA Analytics",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Base font & background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] p {
        color: #94a3b8 !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
        font-family: 'DM Serif Display', serif !important;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-family: 'DM Serif Display', serif;
        color: #f1f5f9;
        font-size: 2rem;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #7dd3fc;
        margin: 0.3rem 0 0;
        font-size: 0.9rem;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        flex: 1;
    }
    .metric-card .label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }
    .metric-card .value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
        margin-top: 0.2rem;
    }
    .metric-card .sub {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }

    /* Section headers */
    .section-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.25rem;
        color: #0f172a;
        border-bottom: 2px solid #0f172a;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem;
    }

    /* Status badge */
    .badge-promo {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.15rem 0.6rem;
        border-radius: 99px;
        margin-left: 0.5rem;
        vertical-align: middle;
    }
    .badge-no-promo {
        display: inline-block;
        background: #f1f5f9;
        color: #64748b;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.15rem 0.6rem;
        border-radius: 99px;
        margin-left: 0.5rem;
        vertical-align: middle;
    }

    /* Upload zone */
    [data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 10px !important;
        padding: 0.5rem !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        color: #0f172a !important;
    }

    /* Model eval box */
    .eval-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }
    .eval-box h4 {
        color: #166534;
        margin: 0 0 0.5rem;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📖 SABDA Analytics")
    st.markdown("---")

    APP_OPTIONS = ['Kamus', 'Alkitab', 'Alkipedia', 'AlkitabGPT', 'Tafsiran']
    APP_NAME = st.selectbox("Aplikasi", APP_OPTIONS, index=0)
    HAS_PROMOTION = APP_NAME == 'Kamus'

    promo_label = "Ada Promosi" if HAS_PROMOTION else "Tanpa Promosi"
    badge_class = "badge-promo" if HAS_PROMOTION else "badge-no-promo"
    st.markdown(f"<span class='{badge_class}'>{promo_label}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Upload CSV**")
    st.caption("Format: Date, Nilai, Notes")

    uploaded_7d  = st.file_uploader("7-Day Retention",    type="csv", key="7d")
    uploaded_dau = st.file_uploader("Daily Active Users", type="csv", key="dau")
    uploaded_fo  = st.file_uploader("First Open",         type="csv", key="fo")
    uploaded_ia  = st.file_uploader("Installed Audience", type="csv", key="ia")

    st.markdown("---")
    run_analysis = st.button("▶ Jalankan Analisis", width="stretch", type="primary")

# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
st.markdown(f"""
<div class="main-header">
    <h1>📖 SABDA — {APP_NAME}</h1>
    <p>Dashboard Analisis Engagement & Prediksi DAU · Periode 180 Hari</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# LITURGICAL DATES (fixed)
# ══════════════════════════════════════════════════════════
liturgical_dates_data = [
    {'Date': date(2025, 12, 25), 'Event': 'Natal'},
    {'Date': date(2026, 1, 1),   'Event': 'Tahun Baru'},
    {'Date': date(2026, 2, 18),  'Event': 'Rabu Abu'},
    {'Date': date(2026, 3, 29),  'Event': 'Minggu Palma'},
    {'Date': date(2026, 4, 2),   'Event': 'Kamis Putih'},
    {'Date': date(2026, 4, 3),   'Event': 'Jumat Agung'},
    {'Date': date(2026, 4, 4),   'Event': 'Sabtu Sunyi'},
    {'Date': date(2026, 4, 5),   'Event': 'Minggu Paskah'},
    {'Date': date(2026, 5, 14),  'Event': 'Kenaikan Isa Almasih'},
    {'Date': date(2026, 5, 24),  'Event': 'Pentakosta'},
]
indonesian_liturgical_df = pd.DataFrame(liturgical_dates_data)
indonesian_liturgical_df['Date'] = pd.to_datetime(indonesian_liturgical_df['Date'])

# ══════════════════════════════════════════════════════════
# DEFAULT CONTENT ENGAGEMENT DATA
# ══════════════════════════════════════════════════════════
DEFAULT_CONTENT = [
    {'Date': '2026-03-23', 'Event Name': 'Jenis kamus dan nomor strong',        'Content Type': 'video',  'Like': 41,  'View': 1446, 'Komen': 3, 'Diposting ulang': 5, 'Disimpan': 5, 'Dibagikan': 2},
    {'Date': '2026-03-07', 'Event Name': 'Pembaruan UI, integrasi AI dan AlkiPEDIA', 'Content Type': 'video', 'Like': 44, 'View': 1366, 'Komen': 4, 'Diposting ulang': 8, 'Disimpan': 3, 'Dibagikan': 7},
    {'Date': '2026-04-20', 'Event Name': 'Fitur search',                         'Content Type': 'video',  'Like': 18,  'View': 443,  'Komen': 0, 'Diposting ulang': 2, 'Disimpan': 2, 'Dibagikan': 1},
    {'Date': '2026-02-27', 'Event Name': 'Pembaruan UI, integrasi AI dan AlkiPEDIA', 'Content Type': 'grafis', 'Like': 34, 'View': 1021, 'Komen': 0, 'Diposting ulang': 4, 'Disimpan': 6, 'Dibagikan': 1},
    {'Date': '2026-05-11', 'Event Name': 'Fitur search',                         'Content Type': 'grafis', 'Like': 14,  'View': 378,  'Komen': 0, 'Diposting ulang': 3, 'Disimpan': 1, 'Dibagikan': 0},
    {'Date': '2026-03-17', 'Event Name': 'Jenis kamus + nomor strong',           'Content Type': 'grafis', 'Like': 11,  'View': 374,  'Komen': 0, 'Diposting ulang': 4, 'Disimpan': 3, 'Dibagikan': 0},
]

# ══════════════════════════════════════════════════════════
# CONTENT ENGAGEMENT INPUT (Tab — hanya jika HAS_PROMOTION)
# ══════════════════════════════════════════════════════════
if HAS_PROMOTION:
    st.markdown('<div class="section-title">📣 Data Promosi Sosmed</div>', unsafe_allow_html=True)
    st.caption("Edit tabel di bawah untuk memperbarui data konten promosi. Gunakan format tanggal YYYY-MM-DD.")

    default_df = pd.DataFrame(DEFAULT_CONTENT)
    edited_df  = st.data_editor(
        default_df,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "Date":           st.column_config.TextColumn("Tanggal (YYYY-MM-DD)"),
            "Content Type":   st.column_config.SelectboxColumn("Tipe", options=["video", "grafis"]),
            "Like":           st.column_config.NumberColumn("Like",   min_value=0),
            "View":           st.column_config.NumberColumn("View",   min_value=0),
            "Komen":          st.column_config.NumberColumn("Komen",  min_value=0),
            "Diposting ulang":st.column_config.NumberColumn("Repost", min_value=0),
            "Disimpan":       st.column_config.NumberColumn("Simpan", min_value=0),
            "Dibagikan":      st.column_config.NumberColumn("Share",  min_value=0),
        },
        key="content_editor"
    )
    edited_df['Date'] = pd.to_datetime(edited_df['Date'])
    content_engagement_df = edited_df
else:
    content_engagement_df = None

# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
def preprocess(file, col_name, strip_comma=True):
    df = pd.read_csv(file)

    if len(df.columns) < 2:
        raise ValueError(
            f"{getattr(file, 'name', 'CSV')}: minimal harus memiliki kolom Date dan Nilai."
        )

    # Only the first three columns are part of the supported input format.
    # Previously, CSV files with more than three columns raised a length mismatch.
    column_count = min(len(df.columns), 3)
    df = df.iloc[:, :column_count].copy()
    df.columns = ['Date', col_name, 'notes'][:column_count]

    raw_dates = df['Date'].copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if df['Date'].isna().any():
        bad_date = raw_dates[df['Date'].isna()].iloc[0]
        raise ValueError(
            f"{getattr(file, 'name', 'CSV')}: tanggal tidak valid: {bad_date!r}."
        )

    df = df.sort_values('Date').reset_index(drop=True)

    raw_values = df[col_name].copy()
    values = df[col_name].astype(str).str.strip().str.replace('%', '', regex=False)
    if strip_comma:
        values = values.str.replace(',', '', regex=False)
    df[col_name] = pd.to_numeric(values, errors='coerce')
    if df[col_name].isna().any():
        bad_value = raw_values[df[col_name].isna()].iloc[0]
        raise ValueError(
            f"{getattr(file, 'name', 'CSV')}: nilai {col_name} tidak valid: {bad_value!r}."
        )

    return df

def create_line_chart(df, col, liturgical_df, promo_df=None, has_promo=False):
    label_map = {'DAU': 'Daily Active Users', '7d': '7-Day Retention (%)', 'FO': 'First Open', 'IA': 'Installed Audience'}
    fig = px.line(df, x='Date', y=col, title=f"{label_map.get(col, col)} — {APP_NAME}")
    fig.update_traces(line=dict(color='#1e3a5f', width=1.8))

    # Weekend shading
    weekends = df[df['Date'].dt.dayofweek >= 5]
    weekend_dates = sorted(weekends['Date'].tolist())
    blocks, i = [], 0
    while i < len(weekend_dates):
        start = weekend_dates[i]; end = start
        while i + 1 < len(weekend_dates) and (weekend_dates[i+1] - weekend_dates[i]).days == 1:
            i += 1; end = weekend_dates[i]
        blocks.append((start, end)); i += 1
    for s, e in blocks:
        fig.add_vrect(x0=s, x1=e + pd.Timedelta(days=1), fillcolor='gray', opacity=0.1, layer='below', line_width=0)

    # Liturgi
    for _, row in liturgical_df.iterrows():
        fig.add_vline(x=row['Date'], line_color='#ef4444', line_dash='dash', line_width=1.2,
                      annotation_text=row['Event'][:8], annotation_font_size=9, annotation_font_color='#ef4444')

    # Promosi
    if has_promo and promo_df is not None:
        for _, row in promo_df.iterrows():
            color = '#16a34a' if row['Content Type'] == 'video' else '#d97706'
            fig.add_vline(x=row['Date'], line_color=color, line_dash='dot', line_width=1.2)

    fig.update_layout(
        height=340, hovermode='x unified',
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Inter', size=12),
        margin=dict(t=45, b=30, l=40, r=20),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(gridcolor='#f1f5f9', tickfont=dict(size=10)),
        title_font=dict(size=13, family='DM Serif Display')
    )
    return fig

# ══════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ══════════════════════════════════════════════════════════
all_uploaded = all([uploaded_7d, uploaded_dau, uploaded_fo, uploaded_ia])

if not all_uploaded:
    st.info("⬅️ Upload 4 file CSV di sidebar, lalu klik **Jalankan Analisis**.")
    st.markdown("**File yang dibutuhkan per app:**")
    cols = st.columns(4)
    for c, label in zip(cols, ['7D · Retention 7 hari', 'DAU · Daily Active', 'FO · First Open', 'IA · Installed Audience']):
        c.markdown(f"📄 `{label}`")
    st.stop()

if not run_analysis:
    st.warning("File sudah diupload. Klik **▶ Jalankan Analisis** di sidebar untuk memulai.")
    st.stop()

# ── Load & preprocess ──────────────────────────────────────
try:
    with st.spinner("Memuat dan memproses data…"):
        sevend = preprocess(uploaded_7d,  '7d',  strip_comma=False)
        dau    = preprocess(uploaded_dau, 'DAU', strip_comma=True)
        fo     = preprocess(uploaded_fo,  'FO',  strip_comma=True)
        ia     = preprocess(uploaded_ia,  'IA',  strip_comma=True)

        merged_df = sevend[['Date', '7d']].copy()
        merged_df = pd.merge(merged_df, dau[['Date', 'DAU']], on='Date', how='left')
        merged_df = pd.merge(merged_df, fo[['Date', 'FO']],  on='Date', how='left')
        merged_df = pd.merge(merged_df, ia[['Date', 'IA']],  on='Date', how='left')
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f"CSV tidak dapat diproses: {exc}")
    st.stop()

PERIODE = f"{merged_df['Date'].min().strftime('%d %b %Y')} – {merged_df['Date'].max().strftime('%d %b %Y')}"

# ── Metric cards ───────────────────────────────────────────
st.markdown(f'<div class="section-title">📊 Ringkasan Metrik · {PERIODE}</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, label, fmt, note in [
    (c1, 'DAU',  '{:,.0f}', 'rata-rata/hari'),
    (c2, 'FO',   '{:,.0f}', 'rata-rata/hari'),
    (c3, '7d',   '{:.1f}%', 'retensi 7 hari'),
    (c4, 'IA',   '{:,.0f}', 'terakhir tercatat'),
]:
    val = merged_df[label].iloc[-1] if label == 'IA' else merged_df[label].mean()
    col.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{fmt.format(val)}</div>
        <div class="sub">{note}</div>
    </div>
    """, unsafe_allow_html=True)

# ── EDA Charts ─────────────────────────────────────────────
st.markdown('<div class="section-title">📈 Visualisasi Data Harian</div>', unsafe_allow_html=True)
st.caption("🔴 Garis merah = hari raya liturgi &nbsp;|&nbsp; 🟩 Hijau = promosi video &nbsp;|&nbsp; 🟨 Kuning = promosi grafis &nbsp;|&nbsp; ⬜ Abu-abu = Sabtu & Minggu")

tab1, tab2, tab3, tab4 = st.tabs(["DAU", "7-Day Retention", "First Open", "Installed Audience"])
for tab, col in zip([tab1, tab2, tab3, tab4], ['DAU', '7d', 'FO', 'IA']):
    with tab:
        fig = create_line_chart(merged_df, col, indonesian_liturgical_df, content_engagement_df, HAS_PROMOTION)
        st.plotly_chart(fig, width="stretch")

# ── Missing values info ────────────────────────────────────
missing = merged_df.isna().sum()
if missing.sum() > 0:
    with st.expander("⚠️ Missing values ditemukan"):
        st.dataframe(missing[missing > 0].rename("Jumlah NaN"), width="stretch")

# ══════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════
analysis_df = merged_df.copy()

liturgical_agg = indonesian_liturgical_df.groupby('Date')['Event'].apply(lambda x: ', '.join(x)).reset_index(name='liturgical_event_names')
analysis_df = pd.merge(analysis_df, liturgical_agg, on='Date', how='left')
analysis_df['is_liturgical_event']     = analysis_df['liturgical_event_names'].notna().astype(int)
analysis_df['liturgical_event_names']  = analysis_df['liturgical_event_names'].fillna('')

if HAS_PROMOTION and content_engagement_df is not None:
    numeric_cols = ['Like', 'View', 'Komen', 'Diposting ulang', 'Disimpan', 'Dibagikan']
    video_agg  = content_engagement_df[content_engagement_df['Content Type'] == 'video'].groupby('Date')[numeric_cols].sum().reset_index()
    video_agg.columns  = ['Date'] + [f'video_{c.lower().replace(" ", "_")}' for c in numeric_cols]
    video_agg['is_content_promotion_video'] = 1
    grafis_agg = content_engagement_df[content_engagement_df['Content Type'] == 'grafis'].groupby('Date')[numeric_cols].sum().reset_index()
    grafis_agg.columns = ['Date'] + [f'grafis_{c.lower().replace(" ", "_")}' for c in numeric_cols]
    grafis_agg['is_content_promotion_grafis'] = 1
    analysis_df = pd.merge(analysis_df, video_agg,  on='Date', how='left')
    analysis_df = pd.merge(analysis_df, grafis_agg, on='Date', how='left')
    promo_cols  = [c for c in analysis_df.columns if c.startswith(('video_', 'grafis_', 'is_content_'))]
    analysis_df[promo_cols] = analysis_df[promo_cols].fillna(0)
else:
    analysis_df['is_content_promotion_video']  = 0
    analysis_df['is_content_promotion_grafis'] = 0

# ══════════════════════════════════════════════════════════
# PROPHET MODEL
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🔮 Model Prediksi Prophet</div>', unsafe_allow_html=True)

try:
    from prophet import Prophet
    from prophet.utilities import regressor_coefficients
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    model_data = analysis_df.dropna(subset=['Date', 'DAU']).sort_values('Date').reset_index(drop=True)
    prophet_df = model_data[['Date', 'DAU']].rename(columns={'Date': 'ds', 'DAU': 'y'})

    base_regressor_cols = [
        'is_liturgical_event', 'is_content_promotion_video', 'is_content_promotion_grafis',
        'video_view', 'video_like', 'grafis_view', 'grafis_like'
    ]
    regressor_cols = [c for c in base_regressor_cols if c in model_data.columns]
    prophet_df = pd.concat([prophet_df, model_data[regressor_cols]], axis=1)

    n = len(prophet_df)
    if n < 21:
        raise ValueError("Data DAU valid minimal 21 hari untuk melatih dan menguji model.")
    test_size = max(7, int(round(n * 0.13)))
    split = n - test_size
    train_df = prophet_df.iloc[:split].copy()
    test_df  = prophet_df.iloc[split:].copy()

    holidays = indonesian_liturgical_df.rename(columns={'Date': 'ds', 'Event': 'holiday'})
    holidays['lower_window'] = 0
    holidays['upper_window'] = 1

    with st.spinner("Melatih model Prophet…"):
        model = Prophet(
            holidays=holidays,
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        for col in regressor_cols:
            model.add_regressor(col)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_df)

        forecast = model.predict(test_df)

    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nonzero = y_true != 0
    mape = (
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        if nonzero.any()
        else np.nan
    )
    mape_display = f"{mape:.2f}%" if np.isfinite(mape) else "N/A"

    # Eval metrics
    col1, col2, col3 = st.columns(3)
    for col, metric, val, desc in [
        (col1, 'MAPE', mape_display, 'Rata-rata % error'),
        (col2, 'MAE',  f"{mae:,.0f}",  'Selisih user/hari'),
        (col3, 'RMSE', f"{rmse:,.0f}", 'Sensitif terhadap spike'),
    ]:
        col.markdown(f"""
        <div class="eval-box">
            <h4>{metric}</h4>
            <div style="font-size:1.6rem;font-weight:700;color:#0f172a">{val}</div>
            <div style="font-size:0.75rem;color:#64748b;margin-top:0.2rem">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # Actual vs Predicted chart
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=test_df['ds'], y=y_true,  name='Aktual',   line=dict(color='#1e3a5f', width=2)))
    fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prediksi', line=dict(color='#f97316', width=2, dash='dash')))
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
        y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(249,115,22,0.12)',
        line=dict(color='rgba(255,255,255,0)'), name='Confidence interval'
    ))
    fig_fc.update_layout(
        title='DAU Aktual vs Prediksi (periode test)',
        height=380, hovermode='x unified',
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Inter'), margin=dict(t=45, b=30, l=40, r=20),
        legend=dict(orientation='h', y=-0.15),
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor='#f1f5f9'),
        title_font=dict(size=13, family='DM Serif Display')
    )
    st.plotly_chart(fig_fc, width="stretch")

    # Koefisien (Kamus only)
    if HAS_PROMOTION:
        st.markdown('<div class="section-title">📐 Dampak Faktor terhadap DAU</div>', unsafe_allow_html=True)
        label_map = {
            'is_content_promotion_video':  'Promosi Video',
            'is_content_promotion_grafis': 'Promosi Grafis',
            'is_liturgical_event':         'Hari Raya Liturgi',
        }
        coefficient_data = regressor_coefficients(model)
        regressor_coeffs = []
        for _, coefficient in coefficient_data.iterrows():
            reg = coefficient['regressor']
            regressor_coeffs.append({
                'Regressor': reg,
                'Label': label_map.get(reg, reg),
                'Dampak (%)': round(float(coefficient['coef']) * 100, 2),
            })
        coeffs_df = pd.DataFrame(regressor_coeffs).sort_values('Dampak (%)', ascending=False)

        fig_coeff = go.Figure(go.Bar(
            x=coeffs_df['Dampak (%)'], y=coeffs_df['Label'], orientation='h',
            marker_color=['#16a34a' if c > 0 else '#ef4444' for c in coeffs_df['Dampak (%)']],
            text=[f"{c:+.2f}%" for c in coeffs_df['Dampak (%)']], textposition='outside'
        ))
        fig_coeff.update_layout(
            title=f'Dampak Faktor — {APP_NAME}', height=320,
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='Inter'), margin=dict(t=45, b=30, l=140, r=60),
            xaxis=dict(showgrid=True, gridcolor='#f1f5f9'), yaxis=dict(showgrid=False),
            title_font=dict(size=13, family='DM Serif Display')
        )
        st.plotly_chart(fig_coeff, width="stretch")
    else:
        coeffs_df = None

    # ── Future forecast ────────────────────────────────────
    st.markdown('<div class="section-title">📅 Prediksi DAU ke Depan</div>', unsafe_allow_html=True)

    last_date = prophet_df['ds'].max()
    horizons  = {'1 Bulan': 30, '3 Bulan': 90, '6 Bulan': 180}
    h_colors  = {'1 Bulan': '#f97316', '3 Bulan': '#16a34a', '6 Bulan': '#ef4444'}
    future_liturgical_dates = indonesian_liturgical_df['Date'].tolist()
    summary_data = []

    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Aktual (historis)', line=dict(color='#1e3a5f', width=1.8)))

    for label, days in horizons.items():
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
        future_df    = pd.DataFrame({'ds': future_dates})
        future_df['is_liturgical_event']         = future_df['ds'].isin(future_liturgical_dates).astype(int)
        future_df['is_content_promotion_video']  = 0
        future_df['is_content_promotion_grafis'] = 0
        for c in regressor_cols:
            if c not in future_df.columns:
                future_df[c] = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast_future = model.predict(future_df)
        color = h_colors[label]
        fig_future.add_trace(go.Scatter(x=forecast_future['ds'], y=forecast_future['yhat'], name=f'Prediksi {label}', line=dict(color=color, dash='dash', width=2)))
        fig_future.add_trace(go.Scatter(
            x=pd.concat([forecast_future['ds'], forecast_future['ds'][::-1]]),
            y=pd.concat([forecast_future['yhat_upper'], forecast_future['yhat_lower'][::-1]]),
            fill='toself', fillcolor=color, opacity=0.08,
            line=dict(color='rgba(255,255,255,0)'), showlegend=False
        ))
        avg   = forecast_future['yhat'].mean()
        start = future_dates[0].strftime('%d %b %Y')
        end   = future_dates[-1].strftime('%d %b %Y')
        summary_data.append({'Horizon': label, 'Periode': f"{start} – {end}", 'Rata-rata DAU': f"{avg:,.0f} user/hari"})

    fig_future.update_layout(
        title='Prediksi DAU — 1, 3, dan 6 Bulan ke Depan (Tanpa Promosi)',
        height=440, hovermode='x unified',
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Inter'), margin=dict(t=45, b=30, l=40, r=20),
        legend=dict(orientation='h', y=-0.15),
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor='#f1f5f9'),
        title_font=dict(size=13, family='DM Serif Display')
    )
    st.plotly_chart(fig_future, width="stretch")

    # Summary table
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, width="stretch", hide_index=True)
    st.caption("* Prediksi bersifat indikatif berdasarkan pola historis, tanpa asumsi promosi atau perubahan produk.")

    # ══════════════════════════════════════════════════════
    # EXPORT PDF
    # ══════════════════════════════════════════════════════
    st.markdown('<div class="section-title">📄 Export Laporan</div>', unsafe_allow_html=True)

    if st.button("⬇️ Generate & Download PDF", type="primary"):
        with st.spinner("Membuat PDF…"):
            try:
                import plotly.io as pio
                from reportlab.lib.pagesizes import A4
                from reportlab.lib import colors as rl_colors
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import cm
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image, PageBreak
                from reportlab.lib.enums import TA_CENTER

                TMP_DIR = tempfile.mkdtemp(prefix="sabda_report_")

                def save_plotly(fig, name, w=1000, h=400):
                    path = os.path.join(TMP_DIR, f"{name}.png")
                    pio.write_image(fig, path, width=w, height=h, scale=2)
                    return path

                # Save all charts
                chart_paths = {}
                for col in ['DAU', '7d', 'FO', 'IA']:
                    chart_paths[col] = save_plotly(create_line_chart(merged_df, col, indonesian_liturgical_df, content_engagement_df, HAS_PROMOTION), f"chart_{col}")
                chart_paths['forecast']       = save_plotly(fig_fc, 'chart_forecast')
                chart_paths['future_forecast'] = save_plotly(fig_future, 'chart_future', w=1000, h=500)
                if HAS_PROMOTION and coeffs_df is not None:
                    chart_paths['koefisien'] = save_plotly(fig_coeff, 'chart_koefisien', w=900, h=320)

                PDF_PATH = os.path.join(
                    TMP_DIR,
                    f"Laporan_{APP_NAME}_{datetime.today().strftime('%Y%m%d')}.pdf",
                )
                doc    = SimpleDocTemplate(PDF_PATH, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm)
                styles = getSampleStyleSheet()
                W = A4[0] - 4*cm

                s_title    = ParagraphStyle('t', fontSize=16, alignment=TA_CENTER, spaceAfter=6,  fontName='Helvetica-Bold')
                s_subtitle = ParagraphStyle('s', fontSize=10, alignment=TA_CENTER, spaceAfter=12, textColor=rl_colors.grey)
                s_h1       = ParagraphStyle('h1', fontSize=13, spaceBefore=14, spaceAfter=6,  fontName='Helvetica-Bold')
                s_h2       = ParagraphStyle('h2', fontSize=11, spaceBefore=10, spaceAfter=4,  fontName='Helvetica-Bold', textColor=rl_colors.HexColor('#333333'))
                s_body     = ParagraphStyle('b', fontSize=9,  spaceAfter=4,   leading=14)
                s_caption  = ParagraphStyle('c', fontSize=8,  alignment=TA_CENTER, textColor=rl_colors.grey, spaceAfter=8)
                s_note     = ParagraphStyle('n', fontSize=8,  textColor=rl_colors.grey, spaceAfter=6)

                def hr(): return HRFlowable(width='100%', thickness=0.5, color=rl_colors.lightgrey, spaceAfter=8)

                def chart_elem(path, caption_text, width=W):
                    return [Image(path, width=width, height=width*0.4), Paragraph(caption_text, s_caption)]

                def mtable(data, col_widths=None):
                    t = Table(data, colWidths=col_widths or [W/len(data[0])]*len(data[0]))
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), rl_colors.HexColor('#f0f0f0')),
                        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE',   (0,0), (-1,-1), 9),
                        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [rl_colors.white, rl_colors.HexColor('#fafafa')]),
                        ('GRID',       (0,0), (-1,-1), 0.4, rl_colors.lightgrey),
                        ('TOPPADDING', (0,0), (-1,-1), 5),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                    ]))
                    return t

                TANGGAL = datetime.today().strftime('%d %B %Y')
                content = [
                    Spacer(1, 1.5*cm),
                    Paragraph("Laporan Analisis Engagement", s_title),
                    Paragraph(f"Aplikasi {APP_NAME} — SABDA", s_title),
                    Spacer(1, 0.3*cm),
                    Paragraph(f"Periode data: {PERIODE}", s_subtitle),
                    Paragraph(f"Dibuat: {TANGGAL}", s_subtitle),
                    hr(), Spacer(1, 0.5*cm),
                ]

                # Metrik ringkasan
                dau_m = merged_df['DAU'].mean(); dau_x = merged_df['DAU'].max(); dau_n = merged_df['DAU'].min()
                fo_m  = merged_df['FO'].mean()
                ia_l  = merged_df['IA'].iloc[-1]
                ret_m = merged_df['7d'].mean()
                content += [
                    Paragraph("1. Ringkasan Metrik", s_h1), hr(),
                    mtable([
                        ['Metrik', 'Rata-rata', 'Tertinggi', 'Terendah'],
                        ['DAU', f"{dau_m:,.0f}", f"{dau_x:,.0f}", f"{dau_n:,.0f}"],
                        ['First Open', f"{fo_m:,.0f}", f"{merged_df['FO'].max():,.0f}", f"{merged_df['FO'].min():,.0f}"],
                        ['7-Day Retention', f"{ret_m:.1f}%", f"{merged_df['7d'].max():.1f}%", f"{merged_df['7d'].min():.1f}%"],
                        ['Installed Audience (terakhir)', f"{ia_l:,.0f}", '—', '—'],
                    ], col_widths=[W*0.4, W*0.2, W*0.2, W*0.2]),
                    Spacer(1, 0.5*cm),
                ]

                # EDA charts
                label_map_col = {'DAU': 'Daily Active Users', '7d': '7-Day Retention (%)', 'FO': 'First Open', 'IA': 'Installed Audience'}
                content += [Paragraph("2. Visualisasi Data Harian", s_h1), hr()]
                for col in ['DAU', '7d', 'FO', 'IA']:
                    content += chart_elem(chart_paths[col], f"Grafik {label_map_col[col]} — {PERIODE}")
                    content += [Spacer(1, 0.3*cm)]

                # Evaluasi model
                content += [
                    PageBreak(),
                    Paragraph("3. Evaluasi Model Prediksi (Prophet)", s_h1), hr(),
                    Paragraph(f"Model dilatih menggunakan data <b>{len(train_df)} hari</b> pertama dan diuji pada <b>{len(test_df)} hari</b> terakhir.", s_body),
                    Spacer(1, 0.3*cm),
                    mtable([
                        ['Metrik', 'Nilai', 'Interpretasi'],
                        ['MAPE',  f"{mape:.2f}%",  f"Model meleset rata-rata {mape:.2f}% per hari"],
                        ['MAE',   f"{mae:,.0f}",   f"Rata-rata selisih {mae:,.0f} user/hari"],
                        ['RMSE',  f"{rmse:,.0f}",  "Sensitif terhadap spike DAU"],
                    ], col_widths=[W*0.2, W*0.2, W*0.6]),
                    Spacer(1, 0.3*cm),
                ] + chart_elem(chart_paths['forecast'], 'Perbandingan DAU Aktual vs Prediksi (periode test)')

                # Koefisien
                if HAS_PROMOTION and coeffs_df is not None:
                    content += [
                        PageBreak(),
                        Paragraph("4. Analisis Dampak Faktor terhadap DAU", s_h1), hr(),
                        Paragraph("Koefisien menunjukkan seberapa besar setiap faktor mendorong naik atau turun DAU dibandingkan baseline organik.", s_body),
                        Spacer(1, 0.3*cm),
                    ] + chart_elem(chart_paths['koefisien'], 'Dampak faktor dalam persen terhadap baseline DAU')
                    coeff_rows = [['Faktor', 'Dampak (%)', 'Arah']]
                    for _, row in coeffs_df.iterrows():
                        arah = '↑ Naik' if row['Dampak (%)'] > 0 else '↓ Turun'
                        coeff_rows.append([row['Label'], f"{row['Dampak (%)']:+.2f}%", arah])
                    content += [Spacer(1, 0.3*cm), mtable(coeff_rows, col_widths=[W*0.5, W*0.25, W*0.25])]

                # Prediksi ke depan
                sec = 5 if HAS_PROMOTION else 4
                content += [
                    PageBreak(),
                    Paragraph(f"{sec}. Prediksi DAU — 1, 3, dan 6 Bulan ke Depan", s_h1), hr(),
                    Paragraph("Prediksi dilakukan menggunakan model Prophet yang telah dilatih, tanpa asumsi aktivitas promosi (baseline organik). Interval kepercayaan ditampilkan sebagai area berbayang.", s_body),
                    Spacer(1, 0.3*cm),
                ] + chart_elem(chart_paths['future_forecast'], 'Prediksi DAU — 1, 3, dan 6 Bulan ke Depan (Tanpa Promosi)', width=W)

                summary_rows = [['Horizon', 'Periode', 'Rata-rata DAU']] + [[r['Horizon'], r['Periode'], r['Rata-rata DAU']] for r in summary_data]
                content += [
                    Spacer(1, 0.4*cm),
                    Paragraph("Ringkasan Prediksi per Horizon", s_h2),
                    Spacer(1, 0.2*cm),
                    mtable(summary_rows, col_widths=[W*0.2, W*0.5, W*0.3]),
                    Spacer(1, 0.3*cm),
                    Paragraph("* Prediksi bersifat indikatif berdasarkan pola historis. Tidak memperhitungkan aktivitas promosi, perubahan produk, atau faktor eksternal mendadak.", s_note),
                ]

                # Footer
                content += [
                    Spacer(1, 1*cm), hr(),
                    Paragraph(f"* Laporan ini dibuat otomatis dari dashboard analisis data Apps SABDA.", s_note),
                    Paragraph(f"* Model: Prophet | MAPE: {mape:.2f}% | Periode: {PERIODE}", s_note),
                    Paragraph(f"* Dibuat: {TANGGAL}", s_note),
                ]

                doc.build(content)

                with open(PDF_PATH, 'rb') as f:
                    pdf_bytes = f.read()

                st.download_button(
                    label="📥 Download PDF Laporan",
                    data=pdf_bytes,
                    file_name=f"Laporan_{APP_NAME}_{datetime.today().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
                st.success("✅ PDF berhasil dibuat!")

            except ImportError as e:
                st.error(f"Library tidak tersedia: {e}. Pastikan `reportlab` dan `kaleido` sudah terinstall.")
            except Exception as e:
                st.error(f"Gagal membuat PDF: {e}")

except ImportError:
    st.error("Library `prophet` tidak tersedia. Install dengan: `pip install prophet`")
except Exception as e:
    st.error(f"Error saat menjalankan model: {e}")
    st.exception(e)
