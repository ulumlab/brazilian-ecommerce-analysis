"""
Dashboard Analisis E-Commerce Olist
Proyek Akhir: Belajar Fundamental Analisis Data - Dicoding
Nama: Chamid Bahrul Ulum

Cara menjalankan:
    streamlit run dashboard/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import os

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Olist E-Commerce Dashboard",
    page_icon="shopping_cart",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Palet warna konsisten
COLOR_PRIMARY = '#2E86AB'
COLOR_ACCENT  = '#E84855'
COLOR_NEUTRAL = '#A8A8A8'
COLOR_SUCCESS = '#3BB273'
COLOR_WARNING = '#F4A261'

# ============================================================
# FUNGSI LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    """
    Memuat dan memproses semua data yang dibutuhkan untuk dashboard.
    Menggunakan cache agar tidak reload setiap kali interaksi.
    Mengembalikan dictionary berisi semua DataFrame yang sudah diproses.
    """
    # Tentukan path data relatif terhadap lokasi file dashboard
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'main_data.csv')

    # Load master data yang sudah diproses dari notebook
    df = pd.read_csv(data_path)

    # Konversi kolom timestamp
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['purchase_yearmonth'] = df['order_purchase_timestamp'].dt.to_period('M')

    return df


@st.cache_data
def compute_monthly_revenue(df):
    """Menghitung total revenue per bulan."""
    monthly = (
        df.groupby('purchase_yearmonth')['total_payment']
        .sum()
        .reset_index()
    )
    monthly.columns = ['yearmonth', 'revenue']
    monthly['yearmonth_str'] = monthly['yearmonth'].astype(str)
    return monthly


@st.cache_data
def compute_category_stats(df):
    """Menghitung statistik per kategori produk."""
    cat_stats = (
        df[df['product_category'] != 'unknown']
        .groupby('product_category')
        .agg(
            total_revenue = ('total_payment', 'sum'),
            n_orders      = ('order_id', 'count'),
            avg_score     = ('review_score', 'mean')
        )
        .reset_index()
        .sort_values('total_revenue', ascending=False)
    )
    return cat_stats


@st.cache_data
def compute_rfm(df):
    """Menghitung nilai dan segmen RFM per pelanggan unik."""
    reference_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby('customer_unique_id')
        .agg(
            recency   = ('order_purchase_timestamp', lambda x: (reference_date - x.max()).days),
            frequency = ('order_id', 'count'),
            monetary  = ('total_payment', 'sum')
        )
        .reset_index()
    )

    rfm['R_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1])
    rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
    rfm['M_score'] = pd.qcut(rfm['monetary'], q=5, labels=[1, 2, 3, 4, 5])

    rfm['R_score'] = rfm['R_score'].astype(int)
    rfm['F_score'] = rfm['F_score'].astype(int)
    rfm['M_score'] = rfm['M_score'].astype(int)

    def segment_customer(row):
        r, f = row['R_score'], row['F_score']
        if r >= 4 and f >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 2:
            return 'New Customers'
        elif r >= 3 and f <= 2:
            return 'Potential Loyalists'
        elif r <= 2 and f >= 3:
            return 'At Risk'
        elif r <= 2 and f <= 2:
            return 'Lost Customers'
        else:
            return 'Need Attention'

    rfm['segment'] = rfm.apply(segment_customer, axis=1)
    return rfm


@st.cache_data
def compute_state_stats(df):
    """Menghitung statistik per negara bagian."""
    state_stats = (
        df.groupby('customer_state')
        .agg(
            n_orders      = ('order_id', 'count'),
            total_revenue = ('total_payment', 'sum'),
            avg_score     = ('review_score', 'mean')
        )
        .reset_index()
        .sort_values('n_orders', ascending=False)
    )
    return state_stats


# ============================================================
# LOAD DATA
# ============================================================
try:
    df = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📊")
    st.title("Olist Dashboard")
    st.markdown("**Analisis E-Commerce Brazil**")
    st.divider()

    st.subheader("Filter Data")

    if data_loaded:
        # Filter rentang waktu
        min_date = df['order_purchase_timestamp'].min().date()
        max_date = df['order_purchase_timestamp'].max().date()

        date_range = st.date_input(
            "Rentang Waktu",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        # Filter negara bagian
        all_states = ['Semua'] + sorted(df['customer_state'].dropna().unique().tolist())
        selected_state = st.selectbox("Negara Bagian", all_states)

        st.divider()

        # Terapkan filter
        df_filtered = df[
            (df['order_purchase_timestamp'].dt.date >= date_range[0]) &
            (df['order_purchase_timestamp'].dt.date <= date_range[1])
        ].copy()

        if selected_state != 'Semua':
            df_filtered = df_filtered[df_filtered['customer_state'] == selected_state]

    st.divider()
    st.caption("Proyek Akhir - Dicoding")
    st.caption("Chamid Bahrul Ulum")
    st.caption("Dataset: Brazilian E-Commerce Olist")


# ============================================================
# KONTEN UTAMA
# ============================================================

if not data_loaded:
    st.error("""
    **File data tidak ditemukan.**

    Pastikan file `main_data.csv` ada di dalam folder `dashboard/`.

    Untuk membuat file ini, jalankan terlebih dahulu notebook analisis
    dan ekspor master DataFrame dengan perintah berikut di akhir notebook:

    ```python
    df_main.to_csv('dashboard/main_data.csv', index=False)
    ```
    """)
    st.stop()

# Header utama
st.title("Dashboard Analisis E-Commerce Olist")
st.markdown("**Periode: Januari 2017 – Agustus 2018 | Platform: Olist, Brazil**")
st.divider()

# ============================================================
# SECTION 1: METRIK RINGKASAN
# ============================================================
st.subheader("Ringkasan Performa")

col1, col2, col3, col4 = st.columns(4)

total_orders   = len(df_filtered)
total_revenue  = df_filtered['total_payment'].sum()
avg_score      = df_filtered['review_score'].mean()
unique_customers = df_filtered['customer_unique_id'].nunique()

with col1:
    st.metric(
        label="Total Order",
        value=f"{total_orders:,}",
        delta=None
    )

with col2:
    st.metric(
        label="Total Revenue",
        value=f"BRL {total_revenue/1e6:.2f}M"
    )

with col3:
    st.metric(
        label="Rata-Rata Review Score",
        value=f"{avg_score:.2f} / 5.0"
    )

with col4:
    st.metric(
        label="Pelanggan Unik",
        value=f"{unique_customers:,}"
    )

st.divider()

# ============================================================
# SECTION 2: TREN REVENUE DAN KATEGORI (Pertanyaan 1)
# ============================================================
st.subheader("Pertanyaan 1: Tren Revenue dan Kategori Produk Teratas")

monthly_revenue = compute_monthly_revenue(df_filtered)
category_stats  = compute_category_stats(df_filtered)
top10_cats      = category_stats.head(10)

col_left, col_right = st.columns([2, 1])

with col_left:
    # Tren revenue bulanan
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    fig1.patch.set_facecolor('none')
    ax1.set_facecolor('none')

    x_vals = range(len(monthly_revenue))

    # Tentukan satuan yang sesuai berdasarkan nilai maksimum revenue
    max_rev = monthly_revenue['revenue'].max() if len(monthly_revenue) > 0 else 1
    if max_rev >= 500_000:
        divisor   = 1_000_000
        rev_label = 'Revenue (Juta BRL)'
        fmt       = '%.2f'
    elif max_rev >= 1_000:
        divisor   = 1_000
        rev_label = 'Revenue (Ribu BRL)'
        fmt       = '%.1f'
    else:
        divisor   = 1
        rev_label = 'Revenue (BRL)'
        fmt       = '%.0f'

    rev_values = monthly_revenue['revenue'] / divisor

    ax1.fill_between(x_vals, rev_values, alpha=0.15, color=COLOR_PRIMARY)
    ax1.plot(x_vals, rev_values,
             color=COLOR_PRIMARY, linewidth=2.5, marker='o',
             markersize=5, markerfacecolor='white', markeredgewidth=2)

    if len(monthly_revenue) > 0:
        peak_idx = monthly_revenue['revenue'].idxmax()
        ax1.plot(peak_idx, rev_values.iloc[peak_idx],
                 'o', color=COLOR_ACCENT, markersize=10, zorder=5)

    ax1.set_xticks(x_vals)
    ax1.set_xticklabels(monthly_revenue['yearmonth_str'],
                        rotation=45, ha='right', fontsize=7)
    ax1.set_ylabel(rev_label, fontsize=9)
    ax1.set_title('Tren Revenue Bulanan', fontsize=11, fontweight='bold')
    ax1.set_ylim(bottom=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))

    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

with col_right:
    st.markdown("**Top 10 Kategori Revenue**")

    n_cats = len(top10_cats)
    fig_height = max(3.5, n_cats * 0.45)
    fig2, ax2 = plt.subplots(figsize=(5, fig_height))
    fig2.patch.set_facecolor('none')
    ax2.set_facecolor('none')

    # Warna: rank 1 = aksen (merah), rank 2-3 = primary (biru), sisanya = netral
    def get_bar_colors(n):
        colors = []
        for i in range(n):
            if i == 0:
                colors.append(COLOR_ACCENT)
            elif i < 3:
                colors.append(COLOR_PRIMARY)
            else:
                colors.append(COLOR_NEUTRAL)
        return colors

    bar_colors = get_bar_colors(n_cats)

    # Satuan adaptif berdasarkan nilai maksimum
    max_cat_rev = top10_cats['total_revenue'].max() if n_cats > 0 else 1
    if max_cat_rev >= 500_000:
        cat_divisor  = 1_000_000
        cat_xlabel   = 'Juta BRL'
        cat_fmt      = lambda v: f'{v:.2f}M'
    elif max_cat_rev >= 1_000:
        cat_divisor  = 1_000
        cat_xlabel   = 'Ribu BRL'
        cat_fmt      = lambda v: f'{v:.1f}K'
    else:
        cat_divisor  = 1
        cat_xlabel   = 'BRL'
        cat_fmt      = lambda v: f'{v:.0f}'

    cat_values = top10_cats['total_revenue'] / cat_divisor

    bars = ax2.barh(
        top10_cats['product_category'],
        cat_values,
        color=bar_colors, edgecolor='white', height=0.6
    )

    max_val = cat_values.max() if len(cat_values) > 0 else 1
    for bar, val in zip(bars, cat_values):
        offset = max_val * 0.03
        ax2.text(val + offset, bar.get_y() + bar.get_height() / 2,
                 cat_fmt(val), va='center', fontsize=7)

    ax2.set_xlim(left=0, right=max_val * 1.3)
    ax2.set_xlabel(cat_xlabel, fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='y', labelsize=7)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.yaxis.grid(False)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

with st.expander("Lihat Insight Pertanyaan 1"):
    if len(monthly_revenue) > 0 and len(top10_cats) > 0:
        peak_month  = monthly_revenue.loc[monthly_revenue['revenue'].idxmax(), 'yearmonth_str']
        peak_val    = monthly_revenue['revenue'].max()
        min_val     = monthly_revenue['revenue'].min()
        avg_val     = monthly_revenue['revenue'].mean()
        top1_cat    = top10_cats.iloc[0]['product_category']
        top1_rev    = top10_cats.iloc[0]['total_revenue']
        top2_cat    = top10_cats.iloc[1]['product_category'] if len(top10_cats) > 1 else '-'

        # Satuan label dinamis
        def fmt_rev(v):
            if v >= 1_000_000: return f"BRL {v/1e6:.2f}M"
            if v >= 1_000:     return f"BRL {v/1e3:.1f}K"
            return f"BRL {v:,.0f}"

        st.info(f"""
    **Insight:**
    - Revenue tertinggi terjadi pada **{peak_month}** sebesar **{fmt_rev(peak_val)}**
    - Revenue terendah: {fmt_rev(min_val)} | Rata-rata bulanan: {fmt_rev(avg_val)}
    - Kategori **{top1_cat}** ({fmt_rev(top1_rev)}) dan **{top2_cat}** adalah kontributor revenue terbesar
    - Puncak {peak_month} sangat kemungkinan dipicu event **Black Friday**
        """)
    else:
        st.info("Tidak ada data untuk filter yang dipilih.")

st.divider()

# ============================================================
# SECTION 3: KEPUASAN PELANGGAN (Pertanyaan 2)
# ============================================================
st.subheader("Pertanyaan 2: Kepuasan Pelanggan per Kategori Produk")

# Ambil 10 kategori dengan volume order terbanyak
top10_by_volume = (
    category_stats.sort_values('n_orders', ascending=False).head(10)
)
top10_names = top10_by_volume['product_category'].tolist()

df_top10 = df_filtered[
    (df_filtered['product_category'].isin(top10_names)) &
    (df_filtered['review_score'].notna())
].copy()

# Hitung proporsi skor per kategori
score_by_cat = (
    df_top10.groupby(['product_category', 'review_score'])
    .size()
    .unstack(fill_value=0)
    .apply(lambda x: x / x.sum() * 100, axis=1)
)

category_score = (
    df_top10.groupby('product_category')['review_score']
    .mean()
    .reset_index()
    .sort_values('review_score', ascending=True)
)

sorted_cats = category_score['product_category'].tolist()
if len(score_by_cat) > 0 and len(sorted_cats) > 0:
    score_by_cat = score_by_cat.reindex(sorted_cats)

col_a, col_b = st.columns([3, 1])

with col_a:
    score_colors = {
        1: '#C0392B',
        2: '#E67E22',
        3: '#F1C40F',
        4: '#2ECC71',
        5: '#1A7A4A'
    }

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    fig3.patch.set_facecolor('none')
    ax3.set_facecolor('none')

    bottom = np.zeros(len(score_by_cat))
    for score in [1, 2, 3, 4, 5]:
        if score in score_by_cat.columns:
            values = score_by_cat[score].values
            ax3.barh(score_by_cat.index, values, left=bottom,
                     color=score_colors[score], label=f'Score {score}',
                     edgecolor='white', linewidth=0.5)
            bottom += values

    ax3.set_xlabel('Proporsi Ulasan (%)', fontsize=9)
    ax3.set_title('Distribusi Review Score per Kategori\n(urut dari kepuasan terendah ke tertinggi)',
                  fontsize=10, fontweight='bold')
    ax3.set_xlim(0, 100)
    ax3.legend(loc='upper right', fontsize=8, title='Score', title_fontsize=8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.tick_params(axis='y', labelsize=7.5)

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

with col_b:
    st.markdown("**Rata-Rata Score**")

    if len(category_score) > 0:
        overall_avg = df_top10['review_score'].mean()

        fig4, ax4 = plt.subplots(figsize=(4, 5))
        fig4.patch.set_facecolor('none')
        ax4.set_facecolor('none')

        bar_colors_score = [
            COLOR_ACCENT if v < overall_avg else COLOR_SUCCESS
            for v in category_score['review_score']
        ]

        bars4 = ax4.barh(
            category_score['product_category'],
            category_score['review_score'],
            color=bar_colors_score, edgecolor='white', height=0.6
        )

        ax4.axvline(x=overall_avg, color='#555', linestyle='--',
                    linewidth=1.5, alpha=0.7, label=f'Rata-rata: {overall_avg:.2f}')

        for bar, val in zip(bars4, category_score['review_score']):
            ax4.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{val:.2f}', va='center', fontsize=7)

        score_min = category_score['review_score'].min()
        score_max = category_score['review_score'].max()
        margin    = max(0.1, (score_max - score_min) * 0.35)
        ax4.set_xlim(
            max(1.0, score_min - margin),
            min(5.2, score_max + margin * 2)
        )
        ax4.set_xlabel('Rata-Rata Score', fontsize=8)
        ax4.legend(fontsize=7, loc='lower right')
        ax4.set_yticks([])
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

with st.expander("Lihat Insight Pertanyaan 2"):
    if len(df_top10) > 0 and len(category_score) > 0:
        overall_avg_score = df_top10['review_score'].mean()
        lowest_cat  = category_score.iloc[0]['product_category']
        lowest_val  = category_score.iloc[0]['review_score']
        highest_cat = category_score.iloc[-1]['product_category']
        highest_val = category_score.iloc[-1]['review_score']
        pct_score5  = (df_top10['review_score'] == 5).sum() / df_top10['review_score'].notna().sum() * 100
        st.info(f"""
    **Insight:**
    - Rata-rata skor kepuasan pada filter ini: **{overall_avg_score:.3f} / 5.0**
    - Kepuasan **terendah**: `{lowest_cat}` (rata-rata {lowest_val:.3f})
    - Kepuasan **tertinggi**: `{highest_cat}` (rata-rata {highest_val:.3f})
    - **{pct_score5:.1f}%** pelanggan memberikan skor 5 (sangat puas)
    - Kategori dengan produk fisik berat/besar cenderung mendapatkan skor lebih rendah
        """)
    else:
        st.info("Tidak ada data review untuk filter yang dipilih.")

st.divider()

# ============================================================
# SECTION 4: RFM ANALYSIS
# ============================================================
st.subheader("Analisis Lanjutan 1: RFM — Segmentasi Pelanggan")

rfm = compute_rfm(df_filtered)

segment_summary = (
    rfm.groupby('segment')
    .agg(
        n_customers   = ('customer_unique_id', 'count'),
        avg_recency   = ('recency', 'mean'),
        avg_frequency = ('frequency', 'mean'),
        avg_monetary  = ('monetary', 'mean')
    )
    .round(1)
    .reset_index()
    .sort_values('n_customers', ascending=False)
)

segment_colors = {
    'Champions'          : '#1A7A4A',
    'Loyal Customers'    : '#2E86AB',
    'Potential Loyalists': '#3BB273',
    'New Customers'      : '#74C69D',
    'Need Attention'     : '#F4A261',
    'At Risk'            : '#E84855',
    'Lost Customers'     : '#C0392B'
}

col_rfm1, col_rfm2 = st.columns([1, 2])

with col_rfm1:
    st.markdown("**Distribusi Segmen Pelanggan**")

    n_segments = len(segment_summary)
    rfm_fig_h  = max(3.0, n_segments * 0.55)
    fig5, ax5  = plt.subplots(figsize=(5, rfm_fig_h))
    fig5.patch.set_facecolor('none')
    ax5.set_facecolor('none')

    sorted_rfm = segment_summary.sort_values('n_customers', ascending=True)
    colors_rfm = [segment_colors.get(s, COLOR_NEUTRAL) for s in sorted_rfm['segment']]

    bars5 = ax5.barh(sorted_rfm['segment'], sorted_rfm['n_customers'],
                     color=colors_rfm, edgecolor='white', height=0.6)

    # Offset label dinamis berdasarkan nilai maksimum
    max_cust = sorted_rfm['n_customers'].max() if len(sorted_rfm) > 0 else 1
    label_offset = max_cust * 0.02

    for bar, val in zip(bars5, sorted_rfm['n_customers']):
        ax5.text(val + label_offset, bar.get_y() + bar.get_height() / 2,
                 f'{val:,}', va='center', fontsize=7.5)

    # xlim dinamis agar label tidak terpotong
    ax5.set_xlim(left=0, right=max_cust * 1.25)
    ax5.set_xlabel('Jumlah Pelanggan', fontsize=9)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.tick_params(axis='y', labelsize=8)
    ax5.yaxis.grid(False)

    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

with col_rfm2:
    st.markdown("**Tabel Ringkasan Segmen RFM**")

    display_rfm = segment_summary.copy()
    display_rfm['avg_monetary'] = display_rfm['avg_monetary'].apply(lambda x: f'BRL {x:,.0f}')
    display_rfm['avg_recency']  = display_rfm['avg_recency'].apply(lambda x: f'{x:.0f} hari')
    display_rfm.columns = ['Segmen', 'Jumlah Pelanggan', 'Rata-rata Recency',
                           'Rata-rata Frequency', 'Rata-rata Monetary']
    st.dataframe(display_rfm, use_container_width=True, hide_index=True)

    st.markdown("""
    **Definisi Segmen:**
    - **Champions**: Baru beli, sering, dan nilai tinggi — pelanggan terbaik
    - **Loyal Customers**: Sering beli, perlu dipertahankan
    - **New Customers**: Baru bergabung, potensi untuk diloyalkan
    - **At Risk**: Dulu aktif, kini sudah lama tidak beli
    - **Lost Customers**: Lama tidak aktif, perlu win-back campaign
    """)

st.divider()

# ============================================================
# SECTION 5: GEOSPATIAL ANALYSIS
# ============================================================
st.subheader("Analisis Lanjutan 2: Geospatial — Distribusi Order di Brazil")

state_stats = compute_state_stats(df_filtered)

col_geo1, col_geo2 = st.columns([1, 2])

with col_geo1:
    st.markdown("**Top 15 Negara Bagian**")

    top15 = state_stats.head(15)
    total_all = state_stats['n_orders'].sum()

    n_states   = len(top15)
    geo_fig_h  = max(3.0, n_states * 0.42)
    fig6, ax6  = plt.subplots(figsize=(5, geo_fig_h))
    fig6.patch.set_facecolor('none')
    ax6.set_facecolor('none')

    # Warna: state pertama (terbesar) = aksen, kedua = primary, sisanya = netral
    def state_color(i):
        if i == 0: return COLOR_ACCENT
        if i == 1: return COLOR_PRIMARY
        return COLOR_NEUTRAL

    bar_colors_state = [state_color(i) for i in range(n_states)]

    bars6 = ax6.barh(top15['customer_state'], top15['n_orders'],
                     color=bar_colors_state, edgecolor='white', height=0.6)

    max_ord   = top15['n_orders'].max() if n_states > 0 else 1
    lbl_offset = max_ord * 0.02

    for bar, n in zip(bars6, top15['n_orders']):
        pct = n / total_all * 100
        ax6.text(n + lbl_offset, bar.get_y() + bar.get_height() / 2,
                 f'{pct:.1f}%', va='center', fontsize=7.5)

    ax6.set_xlim(left=0, right=max_ord * 1.25)
    ax6.set_xlabel('Jumlah Order', fontsize=9)
    ax6.invert_yaxis()
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()

with col_geo2:
    st.markdown("**Peta Distribusi Pelanggan (Heatmap)**")

    # Membuat peta folium sederhana dengan marker per negara bagian
    # Koordinat representatif negara bagian utama Brazil
    state_coords = {
        'SP': (-23.55, -46.63), 'RJ': (-22.91, -43.17), 'MG': (-19.92, -43.94),
        'RS': (-30.03, -51.22), 'PR': (-25.43, -49.27), 'SC': (-27.60, -48.55),
        'BA': (-12.97, -38.50), 'GO': (-16.69, -49.25), 'ES': (-20.32, -40.34),
        'PE': (-8.05, -34.88),  'CE': (-3.72, -38.54),  'MA': (-2.53, -44.30),
        'MT': (-15.60, -56.10), 'MS': (-20.44, -54.65), 'PA': (-1.46, -48.50),
        'AM': (-3.10, -60.02),  'DF': (-15.78, -47.93), 'RN': (-5.79, -35.21),
        'PB': (-7.12, -34.86),  'AL': (-9.66, -35.74),  'PI': (-5.09, -42.80),
        'TO': (-10.25, -48.34), 'SE': (-10.91, -37.07), 'AC': (-9.97, -67.81),
        'RO': (-8.76, -63.90),  'RR': (2.82, -60.68),   'AP': (0.04, -51.07)
    }

    # Buat peta
    m = folium.Map(
        location=[-14.0, -51.0],
        zoom_start=4,
        tiles='CartoDB positron'
    )

    # Skala warna berdasarkan jumlah order
    max_orders = state_stats['n_orders'].max()

    for _, row in state_stats.iterrows():
        state = row['customer_state']
        if state in state_coords:
            lat, lon = state_coords[state]
            # Ukuran circle proporsional dengan jumlah order
            radius = (row['n_orders'] / max_orders) * 40000

            folium.Circle(
                location=[lat, lon],
                radius=radius,
                color=COLOR_PRIMARY,
                fill=True,
                fill_opacity=0.5,
                popup=folium.Popup(
                    f"<b>{state}</b><br>"
                    f"Orders: {row['n_orders']:,}<br>"
                    f"Revenue: BRL {row['total_revenue']:,.0f}<br>"
                    f"Avg Score: {row['avg_score']:.2f}",
                    max_width=200
                ),
                tooltip=f"{state}: {row['n_orders']:,} orders"
            ).add_to(m)

    st_folium(m, width=700, height=400)

with st.expander("Lihat Insight Geospatial"):
    if len(state_stats) > 0:
        top_state     = state_stats.iloc[0]['customer_state']
        top_state_n   = state_stats.iloc[0]['n_orders']
        top_state_pct = top_state_n / state_stats['n_orders'].sum() * 100
        top5_pct      = state_stats.head(5)['n_orders'].sum() / state_stats['n_orders'].sum() * 100
        st.info(f"""
    **Insight:**
    - **{top_state}** mendominasi dengan **{top_state_n:,} order ({top_state_pct:.1f}% dari total filter ini)**
    - Top 5 negara bagian menyumbang **{top5_pct:.1f}%** dari total order pada filter ini
    - Wilayah utara dan barat daya Brazil relatif underserved — peluang ekspansi pasar
    - Klik setiap circle pada peta untuk melihat detail per negara bagian
        """)
    else:
        st.info("Tidak ada data untuk filter yang dipilih.")

st.divider()

# ============================================================
# SECTION 6: CLUSTERING KATEGORI
# ============================================================
st.subheader("Analisis Lanjutan 3: Clustering Kategori Produk")

cat_cluster = category_stats[category_stats['n_orders'] >= 50].copy()

if len(cat_cluster) > 0:
    median_revenue_c = cat_cluster['total_revenue'].median()
    median_score_c   = cat_cluster['avg_score'].median()

    def assign_cluster(row):
        high_rev   = row['total_revenue'] >= median_revenue_c
        high_score = row['avg_score'] >= median_score_c
        if high_rev and high_score:
            return 'Stars'
        elif high_rev and not high_score:
            return 'Cash Cows'
        elif not high_rev and high_score:
            return 'Rising Stars'
        else:
            return 'Underperformers'

    cat_cluster['cluster'] = cat_cluster.apply(assign_cluster, axis=1)

    cluster_colors = {
        'Stars'          : '#1A7A4A',
        'Cash Cows'      : '#2E86AB',
        'Rising Stars'   : '#F4A261',
        'Underperformers': '#E84855'
    }

    cluster_markers = {
        'Stars'          : 'o',
        'Cash Cows'      : 's',
        'Rising Stars'   : '^',
        'Underperformers': 'X'
    }

    # Satuan Y adaptif untuk clustering chart
    max_cluster_rev = cat_cluster['total_revenue'].max() if len(cat_cluster) > 0 else 1
    if max_cluster_rev >= 500_000:
        cluster_div    = 1_000_000
        cluster_ylabel = 'Juta BRL'
    elif max_cluster_rev >= 1_000:
        cluster_div    = 1_000
        cluster_ylabel = 'Ribu BRL'
    else:
        cluster_div    = 1
        cluster_ylabel = 'BRL'

    fig7, ax7 = plt.subplots(figsize=(11, 6))
    fig7.patch.set_facecolor('none')
    ax7.set_facecolor('none')

    for cluster_name, group in cat_cluster.groupby('cluster'):
        ax7.scatter(
            group['avg_score'],
            group['total_revenue'] / cluster_div,
            c=cluster_colors[cluster_name],
            marker=cluster_markers[cluster_name],
            s=np.clip(group['n_orders'] / 5, 20, 800),
            label=f'{cluster_name} (n={len(group)})',
            alpha=0.75, edgecolors='white', linewidths=0.5, zorder=3
        )

    ax7.axvline(x=median_score_c, color='#888', linestyle='--', linewidth=1.2, alpha=0.6)
    ax7.axhline(y=median_revenue_c / cluster_div, color='#888', linestyle='--', linewidth=1.2, alpha=0.6)

    ax7.set_xlabel('Rata-Rata Review Score (Kepuasan)', fontsize=10)
    ax7.set_ylabel(f'Total Revenue ({cluster_ylabel})', fontsize=10)
    ax7.set_title('Matriks Clustering Kategori Produk: Revenue vs Kepuasan Pelanggan\n(Ukuran titik proporsional dengan jumlah order)',
                  fontsize=11, fontweight='bold')
    ax7.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig7)
    plt.close()

    # Tampilkan tabel ringkasan cluster
    cluster_table = (
        cat_cluster.groupby('cluster')
        .agg(
            n_kategori    = ('product_category', 'count'),
            total_revenue = ('total_revenue', 'sum'),
            avg_score     = ('avg_score', 'mean')
        )
        .reset_index()
        .round(2)
    )
    # Format revenue cluster table sesuai satuan yang sama dengan scatter chart
    def fmt_cluster_rev(v):
        if v >= 1_000_000: return f"BRL {v/1e6:.2f}M"
        if v >= 1_000:     return f"BRL {v/1e3:.1f}K"
        return f"BRL {v:,.0f}"
    cluster_table['total_revenue'] = cluster_table['total_revenue'].apply(fmt_cluster_rev)
    cluster_table.columns = ['Cluster', 'Jumlah Kategori', 'Total Revenue', 'Rata-rata Score']
    st.dataframe(cluster_table, use_container_width=True, hide_index=True)

st.divider()

# ============================================================
# SECTION 7: KESIMPULAN
# ============================================================
st.subheader("Kesimpulan dan Rekomendasi")

col_c1, col_c2 = st.columns(2)

with col_c1:
    st.markdown("**Kesimpulan Pertanyaan 1 — Revenue**")
    st.success("""
    Revenue platform Olist tumbuh dari BRL 127.546 (Januari 2017) hingga puncaknya
    BRL 1.153.528 (November 2017) — hampir 9x lipat dalam 10 bulan, sangat kemungkinan
    dipicu event Black Friday. Rata-rata revenue bulanan sepanjang periode analisis
    adalah BRL 768.794. Kategori `health_beauty` (BRL 1,4M), `watches_gifts` (BRL 1,26M),
    dan `bed_bath_table` (BRL 1,22M) mendominasi kontribusi revenue.
    """)

    st.markdown("**Kesimpulan Pertanyaan 2 — Kepuasan**")
    st.warning("""
    Rata-rata kepuasan pelanggan secara keseluruhan adalah **4,156 / 5,0** dengan
    59,2% pelanggan memberikan skor sempurna (5). Namun dari 10 kategori dengan
    volume order tertinggi, `bed_bath_table` (4,010) dan `telephony` (4,056) memiliki
    kepuasan terendah — kemungkinan akibat masalah pengiriman produk besar/berat
    dan ekspektasi pelanggan yang tidak terpenuhi.
    """)

with col_c2:
    st.markdown("**Rekomendasi Action Item**")
    st.info("""
    1. **Maksimalkan Event Musiman (Black Friday)**: Revenue November 2017 mencapai
       BRL 1,15M — 50% di atas rata-rata bulanan. Siapkan stok dan infrastruktur
       minimal 2 bulan sebelumnya; fokus promosi pada `health_beauty` dan `watches_gifts`.

    2. **Perbaiki Kepuasan Kategori Bermasalah**: Lakukan audit seller di kategori
       `bed_bath_table` dan `telephony`, terapkan SLA pengiriman khusus untuk produk
       besar, dan tambahkan program proteksi produk/garansi.

    3. **Program Win-Back untuk Segmen At Risk**: Dari 93.104 pelanggan unik,
       22.079 (23,7%) masuk segmen At Risk dengan rata-rata 393 hari tidak aktif.
       Kampanye email personal dengan voucher diskon dapat efektif mereaktivasi mereka.

    4. **Program Loyalitas untuk New Customers**: Hampir semua pelanggan hanya
       bertransaksi satu kali (rata-rata frequency 1,00–1,09). Cashback atau poin
       reward untuk pembelian kedua dapat meningkatkan repeat purchase rate.

    5. **Ekspansi Geografis**: SP+RJ+MG menyumbang 66% total order. Rekrut seller
       lokal di BA, GO, dan PA yang underserved namun berpopulasi besar.
    """)

# Footer
st.divider()
st.caption("Dashboard dibuat sebagai bagian dari Proyek Akhir Kelas Belajar Fundamental Analisis Data — Dicoding | Chamid Bahrul Ulum")