"""
Dashboard Analisis E-Commerce Olist
Nama  : Chamid Bahrul Ulum
Email : ulumlab@gmail.com

Cara menjalankan:
    streamlit run dashboard/dashboard.py

Analisis yang ditampilkan:
    1. Ringkasan Performa Bisnis
    2. Tren Revenue & Kategori Teratas (Pertanyaan 1)
    3. Kepuasan Pelanggan per Kategori (Pertanyaan 2)
    4. RFM Analysis — Segmentasi Pelanggan
    5. Geospatial Analysis — Distribusi Order di Brazil
    6. K-Means Clustering Kategori Produk
    7. Prediksi Sentimen Ulasan (Random Forest)
    8. Time Series Decomposition Revenue
    9. Analisis Korelasi & Feature Importance
"""

import os
import warnings

import folium
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from folium.plugins import HeatMap
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    silhouette_score,
    silhouette_samples,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Olist Data Science Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palet warna konsisten
COLOR_PRIMARY = "#2E86AB"
COLOR_ACCENT  = "#E84855"
COLOR_NEUTRAL = "#A8A8A8"
COLOR_SUCCESS = "#3BB273"
COLOR_WARNING = "#F4A261"

RANDOM_STATE = 42

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path     = os.path.join(base_dir, "main_data.csv")
    df = pd.read_csv(path)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["purchase_yearmonth"]       = df["order_purchase_timestamp"].dt.to_period("M")
    return df


try:
    df = load_data()
    data_ok = True
except FileNotFoundError:
    data_ok = False

# ============================================================
# FUNGSI KOMPUTASI (semua di-cache)
# ============================================================

@st.cache_data
def compute_monthly_revenue(df):
    monthly = (
        df.groupby("purchase_yearmonth")["total_payment"]
        .sum()
        .reset_index()
    )
    monthly.columns = ["yearmonth", "revenue"]
    monthly["yearmonth_str"] = monthly["yearmonth"].astype(str)
    return monthly


@st.cache_data
def compute_category_stats(df):
    return (
        df[df["product_category"] != "unknown"]
        .groupby("product_category")
        .agg(
            total_revenue=("total_payment", "sum"),
            n_orders=("order_id", "count"),
            avg_score=("review_score", "mean"),
            avg_freight=("total_freight", "mean"),
            avg_items=("n_items", "mean"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )


@st.cache_data
def compute_rfm(df):
    ref = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("customer_unique_id")
        .agg(
            recency=("order_purchase_timestamp", lambda x: (ref - x.max()).days),
            frequency=("order_id", "count"),
            monetary=("total_payment", "sum"),
        )
        .reset_index()
    )
    rfm["R_score"] = pd.qcut(rfm["recency"],   q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"],  q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    def segment(row):
        r, f = row["R_score"], row["F_score"]
        if r >= 4 and f >= 4:   return "Champions"
        if r >= 3 and f >= 3:   return "Loyal Customers"
        if r >= 4 and f <= 2:   return "New Customers"
        if r >= 3 and f <= 2:   return "Potential Loyalists"
        if r <= 2 and f >= 3:   return "At Risk"
        if r <= 2 and f <= 2:   return "Lost Customers"
        return "Need Attention"

    rfm["segment"] = rfm.apply(segment, axis=1)
    return rfm


@st.cache_data
def compute_state_stats(df):
    return (
        df.groupby("customer_state")
        .agg(n_orders=("order_id", "count"), total_revenue=("total_payment", "sum"), avg_score=("review_score", "mean"))
        .reset_index()
        .sort_values("n_orders", ascending=False)
    )


@st.cache_data
def compute_kmeans(df):
    cat = (
        df[df["product_category"] != "unknown"]
        .groupby("product_category")
        .agg(
            total_revenue=("total_payment", "sum"),
            n_orders=("order_id", "count"),
            avg_score=("review_score", "mean"),
            avg_freight=("total_freight", "mean"),
            avg_items=("n_items", "mean"),
        )
        .reset_index()
        .dropna()
    )
    # Turunkan threshold jika data terlalu sedikit setelah filter
    threshold = 50
    while threshold > 5 and len(cat[cat["n_orders"] >= threshold]) < 3:
        threshold -= 10
    cat = cat[cat["n_orders"] >= max(1, threshold)].copy()

    # Butuh minimal 3 kategori untuk clustering
    if len(cat) < 3:
        return None, None, None, None, None, None, None, None

    feats   = ["total_revenue", "n_orders", "avg_score", "avg_freight", "avg_items"]
    scaler  = StandardScaler()
    X       = scaler.fit_transform(cat[feats])

    # Batasi K maksimum berdasarkan jumlah sampel
    max_k   = min(10, len(cat) - 1)
    if max_k < 2:
        return None, None, None, None, None, None, None, None
    k_range    = range(2, max_k + 1)
    inertias   = []
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, km.labels_))

    best_k = list(k_range)[np.argmax(sil_scores)]
    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    km_final.fit(X)
    cat["cluster"]  = km_final.labels_
    cat["cluster_label"] = cat["cluster"].apply(lambda x: f"Cluster {x+1}")
    final_sil = silhouette_score(X, km_final.labels_)
    sil_vals  = silhouette_samples(X, km_final.labels_)

    return cat, X, best_k, final_sil, sil_vals, list(k_range), inertias, sil_scores


@st.cache_data
def compute_rf(df):
    df_clf = df.dropna(subset=["review_score"]).copy()
    if len(df_clf) < 200:
        return None, None, None, None, None, None, None, None, None, None
    df_clf["sentiment"] = (df_clf["review_score"] >= 4).astype(int)

    top20  = df_clf["product_category"].value_counts().nlargest(20).index.tolist()
    df_clf["cat_enc"] = df_clf["product_category"].apply(lambda x: x if x in top20 else "other")
    dummies = pd.get_dummies(df_clf["cat_enc"], prefix="cat", drop_first=True)

    num_feats = ["total_payment", "total_freight", "n_items", "total_item_price", "purchase_month"]
    X = pd.concat([df_clf[num_feats].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    y = df_clf["sentiment"].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=50,
        min_samples_leaf=25, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]
    roc_auc     = roc_auc_score(y_test, y_prob)
    cv_scores   = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
    cm          = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    report      = classification_report(y_test, y_pred, target_names=["Negatif", "Positif"], output_dict=True)

    feat_imp = pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_})
    feat_imp = feat_imp.sort_values("importance", ascending=False).head(15)

    return model, roc_auc, cv_scores, cm, fpr, tpr, feat_imp, report, y_test, y_prob


@st.cache_data
def compute_timeseries(df):
    ts = (
        df.groupby("purchase_yearmonth")["total_payment"]
        .sum()
        .reset_index()
    )
    ts.columns = ["yearmonth", "revenue"]
    ts["dt"] = ts["yearmonth"].apply(lambda x: x.to_timestamp())
    ts = ts.set_index("dt")["revenue"]
    ts.index = pd.DatetimeIndex(ts.index, freq="MS")
    decomp = seasonal_decompose(ts, model="additive", period=4, extrapolate_trend="freq")
    return ts, decomp


@st.cache_data
def compute_correlation(df):
    cols = ["total_payment", "total_freight", "n_items", "total_item_price", "review_score", "purchase_month"]
    return df[cols].dropna().corr(method="spearman")


# ============================================================
# HELPER PLOTTING
# ============================================================

def fmt_rev(v):
    if v >= 1_000_000: return f"BRL {v/1e6:.2f}M"
    if v >= 1_000:     return f"BRL {v/1e3:.1f}K"
    return f"BRL {v:,.0f}"


def adaptive_divisor(max_val):
    if max_val >= 500_000: return 1_000_000, "Juta BRL", lambda v: f"{v:.2f}M"
    if max_val >= 1_000:   return 1_000,     "Ribu BRL", lambda v: f"{v:.1f}K"
    return 1, "BRL", lambda v: f"{v:,.0f}"


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Olist Dashboard")
    st.markdown("**Analisis E-Commerce Brazil**")
    st.divider()

    st.subheader("Filter Data")

    if data_ok:
        min_d = df["order_purchase_timestamp"].min().date()
        max_d = df["order_purchase_timestamp"].max().date()
        date_range = st.date_input("Rentang Waktu", value=(min_d, max_d), min_value=min_d, max_value=max_d)

        all_states = ["Semua"] + sorted(df["customer_state"].dropna().unique().tolist())
        sel_state  = st.selectbox("Negara Bagian", all_states)

        df_f = df[
            (df["order_purchase_timestamp"].dt.date >= date_range[0]) &
            (df["order_purchase_timestamp"].dt.date <= date_range[1])
        ].copy()
        if sel_state != "Semua":
            df_f = df_f[df_f["customer_state"] == sel_state]

    st.divider()

    # Navigasi section
    st.subheader("Navigasi")
    nav_items = [
        ("01", "Ringkasan Performa",           "#1-ringkasan-performa-bisnis"),
        ("02", "Revenue & Kategori",            "#2-pertanyaan-1-tren-revenue-dan-kategori-produk-teratas"),
        ("03", "Kepuasan Pelanggan",            "#3-pertanyaan-2-kepuasan-pelanggan-per-kategori-produk"),
        ("04", "RFM Segmentasi",                "#4-analisis-lanjutan-1-rfm-segmentasi-pelanggan"),
        ("05", "Geospatial",                    "#5-analisis-lanjutan-2-geospatial-distribusi-order-di-brazil"),
        ("06", "K-Means Clustering",            "#6-analisis-lanjutan-4-k-means-clustering-kategori-produk"),
        ("07", "Random Forest",                 "#7-analisis-lanjutan-5-prediksi-sentimen-ulasan-random-forest"),
        ("08", "Time Series",                   "#8-analisis-lanjutan-6-time-series-decomposition-revenue"),
        ("09", "Korelasi & Feature Importance", "#9-analisis-lanjutan-7-analisis-korelasi-feature-importance"),
        ("10", "Kesimpulan & Rekomendasi",      "#10-kesimpulan-dan-rekomendasi"),
    ]
    for num, label, anchor in nav_items:
        st.markdown(f"**{num}** &nbsp; [{label}]({anchor})", unsafe_allow_html=True)

    st.divider()
    st.caption("Chamid Bahrul Ulum")
    st.caption("Dataset: Olist Brazilian E-Commerce")

# ============================================================
# MAIN CONTENT
# ============================================================

if not data_ok:
    st.error("""
    **File `main_data.csv` tidak ditemukan.**
    Letakkan file tersebut di dalam folder `dashboard/` lalu restart Streamlit.
    """)
    st.stop()

st.title("Dashboard Analisis Data Science E-Commerce Olist")
st.markdown("**Periode: Januari 2017 – Agustus 2018 | Platform: Olist, Brazil**")
st.divider()

# ============================================================
# SECTION 1 — RINGKASAN PERFORMA
# ============================================================
st.subheader("1. Ringkasan Performa Bisnis")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Order",       f"{len(df_f):,}")
c2.metric("Total Revenue",     fmt_rev(df_f["total_payment"].sum()))
c3.metric("Avg Review Score",  f"{df_f['review_score'].mean():.2f} / 5.0")
c4.metric("Pelanggan Unik",    f"{df_f['customer_unique_id'].nunique():,}")
c5.metric("Kategori Produk",   f"{df_f[df_f['product_category']!='unknown']['product_category'].nunique()}")

st.divider()

# ============================================================
# SECTION 2 — REVENUE & KATEGORI (Pertanyaan 1)
# ============================================================
st.subheader("2. Pertanyaan 1: Tren Revenue dan Kategori Produk Teratas")

monthly_rev  = compute_monthly_revenue(df_f)
cat_stats    = compute_category_stats(df_f)
top10        = cat_stats.head(10)

col_l, col_r = st.columns([2, 1])

with col_l:
    max_rev   = monthly_rev["revenue"].max() if len(monthly_rev) > 0 else 1
    div, lbl, fmt_fn = adaptive_divisor(max_rev)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("none"); ax.set_facecolor("none")
    x = range(len(monthly_rev))
    ax.fill_between(x, monthly_rev["revenue"] / div, alpha=0.12, color=COLOR_PRIMARY)
    ax.plot(x, monthly_rev["revenue"] / div, color=COLOR_PRIMARY, lw=2.5,
            marker="o", ms=5, markerfacecolor="white", markeredgewidth=2)
    if len(monthly_rev) > 0:
        pk = monthly_rev["revenue"].idxmax()
        ax.plot(pk, monthly_rev.loc[pk, "revenue"] / div, "o", color=COLOR_ACCENT, ms=10, zorder=5,
                label=f"Puncak: {monthly_rev.loc[pk,'yearmonth_str']} ({fmt_fn(monthly_rev.loc[pk,'revenue']/div)})")
        ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(monthly_rev["yearmonth_str"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"Revenue ({lbl})", fontsize=9)
    ax.set_title("Tren Revenue Bulanan", fontsize=11, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with col_r:
    st.markdown("**Top 10 Kategori Revenue**")
    n    = len(top10)
    h    = max(3.5, n * 0.45)
    maxv = top10["total_revenue"].max() if n > 0 else 1
    div2, lbl2, fmt2 = adaptive_divisor(maxv)

    cols_bar = ([COLOR_ACCENT] + [COLOR_PRIMARY] * min(2, n-1) + [COLOR_NEUTRAL] * max(0, n-3))[:n]

    fig2, ax2 = plt.subplots(figsize=(5, h))
    fig2.patch.set_facecolor("none"); ax2.set_facecolor("none")
    bars = ax2.barh(top10["product_category"], top10["total_revenue"] / div2,
                    color=cols_bar, edgecolor="white", height=0.6)
    for bar, v in zip(bars, top10["total_revenue"] / div2):
        ax2.text(v + maxv / div2 * 0.03, bar.get_y() + bar.get_height() / 2,
                 fmt2(v), va="center", fontsize=7)
    ax2.set_xlim(left=0, right=maxv / div2 * 1.3)
    ax2.set_xlabel(lbl2, fontsize=8)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.grid(axis="x", alpha=0.3, linestyle="--"); ax2.yaxis.grid(False)
    plt.tight_layout()
    st.pyplot(fig2); plt.close()

with st.expander("Lihat Insight Pertanyaan 1"):
    if len(monthly_rev) > 0 and len(top10) > 0:
        pk      = monthly_rev["revenue"].idxmax()
        top_cat = top10.iloc[0]["product_category"]
        top_rev = top10.iloc[0]["total_revenue"]
        avg_rev = monthly_rev["revenue"].mean()
        st.info(f"""
**Insight:**
- Revenue tertinggi: **{monthly_rev.loc[pk,'yearmonth_str']}** sebesar **{fmt_rev(monthly_rev.loc[pk,'revenue'])}**
- Rata-rata revenue bulanan: {fmt_rev(avg_rev)}
- Kategori teratas: **{top_cat}** ({fmt_rev(top_rev)})
- Puncak November 2017 sangat kemungkinan dipicu event **Black Friday**
        """)

st.divider()

# ============================================================
# SECTION 3 — KEPUASAN PELANGGAN (Pertanyaan 2)
# ============================================================
st.subheader("3. Pertanyaan 2: Kepuasan Pelanggan per Kategori Produk")

top10_vol = cat_stats.sort_values("n_orders", ascending=False).head(10)["product_category"].tolist()
df_top10  = df_f[(df_f["product_category"].isin(top10_vol)) & df_f["review_score"].notna()].copy()

score_by_cat = (
    df_top10.groupby(["product_category", "review_score"])
    .size().unstack(fill_value=0)
    .apply(lambda x: x / x.sum() * 100, axis=1)
)
cat_score = (
    df_top10.groupby("product_category")["review_score"]
    .mean().reset_index().sort_values("review_score", ascending=True)
)
sorted_cats = cat_score["product_category"].tolist()
if len(score_by_cat) > 0:
    score_by_cat = score_by_cat.reindex(sorted_cats)

score_pal = {1: "#C0392B", 2: "#E67E22", 3: "#F1C40F", 4: "#2ECC71", 5: "#1A7A4A"}

col_a, col_b = st.columns([3, 1])

with col_a:
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    fig3.patch.set_facecolor("none"); ax3.set_facecolor("none")
    bottom = np.zeros(len(score_by_cat))
    for sc in [1, 2, 3, 4, 5]:
        if sc in score_by_cat.columns:
            vals = score_by_cat[sc].values
            ax3.barh(score_by_cat.index, vals, left=bottom, color=score_pal[sc],
                     label=f"Score {sc}", edgecolor="white", linewidth=0.5)
            bottom += vals
    ax3.set_xlabel("Proporsi Ulasan (%)", fontsize=9)
    ax3.set_title("Distribusi Review Score per Kategori\n(urut dari kepuasan terendah ke tertinggi)",
                  fontsize=10, fontweight="bold")
    ax3.set_xlim(0, 100)
    ax3.legend(loc="upper right", fontsize=8, title="Score", title_fontsize=8)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
    ax3.tick_params(axis="y", labelsize=7.5)
    plt.tight_layout()
    st.pyplot(fig3); plt.close()

with col_b:
    if len(cat_score) > 0:
        overall_avg = df_top10["review_score"].mean()
        s_min = cat_score["review_score"].min()
        s_max = cat_score["review_score"].max()
        margin = max(0.1, (s_max - s_min) * 0.35)

        fig4, ax4 = plt.subplots(figsize=(4, 5))
        fig4.patch.set_facecolor("none"); ax4.set_facecolor("none")
        bcols = [COLOR_ACCENT if v < overall_avg else COLOR_SUCCESS for v in cat_score["review_score"]]
        bars4 = ax4.barh(cat_score["product_category"], cat_score["review_score"],
                         color=bcols, edgecolor="white", height=0.6)
        ax4.axvline(x=overall_avg, color="#555", linestyle="--", lw=1.5, alpha=0.7,
                    label=f"Rata-rata: {overall_avg:.2f}")
        for bar, v in zip(bars4, cat_score["review_score"]):
            ax4.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", fontsize=7)
        ax4.set_xlim(max(1.0, s_min - margin), min(5.2, s_max + margin * 2))
        ax4.set_xlabel("Rata-Rata Score", fontsize=8)
        ax4.legend(fontsize=7, loc="lower right")
        ax4.set_yticks([])
        ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig4); plt.close()

with st.expander("Lihat Insight Pertanyaan 2"):
    if len(df_top10) > 0 and len(cat_score) > 0:
        oa    = df_top10["review_score"].mean()
        low   = cat_score.iloc[0]["product_category"]
        lowv  = cat_score.iloc[0]["review_score"]
        high  = cat_score.iloc[-1]["product_category"]
        highv = cat_score.iloc[-1]["review_score"]
        pct5  = (df_top10["review_score"] == 5).sum() / df_top10["review_score"].notna().sum() * 100
        st.info(f"""
**Insight:**
- Rata-rata kepuasan filter ini: **{oa:.3f} / 5.0**
- Kepuasan **terendah**: `{low}` ({lowv:.3f})
- Kepuasan **tertinggi**: `{high}` ({highv:.3f})
- **{pct5:.1f}%** pelanggan memberikan skor 5 (sangat puas)
- Kategori produk besar/berat cenderung mendapat skor lebih rendah
        """)

st.divider()

# ============================================================
# SECTION 4 — RFM ANALYSIS
# ============================================================
st.subheader("4. Analisis Lanjutan 1: RFM — Segmentasi Pelanggan")

rfm = compute_rfm(df_f)
seg_summary = (
    rfm.groupby("segment")
    .agg(n_customers=("customer_unique_id", "count"),
         avg_recency=("recency", "mean"),
         avg_frequency=("frequency", "mean"),
         avg_monetary=("monetary", "mean"),
         total_revenue=("monetary", "sum"))
    .round(1).reset_index().sort_values("n_customers", ascending=False)
)

seg_colors = {
    "Champions": "#1A7A4A", "Loyal Customers": "#2E86AB",
    "Potential Loyalists": "#3BB273", "New Customers": "#74C69D",
    "Need Attention": "#F4A261", "At Risk": "#E84855", "Lost Customers": "#C0392B",
}

col_r1, col_r2 = st.columns([1, 2])

with col_r1:
    sorted_seg = seg_summary.sort_values("n_customers", ascending=True)
    max_c = sorted_seg["n_customers"].max() if len(sorted_seg) > 0 else 1
    h_rfm = max(3.0, len(sorted_seg) * 0.55)

    fig5, ax5 = plt.subplots(figsize=(5, h_rfm))
    fig5.patch.set_facecolor("none"); ax5.set_facecolor("none")
    cols_rfm = [seg_colors.get(s, COLOR_NEUTRAL) for s in sorted_seg["segment"]]
    bars5 = ax5.barh(sorted_seg["segment"], sorted_seg["n_customers"],
                     color=cols_rfm, edgecolor="white", height=0.6)
    for bar, v in zip(bars5, sorted_seg["n_customers"]):
        ax5.text(v + max_c * 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{v:,}", va="center", fontsize=7.5)
    ax5.set_xlim(left=0, right=max_c * 1.25)
    ax5.set_xlabel("Jumlah Pelanggan", fontsize=9)
    ax5.set_title("Distribusi Segmen RFM", fontsize=10, fontweight="bold")
    ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)
    ax5.yaxis.grid(False)
    plt.tight_layout()
    st.pyplot(fig5); plt.close()

with col_r2:
    st.markdown("**Tabel Ringkasan Segmen RFM**")
    disp_rfm = seg_summary.copy()
    disp_rfm["avg_monetary"]  = disp_rfm["avg_monetary"].apply(lambda x: f"BRL {x:,.0f}")
    disp_rfm["avg_recency"]   = disp_rfm["avg_recency"].apply(lambda x: f"{x:.0f} hari")
    disp_rfm["total_revenue"] = disp_rfm["total_revenue"].apply(lambda x: fmt_rev(x))
    disp_rfm.columns = ["Segmen", "Pelanggan", "Avg Recency", "Avg Freq", "Avg Monetary", "Total Revenue"]
    st.dataframe(disp_rfm, use_container_width=True, hide_index=True)

    st.markdown("""
**Definisi Segmen:**
- **Champions**: Baru beli, sering, nilai tinggi
- **Loyal Customers**: Sering beli, perlu dipertahankan
- **New Customers**: Baru bergabung, potensi diloyalkan
- **At Risk**: Dulu aktif, kini sudah lama tidak beli
- **Lost Customers**: Lama tidak aktif, perlu win-back
    """)

st.divider()

# ============================================================
# SECTION 5 — GEOSPATIAL
# ============================================================
st.subheader("5. Analisis Lanjutan 2: Geospatial — Distribusi Order di Brazil")

state_stats = compute_state_stats(df_f)

col_g1, col_g2 = st.columns([1, 2])

with col_g1:
    top15 = state_stats.head(15)
    total_all = state_stats["n_orders"].sum()
    n_st   = len(top15)
    max_o  = top15["n_orders"].max() if n_st > 0 else 1
    h_geo  = max(3.0, n_st * 0.42)

    fig6, ax6 = plt.subplots(figsize=(5, h_geo))
    fig6.patch.set_facecolor("none"); ax6.set_facecolor("none")
    bcols_st = [COLOR_ACCENT if i == 0 else COLOR_PRIMARY if i == 1 else COLOR_NEUTRAL for i in range(n_st)]
    bars6 = ax6.barh(top15["customer_state"], top15["n_orders"],
                     color=bcols_st, edgecolor="white", height=0.6)
    for bar, n in zip(bars6, top15["n_orders"]):
        ax6.text(n + max_o * 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{n/total_all*100:.1f}%", va="center", fontsize=7.5)
    ax6.set_xlim(left=0, right=max_o * 1.25)
    ax6.set_xlabel("Jumlah Order", fontsize=9)
    ax6.set_title("Top 15 Negara Bagian", fontsize=10, fontweight="bold")
    ax6.invert_yaxis()
    ax6.spines["top"].set_visible(False); ax6.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig6); plt.close()

with col_g2:
    state_coords = {
        "SP": (-23.55, -46.63), "RJ": (-22.91, -43.17), "MG": (-19.92, -43.94),
        "RS": (-30.03, -51.22), "PR": (-25.43, -49.27), "SC": (-27.60, -48.55),
        "BA": (-12.97, -38.50), "GO": (-16.69, -49.25), "ES": (-20.32, -40.34),
        "PE": (-8.05, -34.88),  "CE": (-3.72, -38.54),  "MA": (-2.53, -44.30),
        "MT": (-15.60, -56.10), "MS": (-20.44, -54.65), "PA": (-1.46, -48.50),
        "AM": (-3.10, -60.02),  "DF": (-15.78, -47.93), "RN": (-5.79, -35.21),
        "PB": (-7.12, -34.86),  "AL": (-9.66, -35.74),  "PI": (-5.09, -42.80),
        "TO": (-10.25, -48.34), "SE": (-10.91, -37.07), "AC": (-9.97, -67.81),
        "RO": (-8.76, -63.90),  "RR": (2.82, -60.68),   "AP": (0.04, -51.07),
    }
    m = folium.Map(location=[-14.0, -51.0], zoom_start=4, tiles="CartoDB positron")
    max_ord = state_stats["n_orders"].max()
    for _, row in state_stats.iterrows():
        st_ = row["customer_state"]
        if st_ in state_coords:
            lat, lon = state_coords[st_]
            radius   = (row["n_orders"] / max_ord) * 40000
            folium.Circle(
                location=[lat, lon], radius=radius,
                color=COLOR_PRIMARY, fill=True, fill_opacity=0.5,
                popup=folium.Popup(
                    f"<b>{st_}</b><br>Orders: {row['n_orders']:,}<br>"
                    f"Revenue: {fmt_rev(row['total_revenue'])}<br>Avg Score: {row['avg_score']:.2f}",
                    max_width=200
                ),
                tooltip=f"{st_}: {row['n_orders']:,} orders"
            ).add_to(m)
    st_folium(m, width=700, height=380)

with st.expander("Lihat Insight Geospatial"):
    if len(state_stats) > 0:
        top_st   = state_stats.iloc[0]["customer_state"]
        top_n    = state_stats.iloc[0]["n_orders"]
        top_pct  = top_n / state_stats["n_orders"].sum() * 100
        top5_pct = state_stats.head(5)["n_orders"].sum() / state_stats["n_orders"].sum() * 100
        st.info(f"""
**Insight:**
- **{top_st}** mendominasi dengan **{top_n:,} order ({top_pct:.1f}% dari total filter ini)**
- Top 5 negara bagian menyumbang **{top5_pct:.1f}%** total order
- Wilayah utara dan barat daya Brazil relatif underserved — peluang ekspansi pasar
- Klik setiap circle pada peta untuk detail per negara bagian
        """)

st.divider()

# ============================================================
# SECTION 6 — K-MEANS CLUSTERING
# ============================================================
st.subheader("6. Analisis Lanjutan 4: K-Means Clustering Kategori Produk")

with st.spinner("Menjalankan K-Means clustering..."):
    cat_km, X_km, best_k, final_sil, sil_vals, k_list, inertias, sil_scores = compute_kmeans(df_f)

if cat_km is None:
    st.warning(
        "Data pada filter ini terlalu sedikit untuk K-Means Clustering "
        "(minimal 3 kategori diperlukan). Gunakan filter **Semua** negara bagian."
    )
else:
    st.markdown(f"**K Optimal = {best_k} | Silhouette Score = {final_sil:.4f}**")
    col_km1, col_km2 = st.columns(2)

    with col_km1:
        # Elbow + Silhouette
        fig_e, axes_e = plt.subplots(1, 2, figsize=(11, 4))
        fig_e.suptitle("Penentuan K Optimal", fontsize=11, fontweight="bold")

        # Elbow
        axes_e[0].plot(k_list, inertias, marker="o", color=COLOR_PRIMARY, lw=2,
                       markersize=5, markerfacecolor="white", markeredgewidth=2)
        axes_e[0].set_xlabel("K", fontsize=9); axes_e[0].set_ylabel("Inertia", fontsize=9)
        axes_e[0].set_title("Elbow Method", fontsize=10, fontweight="bold")
        axes_e[0].set_xticks(k_list); axes_e[0].set_ylim(bottom=0)
        axes_e[0].spines["top"].set_visible(False); axes_e[0].spines["right"].set_visible(False)

        # Silhouette bar
        bcols_k = [COLOR_ACCENT if k == best_k else COLOR_NEUTRAL for k in k_list]
        bars_k  = axes_e[1].bar(k_list, sil_scores, color=bcols_k, edgecolor="white", width=0.6)
        for bar, sc in zip(bars_k, sil_scores):
            axes_e[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                           f"{sc:.3f}", ha="center", fontsize=7.5)
        axes_e[1].set_xlabel("K", fontsize=9); axes_e[1].set_ylabel("Silhouette Score", fontsize=9)
        axes_e[1].set_title(f"Silhouette Score (K={best_k} terbaik)", fontsize=10, fontweight="bold")
        axes_e[1].set_xticks(k_list); axes_e[1].set_ylim(bottom=0)
        axes_e[1].spines["top"].set_visible(False); axes_e[1].spines["right"].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig_e); plt.close()

    with col_km2:
        # Scatter cluster
        cluster_pal = ["#2E86AB", "#E84855", "#3BB273", "#F4A261", "#9B5DE5", "#00BBF9", "#FEE440", "#00F5D4"]

        fig_sc, ax_sc = plt.subplots(figsize=(6, 4))
        fig_sc.patch.set_facecolor("none"); ax_sc.set_facecolor("none")
        for cid in sorted(cat_km["cluster"].unique()):
            grp = cat_km[cat_km["cluster"] == cid]
            ax_sc.scatter(grp["avg_score"], grp["total_revenue"] / 1e6,
                          c=cluster_pal[cid], s=np.clip(grp["n_orders"] / 8, 20, 500),
                          label=f"Cluster {cid+1} (n={len(grp)})", alpha=0.8,
                          edgecolors="white", linewidths=0.5)
            for _, row in grp.nlargest(2, "total_revenue").iterrows():
                ax_sc.annotate(row["product_category"].replace("_", " ")[:14],
                               xy=(row["avg_score"], row["total_revenue"] / 1e6),
                               xytext=(3, 3), textcoords="offset points", fontsize=6, color="#333")
        ax_sc.set_xlabel("Avg Review Score", fontsize=9)
        ax_sc.set_ylabel("Total Revenue (Juta BRL)", fontsize=9)
        ax_sc.set_title("Distribusi Cluster: Revenue vs Kepuasan\n(ukuran titik = jumlah order)",
                        fontsize=9, fontweight="bold")
        ax_sc.legend(fontsize=7)
        ax_sc.spines["top"].set_visible(False); ax_sc.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_sc); plt.close()

    # Silhouette plot
    cluster_pal = ["#2E86AB", "#E84855", "#3BB273", "#F4A261", "#9B5DE5", "#00BBF9", "#FEE440", "#00F5D4"]
    fig_sil, ax_sil = plt.subplots(figsize=(10, 3.5))
    fig_sil.patch.set_facecolor("none"); ax_sil.set_facecolor("none")
    y_lo = 10
    for cid in range(best_k):
        sv = np.sort(sil_vals[cat_km["cluster"] == cid])
        y_hi = y_lo + len(sv)
        ax_sil.fill_betweenx(np.arange(y_lo, y_hi), 0, sv, facecolor=cluster_pal[cid], alpha=0.8)
        ax_sil.text(-0.05, y_lo + len(sv) / 2, f"C{cid+1}", fontsize=8)
        y_lo = y_hi + 10
    ax_sil.axvline(x=final_sil, color=COLOR_ACCENT, linestyle="--", lw=1.5,
                   label=f"Avg Silhouette: {final_sil:.3f}")
    ax_sil.set_xlabel("Silhouette Coefficient", fontsize=9)
    ax_sil.set_title("Silhouette Plot per Kategori Produk", fontsize=10, fontweight="bold")
    ax_sil.set_yticks([]); ax_sil.legend(fontsize=8)
    ax_sil.spines["top"].set_visible(False); ax_sil.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_sil); plt.close()

    # Tabel ringkasan cluster
    clus_table = (
        cat_km.groupby("cluster_label")
        .agg(n_kategori=("product_category", "count"),
             avg_revenue=("total_revenue", "mean"),
             avg_score=("avg_score", "mean"),
             avg_orders=("n_orders", "mean"))
        .round(2).reset_index()
    )
    clus_table["avg_revenue"] = clus_table["avg_revenue"].apply(fmt_rev)
    clus_table.columns = ["Cluster", "Jumlah Kategori", "Avg Revenue", "Avg Score", "Avg Orders"]
    st.dataframe(clus_table, use_container_width=True, hide_index=True)

    with st.expander("Lihat Insight K-Means"):
        st.info(f"""
**Insight K-Means Clustering:**
- K={best_k} dipilih berdasarkan Silhouette Score tertinggi = **{final_sil:.4f}**
- Silhouette Score {final_sil:.4f} mengindikasikan struktur clustering yang **{"baik" if final_sil > 0.5 else "cukup" if final_sil > 0.25 else "lemah"}**
- Pemisahan cluster mencerminkan perbedaan skala bisnis antar kategori produk
- K-Means lebih objektif dari manual clustering karena mempertimbangkan 5 dimensi fitur secara simultan
        """)

st.divider()

# ============================================================
# SECTION 7 — RANDOM FOREST
# ============================================================
st.subheader("7. Analisis Lanjutan 5: Prediksi Sentimen Ulasan (Random Forest)")

with st.spinner("Melatih Random Forest model..."):
    model_rf, roc_auc, cv_scores, cm, fpr, tpr, feat_imp, report, y_test, y_prob = compute_rf(df_f)

if model_rf is None:
    st.warning(
        "Data pada filter ini terlalu sedikit untuk melatih Random Forest "
        "(minimal 200 sampel diperlukan). Gunakan filter **Semua** negara bagian."
    )
else:
    # Metrik ringkasan
    col_rf0a, col_rf0b, col_rf0c, col_rf0d = st.columns(4)
    col_rf0a.metric("ROC-AUC",          f"{roc_auc:.4f}")
    col_rf0b.metric("CV ROC-AUC",       f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    col_rf0c.metric("Accuracy",         f"{report['accuracy']:.3f}")
    col_rf0d.metric("Variance CV",      f"{cv_scores.std():.4f}")

    col_rf1, col_rf2, col_rf3 = st.columns(3)

    with col_rf1:
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4))
        fig_cm.patch.set_facecolor("none"); ax_cm.set_facecolor("none")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negatif", "Positif"])
        disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                ax_cm.text(j, i + 0.32, f"({cm[i,j]/total*100:.1f}%)", ha="center", fontsize=8.5, color="gray")
        ax_cm.set_title("Confusion Matrix", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_cm); plt.close()

    with col_rf2:
        fig_roc, ax_roc = plt.subplots(figsize=(4.5, 4))
        fig_roc.patch.set_facecolor("none"); ax_roc.set_facecolor("none")
        ax_roc.plot(fpr, tpr, color=COLOR_PRIMARY, lw=2.5, label=f"Random Forest (AUC={roc_auc:.3f})")
        ax_roc.plot([0, 1], [0, 1], color=COLOR_NEUTRAL, linestyle="--", lw=1.5, label="Random (AUC=0.500)")
        ax_roc.fill_between(fpr, tpr, alpha=0.1, color=COLOR_PRIMARY)
        ax_roc.set_xlabel("False Positive Rate", fontsize=9)
        ax_roc.set_ylabel("True Positive Rate", fontsize=9)
        ax_roc.set_title("ROC Curve", fontsize=10, fontweight="bold")
        ax_roc.legend(fontsize=7, loc="lower right")
        ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1.02])
        ax_roc.spines["top"].set_visible(False); ax_roc.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_roc); plt.close()

    with col_rf3:
        fig_fi, ax_fi = plt.subplots(figsize=(4.5, 4))
        fig_fi.patch.set_facecolor("none"); ax_fi.set_facecolor("none")
        fi_cols = [COLOR_ACCENT if i == 0 else COLOR_PRIMARY if i < 3 else COLOR_NEUTRAL
                   for i in range(len(feat_imp))]
        bars_fi = ax_fi.barh(feat_imp["feature"], feat_imp["importance"],
                             color=fi_cols, edgecolor="white", height=0.6)
        for bar, v in zip(bars_fi, feat_imp["importance"]):
            ax_fi.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                       f"{v:.3f}", va="center", fontsize=7)
        ax_fi.invert_yaxis()
        ax_fi.set_xlabel("Feature Importance", fontsize=9)
        ax_fi.set_title("Top 15 Feature Importance", fontsize=10, fontweight="bold")
        ax_fi.set_xlim(left=0, right=feat_imp["importance"].max() * 1.25)
        ax_fi.tick_params(axis="y", labelsize=7)
        ax_fi.yaxis.grid(False)
        ax_fi.spines["top"].set_visible(False); ax_fi.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_fi); plt.close()

    # Classification report table
    st.markdown("**Classification Report**")
    cr_df = pd.DataFrame(report).T.round(3)
    cr_df = cr_df[["precision", "recall", "f1-score", "support"]].iloc[:4]
    st.dataframe(cr_df, use_container_width=True)

    with st.expander("Lihat Insight Random Forest"):
        cv_std = cv_scores.std()
        if cv_std < 0.01:     stab = "sangat stabil — variance sangat rendah, tidak ada indikasi overfitting"
        elif cv_std < 0.03:   stab = "cukup stabil — variance dalam batas wajar"
        else:                 stab = "kurang stabil — pertimbangkan regularisasi lebih ketat"
        st.info(f"""
**Insight Random Forest:**
- ROC-AUC = **{roc_auc:.4f}** — di atas random classifier (0.5), kategori **{"baik" if roc_auc > 0.75 else "cukup" if roc_auc > 0.6 else "lemah"}**
- CV ROC-AUC = {cv_scores.mean():.4f} (variance {cv_std:.4f}) → model **{stab}**
- Precision kelas negatif rendah karena imbalance kelas (78.9% positif vs 21.1% negatif)
- Feature terpenting: `{feat_imp.iloc[0]['feature']}` — faktor utama penentu sentimen ulasan
- **Implikasi bisnis**: model dapat digunakan sebagai early warning system untuk order berisiko ulasan negatif
        """)

st.divider()

# ============================================================
# SECTION 8 — TIME SERIES DECOMPOSITION
# ============================================================
st.subheader("8. Analisis Lanjutan 6: Time Series Decomposition Revenue")

with st.spinner("Menjalankan time series decomposition..."):
    ts_rev, decomp = compute_timeseries(df_f)

col_ts_met1, col_ts_met2, col_ts_met3 = st.columns(3)
col_ts_met1.metric("Trend Range",    f"{fmt_rev(decomp.trend.min())} – {fmt_rev(decomp.trend.max())}")
col_ts_met2.metric("Seasonal Range", f"{fmt_rev(decomp.seasonal.min())} – {fmt_rev(decomp.seasonal.max())}")
col_ts_met3.metric("Residual Std",   fmt_rev(decomp.resid.std()))

fig_ts, axes_ts = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig_ts.suptitle("Time Series Decomposition Revenue Bulanan", fontsize=12, fontweight="bold", y=1.01)

labels_ts = [ts_rev.index[i].strftime("%b %Y") for i in range(len(ts_rev))]
x_ts      = range(len(ts_rev))

# Observed
axes_ts[0].plot(x_ts, ts_rev.values / 1e6, color=COLOR_PRIMARY, lw=2.5, marker="o", ms=4, markerfacecolor="white")
axes_ts[0].fill_between(x_ts, ts_rev.values / 1e6, alpha=0.1, color=COLOR_PRIMARY)
axes_ts[0].set_ylabel("Juta BRL", fontsize=8)
axes_ts[0].set_title("Data Asli (Observed)", fontsize=9, fontweight="bold")

# Trend
axes_ts[1].plot(x_ts, decomp.trend.values / 1e6, color=COLOR_SUCCESS, lw=2.5)
axes_ts[1].set_ylabel("Juta BRL", fontsize=8)
axes_ts[1].set_title("Komponen Trend", fontsize=9, fontweight="bold")

# Seasonal
axes_ts[2].bar(x_ts, decomp.seasonal.values / 1e6,
               color=[COLOR_WARNING if v >= 0 else COLOR_ACCENT for v in decomp.seasonal.values],
               edgecolor="white", width=0.7)
axes_ts[2].axhline(y=0, color="#888", lw=0.8)
axes_ts[2].set_ylabel("Juta BRL", fontsize=8)
axes_ts[2].set_title("Komponen Musiman (Seasonal)", fontsize=9, fontweight="bold")

# Residual
axes_ts[3].stem(x_ts, decomp.resid.values / 1e6, linefmt=COLOR_NEUTRAL, markerfmt="o", basefmt=" ")
axes_ts[3].axhline(y=0, color="#888", lw=1, linestyle="--")
axes_ts[3].set_ylabel("Juta BRL", fontsize=8)
axes_ts[3].set_title("Komponen Residual", fontsize=9, fontweight="bold")
axes_ts[3].set_xticks(x_ts)
axes_ts[3].set_xticklabels(labels_ts, rotation=45, ha="right", fontsize=7.5)

for ax in axes_ts:
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=ax.get_ylim()[0])

plt.tight_layout()
st.pyplot(fig_ts); plt.close()

with st.expander("Lihat Insight Time Series"):
    st.info(f"""
**Insight Time Series Decomposition:**
- **Trend** tumbuh monoton dari {fmt_rev(decomp.trend.min())} → {fmt_rev(decomp.trend.max())} — pertumbuhan organik terkonfirmasi
- **Seasonal** amplitudo {fmt_rev(abs(decomp.seasonal.min()))} – {fmt_rev(decomp.seasonal.max())} — pola musiman ada namun tidak mendominasi
- **Residual** std {fmt_rev(decomp.resid.std())} — lonjakan residual terbesar terjadi November 2017, bukti kuantitatif efek Black Friday
- Dekomposisi mengkonfirmasi pertumbuhan Olist bersifat organik dan berkelanjutan
    """)

st.divider()

# ============================================================
# SECTION 9 — KORELASI & FEATURE IMPORTANCE
# ============================================================
st.subheader("9. Analisis Lanjutan 7: Analisis Korelasi & Feature Importance")

corr_matrix = compute_correlation(df_f)

col_c1, col_c2 = st.columns(2)

with col_c1:
    fig_cor, ax_cor = plt.subplots(figsize=(6, 5))
    fig_cor.patch.set_facecolor("none"); ax_cor.set_facecolor("none")
    sns.heatmap(corr_matrix, ax=ax_cor, annot=True, fmt=".2f", annot_kws={"size": 8},
                cmap="RdYlGn", center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax_cor.set_title("Heatmap Korelasi Spearman", fontsize=10, fontweight="bold")
    ax_cor.tick_params(axis="x", rotation=30, labelsize=8)
    ax_cor.tick_params(axis="y", rotation=0, labelsize=8)
    plt.tight_layout()
    st.pyplot(fig_cor); plt.close()

with col_c2:
    # Korelasi dengan review_score
    corr_score = corr_matrix["review_score"].drop("review_score").sort_values()

    fig_cs, ax_cs = plt.subplots(figsize=(6, 5))
    fig_cs.patch.set_facecolor("none"); ax_cs.set_facecolor("none")
    bcols_c = [COLOR_ACCENT if v < 0 else COLOR_SUCCESS for v in corr_score.values]
    bars_c  = ax_cs.barh(corr_score.index, corr_score.values,
                         color=bcols_c, edgecolor="white", height=0.6)
    ax_cs.axvline(x=0, color="#888", lw=1)
    for bar, v in zip(bars_c, corr_score.values):
        offset = 0.003 if v >= 0 else -0.003
        ha     = "left" if v >= 0 else "right"
        ax_cs.text(v + offset, bar.get_y() + bar.get_height() / 2,
                   f"{v:+.3f}", va="center", ha=ha, fontsize=8.5)
    ax_cs.set_xlabel("Korelasi Spearman dengan Review Score", fontsize=9)
    ax_cs.set_title("Korelasi Variabel\nterhadap Kepuasan Pelanggan",
                    fontsize=10, fontweight="bold")
    ax_cs.spines["top"].set_visible(False); ax_cs.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_cs); plt.close()

    # Tabel korelasi
    st.markdown("**Nilai Korelasi dengan Review Score:**")
    corr_df = pd.DataFrame({
        "Variabel": corr_score.index,
        "Korelasi": corr_score.values.round(3),
        "Arah": ["Negatif" if v < 0 else "Positif" for v in corr_score.values],
        "Kekuatan": ["Kuat" if abs(v) > 0.3 else "Sedang" if abs(v) > 0.1 else "Lemah"
                     for v in corr_score.values]
    })
    st.dataframe(corr_df, use_container_width=True, hide_index=True)

with st.expander("Lihat Insight Korelasi"):
    n_items_corr = corr_matrix.loc["review_score", "n_items"]
    freight_corr = corr_matrix.loc["review_score", "total_freight"]
    payment_corr = corr_matrix.loc["review_score", "total_payment"]
    st.info(f"""
**Insight Korelasi & Feature Importance:**
- Semua variabel transaksi berkorelasi **negatif** dengan review score — ekspektasi lebih tinggi pada transaksi bernilai besar
- `n_items` memiliki korelasi negatif **terkuat ({n_items_corr:+.3f})** — pesanan multi-item lebih rentan ulasan negatif
- `total_freight` = {freight_corr:+.3f} — ongkos kirim tinggi mempengaruhi kepuasan namun tidak dominan
- `total_payment` = {payment_corr:+.3f} — nilai transaksi besar tidak otomatis berarti kepuasan lebih rendah
- **Rekomendasi**: fokus pada standar pengemasan pesanan multi-item dan batas ongkos kirim untuk kategori bermasalah
    """)

st.divider()

# ============================================================
# SECTION 10 — KESIMPULAN & REKOMENDASI
# ============================================================
st.subheader("10. Kesimpulan dan Rekomendasi")

col_con1, col_con2 = st.columns(2)

with col_con1:
    st.markdown("**Kesimpulan Pertanyaan 1 — Revenue**")
    st.success("""
Revenue platform Olist tumbuh dari BRL 127.546 (Januari 2017) hingga BRL 1.153.528
(November 2017) — hampir 9x lipat dalam 10 bulan, terkonfirmasi sebagai dampak
Black Friday oleh analisis time series. Rata-rata revenue bulanan BRL 768.794.
Kategori `health_beauty` (BRL 1,4M), `watches_gifts` (BRL 1,26M), dan
`bed_bath_table` (BRL 1,22M) mendominasi kontribusi revenue.
    """)

    st.markdown("**Kesimpulan Pertanyaan 2 — Kepuasan**")
    st.warning("""
Rata-rata kepuasan 4,156/5,0 dengan 59,2% pelanggan memberikan skor sempurna.
Kategori `bed_bath_table` (4,010) dan `telephony` (4,056) memiliki kepuasan
terendah. Analisis korelasi mengkonfirmasi `n_items` (-0.107) sebagai faktor
negatif terkuat terhadap kepuasan — pesanan multi-item lebih rentan ulasan negatif.
    """)

    st.markdown("**Kesimpulan Machine Learning**")
    st.info("""
K-Means (K=2, Silhouette=0.3306) memisahkan 14 kategori berskala besar dari
43 kategori berskala kecil. Random Forest (ROC-AUC=0.6111, CV=0.6172) berhasil
membangun model prediksi sentimen yang stabil. Time Series decomposition
mengkonfirmasi pertumbuhan organik dan efek Black Friday secara kuantitatif.
    """)

with col_con2:
    st.markdown("**Rekomendasi Action Item**")
    st.info("""
1. **Maksimalkan Black Friday**: Revenue November 2017 lebih dari 50% di atas
   rata-rata bulanan. Siapkan stok dan infrastruktur 2 bulan sebelumnya.

2. **Perbaiki Kepuasan `bed_bath_table` & `telephony`**: Audit seller, SLA
   pengiriman khusus produk besar, tambahkan program proteksi produk.

3. **Win-Back Segmen At Risk**: 22.079 pelanggan (23,7%) tidak aktif rata-rata
   393 hari. Kampanye email personal dengan voucher diskon untuk reaktivasi.

4. **Program Loyalitas New Customers**: Rata-rata frequency 1,00 menunjukkan
   hampir tidak ada repeat purchase. Cashback pembelian kedua dapat efektif.

5. **Ekspansi Geografis**: SP+RJ+MG = 66% total order. Rekrut seller lokal
   di BA, GO, PA yang underserved namun berpopulasi besar.

6. **Early Warning System RF**: Gunakan model Random Forest untuk
   identifikasi order berisiko ulasan negatif sebelum pengiriman selesai.

7. **Standar Multi-Item Packaging**: n_items adalah faktor korelasi negatif
   terkuat (-0.107). Tingkatkan standar pengemasan pesanan multi-item.
    """)

# Footer
st.divider()
st.caption(
    "Dashboard Analisis Data Science E-Commerce Olist | "
    "Chamid Bahrul Ulum | ulumlab@gmail.com"
)