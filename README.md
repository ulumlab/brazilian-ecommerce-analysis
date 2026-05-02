# Proyek Analisis Data: Brazilian E-Commerce Public Dataset (Olist)

**Nama**: Chamid Bahrul Ulum  
**Email**: ulumcourse@gmail.com  
**ID Dicoding**: cbu-dicoding

---

## Deskripsi Proyek

Proyek ini melakukan analisis mendalam terhadap dataset e-commerce publik milik Olist, platform e-commerce terbesar di Brazil. Analisis mencakup seluruh pipeline data mulai dari data wrangling, exploratory data analysis (EDA), visualisasi explanatory, hingga analisis lanjutan berupa RFM Analysis, Geospatial Analysis, dan Clustering Kategori Produk.

**Pertanyaan bisnis yang dijawab:**

1. Bagaimana tren pertumbuhan revenue bulanan dan 10 kategori produk apa yang menghasilkan revenue tertinggi sepanjang Januari 2017 – Agustus 2018?
2. Bagaimana distribusi kepuasan pelanggan (review score) berdasarkan 10 kategori produk dengan volume order tertinggi, dan kategori mana yang memiliki kepuasan terendah?

---

## Struktur Direktori

```
brazilian-ecommerce-analysis/
├── dashboard/
│   ├── dashboard.py        <- Aplikasi Streamlit
│   └── main_data.csv       <- Data hasil olahan dari notebook
├── notebook.ipynb          <- Notebook analisis lengkap
├── README.md               <- File ini
├── requirements.txt        <- Daftar library yang digunakan
└── url.txt                 <- URL dashboard yang sudah di-deploy
```

---

## Cara Menjalankan Dashboard Secara Lokal

### Prasyarat

- Python 3.10 atau lebih baru
- pip (Python package manager)

### Langkah-Langkah

**1. Clone atau extract folder submission**

Pastikan struktur direktori sesuai dengan yang tertera di atas.

**2. Buat virtual environment (direkomendasikan)**

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

**3. Install semua dependensi**

```bash
pip install -r requirements.txt
```

**4. Siapkan file data**

Jalankan terlebih dahulu `notebook.ipynb` hingga selesai, kemudian ekspor master DataFrame dengan menambahkan sel berikut di akhir notebook:

```python
df_main.to_csv('dashboard/main_data.csv', index=False)
```

**5. Jalankan aplikasi Streamlit**

Dari direktori root `submission/`, jalankan perintah berikut:

```bash
streamlit run dashboard/dashboard.py
```

**6. Buka di browser**

Setelah perintah di atas dijalankan, Streamlit akan memberikan URL lokal (biasanya `http://localhost:8501`). Buka URL tersebut di browser.

---

## Fitur Dashboard

| Fitur                   | Deskripsi                                           |
| ----------------------- | --------------------------------------------------- |
| Filter Rentang Waktu    | Menyaring data berdasarkan periode yang dipilih     |
| Filter Negara Bagian    | Fokus analisis pada negara bagian tertentu          |
| Tren Revenue Bulanan    | Line chart interaktif pertumbuhan revenue           |
| Top 10 Kategori         | Bar chart kontributor revenue terbesar              |
| Distribusi Review Score | Stacked bar chart kepuasan per kategori             |
| RFM Analysis            | Segmentasi pelanggan (Champions, At Risk, dll.)     |
| Geospatial Map          | Peta interaktif distribusi order di Brazil (folium) |
| Clustering Matriks      | Scatter plot 4 kuadran kategori produk              |

---

## Analisis Lanjutan yang Diterapkan

1. **RFM Analysis** — Segmentasi pelanggan berdasarkan Recency, Frequency, dan Monetary
2. **Geospatial Analysis** — Distribusi dan konsentrasi order menggunakan peta folium
3. **Clustering** — Pengelompokan kategori produk menggunakan manual binning (Stars, Cash Cows, Rising Stars, Underperformers)

---

## Link Dashboard (Streamlit Cloud)

[Live Dashboard](https://ulumlab-olist-dashboard.streamlit.app)
