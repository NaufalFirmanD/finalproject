# LogiTrack: Sistem Prediksi Keterlambatan Pengiriman

**LogiTrack** adalah sebuah aplikasi web berbasis machine learning yang dibangun menggunakan Streamlit untuk memprediksi potensi keterlambatan pengiriman dalam sebuah jaringan logistik. Aplikasi ini dirancang sebagai alat bantu pengambilan keputusan bagi tim operasional untuk secara proaktif memitigasi risiko keterlambatan dan meningkatkan kepuasan pelanggan.

Proyek ini merupakan tugas akhir dari **Kelompok 7** untuk program **DSGA (Data Science & Generative AI) Camp Batch 4**.

---

## 🚀 Fitur Utama

-   **Prediksi Real-time**: Menganalisis parameter pengiriman untuk memberikan prediksi keterlambatan secara instan.
-   **Antarmuka Interaktif**: UI yang modern dan mudah digunakan, memungkinkan pengguna memasukkan data dengan cepat melalui sidebar.
-   **Visualisasi Probabilitas**: Menampilkan probabilitas prediksi dalam bentuk bagan yang mudah dipahami.
-   **Rekomendasi Tindakan**: Memberikan saran tindakan preventif atau operasional berdasarkan hasil prediksi (apakah pengiriman berisiko terlambat atau aman).
-   **Model Machine Learning**: Ditenagai oleh model *Gradient Boosting Classifier* yang dilatih pada dataset E-Commerce Olist.

---

## 🛠️ Cara Menjalankan Proyek

Untuk menjalankan aplikasi ini di lingkungan lokal Anda, ikuti langkah-langkah berikut:

### 1. Prasyarat

-   Python 3.8 atau versi yang lebih baru.
-   `pip` (Python package installer).

### 2. Kloning atau Unduh Proyek

Dapatkan semua file proyek (seperti `app.py`, `train_model.py`, `requirements.txt`, dan dataset `.csv`) dan letakkan dalam satu folder.

### 3. Buat Lingkungan Virtual (Direkomendasikan)

Sangat disarankan untuk membuat lingkungan virtual agar dependensi proyek tidak tercampur dengan instalasi Python global Anda.

```bash
# Buat lingkungan virtual
python -m venv env

# Aktifkan lingkungan virtual
# Windows
.\env\Scripts\activate
# macOS/Linux
source env/bin/activate
```

### 4. Instal Dependensi

Instal semua pustaka Python yang dibutuhkan yang tercantum dalam file `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 5. Latih Model Machine Learning

Sebelum menjalankan aplikasi utama, Anda **wajib** melatih model terlebih dahulu. Skrip ini akan memproses data dan menghasilkan file-file model (`.pkl`) yang akan digunakan oleh aplikasi Streamlit.

```bash
python train_model.py
```

Pastikan skrip berjalan hingga selesai dan menghasilkan file seperti `best_model.pkl`, `le_customer.pkl`, dll. di direktori yang sama.

### 6. Jalankan Aplikasi Streamlit

Setelah model berhasil dilatih, jalankan aplikasi Streamlit.

```bash
streamlit run app.py
```

Aplikasi akan terbuka secara otomatis di browser default Anda.

---

## 📂 Struktur File

```
.
├── 📄 app.py                    # Skrip utama aplikasi Streamlit
├── 📄 train_model.py             # Skrip untuk melatih model
├── 📄 requirements.txt          # Daftar dependensi Python
├── 📄 olist_cleaned_dataset.csv   # Dataset yang digunakan untuk training
├── 📦 best_model.pkl            # (Dihasilkan) File model utama
├── 📦 le_customer.pkl           # (Dihasilkan) File encoder untuk state customer
├── 📦 le_seller.pkl             # (Dihasilkan) File encoder untuk state seller
└── 📦 le_category.pkl           # (Dihasilkan) File encoder untuk kategori produk
```

---

## Dibuat oleh

**Kelompok 7 - DSGA Camp Batch 4**
