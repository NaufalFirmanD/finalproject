import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import altair as alt

# =================================================================================================
# KONFIGURASI HALAMAN & GAYA
# =================================================================================================
st.set_page_config(
    page_title="LogiTrack - Sistem Prediksi Keterlambatan",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Gaya CSS kustom untuk tampilan yang lebih modern
st.markdown("""
<style>
    .stMetric {
        border: 1px solid #2e3b4e;
        border-radius: 10px;
        padding: 15px;
        background-color: #1a222e;
    }
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #4f8bf9;
        background-color: #4f8bf9;
        color: white;
    }
    .stButton>button:hover {
        border: 1px solid #69a1ff;
        background-color: #69a1ff;
    }
</style>
""", unsafe_allow_html=True)


# =================================================================================================
# LOAD ASSET MODEL & ENCODER
# =================================================================================================
@st.cache_resource
def load_assets():
    """Memuat model dan encoder dari file .pkl dengan penanganan error yang lebih baik."""
    try:
        model = joblib.load('best_model.pkl')
        le_customer = joblib.load('le_customer.pkl')
        le_seller = joblib.load('le_seller.pkl')
        le_category = joblib.load('le_category.pkl')
        return model, le_customer, le_seller, le_category
    except FileNotFoundError as e:
        st.error(f"Error: File model/encoder tidak ditemukan. Pastikan file .pkl ada di direktori yang sama. Detail: {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"Gagal memuat aset. Error: {e}")
        return None, None, None, None

model, le_customer, le_seller, le_category = load_assets()

# Hentikan aplikasi jika aset gagal dimuat
if not all([model, le_customer, le_seller, le_category]):
    st.warning("Aplikasi tidak dapat berjalan karena aset penting (model/encoder) gagal dimuat.")
    st.stop()


# =================================================================================================
# SIDEBAR - PANEL INPUT
# =================================================================================================
with st.sidebar:
    st.image("https://i.imgur.com/sYvLDmU.png", width=150)
    st.header("📊 Parameter Pengiriman")
    st.write("Masukkan detail transaksi dan produk untuk dianalisis.")

    with st.form("input_form"):
        # --- Lokasi ---
        st.subheader("Lokasi")
        customer_state = st.selectbox("Negara Bagian Customer", options=list(le_customer.classes_), help="Pilih lokasi negara bagian customer.")
        seller_state = st.selectbox("Negara Bagian Penjual", options=list(le_seller.classes_), help="Pilih lokasi negara bagian penjual.")
        
        # --- Detail Produk ---
        st.subheader("Detail Produk")
        categories_raw = list(le_category.classes_)
        product_category = st.selectbox(
            "Kategori Produk", 
            options=categories_raw,
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Pilih kategori produk yang dikirim."
        )
        product_weight_g = st.number_input("Berat Produk (Gram)", min_value=0.0, value=1500.0, step=50.0, help="Masukkan berat produk dalam satuan gram.")
        
        # --- Finansial ---
        st.subheader("Finansial")
        price = st.number_input("Harga Barang (BRL)", min_value=0.0, value=120.0, step=10.0, help="Harga produk dalam Real Brasil (BRL).")
        freight_value = st.number_input("Biaya Ongkos Kirim (BRL)", min_value=0.0, value=25.0, step=5.0, help="Biaya pengiriman dalam Real Brasil (BRL).")

        # --- Waktu Transaksi ---
        st.subheader("Waktu Transaksi")
        purchase_date = st.date_input("Tanggal Pembelian", value=datetime.now().date())
        purchase_time = st.time_input("Jam Pembelian", value=datetime.now().time())

        # Tombol submit form
        analyze_button = st.form_submit_button("Analisis Risiko Pengiriman", use_container_width=True)


# =================================================================================================
# PANEL UTAMA - HEADER & HASIL
# =================================================================================================
st.title("🚚 LogiTrack: Sistem Prediksi Keterlambatan")
st.markdown("#### Solusi Cerdas untuk Mitigasi Risiko Logistik oleh Kelompok 7 DSGA")
st.write(
    "Selamat datang di LogiTrack! Aplikasi ini membantu tim operasional mengidentifikasi potensi "
    "keterlambatan pengiriman secara proaktif. Masukkan parameter di sidebar kiri dan klik tombol analisis."
)
st.markdown("---")

# Placeholder untuk hasil prediksi
result_placeholder = st.empty()

if analyze_button:
    # Ekstrak fitur waktu
    purchase_month = purchase_date.month
    purchase_dayofweek = purchase_date.weekday()
    purchase_hour = purchase_time.hour
    
    # Transformasi input
    cust_encoded = le_customer.transform([customer_state])[0]
    sell_encoded = le_seller.transform([seller_state])[0]
    cat_encoded = le_category.transform([product_category])[0]
            
    # Buat DataFrame input
    input_data = pd.DataFrame([{
        'customer_state': cust_encoded,
        'seller_state': sell_encoded,
        'product_category_name_english': cat_encoded,
        'product_weight_g': product_weight_g,
        'price': price,
        'freight_value': freight_value,
        'purchase_month': purchase_month,
        'purchase_dayofweek': purchase_dayofweek,
        'purchase_hour': purchase_hour
    }])
    
    # Lakukan Prediksi
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    
    # Tampilkan Output di Panel Utama
    with result_placeholder.container(border=True):
        st.subheader("🔮 Hasil Analisis Risiko", anchor=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if prediction == 1:
                st.error("### ⚠️ Peringatan: Pengiriman Berisiko TERLAMBAT!", icon="🚨")
                st.metric(label="Probabilitas Keterlambatan", value=f"{prediction_proba[1]*100:.2f}%")
                st.markdown("""
                **Rekomendasi Tindakan Preventif:**
                - **Prioritaskan Paket:** Segera proses paket ini untuk mempercepat waktu sorting di hub.
                - **Evaluasi Kurir:** Pertimbangkan untuk menggunakan layanan kurir alternatif atau premium untuk rute ini.
                - **Komunikasi Proaktif:** Kirim notifikasi otomatis kepada pelanggan mengenai potensi perpanjangan estimasi waktu tiba.
                """)
            else:
                st.success("### ✅ Aman: Pengiriman Diprediksi TEPAT WAKTU", icon="👍")
                st.metric(label="Probabilitas Tepat Waktu", value=f"{prediction_proba[0]*100:.2f}%")
                st.markdown("""
                **Rekomendasi Operasional:**
                - **Proses Standar:** Lanjutkan pengiriman sesuai Prosedur Standar Operasional (SOP) yang berlaku.
                - **Monitoring Kinerja:** Pertahankan kinerja logistik yang baik pada rute ini untuk menjaga kepuasan pelanggan.
                """)
        
        with col2:
            # Visualisasi probabilitas dengan Altair
            prob_data = pd.DataFrame({
                'Status': ['Tepat Waktu', 'Terlambat'],
                'Probabilitas': prediction_proba,
                'Color': ['#3dd56d', '#ff4b4b']
            })
            
            chart = alt.Chart(prob_data).mark_bar(cornerRadius=5).encode(
                x=alt.X('Probabilitas:Q', axis=alt.Axis(format='%', title='Probabilitas')),
                y=alt.Y('Status:N', sort='-x', title=None),
                color=alt.Color('Status:N', scale=alt.Scale(domain=['Tepat Waktu', 'Terlambat'], range=['#3dd56d', '#ff4b4b']), legend=None)
            ).properties(
                title='Distribusi Probabilitas Prediksi'
            )
            st.altair_chart(chart, use_container_width=True)

else:
    # Tampilan default sebelum analisis
    with result_placeholder.container(border=True):
        st.info("Menunggu analisis... Silakan isi parameter di sidebar dan klik tombol 'Analisis Risiko Pengiriman'.")
        st.image("https://i.imgur.com/sYvLDmU.png", caption="LogiTrack siap menganalisis datamu!", use_column_width=True)


# =================================================================================================
# FOOTER & INFORMASI TAMBAHAN
# =================================================================================================
st.markdown("---")
with st.expander("Tentang Proyek LogiTrack & Model"):
    st.markdown("""
    **LogiTrack** adalah proyek akhir dari **Kelompok 7** untuk program **DSGA (Data Science & Generative AI) Camp Batch 4**. 
    
    Aplikasi ini dibangun menggunakan model machine learning yang dilatih pada dataset Olist E-Commerce untuk memprediksi kemungkinan keterlambatan pengiriman.
    
    **Tujuan:**
    - Memberikan alat bantu bagi manajer operasional untuk mengambil keputusan.
    - Mengurangi angka keterlambatan dengan intervensi proaktif.
    - Meningkatkan kepuasan pelanggan.
    
    **Model yang Digunakan:**
    - **Algoritma:** Gradient Boosting Classifier
    - **Fitur Utama:** Lokasi (customer & seller), kategori produk, berat, harga, biaya kirim, dan komponen waktu (bulan, hari, jam).
    """)

st.caption("© 2024 - Aplikasi Prediksi Logistik Olist | Dibuat oleh Kelompok 7 DSGA Camp Batch 4")
