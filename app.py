import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Set page layout professional
st.set_page_config(page_title="LogiTrack - Delay Prediction System", page_icon="🚚", layout="wide")

# Load model dan encoder
@st.cache_resource
def load_assets():
    model = joblib.load('best_model.pkl')
    le_customer = joblib.load('le_customer.pkl')
    le_seller = joblib.load('le_seller.pkl')
    le_category = joblib.load('le_category.pkl')
    return model, le_customer, le_seller, le_category

try:
    model, le_customer, le_seller, le_category = load_assets()
except Exception as e:
    st.error(f"Gagal memuat file .pkl. Pastikan lu run 'python train_model.py' terlebih dahulu! Error: {e}")
    st.stop()

# Header Utama
st.title("🚚 LogiTrack: Sistem Prediksi Keterlambatan Pengiriman")
st.markdown("### Dukungan Pengambilan Keputusan Operasional Logistik Kelompok 7 DSGA")
st.write("Aplikasi cerdas untuk memitigasi risiko keterlambatan pengiriman armada logistik berdasarkan karakteristik barang dan rute asal-tujuan.")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input Parameter Operasional")
    
    # Dropdown interaktif berdasarkan class encoder asli
    customer_state = st.selectbox("Negara Bagian Customer (Customer State)", options=list(le_customer.classes_))
    seller_state = st.selectbox("Negara Bagian Penjual (Seller State)", options=list(le_seller.classes_))
    
    # Trik memformat tampilan dropdown agar bersih tanpa underscore dan berhuruf kapital di awal kata
    categories_raw = list(le_category.classes_)
    product_category = st.selectbox(
        "Kategori Produk (Product Category)", 
        options=categories_raw,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Input Dimensi Fisik & Harga
    product_weight_g = st.number_input("Berat Produk (Gram)", min_value=0.0, value=1500.0, step=50.0)
    price = st.number_input("Harga Barang (BRL)", min_value=0.0, value=120.0, step=10.0)
    freight_value = st.number_input("Biaya Ongkos Kirim (BRL)", min_value=0.0, value=25.0, step=5.0)
    
    # Input Waktu Transaksi
    purchase_date = st.date_input("Tanggal Transaksi Pembelian", value=datetime.now().date())
    purchase_time = st.time_input("Jam Transaksi Pembelian", value=datetime.now().time())

with col2:
    st.subheader("🔮 Hasil Analisis & Manajemen Risiko")
    st.write("Sistem AI akan mengevaluasi parameter input terhadap pola historis keterlambatan Olist.")
    
    if st.button("Analisis Risiko Pengiriman", type="primary"):
        # Ekstraksi fitur komponen waktu
        purchase_month = purchase_date.month
        purchase_dayofweek = purchase_date.weekday()
        purchase_hour = purchase_time.hour
        
        # Transformasi Label Encoding
        cust_encoded = le_customer.transform([customer_state])[0]
        sell_encoded = le_seller.transform([seller_state])[0]
        cat_encoded = le_category.transform([product_category])[0]
            
        # Bentuk input data dengan urutan kolom yang wajib presisi
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
        
        # Hitung prediksi dan probabilitas
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        st.markdown("#### **Status Evaluasi AI:**")
        
        if prediction == 1:
            st.error(f"⚠️ PERINGATAN: Rute & Karakteristik Barang Berisiko TERLAMBAT!")
            st.metric(label="Tingkat Risiko Keterlambatan", value=f"{prediction_proba[1]*100:.2f}%")
            st.markdown("""
            **Rekomendasi Kontingensi Kelompok 7:**
            * Alihkan ke kurir prioritas tinggi (*Express Delivery Link*).
            * Lakukan pengecekan ganda pada hub sortir asal negara bagian terkait.
            * Kirim *pre-alert* otomatis ke customer mengenai potensi pergeseran waktu pengiriman.
            """)
        else:
            st.success("✅ AMAN: Pengiriman Diprediksi TEPAT WAKTU")
            st.metric(label="Tingkat Keyakinan Tepat Waktu", value=f"{prediction_proba[0]*100:.2f}%")
            st.markdown("""
            **Rekomendasi Tindakan:**
            * Jalankan alur logistik reguler sesuai dengan SOP standard warehouse.
            * Monitor pergerakan armada secara berkala via dashboard internal.
            """)

st.markdown("---")
st.caption("LogiTrack Engine v2.0 | CAMP Batch 4 DSGA | Kelompok 7 Mini Project Assignment")