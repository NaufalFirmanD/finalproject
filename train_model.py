import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

print("🔄 Step 1: Ingestion & Merging Data Olist...")
# Load 5 File Utama + 1 Translasi
orders = pd.read_csv('olist_orders_dataset.csv')
items = pd.read_csv('olist_order_items_dataset.csv')
customers = pd.read_csv('olist_customers_dataset.csv')
sellers = pd.read_csv('olist_sellers_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')
translation = pd.read_csv('product_category_name_translation.csv')

# Merge Relasional Pipeline
df = orders.merge(items, on='order_id') \
           .merge(customers, on='customer_id') \
           .merge(sellers, on='seller_id') \
           .merge(products, on='product_id') \
           .merge(translation, on='product_category_name')

print("🧹 Step 2: Cleaning Outliers & Target Labeling...")
# Target Labeling (1 = Telat, 0 = Tepat Waktu)
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
df['is_late'] = (df['order_delivered_customer_date'] > df['order_estimated_delivery_date']).astype(int)

# Filter Outliers Kurir Retail (Maks 30 Kg sesuai revisi)
df = df[(df['product_weight_g'] > 0) & (df['product_weight_g'] <= 30000)]
df = df[(df['product_length_cm'] > 0) & (df['product_height_cm'] > 0) & (df['product_width_cm'] > 0)]

print("⚙️ Step 3: Feature Engineering Berbasis Logistik Bisnis...")
# 1. Rute Distribusi (Gabungan State)
df['rute_distribusi'] = df['seller_state'] + "_to_" + df['customer_state']
# 2. Volume Produk (P x L x T)
df['volume_cm3'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
# 3. Densitas Kargo (Berat / Volume)
df['densitas_kargo'] = df['product_weight_g'] / df['volume_cm3']
# 4. Ongkir per Gram (Freight / Weight)
df['ongkir_per_gram'] = df['price'] / df['product_weight_g']

# Seleksi Fitur Sesuai Revisi (Hapus semua komponen fitur waktu transaksi!)
fitur_numerik = ['product_weight_g', 'volume_cm3', 'densitas_kargo', 'ongkir_per_gram', 'price', 'freight_value']
fitur_kategorikal = ['rute_distribusi', 'product_category_name_english']

X = df[fitur_numerik + fitur_kategorikal]
y = df['is_late']

# Stratified Split untuk Imbalanced Data
# BENER  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("🏗️ Step 4: Membangun Kesatuan Scikit-Learn Pipeline...")
# Preprocessor Pipeline (Encoding & Scaling)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), fitur_numerik),
        ('cat', OneHotEncoder(handle_unknown='ignore'), fitur_kategorikal)
    ])

# Model dengan Penanganan Imbalance Data (class_weight='balanced')
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1))
])

print("🏋️ Step 5: Training Model Pipeline...")
model_pipeline.fit(X_train, y_train)

# Evaluasi Jujur dan Transparan untuk Halaman Dashboard
print("\n📊 HASIL EVALUASI MODEL:")
y_pred = model_pipeline.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_pred))

print("📦 Step 6: Ekspor Berkas Tunggal Terkompresi...")
# Simpan satu file final terkompresi bypass limit github (<100MB)
joblib.dump(model_pipeline, 'logitrack_production_pipeline.pkl', compress=3)
print("🏆 Sukses! Berkas 'logitrack_production_pipeline.pkl' siap mengudara!")