import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, recall_score
import joblib
import os

print("=== 1. MEMBACA & GABUNGKAN SEMUA FILE CSV MENTAH ===")

# Validasi file di folder
required_files = [
    'olist_orders_dataset.csv', 'olist_customers_dataset.csv', 
    'olist_order_items_dataset.csv', 'olist_sellers_dataset.csv', 
    'olist_products_dataset.csv', 'product_category_name_translation.csv'
]

for f in required_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"File {f} belum lu pindahin ke folder proyek VS Code!")

print("-> Loading file CSV...")
o = pd.read_csv('olist_orders_dataset.csv')
c = pd.read_csv('olist_customers_dataset.csv')
oi = pd.read_csv('olist_order_items_dataset.csv')
s = pd.read_csv('olist_sellers_dataset.csv')
p = pd.read_csv('olist_products_dataset.csv')
t = pd.read_csv('product_category_name_translation.csv')

# Filter status delivered & non-null delivery date
o = o[(o['order_status'] == 'delivered') & (o['order_delivered_customer_date'].notnull())].copy()

# Buat TARGET LABEL (is_late)
print("-> Membuat target label 'is_late'...")
o['order_delivered_customer_date'] = pd.to_datetime(o['order_delivered_customer_date'])
o['order_estimated_delivery_date'] = pd.to_datetime(o['order_estimated_delivery_date'])
o['is_late'] = (o['order_delivered_customer_date'] > o['order_estimated_delivery_date']).astype(int)

# Mapping nama kategori ke Bahasa Inggris
print("-> Menerjemahkan kategori produk ke Bahasa Inggris...")
p_mapped = p.merge(t, on='product_category_name', how='left')
p_mapped['product_category_name_english'] = p_mapped['product_category_name_english'].fillna('others')

# Proses Pandas Merge Penggabungan Global
print("-> Menggabungkan (Merging) seluruh tabel data...")
df = o.merge(c, on='customer_id', how='inner')
df = df.merge(oi, on='order_id', how='inner')
df = df.merge(s, on='seller_id', how='inner')
df = df.merge(p_mapped, on='product_id', how='inner')

# Simpan dataset bersih untuk backup tim kelompok lu
df.to_csv('olist_cleaned_dataset.csv', index=False)
print(f"-> Sukses generate 'olist_cleaned_dataset.csv'. Total: {df.shape[0]} baris\n")

print("=== 2. HANDLING MISSING VALUES ===")
df['product_weight_g'] = df['product_weight_g'].fillna(df['product_weight_g'].median())

print("=== 3. EKSTRAKSI FITUR WAKTW ===")
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['purchase_month'] = df['order_purchase_timestamp'].dt.month
df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour

# Seleksi fitur prediktor final
features = [
    'customer_state', 'seller_state', 'product_category_name_english',
    'product_weight_g', 'price', 'freight_value', 
    'purchase_month', 'purchase_dayofweek', 'purchase_hour', 'is_late'
]
df = df[features]

print("=== 4. ENCODING FITUR KATEGORIKAL ===")
le_customer = LabelEncoder()
le_seller = LabelEncoder()
le_category = LabelEncoder()

df['customer_state'] = le_customer.fit_transform(df['customer_state'])
df['seller_state'] = le_seller.fit_transform(df['seller_state'])
df['product_category_name_english'] = le_category.fit_transform(df['product_category_name_english'])

# Simpan semua encoder
joblib.dump(le_customer, 'le_customer.pkl')
joblib.dump(le_seller, 'le_seller.pkl')
joblib.dump(le_category, 'le_category.pkl')
print("-> File 'le_customer.pkl', 'le_seller.pkl', & 'le_category.pkl' aman disimpan.\n")

print("=== 5. MEMISAHKAN DATA TRAIN & TEST ===")
X = df.drop(columns=['is_late'])
y = df['is_late']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("=== 6. MENJALANKAN EKSPERIMEN 3 MODEL ===")
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), random_state=42, eval_metric='logloss', n_jobs=-1)
}

best_model_name = None
best_f1_score = 0
best_model_obj = None
results_summary = {}

for name, model in models.items():
    print(f"\n---> Training Model: {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    results_summary[name] = {"F1-Score": f1, "Recall": recall}
    
    print(f"Hasil Evaluasi {name}:")
    print(classification_report(y_test, y_pred))
    
    if f1 > best_f1_score:
        best_f1_score = f1
        best_model_name = name
        best_model_obj = model

print("\n=== 7. KESIMPULAN EKSPERIMEN MODEL ===")
print("--------------------------------------------------")
print(f"{'Nama Model':<25} | {'F1-Score':<10} | {'Recall':<10}")
print("--------------------------------------------------")
for name, metrics in results_summary.items():
    print(f"{name:<25} | {metrics['F1-Score']:.4f}     | {metrics['Recall']:.4f}")
print("--------------------------------------------------")

print(f"\n=> Model Terbaik Berdasarkan F1-Score: {best_model_name}")

# TAKTIK BYPASS LIMIT GITHUB 100MB: Kompres pkl dengan parameter compress=3
if best_model_name == "Random Forest":
    print("-> Mengompres model Random Forest agar filenya < 100MB...")
    joblib.dump(best_model_obj, 'best_model.pkl', compress=3)
else:
    joblib.dump(best_model_obj, 'best_model.pkl')

print("-> File 'best_model.pkl' sukses disimpan dengan ukuran ekonomis!")