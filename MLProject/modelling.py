import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_modelling():
    # 1. Load Data
    try:
        df = pd.read_csv('MLProject/Fashion_Retail_Sales.csv')
    except FileNotFoundError:
        print("File Fashion_Retail_Sales.csv tidak ditemukan!")
        return

    # 2. Preprocessing (Pembersihan & Transformasi)
    # Menghapus data kosong agar model tidak error
    df_clean = df.dropna(subset=['Review Rating', 'Purchase Amount (USD)', 'Item Purchased', 'Payment Method']).copy()

    # Label Encoding untuk data kategori (Teks -> Angka)
    le_item = LabelEncoder()
    le_payment = LabelEncoder()

    # Gunakan .assign untuk menghindari SettingWithCopyWarning
    df_clean = df_clean.assign(
        Item_Code = le_item.fit_transform(df_clean['Item Purchased']),
        Payment_Code = le_payment.fit_transform(df_clean['Payment Method'])
    )

    # Menentukan Fitur (X) dan Target (y)
    # Menambah Item_Code agar akurasi lebih baik dari sebelumnya
    X = df_clean[['Review Rating', 'Purchase Amount (USD)', 'Item_Code']]
    y = df_clean['Payment_Code']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. MLflow Experiment
    mlflow.set_experiment("Fashion_Retail_Advanced")

    with mlflow.start_run(run_name="Random Forest v2"):
        # Parameter
        n_estimators = 200 # Menaikkan jumlah pohon
        max_depth = 10
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Logging ke MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model_pembayaran")

        print("\n" + "="*40)
        print("📊 HASIL MODELLING SELESAI")
        print("="*40)
        print(f"Akurasi Model : {acc:.2%}")
        print("\nLaporan Klasifikasi:")
        print(classification_report(y_test, y_pred, target_names=le_payment.classes_))
        print("="*40)
        print("INFO: Jalankan 'mlflow ui' di terminal untuk melihat dashboard.")

if __name__ == "__main__":
    run_modelling()