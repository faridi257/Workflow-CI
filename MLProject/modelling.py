import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from sklearn.utils import estimator_html_repr
import mlflow
import mlflow.sklearn

# --- OWAHAN 1: Ora perlu set_experiment amarga wis diatur dening 'mlflow run' ---

df = pd.read_csv('dataset_preprocessing/fashion_sales_processed.csv')
X = df.drop('Purchase Amount (USD)', axis=1)
y = df['Purchase Amount (USD)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
params = {"n_estimators": 150, "max_depth": 10, "random_state": 42}

# --- OWAHAN 2: Hapus 'with mlflow.start_run()' ---
# MLflow Project otomatis nggawe run, dadi awake dhewe langsung logging wae
model = RandomForestRegressor(**params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- A. MANUAL LOGGING PARAMETERS & 5 METRIKS (Syarat Skilled) [cite: 62, 128] ---
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
med_ae = median_absolute_error(y_test, y_pred)

mlflow.log_params(params)
mlflow.log_metric("mse", mse)
mlflow.log_metric("mae", mae)
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("r2_score", r2)
mlflow.log_metric("median_absolute_error", med_ae)

# --- B. NGGAWE ESTIMATOR.HTML (Artefak Skilled) ---
with open("estimator.html", "w", encoding="utf-8") as f:
    f.write(estimator_html_repr(model))
mlflow.log_artifact("estimator.html")

# --- C. ARTEFAK VISUALISASI TAMBAHAN [cite: 66, 180] ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Amount')
plt.ylabel('Predicted Amount')
plt.title('Actual vs Predicted Purchase Amount')

plot_path = "training_evaluation_plot.png"
plt.savefig(plot_path)
mlflow.log_artifact(plot_path)
plt.close()

# --- D. LOG MODEL ---
mlflow.sklearn.log_model(model, "model")

print("Run CI Berhasil! Kabeh artefak lan 5 metriks wis dicathet otomatis dening MLProject.")