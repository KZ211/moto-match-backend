import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# ======================
# 1. CARGAR EL DATASET
# ======================
ruta_archivo = os.path.join("data", "usuarios_motos.csv")
df = pd.read_csv(ruta_archivo)

# ======================
# 2. ENCODEAR CATEGORÍAS
# ======================
# Hay columnas de texto que la IA no puede procesar directamente (como "genero" o "modelo_recomendado").
# Las convertimos a números con LabelEncoder.

columnas_a_encodear = ["genero", "experiencia", "preferencia_uso", "preferencia_estilo", "modelo_recomendado"]
encoders = {}

for col in columnas_a_encodear:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

# ======================
# 3. DIVIDIR X e y
# ======================
X = df.drop("modelo_recomendado", axis=1)
y = df["modelo_recomendado"]

# ======================
# 4. ENTRENAR MODELO
# ======================
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y)

# ======================
# 5. GUARDAR MODELO Y ENCODERS
# ======================
joblib.dump(modelo, "modelo_recomendador.pkl")
joblib.dump(encoders, "encoders.pkl")

print("✅ Modelo y encoders guardados correctamente.")
