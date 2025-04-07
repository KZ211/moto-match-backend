from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# ======================
# 1. CARGAR MODELO Y ENCODERS
# ======================
modelo = joblib.load("modelo_recomendador.pkl")
encoders = joblib.load("encoders.pkl")

# Cargar tabla de motos para buscar info luego
df_motos = pd.read_csv("data/motos.csv")

# ======================
# 2. CONFIGURAR API
# ======================
app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # URL de tu frontend
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP
    allow_headers=["*"],  # Permite todos los headers
)

# Clase para recibir los datos del usuario
class UsuarioInput(BaseModel):
    estatura: int
    peso: int
    genero: str
    experiencia: str
    preferencia_uso: str
    preferencia_estilo: str

# ======================
# 3. ENDPOINT DE RECOMENDACIÓN
# ======================
@app.post("/recomendar")
def recomendar(usuario: UsuarioInput):
    try:
        # Transformar input a formato que el modelo entienda
        entrada = [[
            usuario.estatura,
            usuario.peso,
            encoders["genero"].transform([usuario.genero])[0],
            encoders["experiencia"].transform([usuario.experiencia])[0],
            encoders["preferencia_uso"].transform([usuario.preferencia_uso])[0],
            encoders["preferencia_estilo"].transform([usuario.preferencia_estilo])[0],
        ]]

        # Predecir
        pred = modelo.predict(entrada)[0]
        modelo_nombre = encoders["modelo_recomendado"].inverse_transform([pred])[0]

        # Buscar datos adicionales de la moto
        ficha = df_motos[df_motos["modelo"] == modelo_nombre].iloc[0].to_dict()

        return {
            "modelo_recomendado": modelo_nombre,
            "ficha_tecnica": ficha
        }

    except Exception as e:
        return {"error": str(e)}
