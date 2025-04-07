# MotoMatch Backend

Backend para la aplicación MotoMatch, un recomendador de motos basado en características del usuario.

## Requisitos

- Python 3.9+
- Dependencias listadas en `requirements.txt`

## Instalación local

1. Clonar el repositorio
2. Crear un entorno virtual: `python -m venv venv`
3. Activar el entorno virtual:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Instalar dependencias: `pip install -r requirements.txt`
5. Ejecutar la aplicación: `python main.py`

## Despliegue en Render.com

1. Crear una cuenta en [Render.com](https://render.com)
2. Crear un nuevo Web Service
3. Conectar con tu repositorio de GitHub
4. Configurar el servicio:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py`
   - **Environment Variables**: No se requieren variables de entorno adicionales

## Estructura de archivos

- `app.py`: Aplicación principal FastAPI
- `main.py`: Punto de entrada para el despliegue en Render.com
- `modelo_recomendador.pkl`: Modelo de machine learning para recomendaciones
- `encoders.pkl`: Encoders para transformar datos categóricos
- `data/`: Directorio con datos de motos
- `requirements.txt`: Dependencias del proyecto
- `Procfile`: Configuración para despliegue en Render.com

## API Endpoints

- `POST /recomendar`: Recibe datos del usuario y devuelve una recomendación de moto # moto-match-backend
