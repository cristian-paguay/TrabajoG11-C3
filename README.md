# TrabajoG11-C3
Trabajo práctico G11 - Clase 3
# Análisis Económico Global 2000-2026 - Grupo 11

## 📋 Descripción
Proyecto universitario enfocado en el tratamiento de datos económicos mediante un flujo de ingeniería de datos: Limpieza, Persistencia en SQL y Visualización Interactiva.

## ⚙️ Requisitos
1. Python 3.9+
2. Dataset de Kaggle en `data/finance_economics.csv`

## 🚀 Instalación y Ejecución
1. Instalar dependencias:  
   `pip install -r requirements.txt`
2. Ejecutar la aplicación:  
   `streamlit run app.py`

## 🧹 Proceso de Datos
- **Limpieza:** Se trataron valores nulos mediante la media aritmética para no perder representatividad.
- **DB:** Se utilizó SQLite para asegurar la persistencia de los datos procesados.
- **Análisis:** Uso de mapas de correlación y regresiones lineales para identificar tendencias.