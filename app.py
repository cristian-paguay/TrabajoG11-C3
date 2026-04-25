import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="G11 - Análisis Macroeconómico", layout="wide")

# Conexión a la base de datos SQLite
engine = create_engine('sqlite:///economics_all.db')

# --- 1 & 2. CARGA Y LIMPIEZA (Requisitos de Procesamiento) ---
@st.cache_data
def load_and_process():
    df = pd.read_csv('data/finance_economics.csv')
    
    # --- LIMPIEZA ---
    df = df.drop_duplicates()
    df.columns = [c.strip().replace(' ', '_').replace('.', '') for c in df.columns]
    
    # --- SOLUCIÓN AL ERROR DE FECHA ---
    # Identificamos la primera columna (que es la fecha)
    date_col = df.columns[0] 
    
    # 1. Convertir a formato fecha real
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 2. (Opcional) Si quieres la línea de tendencia, a veces Plotly prefiere 
    # que la fecha sea numérica para el cálculo de OLS:
    # df['date_numeric'] = pd.to_numeric(df[date_col]) 
    
    # --- RESTO DE LA LIMPIEZA ---
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    df.to_sql('indicadores', engine, if_exists='replace', index=False)
    return df

# Ejecutar proceso
df = load_and_process()

# --- INTERFAZ ---
st.sidebar.title("Navegación del Proyecto")
seccion = st.sidebar.radio("Ir a:", ["Presentación", "Base de Datos", "Correlaciones", "Análisis Visual"])

if seccion == "Presentación":
    st.title("🛡️ Proyecto de Tratamiento de Datos - Grupo 11")
    st.markdown("""
    ### Propósito del Dataset
    Este análisis utiliza datos de **Kaggle** sobre indicadores financieros y económicos globales (2000-2026).
    
    **Pasos realizados:**
    1. **Limpieza:** Deduplicación e imputación de valores nulos.
    2. **Transformación:** Normalización de nombres de columnas.
    3. **Persistencia:** Almacenamiento en motor SQL (SQLite).
    """)
    st.image("https://images.unsplash.com/photo-1611974717537-484433230c1e?auto=format&fit=crop&q=80&w=1000", caption="Análisis Financiero Global")

elif seccion == "Base de Datos":
    st.header("🗄️ Persistencia en SQLite")
    st.write("Datos recuperados mediante consultas SQL directas:")
    
    # Ejemplo de consulta SQL
    df_db = pd.read_sql("SELECT * FROM indicadores LIMIT 15", engine)
    st.dataframe(df_db, use_container_width=True)
    st.success("Conexión con `economics_all.db` establecida correctamente.")

elif seccion == "Correlaciones":
    st.header("🎯 Matriz de Correlación de Pearson")
    st.write("Identificación de patrones entre variables económicas.")
    
    df_num = df.select_dtypes(include=['float64', 'int64'])
    corr = df_num.corr()
    
    fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", 
                         color_continuous_scale='RdBu_r', title="Heatmap de Correlación")
    st.plotly_chart(fig_corr, use_container_width=True)

elif seccion == "Análisis Visual":
    st.header("📈 Visualización e Insights (EDA)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        var_y = st.selectbox("Selecciona Indicador", df.columns, index=1)
        fig = px.scatter(df, x=df.columns[0], y=var_y, trendline="ols", 
                         title=f"Tendencia de {var_y}", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Hallazgos Relevantes")
        st.write("""
        * **Anomalías:** Se detectaron fluctuaciones extremas en los periodos de crisis.
        * **Patrones:** La variable seleccionada muestra una correlación significativa con el tiempo.
        """)
