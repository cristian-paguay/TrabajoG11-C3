"""
=============================================================================
G11 - Inteligencia Financiera y Macroeconómica Avanzada
=============================================================================
Descripción: Dashboard interactivo para análisis de mercados financieros
             y variables macroeconómicas. Incluye EDA, ML y base de datos.
Autores    : Grupo 11
Tecnologías: Streamlit, Plotly, Scikit-Learn, SQLAlchemy, Pandas
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import os

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="G11 · Inteligencia Macroeconómica",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border-left: 5px solid #e94560;
}
.main-header h1 { color: #ffffff; margin: 0; font-size: 2.2rem; }
.main-header p  { color: #a0aec0; margin: 0.5rem 0 0; font-size: 1rem; }
.insight-box {
    background: #1a2035;
    border-left: 4px solid #3182ce;
    padding: 0.8rem 1.2rem;
    border-radius: 6px;
    margin: 0.5rem 0;
    color: #e2e8f0;
}
.insight-box strong { color: #63b3ed; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# BASE DE DATOS: Conexión SQLite (sin pyarrow)
# =============================================================================
_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_dir, "economics_all.db")
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})

# =============================================================================
# FUNCIONES DE ANÁLISIS TÉCNICO
# =============================================================================

def calcular_rsi(serie: pd.Series, ventana: int = 14) -> pd.Series:
    """Calcula el Índice de Fuerza Relativa (RSI). >70 sobrecompra, <30 sobreventa."""
    delta = serie.diff()
    ema_g = delta.clip(lower=0).ewm(com=ventana - 1, adjust=False).mean()
    ema_p = (-delta.clip(upper=0)).ewm(com=ventana - 1, adjust=False).mean()
    rs = ema_g / ema_p.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calcular_bandas_bollinger(serie: pd.Series, ventana: int = 20, num_std: float = 2.0):
    """Bandas de Bollinger: (banda_sup, media, banda_inf)."""
    media = serie.rolling(window=ventana).mean()
    desv  = serie.rolling(window=ventana).std()
    return media + num_std * desv, media, media - num_std * desv


def calcular_macd(serie: pd.Series, rapida=12, lenta=26, senal=9):
    """MACD: devuelve (macd, señal, histograma)."""
    ema_r = serie.ewm(span=rapida, adjust=False).mean()
    ema_l = serie.ewm(span=lenta,  adjust=False).mean()
    macd  = ema_r - ema_l
    sig   = macd.ewm(span=senal, adjust=False).mean()
    return macd, sig, macd - sig


# =============================================================================
# ETL: CARGA, LIMPIEZA Y TRANSFORMACIÓN
# =============================================================================

@st.cache_data(show_spinner="⏳ Procesando datos financieros...")
def cargar_y_procesar() -> pd.DataFrame:
    """
    Pipeline ETL completo:
      1. Lectura CSV local
      2. Normalización de columnas
      3. Conversión de tipos (datetime64, float64)
      4. Imputación temporal ffill/bfill
      5. Winsorización con IQR
      6. Feature Engineering por índice: MA, RSI, Bollinger, MACD, retornos
      7. PCA macroeconómico sintético
      8. Persistencia en SQLite
    """
    # 1. Lectura
    ruta_csv = os.path.join(_dir, "Data", "finance_economics_dataset.csv")
    df = pd.read_csv(ruta_csv)

    # 2. Limpieza de nombres
    nuevos = []
    for c in df.columns:
        c = (c.strip()
              .replace(' ', '_').replace('(%)', '').replace('(Billion_USD)', '')
              .replace('(USD_per_Barrel)', '').replace('(USD_per_Ounce)', '')
              .replace('/', '_').replace('&', 'y').replace('__', '_').rstrip('_'))
        nuevos.append(c)
    df.columns = nuevos

    # 3. Tipos
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    cols_num = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for c in cols_num:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # 4. Imputación temporal
    df[cols_num] = df[cols_num].ffill().bfill()

    # 5. Winsorización IQR
    for c in cols_num:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        df[c] = df[c].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

    # 6. Feature Engineering por índice
    piezas = []
    for idx in df['Stock_Index'].unique():
        blq = df[df['Stock_Index'] == idx].copy().reset_index(drop=True)
        blq['MA_20_Close']  = blq['Close_Price'].rolling(20).mean()
        blq['MA_50_Close']  = blq['Close_Price'].rolling(50).mean()
        blq['MA_200_Close'] = blq['Close_Price'].rolling(200).mean()
        blq['RSI_14'] = calcular_rsi(blq['Close_Price'])
        blq['BB_sup'], blq['BB_med'], blq['BB_inf'] = calcular_bandas_bollinger(blq['Close_Price'])
        blq['MACD'], blq['MACD_Signal'], blq['MACD_Hist'] = calcular_macd(blq['Close_Price'])
        blq['Retorno_Diario'] = blq['Close_Price'].pct_change() * 100
        blq['Volatilidad_20d'] = blq['Retorno_Diario'].rolling(20).std()
        if 'GDP_Growth' in blq.columns:
            blq['GDP_YoY'] = blq['GDP_Growth'].pct_change(periods=252) * 100
        piezas.append(blq)
    df = pd.concat(piezas).sort_values(['Date', 'Stock_Index']).reset_index(drop=True)

    # 7. PCA macroeconómico sintético
    cols_pca = ['Inflation_Rate', 'Interest_Rate', 'Unemployment_Rate']
    if all(c in df.columns for c in cols_pca):
        pca    = PCA(n_components=1)
        scaler = StandardScaler()
        df['Indice_Macro_Sintetico'] = pca.fit_transform(
            scaler.fit_transform(df[cols_pca].fillna(0))
        )

    df = df.dropna(subset=['Close_Price', 'Open_Price']).reset_index(drop=True)

    # 8. Persistencia SQLite
    try:
        df.to_sql('indicadores', engine, if_exists='replace', index=False)
    except Exception:
        pass

    return df


# =============================================================================
# CARGA INICIAL
# =============================================================================
try:
    df_global = cargar_y_procesar()
except Exception as e:
    st.error(f"❌ Error crítico en el ciclo ETL: {e}")
    st.stop()

# =============================================================================
# SIDEBAR
# =============================================================================
# Ícono SVG embebido (no depende de URLs externas)
st.sidebar.markdown("""
<div style="text-align:center; padding:1rem 0;">
  <svg width="72" height="72" viewBox="0 0 72 72" xmlns="http://www.w3.org/2000/svg">
    <rect width="72" height="72" rx="14" fill="#0f3460"/>
    <polyline points="10,54 22,32 34,41 48,18 62,27"
              fill="none" stroke="#e94560" stroke-width="3.5" stroke-linejoin="round" stroke-linecap="round"/>
    <circle cx="48" cy="18" r="5" fill="#e94560"/>
    <circle cx="22" cy="32" r="3" fill="#63b3ed" opacity="0.8"/>
    <circle cx="34" cy="41" r="3" fill="#63b3ed" opacity="0.8"/>
    <circle cx="62" cy="27" r="3" fill="#63b3ed" opacity="0.8"/>
    <text x="36" y="68" text-anchor="middle" fill="#718096"
          font-size="9" font-family="monospace" font-weight="bold">G11 · FIN</text>
  </svg>
  <p style="color:#e2e8f0;font-weight:700;margin:0.5rem 0 0;font-size:1rem;">
    Inteligencia Macroeconómica
  </p>
  <p style="color:#718096;font-size:0.75rem;margin:0;">Grupo 11 · 2026</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.title("🧭 Navegación")

seccion = st.sidebar.radio("Sección activa:", [
    "🏠 Inicio y Resumen",
    "📊 Análisis Visual (EDA)",
    "📈 Análisis Técnico",
    "🤖 Machine Learning",
    "🗄️ Base de Datos"
], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Filtros Globales")

indices_disponibles = sorted(df_global['Stock_Index'].unique().tolist())
selected_index = st.sidebar.multiselect(
    "📌 Índices Bursátiles:",
    options=indices_disponibles,
    default=indices_disponibles
)

fecha_min = df_global['Date'].min().date()
fecha_max = df_global['Date'].max().date()
date_range = st.sidebar.date_input(
    "📅 Rango Temporal:",
    value=[fecha_min, fecha_max],
    min_value=fecha_min,
    max_value=fecha_max
)

# Filtros aplicados
if len(date_range) == 2:
    ini, fin = date_range
    mask = (df_global['Date'].dt.date >= ini) & (df_global['Date'].dt.date <= fin)
    df_f = df_global[mask].copy()
else:
    df_f = df_global.copy()

if selected_index:
    df_f = df_f[df_f['Stock_Index'].isin(selected_index)]

n_registros = len(df_f)
st.sidebar.markdown("---")
st.sidebar.caption(f"📊 {n_registros:,} registros activos")
if n_registros:
    st.sidebar.caption(f"📅 {df_f['Date'].min().strftime('%Y-%m-%d')} → {df_f['Date'].max().strftime('%Y-%m-%d')}")


# =============================================================================
# SECCIÓN 1: INICIO Y RESUMEN EJECUTIVO
# =============================================================================
if seccion == "🏠 Inicio y Resumen":
    st.markdown("""
    <div class="main-header">
      <h1>📈 Inteligencia Financiera y Macroeconómica</h1>
      <p>Análisis integral de mercados bursátiles e indicadores económicos globales · Grupo 11</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("🔍 Este dashboard automatiza el ciclo de vida de los datos: desde la ingesta hasta el modelado "
            "predictivo, conectando variables macro (PIB, Inflación, Desempleo) con el comportamiento "
            "de los mercados bursátiles (Dow Jones, S&P 500, NASDAQ).")

    if n_registros == 0:
        st.warning("No hay datos para el rango/índices seleccionados.")
        st.stop()

    # KPIs
    st.subheader("📌 Indicadores Clave del Periodo Seleccionado")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("💵 Precio Cierre Prom.",    f"${df_f['Close_Price'].mean():,.2f}")
    k2.metric("🏆 Precio Máximo",          f"${df_f['Close_Price'].max():,.2f}")
    k3.metric("📉 Volatilidad (σ)",         f"{df_f['Close_Price'].std():,.2f}")
    k4.metric("🔥 Inflación Promedio",      f"{df_f['Inflation_Rate'].mean():.2f}%")
    k5.metric("📊 Crecim. PIB Prom.",       f"{df_f['GDP_Growth'].mean():.2f}%")

    st.markdown("---")

    # Gráficos principales
    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        fig_main = px.line(
            df_f, x='Date', y='Close_Price', color='Stock_Index',
            title="Precio de Cierre — Comparativa de Índices",
            labels={'Close_Price': 'Precio USD', 'Date': 'Fecha', 'Stock_Index': 'Índice'},
            template="plotly_dark",
            color_discrete_map={'Dow Jones': '#3182ce', 'S&P 500': '#38a169', 'NASDAQ': '#e94560'}
        )
        fig_main.update_traces(line_width=1.5)
        fig_main.update_layout(legend_title_text='Índice', height=400)
        st.plotly_chart(fig_main, use_container_width=True)

    with col_g2:
        if 'Retorno_Diario' in df_f.columns:
            fig_ret = px.box(
                df_f.dropna(subset=['Retorno_Diario']),
                x='Stock_Index', y='Retorno_Diario', color='Stock_Index',
                title="Retornos Diarios (%)",
                labels={'Retorno_Diario': 'Retorno (%)', 'Stock_Index': 'Índice'},
                template="plotly_dark",
                color_discrete_map={'Dow Jones': '#3182ce', 'S&P 500': '#38a169', 'NASDAQ': '#e94560'}
            )
            fig_ret.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_ret, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Estadísticas Descriptivas")
    cols_desc = ['Close_Price', 'Trading_Volume', 'GDP_Growth', 'Inflation_Rate',
                 'Interest_Rate', 'Unemployment_Rate', 'Gold_Price', 'Crude_Oil_Price']
    cols_desc_ok = [c for c in cols_desc if c in df_f.columns]
    st.dataframe(df_f[cols_desc_ok].describe().round(3), use_container_width=True)

    st.markdown("---")
    st.subheader("💡 Hallazgos y Conclusiones Estratégicas")
    corr_inf   = df_f[['Inflation_Rate', 'Close_Price']].corr().iloc[0, 1]
    corr_tasa  = df_f[['Interest_Rate', 'Close_Price']].corr().iloc[0, 1]
    corr_ou    = df_f[['Crude_Oil_Price', 'Gold_Price']].corr().iloc[0, 1] if all(c in df_f.columns for c in ['Crude_Oil_Price', 'Gold_Price']) else 0

    with st.expander("📖 Ver análisis completo de hallazgos", expanded=True):
        st.markdown(f"""
        <div class="insight-box">
        <strong>🔑 Hallazgo 1 — Motor Macroeconómico Principal:</strong>
        La correlación tasa de interés–precio de cierre es <strong>{corr_tasa:.3f}</strong>.
        Las variaciones en política monetaria son el principal driver de los índices bursátiles,
        consistente con el canal de descuento de flujos futuros.
        </div>
        <div class="insight-box">
        <strong>🔑 Hallazgo 2 — Inflación y Valuación:</strong>
        Correlación inflación–precio de cierre: <strong>{corr_inf:.3f}</strong>.
        Periodos de inflación elevada erosionan los múltiplos de valuación (P/E) al aumentar
        la tasa de descuento, generando presión bajista sobre los índices.
        </div>
        <div class="insight-box">
        <strong>🔑 Hallazgo 3 — Refugio Seguro (Petróleo vs. Oro):</strong>
        Correlación Petróleo–Oro: <strong>{corr_ou:.3f}</strong>.
        Ambos activos co-mueven ante shocks de oferta, pero el oro actúa como refugio en
        crisis mientras el petróleo refleja la demanda de actividad económica real.
        </div>
        <div class="insight-box">
        <strong>🔑 Hallazgo 4 — Regímenes de Mercado (2000-2008):</strong>
        El clustering K-Means identifica tres regímenes: Expansión (desempleo bajo, PIB positivo),
        Estabilización (métricas mixtas) y Contracción (alto desempleo). La burbuja .com
        (2000–2002) y la antesala a la crisis subprime (2007–2008) son distinguibles en el espacio
        tridimensional macro.
        </div>
        <div class="insight-box">
        <strong>📌 Conclusión General:</strong>
        La integración de análisis técnico (RSI, Bollinger, MACD) con indicadores macro (PCA sintético)
        ofrece una perspectiva holística superior al análisis univariado. El modelo Random Forest
        captura interacciones no lineales que los modelos lineales omiten, logrando predicciones
        de alta precisión sobre datos históricos del periodo 2000–2008.
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# SECCIÓN 2: EDA
# =============================================================================
elif seccion == "📊 Análisis Visual (EDA)":
    st.markdown("""
    <div class="main-header">
      <h1>🔍 Análisis Exploratorio de Datos (EDA)</h1>
      <p>Correlaciones, distribuciones y descomposición temporal</p>
    </div>
    """, unsafe_allow_html=True)

    if n_registros == 0:
        st.warning("No hay datos para el rango seleccionado.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Correlaciones", "📦 Distribuciones", "⏳ Series Temporales", "🛢️ Commodities"])

    # ---- Tab 1: Correlaciones ----
    with tab1:
        st.subheader("Matriz de Correlación Macroeconómica")
        cols_corr = ['Close_Price', 'GDP_Growth', 'Inflation_Rate', 'Interest_Rate',
                     'Unemployment_Rate', 'Trading_Volume', 'Gold_Price', 'Crude_Oil_Price',
                     'Consumer_Confidence_Index', 'Bankruptcy_Rate', 'Real_Estate_Index',
                     'Retorno_Diario', 'Volatilidad_20d']
        cols_corr_ok = [c for c in cols_corr if c in df_f.columns]
        etiq = {
            'Close_Price':'Precio Cierre','GDP_Growth':'PIB','Inflation_Rate':'Inflación',
            'Interest_Rate':'Tasa Interés','Unemployment_Rate':'Desempleo',
            'Trading_Volume':'Volumen','Gold_Price':'Oro','Crude_Oil_Price':'Petróleo',
            'Consumer_Confidence_Index':'Conf. Consumidor','Bankruptcy_Rate':'Quiebras',
            'Real_Estate_Index':'Inmobiliario','Retorno_Diario':'Retorno %','Volatilidad_20d':'Volatilidad'
        }
        corr = df_f[cols_corr_ok].corr()
        corr.index   = [etiq.get(c, c) for c in corr.index]
        corr.columns = [etiq.get(c, c) for c in corr.columns]
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                             color_continuous_scale='RdBu_r', zmin=-1, zmax=1, height=650,
                             title="Mapa de Calor — Correlaciones Financieras y Macroeconómicas")
        fig_corr.update_layout(template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)

        st.write("**📌 Top 5 variables más correlacionadas con el Precio de Cierre:**")
        corr_close = corr['Precio Cierre'].drop('Precio Cierre').abs().sort_values(ascending=False).head(5)
        fig_bar = px.bar(x=corr_close.values, y=corr_close.index, orientation='h',
                         title="Importancia Correlacional — Precio de Cierre",
                         labels={'x': '|Correlación|', 'y': 'Variable'},
                         template="plotly_dark", color=corr_close.values,
                         color_continuous_scale='Blues')
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---- Tab 2: Distribuciones ----
    with tab2:
        st.subheader("Distribuciones y Análisis Estadístico")
        c1, c2 = st.columns(2)
        with c1:
            if 'Retorno_Diario' in df_f.columns:
                fig_h = px.histogram(df_f.dropna(subset=['Retorno_Diario']),
                                     x='Retorno_Diario', color='Stock_Index', nbins=60,
                                     title="Distribución de Retornos Diarios (%)",
                                     labels={'Retorno_Diario': 'Retorno (%)'},
                                     template="plotly_dark", barmode='overlay', opacity=0.7,
                                     color_discrete_map={'Dow Jones':'#3182ce','S&P 500':'#38a169','NASDAQ':'#e94560'})
                fig_h.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
                st.plotly_chart(fig_h, use_container_width=True)
        with c2:
            fig_v = px.violin(df_f, y='Trading_Volume', x='Stock_Index', color='Stock_Index',
                              box=True, points="outliers",
                              title="Volumen de Negociación por Índice",
                              labels={'Trading_Volume': 'Volumen', 'Stock_Index': 'Índice'},
                              template="plotly_dark",
                              color_discrete_map={'Dow Jones':'#3182ce','S&P 500':'#38a169','NASDAQ':'#e94560'})
            fig_v.update_layout(showlegend=False)
            st.plotly_chart(fig_v, use_container_width=True)

        st.subheader("Panel Macroeconómico")
        vars_m = [v for v in ['Inflation_Rate','Interest_Rate','Unemployment_Rate','GDP_Growth'] if v in df_f.columns]
        nom_m = {'Inflation_Rate':'Inflación (%)','Interest_Rate':'Tasa Interés (%)','Unemployment_Rate':'Desempleo (%)','GDP_Growth':'PIB (%)'}
        df_mm = df_f.groupby('Date')[vars_m].mean().reset_index()
        df_ml = df_mm.melt(id_vars='Date', value_vars=vars_m, var_name='Indicador', value_name='Valor')
        df_ml['Indicador'] = df_ml['Indicador'].map(nom_m)
        fig_mp = px.line(df_ml, x='Date', y='Valor', color='Indicador',
                         title="Indicadores Macroeconómicos — Promedio Histórico",
                         labels={'Valor': 'Valor (%)', 'Date': 'Fecha'}, template="plotly_dark")
        st.plotly_chart(fig_mp, use_container_width=True)

    # ---- Tab 3: Descomposición Temporal ----
    with tab3:
        st.subheader("Descomposición de Series Temporales (STL)")
        idx_ts = selected_index[0] if selected_index else indices_disponibles[0]
        df_ts_idx = df_f[df_f['Stock_Index'] == idx_ts].set_index('Date')['Close_Price']
        df_ts_w   = df_ts_idx.resample('W').mean().dropna()

        if len(df_ts_w) >= 104:
            res = seasonal_decompose(df_ts_w, model='additive', period=52)
            fig_d = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                  subplot_titles=["Serie Original","Tendencia","Estacionalidad","Residuos"],
                                  vertical_spacing=0.06)
            pares = [("Serie Original","#63b3ed",df_ts_w),("Tendencia","#f6ad55",res.trend),
                     ("Estacionalidad","#68d391",res.seasonal),("Residuos","#fc8181",res.resid)]
            for i, (nom, col, s) in enumerate(pares, 1):
                fig_d.add_trace(go.Scatter(x=s.index, y=s.values, name=nom,
                                           line=dict(color=col, width=1.5)), row=i, col=1)
            fig_d.update_layout(height=800, template="plotly_dark",
                                title_text=f"Descomposición Aditiva — {idx_ts} (Semanal)")
            st.plotly_chart(fig_d, use_container_width=True)
            st.caption("Modelo aditivo: Serie = Tendencia + Estacionalidad + Residuos. Periodo = 52 semanas.")
        else:
            st.warning(f"Se necesitan ≥104 semanas. Solo hay {len(df_ts_w)}. Amplíe el rango de fechas.")

    # ---- Tab 4: Commodities ----
    with tab4:
        st.subheader("Análisis de Commodities: Petróleo vs. Oro")
        if all(c in df_f.columns for c in ['Crude_Oil_Price', 'Gold_Price']):
            df_c = df_f.groupby('Date')[['Crude_Oil_Price','Gold_Price']].mean().reset_index()
            fig_cm = make_subplots(specs=[[{"secondary_y": True}]])
            fig_cm.add_trace(go.Scatter(x=df_c['Date'], y=df_c['Crude_Oil_Price'],
                                        name='Petróleo (USD/bbl)', line=dict(color='#f6ad55', width=1.5)),
                             secondary_y=False)
            fig_cm.add_trace(go.Scatter(x=df_c['Date'], y=df_c['Gold_Price'],
                                        name='Oro (USD/oz)', line=dict(color='#ffd700', width=1.5)),
                             secondary_y=True)
            fig_cm.update_layout(title="Evolución de Commodities", template="plotly_dark", height=400)
            fig_cm.update_yaxes(title_text="Petróleo (USD/bbl)", secondary_y=False)
            fig_cm.update_yaxes(title_text="Oro (USD/oz)", secondary_y=True)
            st.plotly_chart(fig_cm, use_container_width=True)

            fig_sc = px.scatter(df_c, x='Crude_Oil_Price', y='Gold_Price', trendline='ols',
                                title="Relación Petróleo–Oro (Scatter + Regresión OLS)",
                                labels={'Crude_Oil_Price':'Petróleo (USD/bbl)','Gold_Price':'Oro (USD/oz)'},
                                template="plotly_dark", color_discrete_sequence=['#e94560'])
            st.plotly_chart(fig_sc, use_container_width=True)


# =============================================================================
# SECCIÓN 3: ANÁLISIS TÉCNICO
# =============================================================================
elif seccion == "📈 Análisis Técnico":
    st.markdown("""
    <div class="main-header">
      <h1>📈 Análisis Técnico Avanzado</h1>
      <p>Velas japonesas OHLC, RSI, Bandas de Bollinger y MACD</p>
    </div>
    """, unsafe_allow_html=True)

    if not selected_index:
        st.warning("⚠️ Seleccione al menos un índice en el menú lateral.")
        st.stop()

    idx_sel = st.selectbox("Seleccionar índice:", selected_index)
    df_idx  = df_f[df_f['Stock_Index'] == idx_sel].copy().sort_values('Date')
    n_velas = st.slider("Sesiones a visualizar:", 20, 200, 80, 10)
    df_v    = df_idx.tail(n_velas).copy()

    if len(df_v) < 5:
        st.warning("Datos insuficientes para el rango seleccionado.")
        st.stop()

    # ---- CANDLESTICK + VOLUMEN ----
    st.subheader(f"🕯️ Velas Japonesas — {idx_sel} (Últimas {len(df_v)} sesiones)")

    fig_c = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          row_heights=[0.75, 0.25], vertical_spacing=0.03,
                          subplot_titles=[f"OHLC — {idx_sel}", "Volumen"])

    # Candlestick — usando xaxis tipo date para que las mechas se dibujen correctamente
    fig_c.add_trace(go.Candlestick(
        x=df_v['Date'],
        open=df_v['Open_Price'],
        high=df_v['Daily_High'],
        low=df_v['Daily_Low'],
        close=df_v['Close_Price'],
        name="OHLC",
        increasing=dict(line=dict(color='#26a69a', width=1), fillcolor='#26a69a'),
        decreasing=dict(line=dict(color='#ef5350', width=1), fillcolor='#ef5350'),
        whiskerwidth=0.8,
    ), row=1, col=1)

    # Medias Móviles
    for col_ma, color_ma, label_ma in [
        ('MA_20_Close', '#f6ad55', 'MA 20'),
        ('MA_50_Close', '#9f7aea', 'MA 50'),
    ]:
        if col_ma in df_v.columns:
            fig_c.add_trace(go.Scatter(x=df_v['Date'], y=df_v[col_ma],
                                       name=label_ma, line=dict(color=color_ma, width=1.2),
                                       opacity=0.9), row=1, col=1)

    # Bandas de Bollinger
    if all(c in df_v.columns for c in ['BB_sup', 'BB_inf']):
        fig_c.add_trace(go.Scatter(x=df_v['Date'], y=df_v['BB_sup'],
                                   name='BB Superior', line=dict(color='#63b3ed', width=1, dash='dot'),
                                   opacity=0.6), row=1, col=1)
        fig_c.add_trace(go.Scatter(x=df_v['Date'], y=df_v['BB_inf'],
                                   name='BB Inferior', line=dict(color='#63b3ed', width=1, dash='dot'),
                                   fill='tonexty', fillcolor='rgba(99,179,237,0.05)',
                                   opacity=0.6), row=1, col=1)

    # Volumen coloreado
    colores_vol = ['#26a69a' if c >= o else '#ef5350'
                   for c, o in zip(df_v['Close_Price'], df_v['Open_Price'])]
    fig_c.add_trace(go.Bar(x=df_v['Date'], y=df_v['Trading_Volume'],
                           name='Volumen', marker_color=colores_vol, opacity=0.7), row=2, col=1)

    # IMPORTANTE: xaxis de tipo 'date' (no 'category') para que las mechas se conecten correctamente
    fig_c.update_xaxes(type='date')
    fig_c.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig_c.update_layout(height=600, template="plotly_dark",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=60, r=20, t=60, b=40), hovermode='x unified')
    fig_c.update_yaxes(title_text="Precio (USD)", row=1, col=1)
    fig_c.update_yaxes(title_text="Volumen", row=2, col=1)
    st.plotly_chart(fig_c, use_container_width=True)
    st.caption("🟢 Verde = sesión alcista · 🔴 Rojo = sesión bajista · Las líneas verticales son las mechas (High/Low)")

    st.markdown("---")

    # ---- RSI ----
    st.subheader("📊 RSI — Índice de Fuerza Relativa (14 periodos)")
    if 'RSI_14' in df_v.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_v['Date'], y=df_v['RSI_14'],
                                     name='RSI 14', line=dict(color='#63b3ed', width=1.5)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef5350", opacity=0.7,
                          annotation_text="Sobrecompra (70)", annotation_position="top left")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#26a69a", opacity=0.7,
                          annotation_text="Sobreventa (30)", annotation_position="bottom left")
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="#ef5350", opacity=0.04, line_width=0)
        fig_rsi.add_hrect(y0=0, y1=30, fillcolor="#26a69a", opacity=0.04, line_width=0)
        fig_rsi.update_layout(height=280, template="plotly_dark", yaxis_title="RSI",
                               yaxis=dict(range=[0, 100]), xaxis_title="Fecha",
                               margin=dict(l=60, r=20, t=30, b=40))
        st.plotly_chart(fig_rsi, use_container_width=True)
        st.caption("RSI >70: sobrecompra (posible corrección). RSI <30: sobreventa (posible rebote).")

    st.markdown("---")

    # ---- MACD ----
    st.subheader("📊 MACD — Convergencia/Divergencia de Medias Móviles (12, 26, 9)")
    if all(c in df_v.columns for c in ['MACD', 'MACD_Signal', 'MACD_Hist']):
        fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  row_heights=[0.6, 0.4], vertical_spacing=0.05)
        fig_macd.add_trace(go.Scatter(x=df_v['Date'], y=df_v['MACD'],
                                       name='MACD', line=dict(color='#63b3ed', width=1.5)), row=1, col=1)
        fig_macd.add_trace(go.Scatter(x=df_v['Date'], y=df_v['MACD_Signal'],
                                       name='Señal', line=dict(color='#f6ad55', width=1.5)), row=1, col=1)
        colores_hist = ['#26a69a' if v >= 0 else '#ef5350' for v in df_v['MACD_Hist']]
        fig_macd.add_trace(go.Bar(x=df_v['Date'], y=df_v['MACD_Hist'],
                                   name='Histograma', marker_color=colores_hist), row=2, col=1)
        fig_macd.update_layout(height=400, template="plotly_dark",
                                title=f"MACD — {idx_sel}",
                                margin=dict(l=60, r=20, t=50, b=40), hovermode='x unified')
        st.plotly_chart(fig_macd, use_container_width=True)
        st.caption("Cruce MACD sobre señal → señal alcista. Cruce por debajo → señal bajista.")


# =============================================================================
# SECCIÓN 4: MACHINE LEARNING
# =============================================================================
elif seccion == "🤖 Machine Learning":
    st.markdown("""
    <div class="main-header">
      <h1>🤖 Modelado Predictivo y Segmentación</h1>
      <p>Clustering K-Means, Random Forest, Gradient Boosting y simulador de escenarios</p>
    </div>
    """, unsafe_allow_html=True)

    if n_registros < 100:
        st.warning("Se necesitan ≥100 registros. Amplíe el rango de fechas.")
        st.stop()

    # ---- 1. CLUSTERING ----
    st.header("1️⃣ Segmentación de Regímenes Económicos (K-Means)")
    st.write("Agrupación de periodos históricos según condiciones macroeconómicas dominantes.")

    ckg1, ckg2 = st.columns([3, 1])
    with ckg2:
        n_clusters = st.number_input("Número de clústeres:", min_value=2, max_value=6, value=3)

    feats_cl = [f for f in ['Inflation_Rate','Unemployment_Rate','Interest_Rate','GDP_Growth'] if f in df_f.columns]
    X_cl = df_f[feats_cl].dropna()

    if len(X_cl) > 20:
        sc_km = StandardScaler()
        km    = KMeans(n_clusters=int(n_clusters), random_state=42, n_init=10)
        km.fit(sc_km.fit_transform(X_cl))
        df_cl = df_f.loc[X_cl.index].copy()
        df_cl['Régimen'] = [f'Régimen {i}' for i in km.labels_]

        with ckg1:
            fig_3d = px.scatter_3d(
                df_cl, x='Inflation_Rate', y='Unemployment_Rate', z='Interest_Rate',
                color='Régimen', opacity=0.75,
                title="Espacio Macroeconómico 3D — Regímenes de Mercado",
                labels={'Inflation_Rate':'Inflación %','Unemployment_Rate':'Desempleo %','Interest_Rate':'Tasa Interés %'},
                template="plotly_dark", height=500
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        fig_rt = px.scatter(
            df_cl.groupby(['Date','Régimen']).first().reset_index(),
            x='Date', y='Close_Price', color='Régimen',
            title="Precio de Cierre Coloreado por Régimen Económico",
            labels={'Close_Price':'Precio (USD)','Date':'Fecha'},
            template="plotly_dark", opacity=0.6
        )
        st.plotly_chart(fig_rt, use_container_width=True)

        st.write("**Estadísticas promedio por régimen:**")
        st.dataframe(df_cl.groupby('Régimen')[feats_cl + ['Close_Price']].mean().round(3),
                     use_container_width=True)

    st.markdown("---")

    # ---- 2. REGRESIÓN SUPERVISADA ----
    st.header("2️⃣ Pronóstico del Precio de Cierre (Supervisado)")
    cm1, cm2 = st.columns([2, 1])
    with cm2:
        modelo_sel  = st.selectbox("Algoritmo:", ["Random Forest", "Gradient Boosting"])
        test_pct    = st.slider("% datos prueba:", 10, 40, 20)

    feats_reg = [f for f in ['Trading_Volume','GDP_Growth','Inflation_Rate','Interest_Rate',
                              'Unemployment_Rate','Consumer_Confidence_Index','Crude_Oil_Price',
                              'Gold_Price','Real_Estate_Index','Bankruptcy_Rate'] if f in df_f.columns]
    X_reg = df_f[feats_reg].dropna()
    y_reg = df_f.loc[X_reg.index, 'Close_Price']

    if len(X_reg) > 100:
        X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=test_pct/100,
                                                    random_state=42, shuffle=False)
        sc_ml = StandardScaler()
        X_tr_s, X_te_s = sc_ml.fit_transform(X_tr), sc_ml.transform(X_te)

        with st.spinner("🔄 Entrenando modelo..."):
            if modelo_sel == "Random Forest":
                modelo = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
            else:
                modelo = GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)
            modelo.fit(X_tr_s, y_tr)
            y_pred = modelo.predict(X_te_s)

        r2   = r2_score(y_te, y_pred)
        mae  = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))

        with cm1:
            ma, mb, mc = st.columns(3)
            ma.metric("R² (Precisión)", f"{r2:.4f}")
            mb.metric("MAE", f"${mae:.2f}")
            mc.metric("RMSE", f"${rmse:.2f}")

            df_pred = pd.DataFrame({'Fecha': df_f.loc[X_te.index,'Date'].values,
                                    'Real': y_te.values, 'Predicho': y_pred})
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=df_pred['Fecha'], y=df_pred['Real'],
                                          name='Real', line=dict(color='#63b3ed', width=1.5)))
            fig_pred.add_trace(go.Scatter(x=df_pred['Fecha'], y=df_pred['Predicho'],
                                          name='Predicción', line=dict(color='#f6ad55', width=1.5, dash='dot')))
            fig_pred.update_layout(title=f"Predicción vs. Realidad — {modelo_sel}",
                                   template="plotly_dark", height=350, hovermode='x unified')
            st.plotly_chart(fig_pred, use_container_width=True)

        if hasattr(modelo, 'feature_importances_'):
            imp = pd.Series(modelo.feature_importances_, index=feats_reg).sort_values(ascending=True)
            fig_imp = px.bar(x=imp.values, y=imp.index, orientation='h',
                             title=f"Importancia de Variables — {modelo_sel}",
                             labels={'x':'Importancia','y':'Variable'},
                             template="plotly_dark", color=imp.values, color_continuous_scale='Viridis')
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("---")

        # ---- 3. SIMULADOR ----
        st.header("3️⃣ Simulador de Escenarios Macroeconómicos")
        feats_sim = [f for f in ['Trading_Volume','GDP_Growth','Inflation_Rate','Interest_Rate',
                                  'Unemployment_Rate','Consumer_Confidence_Index'] if f in df_f.columns]
        etiq_sim = {'Trading_Volume':'Volumen','GDP_Growth':'PIB (%)','Inflation_Rate':'Inflación (%)',
                    'Interest_Rate':'Tasa Interés (%)','Unemployment_Rate':'Desempleo (%)','Consumer_Confidence_Index':'Conf. Consumidor'}
        with st.form("simulador"):
            cols_s = st.columns(len(feats_sim))
            vals_s = []
            for i, f in enumerate(feats_sim):
                with cols_s[i]:
                    vals_s.append(st.number_input(etiq_sim.get(f, f),
                                                   value=float(df_global[f].median()), format="%.2f"))
            enviado = st.form_submit_button("🤖 Calcular Predicción")
        if enviado:
            ent = [df_global[f].median() for f in feats_reg]
            for i, f in enumerate(feats_sim):
                if f in feats_reg:
                    ent[feats_reg.index(f)] = vals_s[i]
            pred = modelo.predict(sc_ml.transform([ent]))[0]
            delta = (pred - df_global['Close_Price'].median()) / df_global['Close_Price'].median() * 100
            st.success(f"### 💰 Precio Estimado: **${pred:,.2f}**")
            st.info(f"Δ vs mediana histórica: **{delta:+.2f}%**")


# =============================================================================
# SECCIÓN 5: BASE DE DATOS
# =============================================================================
elif seccion == "🗄️ Base de Datos":
    st.markdown("""
    <div class="main-header">
      <h1>🗄️ Gestión de Base de Datos SQLite</h1>
      <p>Consultas directas, exploración de esquema y exportación de datos</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("La capa de persistencia usa SQLite + SQLAlchemy (sin pyarrow). "
            "El ETL persiste automáticamente la tabla `indicadores` al iniciar la app.")

    # Estado de conexión
    try:
        with engine.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM indicadores")).fetchone()[0]
        st.success(f"✅ Conexión activa con SQLite · {total:,} registros en tabla `indicadores`")
        db_ok = True
    except Exception as e:
        st.error(f"❌ Error de conexión: {e}")
        db_ok = False

    if db_ok:
        tab_d1, tab_d2, tab_d3 = st.tabs(["📋 Vista Previa", "🔎 Consola SQL", "📥 Exportar"])

        with tab_d1:
            n_prev = st.slider("Filas a mostrar:", 10, 200, 50)
            try:
                with engine.connect() as conn:
                    filas = [dict(r) for r in conn.execute(text(f"SELECT * FROM indicadores LIMIT {n_prev}")).mappings()]
                if filas:
                    # Convertir a DataFrame y mostrar sin Arrow
                    df_prev = pd.DataFrame(filas)
                    # Forzar tipos seguros para Streamlit antiguo
                    for col in df_prev.select_dtypes(include='object').columns:
                        df_prev[col] = df_prev[col].astype(str)
                    st.dataframe(df_prev, use_container_width=True)
                else:
                    st.info("Tabla vacía.")
            except Exception as e:
                st.error(f"Error: {e}")

        with tab_d2:
            st.warning("Solo se permiten consultas SELECT.")
            q_usr = st.text_area("Consulta SQL:", height=110,
                                  value="SELECT Stock_Index, ROUND(AVG(Close_Price),2) AS precio_prom,\n"
                                        "ROUND(AVG(Inflation_Rate),3) AS inflacion\nFROM indicadores\nGROUP BY Stock_Index")
            if st.button("▶️ Ejecutar"):
                if not q_usr.strip().upper().startswith("SELECT"):
                    st.error("Solo consultas SELECT.")
                else:
                    try:
                        with engine.connect() as conn:
                            filas_q = [dict(r) for r in conn.execute(text(q_usr)).mappings()]
                        if filas_q:
                            df_q = pd.DataFrame(filas_q)
                            for col in df_q.select_dtypes(include='object').columns:
                                df_q[col] = df_q[col].astype(str)
                            st.dataframe(df_q, use_container_width=True)
                            st.caption(f"{len(df_q)} filas devueltas.")
                        else:
                            st.info("Sin resultados.")
                    except Exception as e:
                        st.error(f"Error SQL: {e}")

            with st.expander("💡 Ejemplos de consultas"):
                st.code("""-- Resumen por índice
SELECT Stock_Index,
       ROUND(AVG(Close_Price),2) AS precio_prom,
       ROUND(MIN(Close_Price),2) AS precio_min,
       ROUND(MAX(Close_Price),2) AS precio_max,
       COUNT(*) AS registros
FROM indicadores
GROUP BY Stock_Index;

-- Periodos de alta inflación (>3.5%)
SELECT Date, Stock_Index, Inflation_Rate, Close_Price
FROM indicadores WHERE Inflation_Rate > 3.5
ORDER BY Inflation_Rate DESC LIMIT 20;

-- Tendencia anual
SELECT strftime('%Y', Date) AS anio,
       ROUND(AVG(Inflation_Rate),3) AS inflacion,
       ROUND(AVG(Close_Price),2) AS cierre
FROM indicadores
GROUP BY anio ORDER BY anio;
""", language="sql")

        with tab_d3:
            c_e1, c_e2 = st.columns(2)
            with c_e1:
                st.download_button("📥 Dataset Filtrado (CSV)",
                                   data=df_f.to_csv(index=False).encode('utf-8'),
                                   file_name="g11_datos_filtrados.csv", mime="text/csv")
            with c_e2:
                resumen = df_f.select_dtypes(include='number').describe().round(4)
                st.download_button("📊 Resumen Estadístico (CSV)",
                                   data=resumen.to_csv().encode('utf-8'),
                                   file_name="g11_resumen.csv", mime="text/csv")
            st.markdown("---")
            st.write("**Esquema de la tabla `indicadores`:**")
            try:
                with engine.connect() as conn:
                    cols_db = [dict(r) for r in conn.execute(text("PRAGMA table_info(indicadores)")).mappings()]
                df_sch = pd.DataFrame(cols_db)[['name','type','notnull']]
                df_sch.columns = ['Columna','Tipo','No Nulo']
                st.dataframe(df_sch, use_container_width=True)
            except Exception as e:
                st.error(f"No se pudo obtener el esquema: {e}")