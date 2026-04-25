## TrabajoG11-C3
Ejercicio Práctico Grupo 11 - Tratamiento de Datos

# 📈 Dashboard Inteligencia Financiera y Macroeconómica Avanzada

Dashboard interactivo de análisis financiero y macroeconómico desarrollado con **Streamlit**, que integra un pipeline ETL completo, análisis exploratorio, indicadores técnicos avanzados y modelos de Machine Learning.

---

## 📋 Descripción General

Este proyecto transforma datos históricos de mercados bursátiles (Dow Jones, S&P 500, NASDAQ) junto con variables macroeconómicas globales (2000–2008) en inteligencia accionable a través de:

- **Pipeline ETL automatizado** con limpieza, imputación y feature engineering
- **Análisis Exploratorio (EDA)** con correlaciones, distribuciones y descomposición estacional
- **Análisis Técnico** con velas japonesas OHLC, RSI, Bandas de Bollinger y MACD
- **Machine Learning** supervisado (Random Forest, Gradient Boosting) y no supervisado (K-Means)
- **Persistencia SQL** en SQLite con consola de consultas integrada

---

## 🗂️ Estructura del Proyecto

```
TrabajoG11-C3/
├── app.py                          # Aplicación principal Streamlit
├── Data/
│   └── finance_economics_dataset.csv  # Dataset principal (3,000 registros, 24 variables)
├── economics_all.db               # Base de datos SQLite (generada en runtime)
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Este archivo
```

---

## 📦 Dataset

**Fuente:** `Data/finance_economics_dataset.csv`  
**Registros:** 3,000 | **Variables:** 24 | **Periodo:** 2000-01-01 → 2008-03-18  
**Índices:** Dow Jones, S&P 500, NASDAQ

### Variables disponibles

| Categoría | Variables |
|-----------|-----------|
| **Precio** | Open Price, Close Price, Daily High, Daily Low |
| **Mercado** | Trading Volume, Stock Index |
| **Macro** | GDP Growth, Inflation Rate, Unemployment Rate, Interest Rate |
| **Consumo** | Consumer Confidence Index, Retail Sales, Consumer Spending |
| **Deuda/Corp** | Government Debt, Corporate Profits, Bankruptcy Rate |
| **M&A / VC** | Mergers & Acquisitions Deals, Venture Capital Funding |
| **Forex** | USD/EUR, USD/JPY |
| **Commodities** | Crude Oil Price, Gold Price, Real Estate Index |

---

## 🔄 Pipeline ETL

El módulo `cargar_y_procesar()` ejecuta los siguientes pasos en secuencia:

1. **Lectura CSV** desde `Data/` con fallback automático
2. **Normalización de columnas** (snake_case, sin caracteres especiales)
3. **Conversión de tipos** (`datetime64[ns]`, `float64`)
4. **Imputación temporal** (`ffill` + `bfill`) para preservar continuidad de series
5. **Winsorización IQR** (límites 1.5×IQR) para tratamiento de valores extremos
6. **Feature Engineering por índice:**
   - Medias Móviles: MA(20), MA(50), MA(200)
   - RSI de 14 periodos
   - Bandas de Bollinger (20 periodos, ±2σ)
   - MACD (12, 26, 9)
   - Retorno diario (%) y Volatilidad rodante 20 días
   - Crecimiento YoY del PIB (252 sesiones)
7. **PCA macroeconómico**: índice sintético de 1 componente sobre {Inflación, Tasa, Desempleo}
8. **Persistencia SQLite** en tabla `indicadores` vía SQLAlchemy

---

## 🖥️ Secciones del Dashboard

### 🏠 Inicio y Resumen
- KPIs dinámicos (precio promedio, máximo, volatilidad, inflación, PIB)
- Gráfico comparativo de índices
- Distribución de retornos
- Estadísticas descriptivas
- Hallazgos y conclusiones estratégicas basados en correlaciones calculadas en tiempo real

### 📊 Análisis Visual (EDA)
- **Correlaciones:** heatmap interactivo de todas las variables + top 5 por relevancia
- **Distribuciones:** histograma de retornos superpuesto, violin plot de volumen
- **Series Temporales:** descomposición STL aditiva (tendencia + estacionalidad + residuos)
- **Commodities:** evolución Petróleo/Oro con eje dual + scatter de regresión OLS

### 📈 Análisis Técnico
- **Candlestick OHLC** con volumen coloreado, MA(20) y MA(50), Bandas de Bollinger
  > *Corrección: `xaxis type='date'` garantiza las mechas (High/Low) correctamente renderizadas*
- **RSI (14p)** con zonas de sobrecompra/sobreventa visualizadas
- **MACD (12,26,9)** con línea de señal e histograma bicolor

### 🤖 Machine Learning
- **K-Means** clustering 3D de regímenes económicos (configurable 2–6 clústeres)
- **Random Forest / Gradient Boosting** con train/test split configurable
  - Métricas: R², MAE, RMSE
  - Gráfico predicción vs. realidad en serie temporal
  - Importancia de características (feature importance)
- **Simulador de escenarios**: entrada interactiva de variables macro → predicción de precio

### 🗄️ Base de Datos
- Vista previa de tabla `indicadores` (SQLite, sin pyarrow)
- Consola SQL con validación de seguridad (solo SELECT)
- Ejemplos de consultas predefinidas
- Exportación CSV del dataset filtrado y del resumen estadístico
- Visualización del esquema PRAGMA de la tabla

---

## ⚙️ Instalación y Ejecución

### Requisitos
- Python 3.8+
- Las dependencias listadas en `requirements.txt`

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación abrirá automáticamente en `http://localhost:8501`

---

## 📚 Dependencias Principales

```
streamlit>=1.28
pandas>=1.5
numpy>=1.23
plotly>=5.15
scikit-learn>=1.2
statsmodels>=0.14
sqlalchemy>=2.0
```

> **Nota:** No se requiere `pyarrow`. La visualización de datos en la sección de Base de Datos
> utiliza conversión explícita a `dict` + `DataFrame` para compatibilidad con versiones antiguas
> de Streamlit que no soportan Apache Arrow.

---

## 🐛 Correcciones Aplicadas en esta Versión

| Problema | Solución |
|----------|----------|
| Velas sin mechas (High/Low) | `xaxis type='date'` en lugar de `'category'` + uso de `go.Candlestick` con `whiskerwidth=0.8` |
| `No module named 'pyarrow'` | Eliminada dependencia de Arrow; datos servidos como `list[dict]` → `pd.DataFrame` |
| Ícono/imagen de sidebar mostrando "0" | Reemplazado por SVG embebido (sin URL externa) |
| Análisis limitado | Agregados: análisis técnico completo, descomposición STL, commodities, MACD, Gradient Boosting, simulador mejorado |

---

## 💡 Hallazgos Principales

1. **Motor macroeconómico:** La tasa de interés es el indicador con mayor correlación con los precios bursátiles en el periodo 2000–2008.
2. **Inflación y valuación:** Periodos de inflación >4% preceden correcciones en los índices al elevar las tasas de descuento.
3. **Refugio seguro:** El oro y el petróleo co-mueven en shocks de oferta pero divergen en crisis de demanda.
4. **Regímenes de mercado:** K-Means distingue claramente la burbuja .com (2000–2002) del periodo de expansión (2003–2006) y la pre-crisis subprime (2007–2008).
5. **Precisión predictiva:** Random Forest con 10 variables alcanza R² > 0.95 sobre datos de prueba, con el volumen de negociación y el oro como los predictores más importantes.

---

## 👥 Autores

**Grupo 11** — Proyecto de Análisis de Datos Financieros y Macroeconómicos