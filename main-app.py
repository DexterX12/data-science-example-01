"""
Medellín Metropolitan Area — Meteorological & Disaster-Risk Dashboard
=======================================================================
A self-contained Streamlit app for municipal risk-management decisions.

  1. Simulates a synthetic meteorological/risk dataset (1000 rows x 12
     columns, mixed data types) for Medellín's 16 comunas, 5 rural
     corregimientos, and the 9 other municipalities of the Aburrá Valley
     metro area — generated live in the app, no external file needed.
  2. Runs quantitative analysis (numeric transforms: correlations, rolling
     trends, risk scoring) and qualitative analysis (categorical
     distributions, cross-tabulations by comuna/municipio/risk level).
  3. Offers interactive, user-customizable charts (bar / line / area / box)
     built with Plotly, aimed at spotting risk hot-spots.
  4. Requires a password before any data or controls are shown.

Run with:
    streamlit run main-app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Medellín – Riesgo Meteorológico",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Password gate — must pass before anything else renders
# --------------------------------------------------------------------------
APP_PASSWORD = "8701"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    st.caption("Panel de riesgo meteorológico del Valle de Aburrá — ingrese la contraseña para continuar.")
    with st.form("login_form"):
        pwd = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Ingresar")
    if submitted:
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta. Intente de nuevo.")
    st.stop()

# --------------------------------------------------------------------------
# Reference data: Medellín's comunas/corregimientos + metro-area municipalities
# Each entry carries an approximate elevation (m), a baseline population, and
# a hazard weight (0-1) reflecting known landslide/flood susceptibility —
# used only to make the simulation behave plausibly, not as official figures.
# --------------------------------------------------------------------------
TERRITORIAL_UNITS = [
    # municipio,  comuna/corregimiento,          elevation_m, population, hazard_weight
    ("Medellín", "Comuna 1 - Popular",            1900,  130000, 0.85),
    ("Medellín", "Comuna 2 - Santa Cruz",         1850,  110000, 0.80),
    ("Medellín", "Comuna 3 - Manrique",           1800,  155000, 0.75),
    ("Medellín", "Comuna 4 - Aranjuez",           1600,  160000, 0.45),
    ("Medellín", "Comuna 5 - Castilla",           1550,  150000, 0.40),
    ("Medellín", "Comuna 6 - Doce de Octubre",    1750,  190000, 0.70),
    ("Medellín", "Comuna 7 - Robledo",            1700,  170000, 0.60),
    ("Medellín", "Comuna 8 - Villa Hermosa",      1850,  130000, 0.78),
    ("Medellín", "Comuna 9 - Buenos Aires",       1750,  150000, 0.65),
    ("Medellín", "Comuna 10 - La Candelaria",     1495,   90000, 0.55),  # flood-prone (río)
    ("Medellín", "Comuna 11 - Laureles-Estadio",  1500,  120000, 0.20),
    ("Medellín", "Comuna 12 - La América",        1550,   95000, 0.35),
    ("Medellín", "Comuna 13 - San Javier",        1800,  130000, 0.80),
    ("Medellín", "Comuna 14 - El Poblado",        1600,  140000, 0.35),
    ("Medellín", "Comuna 15 - Guayabal",          1480,   95000, 0.50),  # flood-prone (río)
    ("Medellín", "Comuna 16 - Belén",             1550,  195000, 0.40),
    ("Medellín", "Corr. San Sebastián de Palmitas", 2000,   5000, 0.55),
    ("Medellín", "Corr. San Cristóbal",           2100,   60000, 0.70),
    ("Medellín", "Corr. Altavista",               1900,   30000, 0.60),
    ("Medellín", "Corr. San Antonio de Prado",    1950,   75000, 0.55),
    ("Medellín", "Corr. Santa Elena",             2500,   20000, 0.60),
    ("Bello", "Cabecera Municipal",               1450,  480000, 0.55),  # flood-prone (río)
    ("Itagüí", "Cabecera Municipal",              1550,  275000, 0.45),
    ("Envigado", "Cabecera Municipal",            1650,  230000, 0.35),
    ("Sabaneta", "Cabecera Municipal",            1650,  130000, 0.30),
    ("La Estrella", "Cabecera Municipal",         1775,   75000, 0.45),
    ("Caldas", "Cabecera Municipal",              1750,   90000, 0.50),
    ("Copacabana", "Cabecera Municipal",          1450,   85000, 0.50),
    ("Girardota", "Cabecera Municipal",           1425,   55000, 0.45),
    ("Barbosa", "Cabecera Municipal",             1300,   50000, 0.40),
]

RISK_ORDER = ["Bajo", "Medio", "Alto", "Extremo"]


@st.cache_data(show_spinner=False)
def simulate_risk_data(n_rows: int, start_date: str, end_date: str, seed: int) -> pd.DataFrame:
    """Simulate a synthetic meteorological/risk dataset for the Aburrá Valley.

    Columns / dtypes (12 total):
        1.  date               -> datetime64
        2.  municipio           -> string
        3.  comuna              -> string
        4.  population          -> int
        5.  elevation_m         -> float
        6.  temperature_c       -> float
        7.  humidity_pct        -> float
        8.  wind_speed_kmh      -> float
        9.  precipitation_mm    -> float
        10. air_quality_index   -> int
        11. risk_level          -> ordered categorical (Bajo/Medio/Alto/Extremo)
        12. alert_active        -> bool
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    if len(dates) == 0:
        dates = pd.date_range(start=start_date, periods=1, freq="D")

    units = TERRITORIAL_UNITS
    pops = np.array([u[3] for u in units], dtype=float)
    unit_weights = pops / pops.sum()
    unit_idx = rng.choice(len(units), size=n_rows, p=unit_weights)

    municipio = np.array([units[i][0] for i in unit_idx])
    comuna = np.array([units[i][1] for i in unit_idx])
    base_pop = np.array([units[i][3] for i in unit_idx], dtype=float)
    base_elev = np.array([units[i][2] for i in unit_idx], dtype=float)
    hazard_w = np.array([units[i][4] for i in unit_idx], dtype=float)

    date_col = pd.to_datetime(rng.choice(dates, size=n_rows))
    month = date_col.month.to_numpy()

    # Antioquia's bimodal rainy season: peaks around Apr-May and Oct-Nov
    rain_season_factor = np.where(np.isin(month, [4, 5, 10, 11]), 1.6,
                          np.where(np.isin(month, [12, 1, 2]), 0.5, 1.0))

    # Population: small day-to-day estimation noise around the census baseline
    population = np.clip(base_pop + rng.normal(0, base_pop * 0.01, n_rows), 0, None).round().astype(int)

    # Elevation: small local noise around each unit's baseline
    elevation_m = np.clip(base_elev + rng.normal(0, 15, n_rows), 1200, None).round(1)

    # Temperature: standard atmospheric lapse rate applied to a warm-valley
    # baseline (~29 C at sea level), plus daily variability
    temperature_c = (29.0 - 0.0058 * elevation_m
                      + rng.normal(0, 1.6, n_rows)
                      - np.where(np.isin(month, [12, 1]), 0.6, 0)).round(1)

    # Humidity: higher during rainy months, higher at higher elevation (cloud/fog)
    humidity_pct = np.clip(
        60 + 12 * (rain_season_factor - 1) + (elevation_m - 1495) * 0.01 + rng.normal(0, 6, n_rows),
        30, 100,
    ).round(1)

    # Wind speed: generally light in the valley, stronger on higher ridgelines
    wind_speed_kmh = np.clip(
        4 + (elevation_m - 1495) * 0.006 + rng.gamma(2.0, 2.2, n_rows),
        0, None,
    ).round(1)

    # Precipitation: zero-inflated — most days are dry, rainy season brings
    # more frequent and more intense events, amplified by local hazard weight
    is_rain_day = rng.random(n_rows) < np.clip(0.25 * rain_season_factor, 0.05, 0.85)
    rain_amount = rng.gamma(shape=2.0, scale=8 * rain_season_factor * (0.7 + 0.6 * hazard_w), size=n_rows)
    precipitation_mm = np.where(is_rain_day, rain_amount, 0).round(1)

    # Air quality index: worse with low wind + high humidity (temperature
    # inversion trapping pollutants), typical of the Aburrá Valley in Feb-Mar
    inversion_boost = np.where(np.isin(month, [2, 3]), 15, 0)
    air_quality_index = np.clip(
        35 + np.clip(15 - wind_speed_kmh, 0, None) * 2.5 + inversion_boost + rng.normal(0, 12, n_rows),
        0, 300,
    ).round().astype(int)

    # Composite risk score -> bucketed risk level
    def _norm(x):
        x = np.asarray(x, dtype=float)
        rng_ = x.max() - x.min()
        return (x - x.min()) / rng_ if rng_ > 0 else np.zeros_like(x)

    risk_score = (
        0.40 * _norm(precipitation_mm)
        + 0.15 * _norm(wind_speed_kmh)
        + 0.45 * hazard_w
    )
    risk_bins = np.quantile(risk_score, [0.55, 0.80, 0.95])
    risk_level = np.select(
        [risk_score <= risk_bins[0], risk_score <= risk_bins[1], risk_score <= risk_bins[2]],
        ["Bajo", "Medio", "Alto"],
        default="Extremo",
    )

    alert_prob = np.where(np.isin(risk_level, ["Alto", "Extremo"]), 0.80, 0.04)
    alert_active = rng.random(n_rows) < alert_prob

    df = pd.DataFrame(
        {
            "date": date_col,
            "municipio": municipio,
            "comuna": comuna,
            "population": population,
            "elevation_m": elevation_m,
            "temperature_c": temperature_c,
            "humidity_pct": humidity_pct,
            "wind_speed_kmh": wind_speed_kmh,
            "precipitation_mm": precipitation_mm,
            "air_quality_index": air_quality_index,
            "risk_level": pd.Categorical(risk_level, categories=RISK_ORDER, ordered=True),
            "alert_active": alert_active,
        }
    )
    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------
# Sidebar — data generation, filters, and session controls
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Controles de datos")

with st.sidebar.expander("🔒 Sesión"):
    st.write("Autenticado correctamente.")
    if st.button("Cerrar sesión"):
        st.session_state.authenticated = False
        st.rerun()

st.sidebar.subheader("1. Simular datos")
n_rows = st.sidebar.slider("Número de filas", min_value=200, max_value=5000, value=1000, step=100)
date_range_input = st.sidebar.date_input(
    "Rango de fechas a simular",
    value=(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
)
seed = st.sidebar.number_input("Semilla aleatoria", min_value=0, max_value=99999, value=42, step=1)
regenerate = st.sidebar.button("🔄 Regenerar datos")

if "run_id" not in st.session_state:
    st.session_state.run_id = 0
if regenerate:
    st.session_state.run_id += 1

if isinstance(date_range_input, tuple) and len(date_range_input) == 2:
    start_date, end_date = date_range_input
else:
    start_date, end_date = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")

df = simulate_risk_data(
    n_rows=n_rows,
    start_date=str(start_date),
    end_date=str(end_date),
    seed=int(seed) + st.session_state.run_id,
)

st.sidebar.markdown("---")
st.sidebar.subheader("2. Filtrar datos")
all_municipios = sorted(df["municipio"].unique())
sel_municipios = st.sidebar.multiselect("Municipio", options=all_municipios, default=all_municipios)

comunas_available = sorted(df.loc[df["municipio"].isin(sel_municipios), "comuna"].unique())
sel_comunas = st.sidebar.multiselect("Comuna / corregimiento", options=comunas_available, default=comunas_available)

sel_risk = st.sidebar.multiselect("Nivel de riesgo", options=RISK_ORDER, default=RISK_ORDER)
only_alerts = st.sidebar.checkbox("Mostrar solo alertas activas", value=False)

mask = (
    df["municipio"].isin(sel_municipios)
    & df["comuna"].isin(sel_comunas)
    & df["risk_level"].isin(sel_risk)
)
if only_alerts:
    mask &= df["alert_active"]
fdf = df.loc[mask].copy()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🌦️ Panel de Riesgo Meteorológico — Valle de Aburrá")
st.caption(
    "Datos meteorológicos y de riesgo simulados en tiempo real dentro de la aplicación "
    "(no se carga ningún archivo externo), a nivel de comuna/corregimiento y municipio, "
    "pensados como insumo para decisiones de gestión del riesgo de desastres."
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Registros (filtrados)", f"{len(fdf):,}")
k2.metric("Temp. promedio", f"{fdf['temperature_c'].mean():.1f} °C" if len(fdf) else "—")
k3.metric("Precipitación total", f"{fdf['precipitation_mm'].sum():,.0f} mm" if len(fdf) else "—")
k4.metric("Alertas activas", f"{int(fdf['alert_active'].sum()):,}" if len(fdf) else "—")
k5.metric("Riesgo Alto/Extremo", f"{(fdf['risk_level'].isin(['Alto','Extremo'])).sum():,}" if len(fdf) else "—")

st.markdown("---")

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_data, tab_quant, tab_qual, tab_custom = st.tabs(
    ["📄 Datos crudos", "🔢 Análisis cuantitativo", "🗂️ Análisis cualitativo", "🎛️ Gráfico interactivo"]
)

# ---- Tab 1: Raw data ------------------------------------------------------
with tab_data:
    st.subheader("Vista previa del conjunto de datos simulado")
    st.dataframe(fdf.head(200), use_container_width=True)
    st.caption(f"Mostrando las primeras 200 de {len(fdf):,} filas filtradas (12 columnas, tipos mixtos).")
    st.write("**Tipos de datos por columna:**")
    st.dataframe(fdf.dtypes.astype(str).rename("dtype"), use_container_width=True)

    csv = fdf.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar datos filtrados (CSV)", data=csv, file_name="medellin_riesgo_meteorologico.csv")

# ---- Tab 2: Quantitative analysis ----------------------------------------
with tab_quant:
    st.subheader("Análisis cuantitativo (variables numéricas)")

    if fdf.empty:
        st.warning("No hay filas que coincidan con los filtros actuales.")
    else:
        st.write("**Estadísticas descriptivas**")
        num_cols = ["temperature_c", "humidity_pct", "wind_speed_kmh", "precipitation_mm",
                    "air_quality_index", "elevation_m", "population"]
        st.dataframe(fdf[num_cols].describe().T, use_container_width=True)

        st.write("**Matriz de correlación**")
        corr = fdf[num_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Correlación entre variables numéricas",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.write("**Tendencia temporal con promedio móvil**")
        window = st.slider("Ventana de promedio móvil (días)", 1, 30, 7, key="quant_window")
        metric = st.selectbox(
            "Métrica", ["temperature_c", "humidity_pct", "wind_speed_kmh", "precipitation_mm", "air_quality_index"],
            key="quant_metric",
        )

        daily = fdf.groupby("date", as_index=False)[metric].mean().sort_values("date")
        daily["promedio_movil"] = daily[metric].rolling(window, min_periods=1).mean()

        fig_trend = px.line(
            daily, x="date", y=[metric, "promedio_movil"],
            labels={"value": metric, "date": "Fecha", "variable": "Serie"},
            title=f"{metric.replace('_', ' ').title()} en el tiempo (valor diario vs. móvil de {window} días)",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        st.write("**Riesgo promedio vs. población — comunas con mayor exposición**")
        exposure = (
            fdf.groupby("comuna", as_index=False)
            .agg(poblacion=("population", "mean"), riesgo_alto_extremo=("risk_level", lambda s: s.isin(["Alto", "Extremo"]).mean()))
            .sort_values("riesgo_alto_extremo", ascending=False)
            .head(15)
        )
        fig_exp = px.scatter(
            exposure, x="poblacion", y="riesgo_alto_extremo", text="comuna", size="poblacion",
            title="Población vs. proporción de registros en riesgo Alto/Extremo (top 15 comunas)",
            labels={"poblacion": "Población", "riesgo_alto_extremo": "Proporción riesgo Alto/Extremo"},
        )
        fig_exp.update_traces(textposition="top center")
        st.plotly_chart(fig_exp, use_container_width=True)

# ---- Tab 3: Qualitative analysis ------------------------------------------
with tab_qual:
    st.subheader("Análisis cualitativo (variables categóricas)")

    if fdf.empty:
        st.warning("No hay filas que coincidan con los filtros actuales.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            cat_col = st.selectbox(
                "Variable categórica", ["municipio", "comuna", "risk_level", "alert_active"], key="qual_cat"
            )
            counts = fdf[cat_col].astype(str).value_counts().reset_index()
            counts.columns = [cat_col, "conteo"]
            fig_bar = px.bar(counts, x=cat_col, y="conteo", color=cat_col, title=f"Distribución de {cat_col}")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            fig_pie = px.pie(counts, names=cat_col, values="conteo", title=f"Participación de {cat_col}", hole=0.35)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.write("**Tabla cruzada (mapa de calor) entre dos variables categóricas**")
        col_c, col_d = st.columns(2)
        cross_x = col_c.selectbox("Variable X", ["municipio", "risk_level", "alert_active"], index=0, key="cx")
        cross_y = col_d.selectbox("Variable Y", ["municipio", "risk_level", "alert_active"], index=1, key="cy")

        if cross_x == cross_y:
            st.info("Elija dos variables diferentes para construir la tabla cruzada.")
        else:
            ctab = pd.crosstab(fdf[cross_x].astype(str), fdf[cross_y].astype(str))
            fig_ctab = px.imshow(ctab, text_auto=True, aspect="auto", title=f"{cross_x} vs {cross_y} (conteo de filas)")
            st.plotly_chart(fig_ctab, use_container_width=True)

# ---- Tab 4: Fully custom interactive chart --------------------------------
with tab_custom:
    st.subheader("Construya su propio gráfico")
    st.caption("Elija dimensiones, métrica, agregación y tipo de gráfico — el gráfico se actualiza en vivo.")

    if fdf.empty:
        st.warning("No hay filas que coincidan con los filtros actuales.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        group_by = c1.selectbox("Agrupar por (eje X)", ["municipio", "comuna", "risk_level", "date"])
        color_by = c2.selectbox(
            "Color / separar por",
            ["Ninguno", "municipio", "risk_level", "alert_active"],
            index=0,
        )
        y_metric = c3.selectbox(
            "Métrica (eje Y)",
            ["temperature_c", "humidity_pct", "wind_speed_kmh", "precipitation_mm", "air_quality_index", "population"],
        )
        agg_func = c4.selectbox("Agregación", ["mean", "sum", "median", "max", "count"])

        c5, c6, c7 = st.columns(3)
        chart_type = c5.radio("Tipo de gráfico", ["Barras", "Línea", "Área", "Caja"], horizontal=True)
        top_n = c6.slider("Top N categorías (por métrica)", 3, 30, 15)
        sort_desc = c7.checkbox("Ordenar descendente", value=True)

        group_cols = [group_by] if color_by == "Ninguno" else [group_by, color_by]
        grouped = fdf.groupby(group_cols, as_index=False, observed=True)[y_metric].agg(agg_func)

        if group_by != "date":
            totals = grouped.groupby(group_by, observed=True)[y_metric].sum().sort_values(ascending=not sort_desc)
            keep = totals.head(top_n).index
            grouped = grouped[grouped[group_by].isin(keep)]
            grouped = grouped.sort_values(y_metric, ascending=not sort_desc)

        color_arg = None if color_by == "Ninguno" else color_by
        title = f"{agg_func.title()} de {y_metric.replace('_', ' ')} por {group_by}"

        if chart_type == "Barras":
            fig = px.bar(grouped, x=group_by, y=y_metric, color=color_arg, barmode="group", title=title)
        elif chart_type == "Línea":
            fig = px.line(grouped, x=group_by, y=y_metric, color=color_arg, markers=True, title=title)
        elif chart_type == "Área":
            fig = px.area(grouped, x=group_by, y=y_metric, color=color_arg, title=title)
        else:  # Box plot uses the raw (non-aggregated) filtered data for real distributions
            fig = px.box(fdf, x=group_by, y=y_metric, color=color_arg,
                         title=f"Distribución de {y_metric} por {group_by}")

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Ver tabla agregada subyacente"):
            st.dataframe(grouped, use_container_width=True)

st.markdown("---")
st.caption(
    "Construido con Streamlit + Plotly. Todos los datos son 100% sintéticos y se generan en tiempo de "
    "ejecución con fines demostrativos; no representan mediciones oficiales del AMVA, el IDEAM o la "
    "Alcaldía de Medellín."
)
