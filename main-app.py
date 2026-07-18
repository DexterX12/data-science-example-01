"""
COVID-19 Simulated Data Dashboard
==================================
A self-contained Streamlit app that:
  1. Simulates a synthetic COVID-19 dataset (5000 rows x 10 columns, mixed
     data types) directly in-platform, controllable by the user.
  2. Runs quantitative analysis (numeric transforms: rolling averages,
     correlations, aggregations) and qualitative analysis (categorical
     distributions, cross-tabulations).
  3. Offers interactive, user-customizable charts (bar / line / area / box)
     built with Plotly.

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
    page_title="COVID-19 Simulated Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Data simulation
# --------------------------------------------------------------------------
COUNTRIES = ["USA", "Brazil", "India", "Colombia", "Germany", "South Africa", "Japan"]
AGE_GROUPS = ["0-17", "18-29", "30-44", "45-59", "60-74", "75+"]
GENDERS = ["Female", "Male", "Other"]
VARIANTS = ["Alpha", "Delta", "Omicron", "Original", "Other"]


@st.cache_data(show_spinner=False)
def simulate_covid_data(n_rows: int, start_date: str, end_date: str, seed: int) -> pd.DataFrame:
    """Simulate a synthetic COVID-19 dataset with 10 columns of mixed types.

    Columns / dtypes:
        1. date                 -> datetime64
        2. country              -> categorical (string)
        3. region_code          -> string (short code)
        4. age_group            -> ordered categorical
        5. gender               -> categorical (string)
        6. variant              -> categorical (string)
        7. daily_cases          -> int
        8. daily_deaths         -> int
        9. vaccination_rate     -> float (0-100 %)
        10. hospitalized        -> bool
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    if len(dates) == 0:
        dates = pd.date_range(start=start_date, periods=1, freq="D")

    date_col = rng.choice(dates, size=n_rows)
    country_col = rng.choice(COUNTRIES, size=n_rows, p=_weighted(COUNTRIES, rng))
    region_col = ["R" + str(x).zfill(2) for x in rng.integers(1, 25, size=n_rows)]
    age_col = rng.choice(AGE_GROUPS, size=n_rows, p=_weighted(AGE_GROUPS, rng, skew=True))
    gender_col = rng.choice(GENDERS, size=n_rows, p=[0.49, 0.49, 0.02])
    variant_col = rng.choice(VARIANTS, size=n_rows, p=[0.15, 0.25, 0.35, 0.10, 0.15])

    # Numeric columns with some correlation structure baked in for realism:
    base_intensity = rng.gamma(shape=2.0, scale=40, size=n_rows)
    daily_cases = np.clip(base_intensity + rng.normal(0, 15, n_rows), 0, None).round().astype(int)
    death_rate_noise = rng.normal(0.02, 0.01, n_rows).clip(0.001, 0.08)
    daily_deaths = np.clip(daily_cases * death_rate_noise, 0, None).round().astype(int)

    vaccination_rate = np.clip(rng.beta(5, 2, n_rows) * 100, 0, 100).round(2)
    hospitalized = rng.random(n_rows) < np.clip(0.05 + death_rate_noise * 2, 0, 0.4)

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(date_col),
            "country": country_col,
            "region_code": region_col,
            "age_group": pd.Categorical(age_col, categories=AGE_GROUPS, ordered=True),
            "gender": gender_col,
            "variant": variant_col,
            "daily_cases": daily_cases,
            "daily_deaths": daily_deaths,
            "vaccination_rate": vaccination_rate,
            "hospitalized": hospitalized,
        }
    )
    return df.sort_values("date").reset_index(drop=True)


def _weighted(categories, rng, skew: bool = False):
    """Generate a plausible (non-uniform) probability vector for a category list."""
    n = len(categories)
    if skew:
        weights = np.linspace(1, 3, n)
    else:
        weights = rng.uniform(0.7, 1.3, n)
    return weights / weights.sum()


# --------------------------------------------------------------------------
# Sidebar — data generation & filter controls
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Data Controls")

st.sidebar.subheader("1. Simulate dataset")
n_rows = st.sidebar.slider("Number of rows", min_value=1000, max_value=20000, value=5000, step=500)
date_range_input = st.sidebar.date_input(
    "Date range to simulate over",
    value=(pd.Timestamp("2020-03-01"), pd.Timestamp("2023-06-30")),
)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
regenerate = st.sidebar.button("🔄 Regenerate data")

# Ensure a stable seed is stored so the button forces a fresh cache key
if "run_id" not in st.session_state:
    st.session_state.run_id = 0
if regenerate:
    st.session_state.run_id += 1

if isinstance(date_range_input, tuple) and len(date_range_input) == 2:
    start_date, end_date = date_range_input
else:
    start_date, end_date = pd.Timestamp("2020-03-01"), pd.Timestamp("2023-06-30")

df = simulate_covid_data(
    n_rows=n_rows,
    start_date=str(start_date),
    end_date=str(end_date),
    seed=int(seed) + st.session_state.run_id,
)

st.sidebar.markdown("---")
st.sidebar.subheader("2. Filter data")
sel_countries = st.sidebar.multiselect("Country", options=COUNTRIES, default=COUNTRIES)
sel_age = st.sidebar.multiselect("Age group", options=AGE_GROUPS, default=AGE_GROUPS)
sel_gender = st.sidebar.multiselect("Gender", options=GENDERS, default=GENDERS)
sel_variant = st.sidebar.multiselect("Variant", options=VARIANTS, default=VARIANTS)

mask = (
    df["country"].isin(sel_countries)
    & df["age_group"].isin(sel_age)
    & df["gender"].isin(sel_gender)
    & df["variant"].isin(sel_variant)
)
fdf = df.loc[mask].copy()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🦠 COVID-19 Simulated Data Dashboard")
st.caption(
    "All data on this page is synthetically generated in real time inside the app — "
    "no external file is loaded. Use the sidebar to resimulate, filter, and explore."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows (filtered)", f"{len(fdf):,}")
k2.metric("Total cases", f"{int(fdf['daily_cases'].sum()):,}")
k3.metric("Total deaths", f"{int(fdf['daily_deaths'].sum()):,}")
k4.metric("Avg. vaccination rate", f"{fdf['vaccination_rate'].mean():.1f}%" if len(fdf) else "—")

st.markdown("---")

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_data, tab_quant, tab_qual, tab_custom = st.tabs(
    ["📄 Raw Data", "🔢 Quantitative Analysis", "🗂️ Qualitative Analysis", "🎛️ Custom Interactive Chart"]
)

# ---- Tab 1: Raw data ------------------------------------------------------
with tab_data:
    st.subheader("Simulated dataset preview")
    st.dataframe(fdf.head(200), use_container_width=True)
    st.caption(f"Showing first 200 of {len(fdf):,} filtered rows (10 columns, mixed dtypes).")
    st.write("**Column dtypes:**")
    st.dataframe(fdf.dtypes.astype(str).rename("dtype"), use_container_width=True)

    csv = fdf.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered data as CSV", data=csv, file_name="simulated_covid_data.csv")

# ---- Tab 2: Quantitative analysis ----------------------------------------
with tab_quant:
    st.subheader("Quantitative (numeric) analysis")

    if fdf.empty:
        st.warning("No rows match the current filters.")
    else:
        st.write("**Descriptive statistics**")
        st.dataframe(fdf[["daily_cases", "daily_deaths", "vaccination_rate"]].describe().T, use_container_width=True)

        st.write("**Correlation matrix**")
        corr_cols = ["daily_cases", "daily_deaths", "vaccination_rate"]
        corr = fdf[corr_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Correlation between numeric variables",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.write("**Time-series trend with rolling average**")
        window = st.slider("Rolling average window (days)", 1, 30, 7, key="quant_window")
        metric = st.selectbox("Metric", ["daily_cases", "daily_deaths", "vaccination_rate"], key="quant_metric")

        daily = fdf.groupby("date", as_index=False)[metric].sum().sort_values("date")
        daily["rolling_avg"] = daily[metric].rolling(window, min_periods=1).mean()

        fig_trend = px.line(
            daily, x="date", y=[metric, "rolling_avg"],
            labels={"value": metric, "date": "Date", "variable": "Series"},
            title=f"{metric.replace('_', ' ').title()} over time (raw vs {window}-day rolling avg)",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

# ---- Tab 3: Qualitative analysis ------------------------------------------
with tab_qual:
    st.subheader("Qualitative (categorical) analysis")

    if fdf.empty:
        st.warning("No rows match the current filters.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            cat_col = st.selectbox(
                "Categorical variable", ["country", "age_group", "gender", "variant"], key="qual_cat"
            )
            counts = fdf[cat_col].value_counts().reset_index()
            counts.columns = [cat_col, "count"]
            fig_bar = px.bar(counts, x=cat_col, y="count", color=cat_col, title=f"Distribution of {cat_col}")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            fig_pie = px.pie(counts, names=cat_col, values="count", title=f"Share of {cat_col}", hole=0.35)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.write("**Cross-tabulation (heatmap) between two categorical variables**")
        col_c, col_d = st.columns(2)
        cross_x = col_c.selectbox("Variable X", ["country", "age_group", "gender", "variant"], index=0, key="cx")
        cross_y = col_d.selectbox("Variable Y", ["country", "age_group", "gender", "variant"], index=3, key="cy")

        if cross_x == cross_y:
            st.info("Choose two different variables to build a cross-tabulation.")
        else:
            ctab = pd.crosstab(fdf[cross_x], fdf[cross_y])
            fig_ctab = px.imshow(ctab, text_auto=True, aspect="auto", title=f"{cross_x} vs {cross_y} (row counts)")
            st.plotly_chart(fig_ctab, use_container_width=True)

# ---- Tab 4: Fully custom interactive chart --------------------------------
with tab_custom:
    st.subheader("Build your own chart")
    st.caption("Pick the dimensions, metric, aggregation, and chart type — the plot updates live.")

    if fdf.empty:
        st.warning("No rows match the current filters.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        group_by = c1.selectbox("Group by (X-axis)", ["country", "age_group", "gender", "variant", "date"])
        color_by = c2.selectbox(
            "Color / split by",
            ["None", "country", "age_group", "gender", "variant"],
            index=0,
        )
        y_metric = c3.selectbox("Metric (Y-axis)", ["daily_cases", "daily_deaths", "vaccination_rate"])
        agg_func = c4.selectbox("Aggregation", ["sum", "mean", "median", "max", "count"])

        c5, c6, c7 = st.columns(3)
        chart_type = c5.radio("Chart type", ["Bar", "Line", "Area", "Box"], horizontal=True)
        top_n = c6.slider("Top N categories (by metric)", 3, len(COUNTRIES) * 2, 10)
        sort_desc = c7.checkbox("Sort descending", value=True)

        group_cols = [group_by] if color_by == "None" else [group_by, color_by]
        grouped = fdf.groupby(group_cols, as_index=False)[y_metric].agg(agg_func)

        if group_by != "date":
            # Limit to top N groups by total metric value for readability
            totals = grouped.groupby(group_by)[y_metric].sum().sort_values(ascending=not sort_desc)
            keep = totals.head(top_n).index
            grouped = grouped[grouped[group_by].isin(keep)]
            grouped = grouped.sort_values(y_metric, ascending=not sort_desc)

        color_arg = None if color_by == "None" else color_by
        title = f"{agg_func.title()} of {y_metric.replace('_', ' ')} by {group_by}"

        if chart_type == "Bar":
            fig = px.bar(grouped, x=group_by, y=y_metric, color=color_arg, barmode="group", title=title)
        elif chart_type == "Line":
            fig = px.line(grouped, x=group_by, y=y_metric, color=color_arg, markers=True, title=title)
        elif chart_type == "Area":
            fig = px.area(grouped, x=group_by, y=y_metric, color=color_arg, title=title)
        else:  # Box plot uses the raw (non-aggregated) filtered data for real distributions
            fig = px.box(fdf, x=group_by, y=y_metric, color=color_arg, title=f"Distribution of {y_metric} by {group_by}")

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("See underlying aggregated table"):
            st.dataframe(grouped, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit + Plotly. Data is 100% synthetic and generated at runtime for demonstration purposes.")
