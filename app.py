# app.py
# Global Statistics Explorer (CSV / XLSX only)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# Config
# --------------------------------------------------

st.set_page_config(
    page_title="Global Stats Explorer",
    layout="wide",
)

DATA_ROOT = Path("data")

CATEGORIES = {
    "Birth & Death Rates": DATA_ROOT / "WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT",
    "Wealth Distribution": DATA_ROOT / "wealth",
    "Education & Indices": DATA_ROOT / "HDR25_Statistical_Annex_HDI_Table",
    "Crime Rates": DATA_ROOT / "crime",
    "Immigration & Migration": DATA_ROOT / "undesa_pd_2024_ims_stock_by_sex_destination_and_origin",
    "Authoritarianism / Regime": DATA_ROOT / "regime",
}

# --------------------------------------------------
# Utilities
# --------------------------------------------------

@st.cache_data
def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError("Unsupported file type")


def dataset_profile(df: pd.DataFrame):
    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Numeric columns": df.select_dtypes(include=np.number).columns.tolist(),
        "Categorical columns": df.select_dtypes(exclude=np.number).columns.tolist(),
        "Missing values": int(df.isna().sum().sum()),
    }


# --------------------------------------------------
# Sidebar – Dataset selection
# --------------------------------------------------

st.sidebar.title("📂 Dataset Library")

category = st.sidebar.selectbox(
    "Select category",
    list(CATEGORIES.keys())
)

folder = CATEGORIES[category]
files = sorted([f for f in folder.glob("*") if f.suffix in [".csv", ".xlsx"]])

uploaded_file = st.sidebar.file_uploader(
    "Or upload your own CSV / XLSX",
    type=["csv", "xlsx"]
)

dataset_name = None
df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    dataset_name = uploaded_file.name
elif files:
    dataset_name = st.sidebar.selectbox(
        "Select dataset",
        files,
        format_func=lambda x: x.name
    )
    df = load_dataset(dataset_name)

# --------------------------------------------------
# Main layout
# --------------------------------------------------

st.title("🌍 Global Statistics Explorer")

if df is None:
    st.info("Please select or upload a dataset.")
    st.stop()

# --------------------------------------------------
# Section 1: Dataset Overview
# --------------------------------------------------

st.header("1️⃣ Dataset Overview")

profile = dataset_profile(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", profile["Rows"])
col2.metric("Columns", profile["Columns"])
col3.metric("Numeric cols", len(profile["Numeric columns"]))
col4.metric("Missing values", profile["Missing values"])

with st.expander("Preview data"):
    st.dataframe(df.head(50), use_container_width=True)

# --------------------------------------------------
# Section 2: Analyst Graph Area (guided)
# --------------------------------------------------

st.header("2️⃣ Curated Graph Area")

num_cols = profile["Numeric columns"]
cat_cols = profile["Categorical columns"]

if num_cols:
    col_a, col_b, col_c = st.columns(3)

    x_axis = col_a.selectbox("X-axis", options=df.columns)
    y_axis = col_b.selectbox("Y-axis", options=num_cols)
    color_by = col_c.selectbox("Color (optional)", options=[None] + cat_cols)

    chart_type = st.selectbox(
        "Chart type",
        ["Line", "Bar", "Scatter", "Choropleth (ISO-3)"]
    )

    fig = None

    if chart_type == "Line":
        fig = px.line(df, x=x_axis, y=y_axis, color=color_by)
    elif chart_type == "Bar":
        fig = px.bar(df, x=x_axis, y=y_axis, color=color_by)
    elif chart_type == "Scatter":
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by)
    elif chart_type == "Choropleth (ISO-3)":
        if "iso3" not in df.columns:
            st.warning("Dataset must contain an 'iso3' column.")
        else:
            fig = px.choropleth(
                df,
                locations="iso3",
                color=y_axis,
                hover_name=color_by if color_by else None,
            )

    if fig:
        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No numeric columns available for charting.")

# --------------------------------------------------
# Section 3: Exploratory Zone (client-built)
# --------------------------------------------------

st.header("3️⃣ Exploratory Zone (Build Your Own)")

st.markdown("""
This area allows **free-form exploration**.
Clients can:
- Pick any columns
- Filter data
- Create multiple charts
""")

with st.expander("🔍 Filter data"):
    filters = {}
    for col in cat_cols:
        values = st.multiselect(
            f"Filter {col}",
            options=df[col].dropna().unique()
        )
        if values:
            filters[col] = values

df_filtered = df.copy()
for col, vals in filters.items():
    df_filtered = df_filtered[df_filtered[col].isin(vals)]

st.subheader("Custom Chart Builder")

x = st.selectbox("X", options=df_filtered.columns, key="explore_x")
y = st.selectbox("Y", options=num_cols, key="explore_y")
color = st.selectbox("Color", options=[None] + cat_cols, key="explore_color")
size = st.selectbox("Size", options=[None] + num_cols, key="explore_size")

chart_kind = st.radio(
    "Chart",
    ["Scatter", "Line", "Bar"],
    horizontal=True
)

if chart_kind == "Scatter":
    fig2 = px.scatter(df_filtered, x=x, y=y, color=color, size=size)
elif chart_kind == "Line":
    fig2 = px.line(df_filtered, x=x, y=y, color=color)
else:
    fig2 = px.bar(df_filtered, x=x, y=y, color=color)

st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.caption("© Global Stats Explorer – CSV/XLSX-only analytical framework")
