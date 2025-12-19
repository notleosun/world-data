# =========================================================
# Global Stats Explorer — page routing like cov19st.py
# (CSV / XLSX ONLY)
# =========================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu

# ----------------------------
# Config
# ----------------------------
st.set_page_config(layout="wide", page_title="Global Stats Explorer")

DATA_ROOT = Path("data")
CATEGORIES: Dict[str, Path] = {
    "Birth rate & Death rate": DATA_ROOT / "demographics",
    "Wealth distribution": DATA_ROOT / "wealth",
    "Education & indices": DATA_ROOT / "education",
    "Crime rates": DATA_ROOT / "crime",
    "Immigration / migration": DATA_ROOT / "migration",
    "Authoritarianism / regime": DATA_ROOT / "regime",
}

SUPPORTED_EXTS = {".csv", ".xlsx", ".xls"}


# =========================================================
# Helpers — CSV/XLSX only
# =========================================================
@st.cache_data(show_spinner=False)
def load_file(path: str) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(p)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(p)
    raise ValueError("Unsupported file type (CSV/XLSX only).")


@st.cache_data(show_spinner=False)
def load_upload_bytes(filename: str, b: bytes) -> pd.DataFrame:
    suf = Path(filename).suffix.lower()
    bio = pd.io.common.BytesIO(b)
    if suf == ".csv":
        return pd.read_csv(bio)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(bio)
    raise ValueError("Unsupported file type (CSV/XLSX only).")


def df_profile(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing": int(df.isna().sum().sum()),
        "numeric": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical": df.select_dtypes(exclude=[np.number]).columns.tolist(),
    }


def ss_key(page: str, name: str) -> str:
    return f"{page}__{name}"


# =========================================================
# Dataset loader per "page"
# - supports: load from folder + upload
# - supports loading MULTIPLE datasets simultaneously
# =========================================================
def dataset_loader_ui(page_name: str, folder: Path) -> Dict[str, pd.DataFrame]:
    folder.mkdir(parents=True, exist_ok=True)

    loaded_key = ss_key(page_name, "datasets")
    if loaded_key not in st.session_state:
        st.session_state[loaded_key] = {}  # label -> df

    loaded: Dict[str, pd.DataFrame] = st.session_state[loaded_key]

    st.subheader("1) Datasets (CSV/XLSX only)")

    files = sorted([p for p in folder.glob("*") if p.suffix.lower() in SUPPORTED_EXTS])

    colL, colR = st.columns([1.25, 1], gap="large")

    with colL:
        with st.expander("Load from folder", expanded=True):
            if files:
                chosen = st.multiselect(
                    "Select one or more datasets",
                    options=files,
                    format_func=lambda p: p.name,
                    key=ss_key(page_name, "disk_pick"),
                )
                if st.button("Load selected", key=ss_key(page_name, "load_disk_btn")):
                    for p in chosen:
                        label = p.name
                        if label in loaded:
                            continue
                        loaded[label] = load_file(str(p))
                    st.session_state[loaded_key] = loaded
                    st.success("Loaded selected datasets.")
            else:
                st.info(f"No CSV/XLSX files found in `{folder}` yet.")

        with st.expander("Upload dataset", expanded=False):
            up = st.file_uploader(
                "Upload CSV/XLSX",
                type=["csv", "xlsx", "xls"],
                key=ss_key(page_name, "uploader"),
            )
            if up is not None:
                df = load_upload_bytes(up.name, up.getvalue())
                label = up.name
                # avoid collisions
                if label in loaded:
                    base = Path(label).stem
                    ext = Path(label).suffix
                    i = 2
                    while f"{base} ({i}){ext}" in loaded:
                        i += 1
                    label = f"{base} ({i}){ext}"
                loaded[label] = df
                st.session_state[loaded_key] = loaded
                st.success(f"Uploaded and loaded: {label}")

    with colR:
        with st.expander("Loaded datasets", expanded=True):
            if not loaded:
                st.warning("No datasets loaded yet.")
            else:
                labels = list(loaded.keys())
                for i, lbl in enumerate(labels):
                    prof = df_profile(loaded[lbl])
                    a, b, c, d, e = st.columns([3.2, 1, 1, 1, 1])
                    a.markdown(f"**{lbl}**")
                    b.caption(f"{prof['rows']:,} rows")
                    c.caption(f"{prof['cols']:,} cols")
                    d.caption(f"{prof['missing']:,} missing")
                    if e.button("Remove", key=ss_key(page_name, f"rm_{i}")):
                        loaded.pop(lbl, None)
                        st.session_state[loaded_key] = loaded
                        st.rerun()

                preview_lbl = st.selectbox(
                    "Preview",
                    labels,
                    key=ss_key(page_name, "preview_lbl"),
                )
                st.dataframe(loaded[preview_lbl].head(50), use_container_width=True)

    return loaded


# =========================================================
# Exploratory builder (client makes their own chart)
# =========================================================
def exploratory_builder(page_name: str, datasets: Dict[str, pd.DataFrame]) -> None:
    st.subheader("3) Exploratory Zone (client can build charts)")

    if not datasets:
        st.info("Load a dataset to enable exploration.")
        return

    ds_lbl = st.selectbox("Dataset", list(datasets.keys()), key=ss_key(page_name, "exp_ds"))
    df = datasets[ds_lbl]
    prof = df_profile(df)

    if not prof["numeric"]:
        st.warning("No numeric columns available for Y-axis in this dataset.")
        return

    # optional filters (lightweight)
    with st.expander("Filters (optional)", expanded=False):
        df_f = df.copy()
        for col in prof["categorical"][:10]:
            uniq = df[col].dropna().unique()
            if len(uniq) == 0 or len(uniq) > 3000:
                continue
            chosen = st.multiselect(col, sorted(uniq)[:5000], key=ss_key(page_name, f"flt_{ds_lbl}_{col}"))
            if chosen:
                df_f = df_f[df_f[col].isin(chosen)]
    st.caption(f"Rows after filters: {len(df_f):,}")

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    kind = c1.selectbox("Chart", ["Scatter", "Line", "Bar", "Box", "Map (iso3)"], key=ss_key(page_name, "exp_kind"))
    x = c2.selectbox("X", list(df_f.columns), key=ss_key(page_name, "exp_x"))
    y = c3.selectbox("Y", prof["numeric"], key=ss_key(page_name, "exp_y"))
    color = c4.selectbox("Color", [None] + prof["categorical"], key=ss_key(page_name, "exp_color"))
    size = c5.selectbox("Size", [None] + prof["numeric"], key=ss_key(page_name, "exp_size"))

    if kind == "Scatter":
        fig = px.scatter(df_f, x=x, y=y, color=color, size=size)
    elif kind == "Line":
        fig = px.line(df_f, x=x, y=y, color=color)
    elif kind == "Bar":
        fig = px.bar(df_f, x=x, y=y, color=color)
    elif kind == "Box":
        fig = px.box(df_f, x=x, y=y, color=color)
    else:
        if "iso3" not in df_f.columns:
            st.warning("Map requires a column named 'iso3' (ISO-3 codes).")
            return
        fig = px.choropleth(df_f, locations="iso3", color=y, hover_name="country" if "country" in df_f.columns else None)

    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# Page renderers — ADD YOUR GRAPH ZONES HERE
# Each page has explicit code blocks for you to extend
# =========================================================
def render_birth_death():
    st.title("👶💀 Birth rate & Death rate")
    datasets = dataset_loader_ui("Birth rate & Death rate", CATEGORIES["Birth rate & Death rate"])

    st.divider()
    st.header("2) Graph Area (add your indicator charts here)")

    # -------------------------------------------------
    # GRAPH ZONE A: (ADD YOUR GRAPHS HERE)
    # Example scaffold: time series
    # -------------------------------------------------
    st.subheader("Graph Zone A — Time series (scaffold)")
    if datasets:
        ds_lbl = st.selectbox("Dataset (Zone A)", list(datasets.keys()), key="bd_zoneA_ds")
        df = datasets[ds_lbl]
        prof = df_profile(df)
        if prof["numeric"]:
            x = st.selectbox("X (e.g., year/date)", list(df.columns), key="bd_zoneA_x")
            y = st.selectbox("Y (indicator)", prof["numeric"], key="bd_zoneA_y")
            color = st.selectbox("Color (optional)", [None] + prof["categorical"], key="bd_zoneA_color")
            st.plotly_chart(px.line(df, x=x, y=y, color=color), use_container_width=True)
        else:
            st.warning("No numeric columns in selected dataset.")
    else:
        st.info("Load at least one dataset to enable Zone A.")

    # -------------------------------------------------
    # GRAPH ZONE B: (ADD YOUR GRAPHS HERE)
    # Example scaffold: choropleth map
    # -------------------------------------------------
    st.subheader("Graph Zone B — Choropleth (scaffold)")
    if datasets:
        ds_lbl = st.selectbox("Dataset (Zone B)", list(datasets.keys()), key="bd_zoneB_ds")
        df = datasets[ds_lbl]
        prof = df_profile(df)
        if "iso3" in df.columns and prof["numeric"]:
            y = st.selectbox("Value column (map color)", prof["numeric"], key="bd_zoneB_y")
            st.plotly_chart(px.choropleth(df, locations="iso3", color=y), use_container_width=True)
        else:
            st.info("Requires `iso3` column + at least one numeric column.")
    else:
        st.info("Load at least one dataset to enable Zone B.")

    st.divider()
    exploratory_builder("Birth rate & Death rate", datasets)


def render_wealth():
    st.title("💰 Wealth distribution / inequality")
    datasets = dataset_loader_ui("Wealth distribution", CATEGORIES["Wealth distribution"])

    st.divider()
    st.header("2) Graph Area (add your indicator charts here)")

    # -------------------------------------------------
    # GRAPH ZONE A (ADD YOUR GRAPHS HERE)
    # -------------------------------------------------
    st.subheader("Graph Zone A — Inequality trends (placeholder)")
    st.info("Add your wealth/inequality indicator graphs here (e.g., top1%, Gini, wealth shares).")

    # -------------------------------------------------
    # GRAPH ZONE B (ADD YOUR GRAPHS HERE)
    # -------------------------------------------------
    st.subheader("Graph Zone B — Cross-country comparison (placeholder)")
    st.info("Add your cross-sectional comparisons here (selected year, selected group).")

    st.divider()
    exploratory_builder("Wealth distribution", datasets)


def render_education():
    st.title("🎓 Education levels & indices")
    datasets = dataset_loader_ui("Education & indices", CATEGORIES["Education & indices"])

    st.divider()
    st.header("2) Graph Area (add your indicator charts here)")

    # -------------------------------------------------
    # GRAPH ZONE A (ADD YOUR GRAPHS HERE)
    # -------------------------------------------------
    st.subheader("Graph Zone A — Attainment / enrollment (placeholder)")
    st.info("Add your education indicator graphs here (enrollment, attainment, literacy, etc.).")

    # -------------------------------------------------
    # GRAPH ZONE B (ADD YOUR GRAPHS HERE)
    # -------------------------------------------------
    st.subheader("Graph Zone B — Composite education index (placeholder)")
    st.info("Add your multi-element index graphs here (weights, subcomponents, comparisons).")

    st.divider()
    exploratory_builder("Education & indices", datasets)


def render_crime():
    st.title("🚔 Crime rates")
    datasets = dataset_loader_ui("Crime rates", CATEGORIES["Crime rates"])

    st.divider()
    st.header("2) Graph Area (add your indicator charts here)")

    # -------------------------------------------------
    # GRAPH ZONE A (ADD YOUR GRAPHS HERE)
    # -------------------------------------------------
    st.subheader("Graph Zone A — Homicide / key crime indicators (placeholder)")
    st.info("Add crime indicator graphs here (homicide rates, property crime, etc.).")

    # -------------------------------------------------
    # GRAPH ZONE B (ADD YOUR GRAPHS HERE)
    # -------------------------------------------------
    st.subheader("Graph Zone B — Crime vs other variables (placeholder)")
    st.info("Add correlation/scatter panels here (crime vs inequality, regime score, etc.).")

    st.divider()
    exploratory_builder("Crime rates", datasets)


def render_migration():
    st.title("🧳 Immigration / migration")
    datasets = dataset_loader_ui("Immigration / migration", CATEGORIES["Immigration / migration"])

    st.divider()
    st.header("2) Graph Area (add your indicator charts here)")

    # -------------------------------------------------
    # GRAPH ZONE A (ADD YOUR GRAPHS HERE)
    # -------------------------------------------------
    st.subheader("Graph Zone A — Migrant stock/flows (placeholder)")
    st.info("Add migration indicators here (stock, flows, refugees, asylum, etc.).")

    # -------------------------------------------------
    # GRAPH ZONE B (ADD YOUR GRAPHS HERE)
    # -------------------------------------------------
    st.subheader("Graph Zone B — Origin/destination matrix (placeholder)")
    st.info("Add heatmaps / sankey style charts here if your data supports it.")

    st.divider()
    exploratory_builder("Immigration / migration", datasets)


def render_regime():
    st.title("🏛️ Authoritarianism / regime indices")
    datasets = dataset_loader_ui("Authoritarianism / regime", CATEGORIES["Authoritarianism / regime"])

    st.divider()
    st.header("2) Graph Area (add your indicator charts here)")

    # -------------------------------------------------
    # GRAPH ZONE A (ADD YOUR GRAPHS HERE)
    # -------------------------------------------------
    st.subheader("Graph Zone A — Regime score over time (placeholder)")
    st.info("Add regime indicator trends here (V-Dem style indices, Freedom scores, Polity, etc.).")

    # -------------------------------------------------
    # GRAPH ZONE B (ADD YOUR GRAPHS HERE)
    # -------------------------------------------------
    st.subheader("Graph Zone B — Regime map / clusters (placeholder)")
    st.info("Add choropleths, clustering, or quadrant charts here.")

    st.divider()
    exploratory_builder("Authoritarianism / regime", datasets)


# =========================================================
# Sidebar navigation (same pattern as your cov19st.py)
# =========================================================
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=list(CATEGORIES.keys()),
        menu_icon="arrow-down-right-circle-fill",
        icons=["activity", "cash-coin", "mortarboard", "shield", "globe", "bank"],
        default_index=0,
    )

# =========================================================
# Route to the selected "page"
# =========================================================
if selected == "Birth rate & Death rate":
    render_birth_death()
elif selected == "Wealth distribution":
    render_wealth()
elif selected == "Education & indices":
    render_education()
elif selected == "Crime rates":
    render_crime()
elif selected == "Immigration / migration":
    render_migration()
elif selected == "Authoritarianism / regime":
    render_regime()
else:
    st.title("Global Stats Explorer")
    st.info("Choose a page from the sidebar.")
