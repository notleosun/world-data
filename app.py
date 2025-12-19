# app.py
# ------------------------------------------------------------
# Global Stats Explorer (cov19-style routing)
# - Sidebar option_menu navigation
# - Folder-only dataset loader + preview (NO uploads)
# - NO year/type filtering (datasets assumed pre-filtered)
# - ONLY Graph Zone C (Exploratory chart builder)
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="Global Stats Explorer", layout="wide")

# ----------------------------
# Paths (SAFE on Streamlit Cloud)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"

CATEGORIES = {
    "Birth rate & Death rate": DATA_ROOT / "demographics",
    "Wealth distribution / inequality": DATA_ROOT / "wealth",
    "Education levels & indices": DATA_ROOT / "education",
    "Crime rates": DATA_ROOT / "crime",
    "Immigration / migration": DATA_ROOT / "migration",
    "Authoritarianism / regime indices": DATA_ROOT / "regime",
}

SUPPORTED_EXTS = {".csv", ".xlsx", ".xlsm", ".xls"}  # remove .xls if you want


# ----------------------------
# Helpers
# ----------------------------
def ss_key(page: str, name: str) -> str:
    return f"{page}__{name}"


@st.cache_data(show_spinner=False)
def load_file(path: str) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()

    if suf == ".csv":
        return pd.read_csv(p)

    if suf in (".xlsx", ".xlsm"):
        return pd.read_excel(p, engine="openpyxl")

    if suf == ".xls":
        return pd.read_excel(p, engine="xlrd")

    raise ValueError(f"Unsupported file type: {suf}")


def df_profile(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "numeric": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical": df.select_dtypes(exclude=[np.number]).columns.tolist(),
    }


def get_datasets_map(page_name: str) -> Dict[str, pd.DataFrame]:
    key = ss_key(page_name, "datasets_map")
    if key not in st.session_state:
        st.session_state[key] = {}
    return st.session_state[key]


# ============================================================
# Page renderer (Loader + Preview + Zone C only)
# ============================================================
def render_page(*, page_name: str, data_folder: Path, description: str) -> None:
    st.title(page_name)
    st.caption(description)

    data_folder.mkdir(parents=True, exist_ok=True)

    left, right = st.columns([1.1, 1.9], gap="large")

    # ----------------------------
    # LEFT: Loader + Preview
    # ----------------------------
    with left:
        st.subheader("1) Load & Preview (folder-only)")
        st.caption(f"Folder: `{data_folder}`")

        datasets_map = get_datasets_map(page_name)

        files = sorted([p for p in data_folder.glob("*") if p.suffix.lower() in SUPPORTED_EXTS])

        if not files:
            st.info("No CSV/XLSX files found.")
        else:
            with st.expander("Load datasets from folder", expanded=True):
                chosen = st.multiselect(
                    "Select datasets",
                    options=files,
                    format_func=lambda p: p.name,
                    key=ss_key(page_name, "disk_pick"),
                )

                if st.button("Load selected", key=ss_key(page_name, "load_btn")):
                    for p in chosen:
                        if p.name not in datasets_map:
                            datasets_map[p.name] = load_file(str(p))
                    st.session_state[ss_key(page_name, "datasets_map")] = datasets_map
                    st.rerun()

            with st.expander("Loaded datasets + preview", expanded=True):
                if not datasets_map:
                    st.warning("No datasets loaded yet.")
                else:
                    labels = list(datasets_map.keys())

                    for i, lbl in enumerate(labels):
                        c1, c2 = st.columns([6, 1])
                        c1.markdown(f"**{lbl}**")
                        if c2.button("Remove", key=ss_key(page_name, f"rm_{i}")):
                            datasets_map.pop(lbl)
                            st.session_state[ss_key(page_name, "datasets_map")] = datasets_map
                            st.rerun()

                    if datasets_map:
                        preview_lbl = st.selectbox(
                            "Preview dataset",
                            options=list(datasets_map.keys()),
                            key=ss_key(page_name, "preview_lbl"),
                        )
                        df_prev = datasets_map[preview_lbl]
                        st.caption(f"Rows: {len(df_prev):,} | Columns: {df_prev.shape[1]:,}")
                        st.dataframe(df_prev.head(50), width="stretch")

    # ----------------------------
    # RIGHT: Graph Zone C ONLY
    # ----------------------------
    with right:
        datasets_map = get_datasets_map(page_name)

        st.subheader("2) Graph Zone C — Exploratory Builder")

        if not datasets_map:
            st.info("Load a dataset on the left to build charts.")
            return

        active_ds = st.selectbox(
            "Dataset",
            options=list(datasets_map.keys()),
            key=ss_key(page_name, "active_ds"),
        )

        df = datasets_map[active_ds].copy()
        prof = df_profile(df)

        if not prof["numeric"]:
            st.warning("No numeric columns available for charts.")
            return

        # Optional categorical filters
        with st.expander("Filters (optional)", expanded=False):
            for col in prof["categorical"][:10]:
                uniq = df[col].dropna().unique()
                if len(uniq) == 0 or len(uniq) > 3000:
                    continue
                uniq_str = sorted({str(v) for v in uniq})
                chosen = st.multiselect(
                    col,
                    uniq_str,
                    key=ss_key(page_name, f"f_{col}"),
                )
                if chosen:
                    df = df[df[col].astype(str).isin(chosen)]

        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])

        chart_type = c1.selectbox("Chart", ["Scatter", "Line", "Bar", "Box"])
        x = c2.selectbox("X", df.columns)
        y = c3.selectbox("Y (numeric)", prof["numeric"])
        color = c4.selectbox("Color", [None] + prof["categorical"])
        size = c5.selectbox("Size", [None] + prof["numeric"])

        if chart_type == "Scatter":
            fig = px.scatter(df, x=x, y=y, color=color, size=size)
        elif chart_type == "Line":
            fig = px.line(df, x=x, y=y, color=color)
        elif chart_type == "Bar":
            fig = px.bar(df, x=x, y=y, color=color)
        else:
            fig = px.box(df, x=x, y=y, color=color)

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page routing
# ============================================================
def render_birth_death():
    render_page(
        page_name="Birth rate & Death rate",
        data_folder=CATEGORIES["Birth rate & Death rate"],
        description="Vital statistics (pre-filtered).",
    )


def render_wealth():
    render_page(
        page_name="Wealth distribution / inequality",
        data_folder=CATEGORIES["Wealth distribution / inequality"],
        description="Wealth and inequality indicators (pre-filtered).",
    )


def render_education():
    render_page(
        page_name="Education levels & indices",
        data_folder=CATEGORIES["Education levels & indices"],
        description="Education indicators and indices (pre-filtered).",
    )


def render_crime():
    render_page(
        page_name="Crime rates",
        data_folder=CATEGORIES["Crime rates"],
        description="Crime indicators (pre-filtered).",
    )


def render_migration():
    render_page(
        page_name="Immigration / migration",
        data_folder=CATEGORIES["Immigration / migration"],
        description="Migration indicators (pre-filtered).",
    )


def render_regime():
    render_page(
        page_name="Authoritarianism / regime indices",
        data_folder=CATEGORIES["Authoritarianism / regime indices"],
        description="Political regime indicators (pre-filtered).",
    )


# ============================================================
# Sidebar navigation (cov19-style)
# ============================================================
with st.sidebar:
    selected = option_menu(
        "Navigation",
        list(CATEGORIES.keys()),
        icons=["activity", "cash-coin", "mortarboard", "shield", "globe", "bank"],
        default_index=0,
    )

# ============================================================
# Routing
# ============================================================
if selected == "Birth rate & Death rate":
    render_birth_death()
elif selected == "Wealth distribution / inequality":
    render_wealth()
elif selected == "Education levels & indices":
    render_education()
elif selected == "Crime rates":
    render_crime()
elif selected == "Immigration / migration":
    render_migration()
elif selected == "Authoritarianism / regime indices":
    render_regime()
