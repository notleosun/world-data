# app.py
# ------------------------------------------------------------
# Global Stats Explorer (cov19-style routing)
# - Sidebar option_menu navigation
# - Each category is a separate "page"
# - Folder-only dataset loader + preview (NO uploads)
# - NO year/type filter functions (datasets are assumed pre-filtered)
# - Clear Graph Zones A/B/C for you to add indicator-specific charts
# - CSV/XLSX only; explicit excel engines
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
DATA_ROOT = BASE_DIR / "data"  # keep data inside repo folder

CATEGORIES = {
    "Birth rate & Death rate": DATA_ROOT / "demographics",
    "Wealth distribution / inequality": DATA_ROOT / "wealth",
    "Education levels & indices": DATA_ROOT / "education",
    "Crime rates": DATA_ROOT / "crime",
    "Immigration / migration": DATA_ROOT / "migration",
    "Authoritarianism / regime indices": DATA_ROOT / "regime",
}

SUPPORTED_EXTS = {".csv", ".xlsx", ".xlsm", ".xls"}  # remove ".xls" if you don't want xlrd


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
        # If you keep .xls support, add xlrd to requirements: pip install xlrd
        return pd.read_excel(p, engine="xlrd")

    raise ValueError(f"Unsupported file type: {suf}")


def df_profile(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing": int(df.isna().sum().sum()),
        "numeric": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical": df.select_dtypes(exclude=[np.number]).columns.tolist(),
    }


def get_datasets_map(page_name: str) -> Dict[str, pd.DataFrame]:
    """
    Session-state storage for datasets loaded on a given page.
    """
    key = ss_key(page_name, "datasets_map")
    if key not in st.session_state:
        st.session_state[key] = {}
    return st.session_state[key]


# ============================================================
# Page renderer template
# ============================================================
def render_page(*, page_name: str, data_folder: Path, description: str) -> None:
    st.title(page_name)
    st.caption(description)

    # Ensure folder exists inside repo
    data_folder.mkdir(parents=True, exist_ok=True)

    left, right = st.columns([1.1, 1.9], gap="large")

    # ----------------------------
    # LEFT: Folder-only loader + preview (NO uploads)
    # ----------------------------
    with left:
        st.subheader("1) Load & Preview (folder-only)")
        st.caption(f"Folder: `{data_folder}`")

        datasets_map = get_datasets_map(page_name)

        files = sorted([p for p in data_folder.glob("*") if p.suffix.lower() in SUPPORTED_EXTS])

        if not files:
            st.info("No CSV/XLSX files found. Add files into the folder above.")
        else:
            with st.expander("Load datasets from folder", expanded=True):
                chosen = st.multiselect(
                    "Select one or more files to load",
                    options=files,
                    format_func=lambda p: p.name,
                    key=ss_key(page_name, "disk_pick"),
                )

                if st.button("Load selected", key=ss_key(page_name, "load_btn")):
                    for p in chosen:
                        label = p.name
                        if label in datasets_map:
                            continue
                        try:
                            datasets_map[label] = load_file(str(p))
                        except Exception as e:
                            st.error(f"Failed to load `{p.name}`: {e}")
                    st.session_state[ss_key(page_name, "datasets_map")] = datasets_map
                    st.rerun()

            with st.expander("Loaded datasets + preview", expanded=True):
                datasets_map = get_datasets_map(page_name)  # refresh
                if len(datasets_map) == 0:
                    st.warning("No datasets loaded yet.")
                else:
                    labels = list(datasets_map.keys())

                    # Remove buttons
                    for i, lbl in enumerate(list(labels)):
                        c1, c2 = st.columns([6, 1])
                        c1.markdown(f"**{lbl}**")
                        if c2.button("Remove", key=ss_key(page_name, f"rm_{i}")):
                            datasets_map.pop(lbl, None)
                            st.session_state[ss_key(page_name, "datasets_map")] = datasets_map
                            st.rerun()

                    if len(datasets_map) > 0:
                        labels = list(datasets_map.keys())
                        preview_lbl = st.selectbox(
                            "Preview dataset",
                            options=labels,
                            key=ss_key(page_name, "preview_lbl"),
                        )

                        df_prev = datasets_map[preview_lbl].copy()
                        st.caption(f"Preview rows: {len(df_prev):,} | cols: {df_prev.shape[1]:,}")
                        st.dataframe(df_prev.head(50), width="stretch")

                        with st.expander("Columns", expanded=False):
                            st.dataframe(
                                pd.DataFrame(
                                    {"column": df_prev.columns, "dtype": df_prev.dtypes.astype(str).values}
                                ),
                                width="stretch",
                            )

    # ----------------------------
    # RIGHT: Graph zones (you edit/add graphs here)
    # ----------------------------
    with right:
        datasets_map = get_datasets_map(page_name)  # refresh

        st.subheader("2) Graph Areas (add graphs per indicator)")
        st.caption("No global filters here — datasets are assumed pre-filtered before loading.")

        if not isinstance(datasets_map, dict) or len(datasets_map) == 0:
            st.info("Load at least one dataset on the left to enable graphs.")
            return

        active_ds = st.selectbox(
            "Active dataset (used by Graph Zones unless overridden)",
            options=list(datasets_map.keys()),
            key=ss_key(page_name, "active_ds"),
        )

        df_work = datasets_map[active_ds].copy()
        prof = df_profile(df_work)

        # =========================================================
        # GRAPH ZONE A — Time series / trend scaffold
        # =========================================================
        st.markdown("### Graph Zone A — Trend / Time series")
        with st.expander("Zone A controls", expanded=True):
            if len(prof["numeric"]) == 0:
                st.warning("No numeric columns available to plot.")
            else:
                x_col = st.selectbox("X", options=list(df_work.columns), key=ss_key(page_name, "za_x"))
                y_col = st.selectbox("Y (numeric)", options=prof["numeric"], key=ss_key(page_name, "za_y"))
                color_col = st.selectbox("Color (optional)", options=[None] + prof["categorical"], key=ss_key(page_name, "za_c"))

                fig = px.line(df_work, x=x_col, y=y_col, color=color_col, title=f"Zone A: {y_col} vs {x_col}")
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # =========================================================
        # GRAPH ZONE B — Map / cross-sectional scaffold
        # =========================================================
        st.markdown("### Graph Zone B — Map / Cross-section")
        with st.expander("Zone B controls", expanded=True):
            if "ISO3 Alpha-code" in df_work.columns and len(prof["numeric"]) > 0:
                iso_col = "ISO3 Alpha-code"
                val_col = st.selectbox("Map value (numeric)", options=prof["numeric"], key=ss_key(page_name, "zb_val"))
                hover = "Region, subregion, country or area *" if "Region, subregion, country or area *" in df_work.columns else None
                fig = px.choropleth(df_work, locations=iso_col, color=val_col, hover_name=hover, title=f"Zone B: {val_col} map")
                st.plotly_chart(fig, use_container_width=True)
            elif "iso3" in df_work.columns and len(prof["numeric"]) > 0:
                iso_col = "iso3"
                val_col = st.selectbox("Map value (numeric)", options=prof["numeric"], key=ss_key(page_name, "zb_val2"))
                hover = "country" if "country" in df_work.columns else None
                fig = px.choropleth(df_work, locations=iso_col, color=val_col, hover_name=hover, title=f"Zone B: {val_col} map")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Map needs an ISO3 column (`ISO3 Alpha-code` or `iso3`) and at least one numeric column.")

        st.divider()

        # =========================================================
        # GRAPH ZONE C — Exploratory Builder (optional)
        # =========================================================
        st.markdown("### Graph Zone C — Exploratory (optional)")
        with st.expander("Zone C builder", expanded=False):
            prof2 = df_profile(df_work)
            if len(prof2["numeric"]) == 0:
                st.warning("No numeric columns available for charts.")
            else:
                # Optional categorical filters (safe string sorting)
                df_exp = df_work.copy()
                with st.expander("Filters (optional)", expanded=False):
                    for col in prof2["categorical"][:10]:
                        uniq = df_exp[col].dropna().unique()
                        if len(uniq) == 0:
                            continue
                        if len(uniq) > 3000:
                            st.caption(f"Skipping '{col}' (too many unique values).")
                            continue
                        uniq_str = sorted({str(v) for v in uniq})
                        chosen = st.multiselect(
                            f"{col}",
                            options=uniq_str[:5000],
                            key=ss_key(page_name, f"zc_f_{col}"),
                        )
                        if chosen:
                            df_exp = df_exp[df_exp[col].astype(str).isin(chosen)]

                c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
                kind = c1.selectbox("Chart", ["Scatter", "Line", "Bar", "Box"], key=ss_key(page_name, "zc_kind"))
                x = c2.selectbox("X", options=list(df_exp.columns), key=ss_key(page_name, "zc_x"))
                y = c3.selectbox("Y (numeric)", options=prof2["numeric"], key=ss_key(page_name, "zc_y"))
                color = c4.selectbox("Color", options=[None] + prof2["categorical"], key=ss_key(page_name, "zc_color"))
                size = c5.selectbox("Size", options=[None] + prof2["numeric"], key=ss_key(page_name, "zc_size"))

                if kind == "Scatter":
                    fig = px.scatter(df_exp, x=x, y=y, color=color, size=size)
                elif kind == "Line":
                    fig = px.line(df_exp, x=x, y=y, color=color)
                elif kind == "Bar":
                    fig = px.bar(df_exp, x=x, y=y, color=color)
                else:
                    fig = px.box(df_exp, x=x, y=y, color=color)

                st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Individual page renderers (cov19-style routing)
# ============================================================
def render_birth_death():
    render_page(
        page_name="Birth rate & Death rate",
        data_folder=CATEGORIES["Birth rate & Death rate"],
        description="Vital statistics: births, deaths, fertility, mortality indicators (assumed pre-filtered).",
    )


def render_wealth():
    render_page(
        page_name="Wealth distribution / inequality",
        data_folder=CATEGORIES["Wealth distribution / inequality"],
        description="Wealth/income shares, Gini, top percentiles, inequality measures (assumed pre-filtered).",
    )


def render_education():
    render_page(
        page_name="Education levels & indices",
        data_folder=CATEGORIES["Education levels & indices"],
        description="Attainment, enrollment, learning outcomes, composite education indices (assumed pre-filtered).",
    )


def render_crime():
    render_page(
        page_name="Crime rates",
        data_folder=CATEGORIES["Crime rates"],
        description="Crime indicators (homicide etc.), comparisons and dashboards (assumed pre-filtered).",
    )


def render_migration():
    render_page(
        page_name="Immigration / migration",
        data_folder=CATEGORIES["Immigration / migration"],
        description="Migrant stock/flows, refugees, asylum indicators (assumed pre-filtered).",
    )


def render_regime():
    render_page(
        page_name="Authoritarianism / regime indices",
        data_folder=CATEGORIES["Authoritarianism / regime indices"],
        description="Democracy/autocracy indices, governance measures (assumed pre-filtered).",
    )


# ============================================================
# Sidebar nav (cov19-style)
# ============================================================
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=list(CATEGORIES.keys()),
        menu_icon="diagram-3-fill",
        icons=["activity", "cash-coin", "mortarboard", "shield", "globe", "bank"],
        default_index=0,
    )
    st.caption(f"Data root: `{DATA_ROOT}`")


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
else:
    st.title("Global Stats Explorer")
    st.info("Choose a page from the sidebar.")
