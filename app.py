# app.py
# ------------------------------------------------------------
# Global Stats Explorer (from scratch, cov19-style routing)
# - Sidebar option_menu navigation (like cov19 file)
# - Each category is its own "page" (render_* functions)
# - Folder-only dataset loader + preview (NO uploads)
# - Clear code zones in each page where YOU add graphs
# - CSV/XLSX only; robust Excel engine handling
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

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

# If you want a single shared data folder (no categories), set:
# DATA_ROOT = BASE_DIR / "data_all"
# and point all pages to that folder.

CATEGORIES = {
    "Birth & Death": DATA_ROOT / "demographics",
    "Wealth Distribution": DATA_ROOT / "wealth",
    "Education & Indices": DATA_ROOT / "education",
    "Crime Rates": DATA_ROOT / "crime",
    "Migration": DATA_ROOT / "migration",
    "Regime / Authoritarianism": DATA_ROOT / "regime",
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

    # Modern excel
    if suf in (".xlsx", ".xlsm"):
        return pd.read_excel(p, engine="openpyxl")

    # Legacy .xls (requires xlrd)
    if suf == ".xls":
        # If you keep .xls, add xlrd to requirements: pip install xlrd
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


def apply_base_filter(df: pd.DataFrame, year: Optional[str] = None, type_value: Optional[str] = None) -> pd.DataFrame:
    """
    Optional standard filters used across graph zones.
    - year: e.g. "2023"
    - type_value: e.g. "country"
    """
    out = df.copy()

    if year is not None and "year" in out.columns:
        out = out[out["year"].astype(str) == str(year)]

    if type_value is not None and "type" in out.columns:
        out = out[out["type"].astype(str).str.lower() == str(type_value).lower()]

    return out


# ============================================================
# Folder-only Loader + Preview (CODE STYLE, not a function)
# We'll "inline" this pattern inside each page renderer.
# To reduce repetition, we do keep a tiny helper to get datasets,
# but you can paste the block if you prefer.
# ============================================================
def get_datasets_map(page_name: str) -> Dict[str, pd.DataFrame]:
    key = ss_key(page_name, "datasets_map")
    if key not in st.session_state:
        st.session_state[key] = {}
    return st.session_state[key]


# ============================================================
# PAGE TEMPLATE (each page uses this structure)
# - left: folder loader + preview
# - right: graph zones (you add graphs)
# ============================================================
def render_page(
    *,
    page_name: str,
    data_folder: Path,
    description: str,
    default_year: str = "2023",
    default_type: str = "country",
) -> None:
    st.title(page_name)
    st.caption(description)

    # Ensure folder exists inside repo
    data_folder.mkdir(parents=True, exist_ok=True)

    # Two-column layout
    left, right = st.columns([1.1, 1.9], gap="large")

    # ----------------------------
    # LEFT: Loader + Preview (NO uploads)
    # ----------------------------
    with left:
        st.subheader("1) Load & Preview (folder-only)")
        st.caption(f"Folder: `{data_folder}`")

        datasets_map = get_datasets_map(page_name)

        files = sorted([p for p in data_folder.glob("*") if p.suffix.lower() in SUPPORTED_EXTS])

        if not files:
            st.info("No CSV/XLSX files found. Add files into the folder above.")
        else:
            with st.expander("Load datasets", expanded=True):
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

                        # OPTIONAL: preview filters
                        with st.expander("Preview filters (optional)", expanded=False):
                            year_val = st.text_input("year =", value=default_year, key=ss_key(page_name, "prev_year"))
                            type_val = st.text_input("type =", value=default_type, key=ss_key(page_name, "prev_type"))
                            apply_prev_filters = st.checkbox("Apply preview filters", value=True, key=ss_key(page_name, "prev_apply"))

                        if apply_prev_filters:
                            df_prev = apply_base_filter(df_prev, year=year_val, type_value=type_val)

                        st.caption(f"Preview rows: {len(df_prev):,} | cols: {df_prev.shape[1]:,}")
                        st.dataframe(df_prev.head(50), width="stretch")

    # ----------------------------
    # RIGHT: Graph Zones (YOU add graphs here)
    # ----------------------------
    with right:
        datasets_map = get_datasets_map(page_name)  # refresh

        st.subheader("2) Graph Areas (add graphs per indicator)")
        st.caption("Each zone is a separate code area for you to add charts.")

        if not isinstance(datasets_map, dict) or len(datasets_map) == 0:
            st.info("Load at least one dataset on the left to enable graphs.")
            return

        # Common controls for your graphs
        with st.expander("Common filters for graphs (optional)", expanded=False):
            g_year = st.text_input("Filter year =", value=default_year, key=ss_key(page_name, "g_year"))
            g_type = st.text_input("Filter type =", value=default_type, key=ss_key(page_name, "g_type"))
            use_common_filters = st.checkbox("Apply these filters to Graph Zones", value=True, key=ss_key(page_name, "g_apply"))

        # Pick active dataset for zones (you can change per-zone too)
        active_ds = st.selectbox(
            "Active dataset (used by Graph Zones unless overridden)",
            options=list(datasets_map.keys()),
            key=ss_key(page_name, "active_ds"),
        )
        df_base = datasets_map[active_ds]
        df_work = apply_base_filter(df_base, g_year, g_type) if use_common_filters else df_base.copy()

        st.caption(f"Active dataset rows after common filters: {len(df_work):,}")

        # =========================================================
        # GRAPH ZONE A (YOU ADD GRAPHS HERE)
        # Example: simple line chart scaffold
        # =========================================================
        st.markdown("### Graph Zone A — (your indicator chart)")
        with st.expander("Zone A controls", expanded=True):
            prof = df_profile(df_work)
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
        # GRAPH ZONE B (YOU ADD GRAPHS HERE)
        # Example: choropleth scaffold (requires iso3)
        # =========================================================
        st.markdown("### Graph Zone B — (map / cross-sectional)")
        with st.expander("Zone B controls", expanded=True):
            prof = df_profile(df_work)
            if "iso3" not in df_work.columns:
                st.info("Zone B map requires an `iso3` column (ISO-3 country codes).")
            elif len(prof["numeric"]) == 0:
                st.warning("No numeric columns available to color the map.")
            else:
                val_col = st.selectbox("Map value (numeric)", options=prof["numeric"], key=ss_key(page_name, "zb_val"))
                hover = "country" if "country" in df_work.columns else None
                fig = px.choropleth(df_work, locations="iso3", color=val_col, hover_name=hover, title=f"Zone B: {val_col} map")
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # =========================================================
        # GRAPH ZONE C (OPTIONAL) — Exploratory builder
        # (If you don't want this, delete this block)
        # =========================================================
        st.markdown("### Graph Zone C — Exploratory (client chart builder)")
        with st.expander("Zone C builder", expanded=False):
            df_exp = df_work.copy()
            prof = df_profile(df_exp)

            if len(prof["numeric"]) == 0:
                st.warning("No numeric columns available for charts.")
            else:
                # Filters (safe string sorting)
                with st.expander("Extra filters", expanded=False):
                    for col in prof["categorical"][:10]:
                        uniq = df_exp[col].dropna().unique()
                        if len(uniq) == 0:
                            continue
                        # safe string conversion prevents sort errors
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
                y = c3.selectbox("Y (numeric)", options=prof["numeric"], key=ss_key(page_name, "zc_y"))
                color = c4.selectbox("Color", options=[None] + prof["categorical"], key=ss_key(page_name, "zc_color"))
                size = c5.selectbox("Size", options=[None] + prof["numeric"], key=ss_key(page_name, "zc_size"))

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
        data_folder=CATEGORIES["Birth & Death"],
        description="Vital statistics: birth rate, death rate, fertility, mortality metrics.",
        default_year="2023",
        default_type="country",
    )


def render_wealth():
    render_page(
        page_name="Wealth distribution / inequality",
        data_folder=CATEGORIES["Wealth Distribution"],
        description="Wealth/income shares, Gini, top percentiles, inequality measures.",
        default_year="2023",
        default_type="country",
    )


def render_education():
    render_page(
        page_name="Education levels & indices",
        data_folder=CATEGORIES["Education & Indices"],
        description="Attainment, enrollment, learning outcomes, composite education indices.",
        default_year="2023",
        default_type="country",
    )


def render_crime():
    render_page(
        page_name="Crime rates",
        data_folder=CATEGORIES["Crime Rates"],
        description="Homicide rates and crime indicators; comparisons and trends.",
        default_year="2023",
        default_type="country",
    )


def render_migration():
    render_page(
        page_name="Immigration / migration",
        data_folder=CATEGORIES["Migration"],
        description="Migrant stock/flows, refugees, asylum, origin/destination indicators.",
        default_year="2023",
        default_type="country",
    )


def render_regime():
    render_page(
        page_name="Authoritarianism / regime indices",
        data_folder=CATEGORIES["Regime / Authoritarianism"],
        description="Democracy/autocracy indices, governance measures, regime scores.",
        default_year="2023",
        default_type="country",
    )


# ============================================================
# Sidebar nav (cov19-style)
# ============================================================
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=[
            "Birth rate & Death rate",
            "Wealth distribution / inequality",
            "Education levels & indices",
            "Crime rates",
            "Immigration / migration",
            "Authoritarianism / regime indices",
        ],
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
