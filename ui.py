"""Streamlit UI helpers (theme CSS, KPI cards, width-API compatibility wrappers)."""
import html
import streamlit as st

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
section[data-testid="stSidebar"] {border-right: 1px solid #232938;}
.kpi {background:#171b26; border:1px solid #252b3b; border-radius:12px; padding:16px 18px; height:100%;}
.kpi .l {color:#8b93a7; font-size:.76rem; text-transform:uppercase; letter-spacing:.06em;}
.kpi .v {color:#e8ecf5; font-size:1.65rem; font-weight:650; margin-top:4px;}
.kpi .s {color:#5b6478; font-size:.72rem; margin-top:2px;}
.tag-calc, .tag-ai {display:inline-block; font-size:.7rem; padding:2px 9px; border-radius:99px; margin-bottom:6px;}
.tag-calc {background:#12372d; color:#38d9a9;} .tag-ai {background:#2b2a4d; color:#a5b4ff;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{html.escape(str(sub))}</div></div>', unsafe_allow_html=True)


def tag(kind: str):
    if kind == "calc":
        st.markdown('<span class="tag-calc">✔ CALCULATED from your data</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="tag-ai">✦ AI-GENERATED interpretation - verify before use</span>', unsafe_allow_html=True)


def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, **kw)


def show_fig(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, key=key)


def fmt(v, prefix="", decimals=0):
    return "N/A" if v is None else f"{prefix}{v:,.{decimals}f}"
