"""Plotly charts shared by the Dashboard and the Export Center."""
from __future__ import annotations
import pandas as pd
import plotly.express as px
from . import kpis as K
from .utils import period_rule

COLORS = ["#6c8cff", "#38d9a9", "#ffa94d", "#f06595", "#74c0fc"]


def style(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=50, b=10), colorway=COLORS)
    return fig


def build_charts(df: pd.DataFrame, roles: dict) -> dict:
    figs, rev = {}, roles["revenue"]
    if not rev:
        return figs
    cur = K.currency(roles)
    if roles["date"]:
        dd = df.dropna(subset=[roles["date"]])
        if not dd.empty:
            rule, label = period_rule((dd[roles["date"]].max() - dd[roles["date"]].min()).days)
            ts = dd.set_index(roles["date"])[rev].resample(rule).sum().reset_index()
            figs[f"Revenue by {label}"] = style(px.line(ts, x=roles["date"], y=rev, markers=True,
                                                        title=f"Revenue by {label}"))
    for role, title in (("category", "Revenue by category"), ("city", "Revenue by city"), ("product", "Top products")):
        c = roles[role]
        if c:
            t = K.top_by(df, c, rev, 10)
            t[c] = t[c].astype(str)
            figs[title] = style(px.bar(t, x=c, y=rev, title=title))
    for f in figs.values():
        f.update_yaxes(tickprefix=cur)
    return figs
