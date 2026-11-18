"""Computes verifiable facts from the dataset, and answers common questions WITHOUT any AI.
Everything here is calculated with pandas; nothing is estimated or invented."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from . import kpis as K
from .utils import py, is_text, period_rule, last_period_partial


def top_list(df, col, val, n=5):
    t = K.top_by(df, col, val, n)
    total = float(df[val].sum())
    return [{"name": str(r[col]), "value": float(r[val]),
             "share_pct": round(100 * float(r[val]) / total, 1) if total else None}
            for _, r in t.iterrows()]


def periodic(df, roles):
    d, v = roles["date"], roles["revenue"]
    if not (d and v):
        return None
    dd = df.dropna(subset=[d])
    if dd.empty:
        return None
    span = (dd[d].max() - dd[d].min()).days
    rule, label = period_rule(span)
    s = dd.set_index(d)[v].resample(rule).sum()
    if len(s) < 3:
        return None
    slope = float(np.polyfit(np.arange(len(s)), s.values.astype(float), 1)[0])
    first, last = float(s.iloc[0]), float(s.iloc[-1])
    return {
        "granularity": label, "periods": len(s),
        "best_period": str(s.idxmax().date()), "best_value": float(s.max()),
        "worst_period": str(s.idxmin().date()), "worst_value": float(s.min()),
        "slope_per_period": slope,
        "first_to_last_pct_change": round(100 * (last - first) / first, 1) if first else None,
        "last_period_may_be_partial": bool(last_period_partial(dd[d].max(), s.index[-1], rule)),
        "series": [{"period": str(i.date()), "value": float(x)} for i, x in s.tail(24).items()],
    }


def outliers(df, roles):
    v = roles["revenue"]
    if not v:
        return None
    s = df[v].dropna()
    if len(s) < 8:
        return None
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    if iqr == 0:
        return None
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (df[v] < lo) | (df[v] > hi)
    cols = [c for c in (roles["date"], roles["product"], roles["city"], v) if c]
    ex = df.loc[mask, cols].sort_values(v, ascending=False).head(5)
    return {"method": "IQR rule (1.5 x IQR) on " + v, "lower_bound": float(lo), "upper_bound": float(hi),
            "count": int(mask.sum()), "examples": py(ex.to_dict("records"))}


def build_facts(df: pd.DataFrame, roles: dict | None = None) -> dict:
    roles = roles or K.detect(df)
    f = {
        "rows": len(df), "columns": df.shape[1],
        "column_types": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(n) for c, n in df.isna().sum().items() if n},
        "duplicate_rows": int(df.duplicated().sum()),
        "detected_columns": {r: c for r, c in roles.items() if c},
        "kpis": K.compute(df, roles),
        "currency_symbol": K.currency(roles),
    }
    rev = roles["revenue"]
    if rev:
        for role in ("category", "city", "product"):
            if roles[role]:
                f[f"top_{role}"] = top_list(df, roles[role], rev)
        f["trend"] = periodic(df, roles)
        f["outliers"] = outliers(df, roles)
    f["numeric_summary"] = {c: {"min": float(df[c].min()), "mean": float(df[c].mean()), "max": float(df[c].max())}
                            for c in df.select_dtypes("number").columns[:12] if df[c].notna().any()}
    if roles["date"]:
        f["date_range"] = [str(df[roles["date"]].min().date()), str(df[roles["date"]].max().date())]
    return py(f)


def _m(v, cur):
    return "N/A" if v is None else f"{cur}{v:,.2f}"


def insights_text(f: dict) -> str:
    """Plain-language findings, 100% derived from `build_facts`."""
    cur, k, out = f["currency_symbol"], f["kpis"], []
    out.append(f"The dataset has {f['rows']:,} rows and {f['columns']} columns.")
    if k["revenue"] is not None:
        out.append(f"Total revenue is {_m(k['revenue'], cur)} across {k['orders']:,} orders "
                   f"(average order value {_m(k['aov'], cur)}).")
    for role in ("product", "category", "city"):
        t = f.get(f"top_{role}")
        if t:
            out.append(f"Top {role}: {t[0]['name']} with {_m(t[0]['value'], cur)} "
                       f"({t[0]['share_pct']}% of revenue).")
    tr = f.get("trend")
    if tr:
        direction = "upward" if tr["slope_per_period"] > 0 else "downward"
        out.append(f"Revenue trend is {direction} over {tr['periods']} {tr['granularity']}s "
                   f"(best {tr['granularity']}: {tr['best_period']}, weakest: {tr['worst_period']}).")
        if tr["last_period_may_be_partial"]:
            out.append("Note: the final period may be incomplete, which can distort the trend.")
    o = f.get("outliers")
    if o:
        out.append(f"{o['count']} unusual revenue value(s) found using the IQR rule." if o["count"]
                   else "No unusual revenue values found using the IQR rule.")
    if f["missing_values"]:
        out.append(f"Missing values in {len(f['missing_values'])} column(s); {f['duplicate_rows']} duplicate row(s).")
    else:
        out.append(f"No missing values; {f['duplicate_rows']} duplicate row(s).")
    return "\n".join("- " + x for x in out)


def _mentions_data_value(q: str, df: pd.DataFrame) -> bool:
    """If the question names a specific city/product/etc., the simple rules would ignore it -> defer."""
    ql = q.lower()
    for c in df.columns:
        if is_text(df[c]) and df[c].nunique() <= 200:
            for val in df[c].dropna().unique():
                v = str(val).lower()
                if len(v) >= 3 and re.search(r"\b" + re.escape(v) + r"\b", ql):
                    return True
    return False


def answer_local(q: str, df: pd.DataFrame, roles: dict, f: dict) -> str | None:
    """Rule-based answers. Returns None when the question needs AI or a custom query."""
    ql, cur, k = q.lower(), f["currency_symbol"], f["kpis"]
    if _mentions_data_value(q, df) or re.search(r"\b(in|for|during|between|since|last|this)\b.*\b(\d{4}|month|year|week|quarter)\b", ql):
        return None
    rules = [
        (r"average order|aov", lambda: f"Average order value is {_m(k['aov'], cur)}." if k["aov"] is not None else None),
        (r"total (revenue|sales)|how much (revenue|sales)|^revenue|^what is my revenue",
         lambda: f"Total revenue is {_m(k['revenue'], cur)}." if k["revenue"] is not None else None),
        (r"how many orders|total orders|number of orders", lambda: f"There are {k['orders']:,} orders."),
        (r"quantity|units sold", lambda: f"Total quantity is {k['quantity']:,.0f}." if k["quantity"] is not None else None),
        (r"product", lambda: _top(f, "product", cur)),
        (r"cit(y|ies)|location|region", lambda: _top(f, "city", cur)),
        (r"categor", lambda: _top(f, "category", cur)),
        (r"trend|growth|growing|decline|over time", lambda: _trend(f)),
        (r"unusual|outlier|anomal|odd", lambda: _outl(f, cur)),
        (r"missing|duplicate|quality|clean", lambda: _quality(f)),
    ]
    for pat, fn in rules:
        if re.search(pat, ql):
            return fn()
    return None


def _top(f, role, cur):
    t = f.get(f"top_{role}")
    if not t:
        return None
    lines = [f"{i + 1}. {x['name']}: {_m(x['value'], cur)} ({x['share_pct']}%)" for i, x in enumerate(t)]
    return f"Top {role}s by revenue:\n" + "\n".join(lines)


def _trend(f):
    tr = f.get("trend")
    if not tr:
        return None
    s = (f"Over {tr['periods']} {tr['granularity']}s, the fitted trend is "
         f"{'upward' if tr['slope_per_period'] > 0 else 'downward'} "
         f"({tr['slope_per_period']:+,.2f} per {tr['granularity']}). Best {tr['granularity']}: "
         f"{tr['best_period']}; weakest: {tr['worst_period']}.")
    if tr["last_period_may_be_partial"]:
        s += " The last period may be incomplete."
    return s


def _outl(f, cur):
    o = f.get("outliers")
    if not o:
        return "Not enough variation or rows to test for unusual values."
    if not o["count"]:
        return f"No unusual values found ({o['method']})."
    ex = "; ".join(", ".join(f"{k}={v}" for k, v in r.items()) for r in o["examples"])
    return f"{o['count']} unusual value(s) ({o['method']}). Examples: {ex}"


def _quality(f):
    m = f["missing_values"]
    ms = ", ".join(f"{c}: {n}" for c, n in m.items()) if m else "none"
    return f"Missing values: {ms}. Duplicate rows: {f['duplicate_rows']}."
