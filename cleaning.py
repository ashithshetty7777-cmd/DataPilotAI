"""Issue detection and cleaning operations. All ops are pure functions: they return a NEW frame,
so callers can preview before applying and the original is never modified."""
from __future__ import annotations
import warnings
import pandas as pd
from .utils import is_text


def _numeric_like(s: pd.Series) -> pd.Series:
    cleaned = s.astype("string").str.replace(r"[₹$€£,\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _date_like(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce")


def detect_issues(df: pd.DataFrame) -> list[dict]:
    issues = []
    dup = int(df.duplicated().sum())
    if dup:
        issues.append({"kind": "duplicates", "column": None, "count": dup,
                       "message": f"{dup} fully duplicated row(s).", "op": {"op": "drop_duplicates"}})
    for c in df.columns:
        s = df[c]
        miss = int(s.isna().sum())
        if miss:
            default = "median" if pd.api.types.is_numeric_dtype(s) else "mode"
            issues.append({"kind": "missing", "column": c, "count": miss, "default_method": default,
                           "message": f"'{c}' has {miss} missing value(s) ({100 * miss / len(df):.1f}%).",
                           "op": {"op": "fill_" + default, "column": c}})
        if not is_text(s):
            continue
        nn = s.dropna().astype(str)
        if nn.empty:
            continue
        ws = int((nn != nn.str.strip()).sum())
        if ws:
            issues.append({"kind": "whitespace", "column": c, "count": ws,
                           "message": f"'{c}' has {ws} value(s) with leading/trailing spaces.",
                           "op": {"op": "strip", "column": c}})
        stripped = nn.str.strip()
        if stripped.nunique() > stripped.str.lower().nunique():
            issues.append({"kind": "case", "column": c, "count": int(stripped.nunique() - stripped.str.lower().nunique()),
                           "message": f"'{c}' has the same text written with different capitalisation.",
                           "op": {"op": "standardize_case", "column": c, "mode": "title"}})
        if _numeric_like(nn).notna().mean() > 0.9:
            issues.append({"kind": "dtype", "column": c, "count": len(nn),
                           "message": f"'{c}' looks numeric but is stored as text.",
                           "op": {"op": "to_numeric", "column": c}})
        elif _date_like(nn).notna().mean() > 0.9:
            issues.append({"kind": "dtype", "column": c, "count": len(nn),
                           "message": f"'{c}' looks like dates but is stored as text.",
                           "op": {"op": "to_datetime", "column": c}})
    return issues


def apply_op(df: pd.DataFrame, op: dict) -> pd.DataFrame:
    kind, c = op["op"], op.get("column")
    out = df.copy()
    if kind == "drop_duplicates":
        return out.drop_duplicates().reset_index(drop=True)
    if c not in out.columns:
        raise ValueError(f"Column '{c}' not found.")
    if kind == "drop_missing_rows":
        return out.dropna(subset=[c]).reset_index(drop=True)
    if kind == "drop_column":
        return out.drop(columns=[c])
    if kind == "fill_median":
        out[c] = out[c].fillna(out[c].median())
    elif kind == "fill_mean":
        out[c] = out[c].fillna(out[c].mean())
    elif kind == "fill_mode":
        m = out[c].mode(dropna=True)
        if m.empty:
            raise ValueError(f"'{c}' has no values to take a mode from.")
        out[c] = out[c].fillna(m.iloc[0])
    elif kind == "fill_value":
        val = op["value"]
        if pd.api.types.is_numeric_dtype(out[c]):
            val = pd.to_numeric(val)
        out[c] = out[c].fillna(val)
    elif kind == "strip":
        out[c] = out[c].where(out[c].isna(), out[c].astype(str).str.strip())
    elif kind == "standardize_case":
        st = out[c].astype(str).str.strip()
        st = {"title": st.str.title, "lower": st.str.lower, "upper": st.str.upper}[op.get("mode", "title")]()
        out[c] = out[c].where(out[c].isna(), st)
    elif kind == "to_numeric":
        out[c] = _numeric_like(out[c])
    elif kind == "to_datetime":
        out[c] = _date_like(out[c])
    else:
        raise ValueError(f"Unknown operation '{kind}'.")
    return out


def preview(df: pd.DataFrame, op: dict) -> tuple[pd.DataFrame, dict]:
    new = apply_op(df, op)
    info = {"rows_before": len(df), "rows_after": len(new), "sample": None, "cells_changed": None, "new_missing": 0}
    c = op.get("column")
    if c and c in new.columns and len(new) == len(df):
        a, b = df[c], new[c]
        changed = ~((a.astype(str) == b.astype(str)) | (a.isna() & b.isna()))
        info["cells_changed"] = int(changed.sum())
        info["new_missing"] = int(b.isna().sum() - a.isna().sum()) if op["op"] in ("to_numeric", "to_datetime") else 0
        if changed.any():
            info["sample"] = pd.DataFrame({"before": a[changed].head(10), "after": b[changed].head(10)})
    elif op["op"] == "drop_duplicates":
        info["cells_changed"] = len(df) - len(new)
    return new, info
