"""KPI calculation from the actual data. A metric is None (shown as N/A) when its column is absent,
so the UI never shows a fake zero."""
from __future__ import annotations
import pandas as pd

HINTS = {
    "revenue": ["netrevenue", "revenue", "totalsales", "sales", "amount", "total"],
    "quantity": ["quantity", "qty", "units"],
    "date": ["orderdate", "date"],
    "order_id": ["orderid", "invoice"],
    "category": ["category"],
    "city": ["city"],
    "product": ["product", "item"],
}


def _norm(s: str) -> str:
    return s.lower().replace(" ", "").replace("-", "").replace("_", "")


def detect(df: pd.DataFrame) -> dict[str, str | None]:
    cols = {_norm(c): c for c in df.columns}
    found: dict[str, str | None] = {}
    for role, hints in HINTS.items():
        found[role] = None
        for h in hints:
            for n, orig in cols.items():
                if h not in n:
                    continue
                if role in ("revenue", "quantity") and not pd.api.types.is_numeric_dtype(df[orig]):
                    continue
                if role == "date" and not pd.api.types.is_datetime64_any_dtype(df[orig]):
                    continue
                found[role] = orig
                break
            if found[role]:
                break
    return found


def currency(roles: dict) -> str:
    return "₹" if "inr" in (roles.get("revenue") or "").lower() else ""


def compute(df: pd.DataFrame, roles: dict[str, str | None]) -> dict:
    rev, qty, oid = roles["revenue"], roles["quantity"], roles["order_id"]
    total_rev = float(df[rev].sum()) if rev else None
    orders = int(df[oid].nunique()) if oid else len(df)   # no order-id column: 1 row = 1 order
    return {
        "revenue": total_rev,
        "orders": orders,
        "quantity": float(df[qty].sum()) if qty else None,
        "aov": (total_rev / orders) if (total_rev is not None and orders) else None,
    }


def top_by(df: pd.DataFrame, group_col: str, value_col: str, n: int = 10) -> pd.DataFrame:
    return (df.groupby(group_col, dropna=False)[value_col].sum()
              .sort_values(ascending=False).head(n).reset_index())
