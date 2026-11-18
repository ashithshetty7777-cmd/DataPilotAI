"""Small shared helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd


def is_text(s: pd.Series) -> bool:
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s)
                or pd.api.types.is_bool_dtype(s))


def py(o):
    """Convert numpy/pandas objects to JSON-safe Python values."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if o != o else float(o)
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    if isinstance(o, dict):
        return {str(k): py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [py(v) for v in o]
    return o


def period_rule(span_days: int) -> tuple[str, str]:
    return ("MS", "month") if span_days >= 180 else ("W", "week")


def last_period_partial(max_date: pd.Timestamp, label: pd.Timestamp, rule: str) -> bool:
    """True if the final resampled period is not fully covered by the data."""
    end = label + pd.offsets.MonthEnd(0) if rule == "MS" else label
    return max_date.normalize() < end.normalize()
