"""Transparent sales forecast: linear trend (+ month-of-year seasonality only when >= 2 full years of
monthly data). Intervals are 95% prediction intervals from the fit residuals. No black box."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .utils import last_period_partial

MIN_PERIODS = 12


def _fit(t, y):
    X = np.column_stack([np.ones_like(t), t])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int, freq: str = "W",
             drop_partial: bool = True) -> dict:
    """freq: 'W' weekly or 'M' monthly."""
    res = {"ok": False, "message": "", "history": None, "forecast": None, "metrics": {}, "assumptions": [], "warnings": []}
    d = df.dropna(subset=[date_col, value_col])
    if d.empty:
        res["message"] = "No rows with both a date and a value."
        return res
    rule = "MS" if freq == "M" else "W"
    s = d.set_index(date_col)[value_col].astype(float).resample(rule).sum()
    if drop_partial and len(s) > 1 and last_period_partial(d[date_col].max(), s.index[-1], rule):
        s = s.iloc[:-1]
        res["warnings"].append("The last period was incomplete, so it was excluded to avoid a false drop.")
    unit = "month" if freq == "M" else "week"
    if len(s) < MIN_PERIODS:
        res["message"] = (f"Only {len(s)} {unit}s of history are available; at least {MIN_PERIODS} are needed for a "
                          f"reliable forecast. Upload more history{' or try weekly mode' if freq == 'M' else ''}.")
        return res
    if (s == 0).mean() > 0.3:
        res["warnings"].append("More than 30% of periods have zero sales; the forecast may be unreliable.")

    n, y = len(s), s.values
    t = np.arange(n, dtype=float)
    seasonal = freq == "M" and n >= 24
    months = s.index.month.values
    beta = _fit(t, y)
    trend_fit = beta[0] + beta[1] * t
    idx = np.zeros(13)
    if seasonal:
        for m in range(1, 13):
            idx[m] = (y - trend_fit)[months == m].mean()
        fit = trend_fit + idx[months]
    else:
        fit = trend_fit
    resid = y - fit
    dof = max(n - (14 if seasonal else 2), 1)
    sigma = float(np.sqrt((resid ** 2).sum() / dof))

    # Hold-out backtest on the last 20% (or `periods`, whichever is smaller)
    h = max(2, min(periods, n // 5))
    tb = _fit(t[:-h], y[:-h])
    pred_bt = tb[0] + tb[1] * t[-h:]
    mask = y[-h:] != 0
    res["metrics"] = {"holdout_periods": h, "MAE": float(np.abs(y[-h:] - pred_bt).mean()),
                      "MAPE_pct": float(np.mean(np.abs((y[-h:] - pred_bt)[mask] / y[-h:][mask])) * 100) if mask.any() else None}

    ft = np.arange(n, n + periods, dtype=float)
    step = pd.offsets.MonthBegin(1) if freq == "M" else pd.Timedelta(days=7)
    fdates = [s.index[-1] + step * (i + 1) for i in range(periods)]
    pred = beta[0] + beta[1] * ft
    if seasonal:
        pred = pred + idx[np.array([x.month for x in fdates])]
    sxx = ((t - t.mean()) ** 2).sum()
    se = sigma * np.sqrt(1 + 1 / n + (ft - t.mean()) ** 2 / sxx)
    pred = np.clip(pred, 0, None)
    res["forecast"] = pd.DataFrame({"period": fdates, "forecast": pred,
                                    "lower_95": np.clip(pred - 1.96 * se, 0, None), "upper_95": pred + 1.96 * se})
    res["history"] = s
    res["assumptions"] = [
        f"Sales are summed per {unit}; {unit}s with no sales count as zero.",
        "Model: straight-line trend fitted to history" + (" plus a month-of-year seasonal adjustment." if seasonal else
                                                          " (no seasonality: needs 24+ months of data)."),
        "Assumes the future behaves like the past: no promotions, price changes, new products or market shocks.",
        "Intervals are 95% prediction ranges from past fit errors; real outcomes can fall outside them.",
    ]
    if periods > n // 2:
        res["warnings"].append("You are forecasting more than half the length of your history; treat it as rough.")
    res["ok"] = True
    return res
