"""Dataset loading + local persistence. Original uploads are stored untouched in data/datasets;
cleaned versions are stored separately in data/cleaned, so the original is always recoverable."""
from __future__ import annotations
import io, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data"
STORE = ROOT / "datasets"
CLEAN = ROOT / "cleaned"
ALLOWED = {".csv", ".xlsx", ".xls"}
MAX_MB = 100
LARGE_ROWS = 200_000


class DatasetError(Exception):
    """User-facing error for bad uploads."""


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]", "_", Path(name).name)


def parse_bytes(name: str, raw: bytes) -> pd.DataFrame:
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED:
        raise DatasetError(f"Unsupported file type '{ext}'. Upload CSV or Excel (.xlsx).")
    if not raw:
        raise DatasetError("The file is empty.")
    if len(raw) > MAX_MB * 1024 * 1024:
        raise DatasetError(f"File is larger than {MAX_MB} MB.")
    try:
        if ext == ".csv":
            df = None
            for enc in ("utf-8-sig", "latin-1"):
                try:
                    df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise DatasetError("Could not decode the CSV file.")
        else:
            df = pd.read_excel(io.BytesIO(raw))
    except DatasetError:
        raise
    except Exception as e:
        raise DatasetError(f"Could not read '{name}': {e}") from e
    if df.empty or df.shape[1] == 0:
        raise DatasetError("The file has no data rows or columns.")
    df.columns = [str(c).strip() for c in df.columns]
    return optimize(_auto_dates(df))


def _auto_dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) and re.search(r"date|time|day", c, re.I):
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                continue
            if parsed.notna().mean() > 0.8:
                df[c] = parsed
    return df


def optimize(df: pd.DataFrame) -> pd.DataFrame:
    """Memory saving for big files only: downcast integer columns (lossless)."""
    if len(df) >= LARGE_ROWS:
        for c in df.select_dtypes(include="integer").columns:
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def save_upload(name: str, raw: bytes) -> str:
    STORE.mkdir(parents=True, exist_ok=True)
    key = safe_name(name)
    (STORE / key).write_bytes(raw)
    return key


def restore_all() -> dict[str, pd.DataFrame]:
    out = {}
    if STORE.exists():
        for p in sorted(STORE.iterdir()):
            if p.suffix.lower() in ALLOWED:
                try:
                    out[p.name] = parse_bytes(p.name, p.read_bytes())
                except DatasetError:
                    continue
    return out


def save_cleaned(key: str, df: pd.DataFrame) -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN / (safe_name(key) + ".csv"), index=False)


def load_cleaned(key: str) -> pd.DataFrame | None:
    p = CLEAN / (safe_name(key) + ".csv")
    if p.exists():
        try:
            return parse_bytes(p.name, p.read_bytes())
        except DatasetError:
            return None
    return None


def delete_cleaned(key: str) -> None:
    p = CLEAN / (safe_name(key) + ".csv")
    if p.exists():
        p.unlink()


def delete_saved(key: str) -> None:
    p = STORE / safe_name(key)
    if p.exists():
        p.unlink()
    delete_cleaned(key)


def describe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Column": df.columns,
        "Type": [str(t) for t in df.dtypes],
        "Missing": df.isna().sum().values,
        "Unique": [df[c].nunique() for c in df.columns],
    })
