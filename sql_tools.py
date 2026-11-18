"""Safe, read-only SQL on the active dataset using in-memory SQLite (table name: data)."""
from __future__ import annotations
import re, sqlite3, time
import pandas as pd
from . import kpis as K

TABLE = "data"
MAX_ROWS = 1000
TIMEOUT_S = 5


class SQLSafetyError(Exception):
    pass


def _strip_noise(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql.strip()


def validate(sql: str) -> str:
    """Returns the cleaned single SELECT statement or raises SQLSafetyError."""
    s = _strip_noise(sql).rstrip(";").strip()
    if not s:
        raise SQLSafetyError("Query is empty.")
    no_lit = re.sub(r"'(?:[^']|'')*'", "''", s)
    no_lit = re.sub(r'"(?:[^"]|"")*"', '""', no_lit)
    if ";" in no_lit:
        raise SQLSafetyError("Only one statement is allowed.")
    if not re.match(r"^(select|with)\b", no_lit, re.I):
        raise SQLSafetyError("Only SELECT queries are allowed.")
    bad = re.search(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex|truncate)\b",
                    no_lit, re.I)
    if bad:
        raise SQLSafetyError(f"'{bad.group(1).upper()}' is not allowed - read-only queries only.")
    return s


def _authorizer(action, *_):
    ok = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    if hasattr(sqlite3, "SQLITE_RECURSIVE"):
        ok.add(sqlite3.SQLITE_RECURSIVE)
    return sqlite3.SQLITE_OK if action in ok else sqlite3.SQLITE_DENY


def run_query(df: pd.DataFrame, sql: str, limit: int = MAX_ROWS) -> pd.DataFrame:
    s = validate(sql)
    conn = sqlite3.connect(":memory:")
    try:
        df.to_sql(TABLE, conn, index=False)
        conn.set_authorizer(_authorizer)          # second line of defence, enforced by SQLite itself
        t0 = time.time()
        conn.set_progress_handler(lambda: 1 if time.time() - t0 > TIMEOUT_S else 0, 10000)
        try:
            return pd.read_sql_query(f"SELECT * FROM ({s}) LIMIT {int(limit)}", conn)
        except Exception as e:
            raise SQLSafetyError(f"SQL error: {e}") from e
    finally:
        conn.close()


def sample_queries(df: pd.DataFrame, roles: dict) -> list[tuple[str, str]]:
    q = [("Preview first rows", f"SELECT * FROM {TABLE} LIMIT 10"),
         ("Count rows", f"SELECT COUNT(*) AS total_rows FROM {TABLE}")]
    r, d = roles["revenue"], roles["date"]
    if r:
        q.append(("Total revenue", f'SELECT SUM("{r}") AS total_revenue FROM {TABLE}'))
        for role, title in (("category", "Revenue by category"), ("city", "Revenue by city"),
                            ("product", "Top 5 products")):
            c = roles[role]
            if c:
                lim = " LIMIT 5" if role == "product" else ""
                q.append((title, f'SELECT "{c}", SUM("{r}") AS revenue FROM {TABLE} '
                                 f'GROUP BY "{c}" ORDER BY revenue DESC{lim}'))
        if d:
            q.append(("Monthly revenue", f"SELECT strftime('%Y-%m', \"{d}\") AS month, SUM(\"{r}\") AS revenue "
                                         f"FROM {TABLE} GROUP BY month ORDER BY month"))
        q.append(("Orders above average",
                  f'SELECT * FROM {TABLE} WHERE "{r}" > (SELECT AVG("{r}") FROM {TABLE}) ORDER BY "{r}" DESC LIMIT 20'))
    return q


def explain(sql: str) -> list[str]:
    """Beginner-friendly, rule-based explanation (no AI needed)."""
    s = _strip_noise(sql)
    out = []
    pats = [
        (r"\bselect\s+(distinct\s+)?(.*?)\s+from\b", lambda m: f"SELECT picks which columns/calculations to show: {m.group(2).strip()[:120]}."),
        (r"\bfrom\s+(\w+)", lambda m: f"FROM says which table to read: '{m.group(1)}' (your active dataset)."),
        (r"\bjoin\b", lambda m: "JOIN combines rows from two tables."),
        (r"\bwhere\s+(.*?)(\bgroup by\b|\border by\b|\blimit\b|$)", lambda m: f"WHERE keeps only rows matching: {m.group(1).strip()[:120]}."),
        (r"\bgroup by\s+(.*?)(\border by\b|\bhaving\b|\blimit\b|$)", lambda m: f"GROUP BY makes one result row per distinct value of: {m.group(1).strip()}."),
        (r"\bhaving\b", lambda m: "HAVING filters the groups after they are calculated."),
        (r"\border by\s+(.*?)(\blimit\b|$)", lambda m: f"ORDER BY sorts the result by: {m.group(1).strip()} (DESC = largest first)."),
        (r"\blimit\s+(\d+)", lambda m: f"LIMIT {m.group(1)} returns only that many rows."),
    ]
    for p, fn in pats:
        m = re.search(p, s, re.I | re.S)
        if m:
            out.append(fn(m))
    aggs = {"sum": "adds up values", "avg": "takes the average", "count": "counts rows", "min": "finds the smallest",
            "max": "finds the largest"}
    for a, t in aggs.items():
        if re.search(rf"\b{a}\s*\(", s, re.I):
            out.append(f"{a.upper()}() {t}.")
    return out or ["Could not break this query into parts - check the syntax."]
