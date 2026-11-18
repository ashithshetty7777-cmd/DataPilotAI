"""Runs user/AI-generated pandas code on a COPY of the dataset after static checks.
Honest limitation: this is an AST allow-list for a local single-user app, not a hardened sandbox."""
from __future__ import annotations
import ast, builtins, contextlib, io
import numpy as np
import pandas as pd

BANNED = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars", "getattr",
          "setattr", "delattr", "breakpoint", "os", "sys", "subprocess", "importlib", "builtins", "quit",
          "exit", "memoryview", "io", "compat", "util", "testing", "core"}
BANNED_ATTRS = BANNED | {"to_csv", "to_excel", "to_pickle", "to_parquet", "to_sql", "to_feather", "to_hdf",
                         "to_clipboard", "to_json", "to_html", "to_latex", "to_markdown", "to_xml", "to_stata"}
SAFE_BUILTINS = {n: getattr(builtins, n) for n in (
    "len", "range", "sum", "min", "max", "abs", "round", "sorted", "list", "dict", "set", "tuple", "str", "int",
    "float", "bool", "enumerate", "zip", "print", "isinstance", "any", "all", "map", "filter", "reversed",
    "divmod", "pow")}


class CodeSafetyError(Exception):
    pass


def check(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeSafetyError(f"Syntax error on line {e.lineno}: {e.msg}") from e
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            raise CodeSafetyError("Imports are not allowed. pandas as `pd`, numpy as `np`, plotly.express as `px` and your data as `df` are already available.")
        if isinstance(n, ast.Name) and n.id in BANNED:
            raise CodeSafetyError(f"'{n.id}' is not allowed.")
        if isinstance(n, ast.Attribute) and (n.attr.startswith("_") or n.attr in BANNED_ATTRS or n.attr.startswith("read_")):
            raise CodeSafetyError(f"Access to '.{n.attr}' is not allowed.")


def run(code: str, df: pd.DataFrame) -> dict:
    """Returns {stdout, result, fig, error}. Set a variable named `result` (or `fig`) to display output."""
    out = {"stdout": "", "result": None, "fig": None, "error": None}
    try:
        check(code)
    except CodeSafetyError as e:
        out["error"] = str(e)
        return out
    ns = {"df": df.copy(), "pd": pd, "np": np, "__builtins__": SAFE_BUILTINS}
    try:
        import plotly.express as px
        ns["px"] = px
    except ImportError:
        pass
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<analysis>", "exec"), ns)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    out["stdout"] = buf.getvalue()
    out["result"] = ns.get("result")
    out["fig"] = ns.get("fig")
    return out
