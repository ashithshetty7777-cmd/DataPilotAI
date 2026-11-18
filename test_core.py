import io, json
import numpy as np, pandas as pd, pytest
from core import datasets as D, kpis as K, facts as F, cleaning as C, sql_tools as S
from core import pyrunner as P, forecast as FC, ai as AI, exports as X, sample_data


def _csv(df): return df.to_csv(index=False).encode()
def _xlsx(df):
    b = io.BytesIO(); df.to_excel(b, index=False); return b.getvalue()

@pytest.fixture
def df(): return sample_data.make()


# ---------- upload / persistence ----------
def test_csv_and_excel_load_identically(df):
    a = D.parse_bytes("s.csv", _csv(df)); b = D.parse_bytes("s.xlsx", _xlsx(df))
    assert a.shape == b.shape == (30, 11)
    assert pd.api.types.is_datetime64_any_dtype(a["OrderDate"])

@pytest.mark.parametrize("name,raw", [("a.txt", b"x"), ("a.csv", b""), ("a.xlsx", b"not excel"), ("a.csv", b"col\n")])
def test_invalid_files_raise_friendly_error(name, raw):
    with pytest.raises(D.DatasetError): D.parse_bytes(name, raw)

def test_save_restore_and_cleaned_copy(tmp_path, monkeypatch, df):
    monkeypatch.setattr(D, "STORE", tmp_path / "o"); monkeypatch.setattr(D, "CLEAN", tmp_path / "c")
    key = D.save_upload("../evil name!.csv", _csv(df))
    assert ".." not in key and "/" not in key
    assert D.restore_all()[key].shape == (30, 11)
    D.save_cleaned(key, df.head(5)); assert len(D.load_cleaned(key)) == 5
    D.delete_saved(key); assert D.restore_all() == {} and D.load_cleaned(key) is None


# ---------- KPIs ----------
def test_kpis_match_manual_calculation(df):
    roles = K.detect(df); k = K.compute(df, roles)
    assert roles["revenue"] == "NetRevenue_INR" and roles["quantity"] == "Quantity"
    assert k["revenue"] == pytest.approx(df.NetRevenue_INR.sum())
    assert k["orders"] == 30 and k["quantity"] == df.Quantity.sum()
    assert K.currency(roles) == "₹"

def test_missing_columns_give_none_not_zero():
    d = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    k = K.compute(d, K.detect(d))
    assert k["revenue"] is None and k["quantity"] is None and k["aov"] is None


# ---------- facts / local answers ----------
def test_facts_are_exact_and_json_safe(df):
    r = K.detect(df); f = F.build_facts(df, r); json.dumps(f)
    top = df.groupby("Product").NetRevenue_INR.sum().idxmax()
    assert f["top_product"][0]["name"] == top
    assert f["kpis"]["revenue"] == pytest.approx(df.NetRevenue_INR.sum())

def test_local_answers_use_real_numbers(df):
    r = K.detect(df); f = F.build_facts(df, r)
    a = F.answer_local("What is my total revenue?", df, r, f)
    assert f"{df.NetRevenue_INR.sum():,.2f}" in a
    best_city = df.groupby("City").NetRevenue_INR.sum().idxmax()
    assert best_city in F.answer_local("Which city performed best?", df, r, f)
    assert F.answer_local("Give me recommendations to improve sales", df, r, f) is None

def test_question_naming_a_value_is_not_answered_from_unfiltered_totals(df):
    r = K.detect(df); f = F.build_facts(df, r)
    assert F.answer_local("What is the total revenue in Mumbai?", df, r, f) is None

def test_no_customer_metrics_invented(df):
    assert "customer" not in json.dumps(F.build_facts(df, K.detect(df))).lower()


# ---------- cleaning ----------
def _dirty():
    return pd.DataFrame({"City": ["Pune ", "pune", "Mumbai", None, "Mumbai"], "Amt": ["1,000", "2000", "x", "300", "300"],
                         "D": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-04"], "N": [1.0, None, 3, 4, 4]})

def test_issue_detection():
    kinds = {(i["kind"], i["column"]) for i in C.detect_issues(pd.concat([_dirty(), _dirty().tail(1)]))}
    assert ("duplicates", None) in kinds and ("missing", "City") in kinds and ("missing", "N") in kinds
    assert ("whitespace", "City") in kinds and ("case", "City") in kinds and ("dtype", "D") in kinds

def test_preview_does_not_modify_original_and_apply_works():
    d = _dirty(); before = d.copy()
    new, info = C.preview(d, {"op": "fill_median", "column": "N"})
    pd.testing.assert_frame_equal(d, before)
    assert new.N.isna().sum() == 0 and info["cells_changed"] == 1
    new, info = C.preview(d, {"op": "to_numeric", "column": "Amt"})
    assert new.Amt.iloc[0] == 1000 and info["new_missing"] == 1
    assert len(C.apply_op(pd.concat([d, d.tail(1)]), {"op": "drop_duplicates"})) == 5
    assert C.apply_op(d, {"op": "strip", "column": "City"}).City.iloc[0] == "Pune"

def test_bad_cleaning_op_raises():
    with pytest.raises(ValueError): C.apply_op(_dirty(), {"op": "nope", "column": "N"})
    with pytest.raises(ValueError): C.apply_op(_dirty(), {"op": "strip", "column": "missing_col"})


# ---------- SQL ----------
def test_sql_runs_and_matches_pandas(df):
    out = S.run_query(df, 'SELECT SUM("NetRevenue_INR") AS t FROM data')
    assert out.t.iloc[0] == pytest.approx(df.NetRevenue_INR.sum())
    for title, q in S.sample_queries(df, K.detect(df)):
        assert len(S.run_query(df, q)) >= 1, title

@pytest.mark.parametrize("q", ["DELETE FROM data", "DROP TABLE data", "SELECT 1; DROP TABLE data",
                               "UPDATE data SET Quantity=0", "PRAGMA table_info(data)", "INSERT INTO data VALUES (1)",
                               "ATTACH DATABASE 'x.db' AS x", "", "-- hi"])
def test_sql_rejects_unsafe(df, q):
    with pytest.raises(S.SQLSafetyError): S.run_query(df, q)

def test_sql_literal_with_keyword_is_fine_and_errors_are_friendly(df):
    assert len(S.run_query(df, "SELECT 'update' AS w FROM data LIMIT 1")) == 1
    with pytest.raises(S.SQLSafetyError): S.run_query(df, "SELECT nope FROM data")
    assert any("GROUP BY" in l for l in S.explain("SELECT City, SUM(x) FROM data GROUP BY City"))


# ---------- python runner ----------
def test_python_runs_on_copy(df):
    r = P.run("df['Quantity'] = 0\nresult = df.Quantity.sum()", df)
    assert r["error"] is None and r["result"] == 0 and df.Quantity.sum() > 0

@pytest.mark.parametrize("code", ["import os", "open('x')", "__import__('os')", "df.to_csv('x.csv')", "pd.read_csv('x')",
                                  "x = df.__class__", "eval('1')", "pd.io.common.os.listdir('.')", "exec('1')"])
def test_python_blocks_unsafe(df, code):
    assert P.run(code, df)["error"]

def test_python_reports_runtime_errors_and_stdout(df):
    assert "KeyError" in P.run("df['nope']", df)["error"]
    assert "30" in P.run("print(len(df))", df)["stdout"]


# ---------- forecast ----------
def test_forecast_refuses_short_history(df):
    r = FC.forecast(df, "OrderDate", "NetRevenue_INR", 4, "M")
    assert not r["ok"] and "at least" in r["message"]

def test_forecast_recovers_known_trend():
    dates = pd.date_range("2024-01-01", periods=104, freq="W-MON")
    d = pd.DataFrame({"d": dates, "v": 100 + 5 * np.arange(104.0)})
    r = FC.forecast(d, "d", "v", 8, "W", drop_partial=False)
    assert r["ok"] and len(r["forecast"]) == 8
    assert r["forecast"].forecast.iloc[0] == pytest.approx(100 + 5 * 104, rel=0.02)
    assert (r["forecast"].lower_95 <= r["forecast"].forecast).all() and r["assumptions"]

def test_forecast_never_negative_and_monthly_seasonality():
    dates = pd.date_range("2023-01-01", periods=36, freq="MS")
    v = 1000 + 20 * np.arange(36) + np.tile([0, 0, 0, 0, 0, 0, 0, 0, 0, 300, 400, 600], 3)
    r = FC.forecast(pd.DataFrame({"d": dates, "v": v}), "d", "v", 12, "M", drop_partial=False)
    assert r["ok"] and (r["forecast"].lower_95 >= 0).all()
    assert any("seasonal" in a for a in r["assumptions"])


# ---------- AI layer (mocked network: no real API call was made) ----------
class _Resp:
    def __init__(s, o): s.o = o
    def read(s): return json.dumps(s.o).encode()
    def __enter__(s): return s
    def __exit__(s, *a): pass

def test_ai_parsing_and_errors(monkeypatch):
    import urllib.error
    monkeypatch.setattr(AI.urllib.request, "urlopen", lambda req, timeout=0: _Resp(
        {"candidates": [{"content": {"parts": [{"text": "hello"}, {"text": "x", "thought": True}]}}]}))
    assert AI.generate("k", "m", "s", "p") == "hello"
    monkeypatch.setattr(AI.urllib.request, "urlopen", lambda req, timeout=0: _Resp({"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(AI.AIError): AI.generate("k", "m", "s", "p")
    def boom(req, timeout=0): raise urllib.error.HTTPError("u", 401, "x", {}, io.BytesIO(b'{"error":{"message":"bad key"}}'))
    monkeypatch.setattr(AI.urllib.request, "urlopen", boom)
    with pytest.raises(AI.AIError): AI.generate("k", "m", "s", "p")
    with pytest.raises(AI.AIError): AI.generate("", "", "s", "p")

def test_model_listing_filters(monkeypatch):
    mk = lambda n, m="generateContent": {"name": "models/" + n, "supportedGenerationMethods": [m]}
    monkeypatch.setattr(AI.urllib.request, "urlopen", lambda req, timeout=0: _Resp({"models": [
        mk("gemini-x-flash"), mk("gemini-x-flash-lite"), mk("gemini-x-image"), mk("gemini-embedding", "embedContent"), mk("other")]}))
    ms = AI.list_models("k")
    assert ms == ["gemini-x-flash", "gemini-x-flash-lite"] and AI.default_model(ms) == "gemini-x-flash-lite"

def test_extract_code_and_prompt_has_no_raw_rows_by_default(df):
    assert AI.extract_code("hi\n```sql\nSELECT 1\n```", "sql") == "SELECT 1"
    assert "Sample rows" not in AI.schema_text(df) and "Sample rows" in AI.schema_text(df, 3)


# ---------- exports ----------
def test_exports(df):
    r = K.detect(df)
    xl = pd.read_excel(io.BytesIO(X.to_excel_bytes({"Cleaned": df, "Cleaned": df, "a/b:c": df})), sheet_name=None)
    assert set(xl) == {"Cleaned", "a_b_c"} and len(xl["Cleaned"]) == 30
    assert len(pd.read_csv(io.BytesIO(X.to_csv_bytes(df)))) == 30
    ks = X.kpi_summary(df, r)
    assert ks.loc[ks.Metric == "Total orders", "Value"].iloc[0] == "30"

def test_pdf_export_if_library_present(df):
    pytest.importorskip("fpdf")
    f = F.build_facts(df, K.detect(df))
    pdf = X.build_pdf("Report", "s.csv", X.kpi_summary(df, K.detect(df)), {"Top": pd.DataFrame(f["top_product"])},
                      F.insights_text(f), "AI text ₹ " + "x" * 200)
    assert pdf[:4] == b"%PDF"
