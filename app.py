"""DataPilot AI - AI-powered analytics platform (Streamlit)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from core import ai as AI, charts as CH, cleaning as CL, datasets as D, exports as X, facts as F
from core import forecast as FC, kpis as K, pyrunner as PR, sql_tools as SQ
from core.ui import fmt, inject_css, kpi, show_df, show_fig, tag

load_dotenv()
st.set_page_config(page_title="DataPilot AI", page_icon="📊", layout="wide")
inject_css()

SECTIONS = ["📈 Dashboard", "🗂️ Datasets", "🧠 AI Analyst", "🧹 Data Cleaning", "💬 AI Chat",
            "🗄️ SQL Assistant", "🐍 Python Assistant", "🔮 Forecasting", "📦 Export Center"]
ss = st.session_state


# ------------------------------------------------------------------ state
def init_state():
    if "originals" in ss:
        return
    ss.originals = D.restore_all()                        # session restoration from disk
    ss.working, ss.cleaned, ss.version = {}, set(), {}
    for k, df in ss.originals.items():
        c = D.load_cleaned(k)
        ss.working[k] = c if c is not None else df.copy()
        if c is not None:
            ss.cleaned.add(k)
        ss.version[k] = 0
    ss.active = next(iter(ss.originals), None)
    ss.seen, ss.chat, ss.insights, ss.exports, ss.facts_cache = set(), {}, {}, {}, {}
    ss.models, ss.ai_model, ss.ai_key, ss.sample_rows, ss.up_n = [], None, "", 0, 0


def bump(key):
    ss.version[key] = ss.version.get(key, 0) + 1
    ss.exports = {k: v for k, v in ss.exports.items() if not k.startswith(key + "|")}


def handle_uploads(files):
    for f in files or []:
        sig = (f.name, f.size)
        if sig in ss.seen:
            continue
        raw = f.getvalue()
        try:
            df = D.parse_bytes(f.name, raw)
        except D.DatasetError as e:
            st.sidebar.error(str(e))
            continue
        key = D.save_upload(f.name, raw)
        D.delete_cleaned(key)
        ss.originals[key], ss.working[key] = df, df.copy()
        ss.cleaned.discard(key)
        ss.active = key
        ss.seen.add(sig)
        bump(key)
        st.sidebar.success(f"Loaded {key}")


def get_facts(key, df, roles):
    ck = (key, ss.version.get(key, 0))
    if ck not in ss.facts_cache:
        ss.facts_cache = {k: v for k, v in ss.facts_cache.items() if k[0] != key}
        ss.facts_cache[ck] = F.build_facts(df, roles)
    return ss.facts_cache[ck]


# ------------------------------------------------------------------ AI helpers
def ai_ready():
    return bool(ss.ai_key and ss.ai_model)


def call_ai(system, prompt):
    try:
        with st.spinner("Asking Gemini..."):
            return AI.generate(ss.ai_key, ss.ai_model, system, prompt)
    except AI.AIError as e:
        st.error(f"AI request failed: {e}")
        return None


def sidebar_ai():
    with st.sidebar.expander("🤖 AI settings (Gemini)"):
        env_key = AI.get_env_key()
        typed = st.text_input("API key", type="password", placeholder="loaded from .env" if env_key else "paste key (memory only)")
        ss.ai_key = typed or env_key
        if ss.ai_key and st.button("Load available models"):
            try:
                ss.models = AI.list_models(ss.ai_key)
                ss.ai_model = AI.default_model(ss.models)
                if not ss.models:
                    st.warning("This key returned no usable text models.")
            except AI.AIError as e:
                st.error(str(e))
        if ss.models:
            idx = ss.models.index(ss.ai_model) if ss.ai_model in ss.models else 0
            ss.ai_model = st.selectbox("Model", ss.models, index=idx)
        else:
            ss.ai_model = st.text_input("Model id", value=ss.ai_model or AI.default_model([]) or "") or None
        ss.sample_rows = 5 if st.checkbox("Also send 5 sample rows", value=False) else 0
        st.caption("By default only the schema and calculated aggregates are sent - never raw rows. "
                   "Free-tier prompts may be used by Google to improve its products, so avoid sensitive data.")
    st.sidebar.caption("AI: ✅ ready" if ai_ready() else "AI: not configured (everything else still works)")


# ------------------------------------------------------------------ pages
def page_dashboard(key, df):
    roles = K.detect(df)
    view = df
    with st.expander("Filters"):
        c = st.columns(3)
        if roles["date"] and df[roles["date"]].notna().any():
            lo, hi = df[roles["date"]].min().date(), df[roles["date"]].max().date()
            rng = c[0].date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
            if isinstance(rng, (tuple, list)) and len(rng) == 2:
                s, e = pd.to_datetime(rng[0]), pd.to_datetime(rng[1]) + pd.Timedelta(days=1)
                view = view[(view[roles["date"]] >= s) & (view[roles["date"]] < e)]
        for i, role in enumerate(("category", "city"), start=1):
            if roles[role]:
                sel = c[i].multiselect(roles[role], sorted(df[roles[role]].dropna().astype(str).unique()))
                if sel:
                    view = view[view[roles[role]].astype(str).isin(sel)]
    if view.empty:
        st.warning("No rows match the current filters.")
        return
    k, cur = K.compute(view, roles), K.currency(roles)
    cols = st.columns(4)
    kpi(cols[0], "Total Revenue", fmt(k["revenue"], cur), roles["revenue"] or "no revenue column found")
    kpi(cols[1], "Total Orders", fmt(k["orders"]), roles["order_id"] or "no order-ID column: 1 row = 1 order")
    kpi(cols[2], "Total Quantity", fmt(k["quantity"]), roles["quantity"] or "no quantity column found")
    kpi(cols[3], "Avg Order Value", fmt(k["aov"], cur), "revenue / orders")
    st.caption(f"Showing {len(view):,} of {len(df):,} rows · detected: "
               + ", ".join(f"{r}→{c}" for r, c in roles.items() if c))
    figs = CH.build_charts(view, roles)
    if not figs:
        st.info("No numeric revenue/sales column detected, so revenue charts are hidden. Use SQL/Python assistants to explore.")
    items = list(figs.items())
    for i in range(0, len(items), 2):
        cc = st.columns(2)
        for col, (name, fig) in zip(cc, items[i:i + 2]):
            with col:
                show_fig(fig, key=f"dash_{key}_{name}")
    with st.expander("Data preview"):
        show_df(view.head(200))


def page_datasets(key, df):
    rows = [{"File": k, "Rows": len(v), "Columns": v.shape[1],
             "Size (MB)": round(v.memory_usage(deep=True).sum() / 1e6, 2),
             "Cleaned": "yes" if k in ss.cleaned else "no", "Active": "●" if k == key else ""}
            for k, v in ss.working.items()]
    show_df(pd.DataFrame(rows))
    st.subheader(f"Schema - {key}")
    show_df(D.describe(df))
    st.subheader("Preview")
    show_df(df.head(100))
    if len(df) >= D.LARGE_ROWS:
        st.info("Large dataset: integer columns were downcast to save memory.")
    if st.button("🗑️ Remove this dataset"):
        D.delete_saved(key)
        for d in (ss.originals, ss.working, ss.chat, ss.insights):
            d.pop(key, None)
        ss.cleaned.discard(key)
        ss.seen = {s for s in ss.seen if D.safe_name(s[0]) != key}
        ss.up_n += 1                                   # resets the uploader so the file can be added again
        ss.active = next(iter(ss.working), None)
        st.rerun()


def page_analyst(key, df):
    roles = K.detect(df)
    facts = get_facts(key, df, roles)
    tag("calc")
    st.markdown(F.insights_text(facts))
    for role in ("product", "category", "city"):
        if facts.get(f"top_{role}"):
            with st.expander(f"Top {role}s"):
                show_df(pd.DataFrame(facts[f"top_{role}"]))
    if facts.get("outliers") and facts["outliers"]["count"]:
        with st.expander("Unusual values"):
            show_df(pd.DataFrame(facts["outliers"]["examples"]))
    with st.expander("Verified facts sent to the AI (JSON)"):
        st.json(facts)
    st.divider()
    st.subheader("AI business report")
    if not ai_ready():
        st.info("Add a Gemini API key in the sidebar to get an AI-written report. The calculated findings above work without it.")
    elif st.button("Generate AI report"):
        q = ("Write a concise business report: 1) key findings, 2) notable patterns or risks, "
             "3) 3-5 practical recommendations. Use simple language.")
        p = AI.analyst_prompt(q, facts) + ("\n\n" + AI.schema_text(df, ss.sample_rows) if ss.sample_rows else "")
        text = call_ai(AI.ANALYST_SYSTEM, p)
        if text:
            ss.insights[key] = text
            bump(key)
    if ss.insights.get(key):
        tag("ai")
        st.markdown(ss.insights[key])


def _missing_options(issue):
    return ["median", "mean", "mode", "drop rows", "constant"] if issue["default_method"] == "median" \
        else ["mode", "drop rows", "constant"]


def page_cleaning(key, df):
    st.caption("Review each issue, preview the change, then apply. Your original upload is never modified.")
    issues = CL.detect_issues(df)
    c1, c2 = st.columns([3, 1])
    c1.metric("Issues found", len(issues))
    if c2.button("↩ Reset to original", disabled=key not in ss.cleaned):
        ss.working[key] = ss.originals[key].copy()
        D.delete_cleaned(key)
        ss.cleaned.discard(key)
        bump(key)
        st.rerun()
    if not issues:
        st.success("No issues detected.")
    for n, iss in enumerate(issues):
        with st.expander(f"{iss['kind'].title()} · {iss['message']}"):
            op = dict(iss["op"])
            if iss["kind"] == "missing":
                m = st.selectbox("Fix with", _missing_options(iss), key=f"m_{key}_{n}")
                if m == "drop rows":
                    op = {"op": "drop_missing_rows", "column": iss["column"]}
                elif m == "constant":
                    v = st.text_input("Value", key=f"v_{key}_{n}")
                    op = {"op": "fill_value", "column": iss["column"], "value": v}
                else:
                    op = {"op": f"fill_{m}", "column": iss["column"]}
            if iss["kind"] == "case":
                mode = st.selectbox("Case", ["title", "lower", "upper"], key=f"c_{key}_{n}")
                op["mode"] = mode
            b1, b2 = st.columns(2)
            try:
                if b1.button("Preview", key=f"p_{key}_{n}"):
                    _, info = CL.preview(df, op)
                    st.write(f"Rows: {info['rows_before']:,} → {info['rows_after']:,}"
                             + (f" · cells changed: {info['cells_changed']}" if info["cells_changed"] is not None else ""))
                    if info["new_missing"] > 0:
                        st.warning(f"{info['new_missing']} value(s) can't be converted and will become missing.")
                    if info["sample"] is not None:
                        show_df(info["sample"])
                if b2.button("Apply", key=f"a_{key}_{n}", type="primary"):
                    ss.working[key] = CL.apply_op(df, op)
                    ss.cleaned.add(key)
                    D.save_cleaned(key, ss.working[key])
                    bump(key)
                    st.rerun()
            except Exception as e:
                st.error(f"Could not apply: {e}")
    st.caption("All other sections use the cleaned version of this dataset.")


def page_chat(key, df):
    roles = K.detect(df)
    facts = get_facts(key, df, roles)
    hist = ss.chat.setdefault(key, [])
    if not hist:
        st.caption("Try: *What is my total revenue?* · *Which product generated the highest sales?* · "
                   "*Which city performed best?* · *Are there unusual values?* · *Give me recommendations to improve sales*")
    for m in hist:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    q = st.chat_input("Ask a question about your data")
    if not q:
        return
    hist.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    local = F.answer_local(q, df, roles, facts)
    parts = []
    if local:
        parts.append("**✔ Calculated from your data**\n\n" + local)
    if ai_ready():
        p = AI.analyst_prompt(q, facts, local, hist[:-1])
        text = call_ai(AI.ANALYST_SYSTEM, p)
        if text:
            parts.append("**✦ AI interpretation** *(written by Gemini from the calculated facts - verify)*\n\n" + text)
    elif not local:
        parts.append("I can't answer that with the built-in calculations. For open-ended questions and recommendations, "
                     "add a Gemini key in the sidebar - or use the **SQL Assistant** for exact custom numbers.")
    ans = "\n\n".join(parts) or "No answer could be produced."
    hist.append({"role": "assistant", "content": ans})
    with st.chat_message("assistant"):
        st.markdown(ans)


def page_sql(key, df):
    roles = K.detect(df)
    st.caption(f"Queries run read-only on a temporary table named **data** · columns: "
               + ", ".join(f"`{c}`" for c in df.columns))
    if "pending_sql" in ss:
        ss.sql_text = ss.pop("pending_sql")
    ss.setdefault("sql_text", "")
    t_run, t_samples, t_ai = st.tabs(["Run SQL", "Sample queries", "Ask AI to write SQL"])
    with t_run:
        st.text_area("SQL", key="sql_text", height=140, placeholder='SELECT * FROM data LIMIT 10')
        b1, b2 = st.columns(2)
        if b1.button("▶ Run query", type="primary"):
            try:
                out = SQ.run_query(df, ss.sql_text)
                st.success(f"{len(out):,} row(s)" + (f" (showing first {SQ.MAX_ROWS})" if len(out) >= SQ.MAX_ROWS else ""))
                show_df(out)
            except SQ.SQLSafetyError as e:
                st.error(str(e))
        if b2.button("📖 Explain this query") and ss.sql_text.strip():
            for line in SQ.explain(ss.sql_text):
                st.markdown("- " + line)
    with t_samples:
        for i, (title, q) in enumerate(SQ.sample_queries(df, roles)):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{title}**")
            c1.code(q, language="sql")
            if c2.button("Use", key=f"sq_{i}"):
                ss.pending_sql = q
                st.rerun()
    with t_ai:
        if not ai_ready():
            st.info("Add a Gemini API key in the sidebar to generate SQL from a plain-English request.")
        else:
            req = st.text_input("What do you want to find out?", placeholder="Top 3 cities by revenue in March")
            if st.button("Generate SQL") and req:
                text = call_ai(AI.SQL_SYSTEM, AI.schema_text(df, ss.sample_rows) + "\n\nRequest: " + req)
                if text:
                    code = AI.extract_code(text, "sql")
                    tag("ai")
                    st.markdown(text)
                    if code:
                        ss.pending_sql = code
                        st.info("Query copied to the **Run SQL** tab (not executed). Review it, then run.")


def page_python(key, df):
    st.caption("Code runs on a **copy** of your data (`df`). `pd`, `np`, `px` are available. "
               "Put the answer in `result` (or a Plotly figure in `fig`). No imports or file access.")
    if "pending_code" in ss:
        ss.py_code = ss.pop("pending_code")
    ss.setdefault("py_code", "")
    examples = {"Summary statistics": "result = df.describe(include='all').T",
                "Missing values": "result = df.isna().sum().rename('missing').to_frame()",
                "Top 5 rows by first numeric column":
                    "col = df.select_dtypes('number').columns[0]\nresult = df.nlargest(5, col)"}
    t_run, t_ai = st.tabs(["Editor", "Ask AI to write code"])
    with t_run:
        ex = st.selectbox("Load an example", ["-"] + list(examples))
        if ex != "-" and st.button("Load example"):
            ss.pending_code = examples[ex]
            st.rerun()
        st.text_area("Python (pandas)", key="py_code", height=180)
        if st.button("▶ Run code", type="primary"):
            out = PR.run(ss.py_code, df)
            if out["error"]:
                st.error(out["error"])
            if out["stdout"]:
                st.code(out["stdout"])
            r = out["result"]
            if isinstance(r, pd.Series):
                r = r.to_frame()
            if isinstance(r, pd.DataFrame):
                show_df(r)
            elif r is not None:
                st.write(r)
            if out["fig"] is not None:
                try:
                    show_fig(out["fig"], key="py_fig")
                except Exception as e:
                    st.error(f"Could not draw `fig`: {e}")
    with t_ai:
        if not ai_ready():
            st.info("Add a Gemini API key in the sidebar to generate code from a plain-English request.")
        else:
            req = st.text_input("What should the code do?", placeholder="Show monthly revenue as a bar chart")
            if st.button("Generate code") and req:
                text = call_ai(AI.PY_SYSTEM, AI.schema_text(df, ss.sample_rows) + "\n\nRequest: " + req)
                if text:
                    tag("ai")
                    st.markdown(text)
                    code = AI.extract_code(text, "python")
                    if code:
                        ss.pending_code = code
                        st.info("Code copied to the **Editor** tab (not executed). Read it, then run.")


def page_forecast(key, df):
    roles = K.detect(df)
    if not (roles["date"] and roles["revenue"]):
        st.warning("Forecasting needs a date column and a numeric revenue/sales column. "
                   f"Detected: date={roles['date']}, revenue={roles['revenue']}.")
        return
    c = st.columns(3)
    mode = c[0].radio("History granularity", ["Weekly", "Monthly"], horizontal=True)
    freq = "W" if mode == "Weekly" else "M"
    periods = c[1].slider("Periods to forecast", 1, 26 if freq == "W" else 12, 8 if freq == "W" else 6)
    drop = c[2].checkbox("Exclude incomplete last period", value=True)
    res = FC.forecast(df, roles["date"], roles["revenue"], periods, freq, drop)
    for w in res["warnings"]:
        st.warning(w)
    if not res["ok"]:
        st.error(res["message"])
        return
    cur, h, f = K.currency(roles), res["history"], res["forecast"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.index, y=h.values, name="Historical", line=dict(color="#6c8cff")))
    fig.add_trace(go.Scatter(x=f.period, y=f.upper_95, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f.period, y=f.lower_95, fill="tonexty", fillcolor="rgba(56,217,169,.18)",
                             line=dict(width=0), name="95% range"))
    fig.add_trace(go.Scatter(x=f.period, y=f.forecast, name="Forecast", line=dict(color="#38d9a9", dash="dash")))
    CH.style(fig)
    fig.update_layout(title="Historical and forecast revenue", yaxis_tickprefix=cur)
    show_fig(fig, key="fc_fig")
    m = res["metrics"]
    mc = st.columns(3)
    mc[0].metric("Forecast total", f"{cur}{f.forecast.sum():,.0f}")
    mc[1].metric("Back-test MAPE", "N/A" if m["MAPE_pct"] is None else f"{m['MAPE_pct']:.1f}%",
                 help=f"Trend model fitted without the last {m['holdout_periods']} periods, then compared with them.")
    mc[2].metric("History used", f"{len(h)} periods")
    with st.expander("Assumptions and limitations", expanded=True):
        for a in res["assumptions"]:
            st.markdown("- " + a)
    show_df(f.assign(period=f.period.dt.date).round(2))
    st.download_button("⬇ Forecast CSV", X.to_csv_bytes(f), "forecast.csv", "text/csv")


def _prepared(label, key, builder, fname, mime):
    ck = f"{key}|{ss.version.get(key, 0)}|{fname}"
    if st.button(f"Prepare {label}", key="prep_" + ck):
        try:
            with st.spinner("Building..."):
                ss.exports[ck] = builder()
        except X.ExportError as e:
            st.error(str(e))
    if ck in ss.exports:
        st.download_button(f"⬇ Download {label}", ss.exports[ck], fname, mime, key="dl_" + ck)


def page_export(key, df):
    roles = K.detect(df)
    facts = get_facts(key, df, roles)
    kp = X.kpi_summary(df, roles)
    base = key.rsplit(".", 1)[0]
    st.subheader("Data")
    c1, c2 = st.columns(2)
    with c1:
        def xl():
            sheets = {"Data": df, "KPI summary": kp}
            if key in ss.cleaned:
                sheets["Original"] = ss.originals[key]
            return X.to_excel_bytes(sheets)
        _prepared("Excel", key, xl, f"{base}_cleaned.xlsx",
                  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c2:
        _prepared("CSV", key, lambda: X.to_csv_bytes(df), f"{base}_cleaned.csv", "text/csv")
    st.subheader("KPI summary")
    show_df(kp)
    st.download_button("⬇ KPI summary CSV", X.to_csv_bytes(kp), f"{base}_kpis.csv", "text/csv")
    st.subheader("Charts")
    figs = CH.build_charts(df, roles)
    if figs:
        _prepared("all charts (ZIP of interactive HTML)", key, lambda: X.charts_zip(figs), f"{base}_charts.zip", "application/zip")
        st.caption("Charts export as interactive HTML (opens in any browser; needs internet for the Plotly library). "
                   "PNG export would require an extra heavy dependency, so it is not included.")
    else:
        st.info("No charts available for this dataset.")
    st.subheader("PDF report")
    ai_txt = ss.insights.get(key)
    st.caption("Includes KPIs, calculated findings and top lists" + (" plus your AI report (clearly labelled)." if ai_txt
               else ". Generate an AI report in AI Analyst first to include it."))
    def pdf():
        tabs = {f"Top {r}s": pd.DataFrame(facts[f"top_{r}"]) for r in ("product", "category", "city") if facts.get(f"top_{r}")}
        return X.build_pdf("DataPilot AI - Analytics Report", key, kp, tabs, F.insights_text(facts), ai_txt)
    _prepared("PDF report", key, pdf, f"{base}_report.pdf", "application/pdf")


PAGES = {"📈 Dashboard": page_dashboard, "🗂️ Datasets": page_datasets, "🧠 AI Analyst": page_analyst,
         "🧹 Data Cleaning": page_cleaning, "💬 AI Chat": page_chat, "🗄️ SQL Assistant": page_sql,
         "🐍 Python Assistant": page_python, "🔮 Forecasting": page_forecast, "📦 Export Center": page_export}


def main():
    init_state()
    with st.sidebar:
        st.markdown("## 📊 DataPilot AI")
        page = st.radio("Navigate", SECTIONS, label_visibility="collapsed")
        st.divider()
        handle_uploads(st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"], accept_multiple_files=True,
                                        key=f"uploader_{ss.up_n}"))
        names = list(ss.working)
        if names:
            ss.active = st.selectbox("Active dataset", names,
                                     index=names.index(ss.active) if ss.active in names else 0)
    sidebar_ai()
    st.title(page.split(" ", 1)[1])
    if not ss.active or ss.active not in ss.working:
        st.info("Upload a CSV or Excel file in the sidebar to begin.")
        return
    st.caption(f"Active dataset: **{ss.active}**" + (" · cleaned version" if ss.active in ss.cleaned else ""))
    try:
        PAGES[page](ss.active, ss.working[ss.active])
    except Exception as e:                                   # last-resort guard: never show a raw traceback
        st.error(f"Something went wrong in this section: {type(e).__name__}: {e}")


main()
