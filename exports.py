"""Export helpers: Excel, CSV, KPI summary, chart HTML (+ZIP) and a PDF report."""
from __future__ import annotations
import io, re, zipfile
import pandas as pd
from . import kpis as K


class ExportError(Exception):
    pass


def _sheet(name: str) -> str:
    return re.sub(r"[\[\]\*\?/\\:]", "_", name)[:31] or "Sheet"


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        used = set()
        for name, df in sheets.items():
            n, i = _sheet(name), 1
            while n in used:
                i += 1
                n = _sheet(name)[:28] + f"_{i}"
            used.add(n)
            df.to_excel(w, sheet_name=n, index=False)
    return buf.getvalue()


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def kpi_summary(df: pd.DataFrame, roles: dict) -> pd.DataFrame:
    k, cur = K.compute(df, roles), K.currency(roles)
    rows = [("Rows analysed", len(df), "dataset"),
            ("Total revenue", k["revenue"], roles["revenue"] or "column not found"),
            ("Total orders", k["orders"], roles["order_id"] or "1 row = 1 order"),
            ("Total quantity", k["quantity"], roles["quantity"] or "column not found"),
            ("Average order value", k["aov"], "revenue / orders")]
    out = pd.DataFrame(rows, columns=["Metric", "Value", "Source"])
    out["Value"] = [("N/A" if v is None else (f"{cur}{v:,.2f}" if m in ("Total revenue", "Average order value") else f"{v:,.0f}"))
                    for m, v in zip(out.Metric, out.Value)]
    return out


def fig_html(fig) -> bytes:
    return fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")


def charts_zip(figs: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, f in figs.items():
            z.writestr(re.sub(r"[^A-Za-z0-9_-]+", "_", name) + ".html", fig_html(f))
    return buf.getvalue()


def _s(t) -> str:
    t = str(t).replace("₹", "Rs. ").replace("–", "-").replace("—", "-").replace("“", '"').replace("”", '"') \
              .replace("’", "'").replace("•", "-").replace("**", "").replace("`", "")
    t = re.sub(r"^#+\s*", "", t, flags=re.M)
    t = re.sub(r"(\S{60})(?=\S)", r"\1 ", t)          # break very long tokens so the PDF layout never fails
    return t.encode("latin-1", "replace").decode("latin-1")


def build_pdf(title: str, dataset: str, kpis: pd.DataFrame, tables: dict[str, pd.DataFrame],
              findings: str, ai_text: str | None = None) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError as e:
        raise ExportError("PDF export needs the 'fpdf2' package: pip install fpdf2") from e
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin

    def h(txt, size=13):
        pdf.set_font("Helvetica", "B", size)
        pdf.cell(0, 9, _s(txt), new_x="LMARGIN", new_y="NEXT")

    def para(txt):
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5.5, _s(txt), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    def table(df, max_rows=15):
        df = df.head(max_rows)
        cw = W / max(len(df.columns), 1)
        pdf.set_font("Helvetica", "B", 9)
        for c in df.columns:
            pdf.cell(cw, 6, _s(c)[:22], border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        for _, r in df.iterrows():
            for v in r:
                pdf.cell(cw, 6, _s(f"{v:,.2f}" if isinstance(v, float) else v)[:24], border=1)
            pdf.ln()
        pdf.ln(3)

    h(title, 18)
    para(f"Dataset: {dataset}")
    h("Key metrics")
    table(kpis)
    h("Calculated findings")
    para(findings)
    for name, t in tables.items():
        h(name)
        table(t)
    if ai_text:
        h("AI-generated interpretation")
        para("The text below was written by an AI model from the calculated facts above. Verify before relying on it.")
        para(ai_text)
    return bytes(pdf.output())
