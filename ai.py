"""Google Gemini via the official REST API (stdlib only - no extra dependency).
Privacy by default: callers send schema + computed aggregates, not raw rows (rows only if the user opts in)."""
from __future__ import annotations
import json, os, re, time
import urllib.error, urllib.request

BASE = "https://generativelanguage.googleapis.com/v1beta"
_SKIP = ("image", "tts", "embedding", "audio", "veo", "imagen", "robotics", "live", "computer-use", "aqa")

ANALYST_SYSTEM = (
    "You are a careful business data analyst inside an analytics app. You are given VERIFIED FACTS computed with "
    "pandas from the user's dataset. Rules: (1) Use ONLY numbers present in the facts; never invent or estimate "
    "statistics. (2) If the facts do not contain what is needed, say exactly that and suggest using the SQL or "
    "Python assistant. (3) Clearly separate interpretation from calculation: begin interpretive statements with "
    "'Interpretation:' and recommendations with 'Suggestion:'. (4) Explain in simple language for a non-expert. "
    "(5) Mention data limitations (small sample, partial period, missing columns) when relevant."
)
SQL_SYSTEM = (
    "You write SQLite SELECT queries. The only table is `data`. Use ONLY the listed columns, double-quoted. "
    "Return exactly one query in a ```sql fenced block, then a short plain-English explanation. Read-only: SELECT only."
)
PY_SYSTEM = (
    "You write short pandas code. The DataFrame is already loaded as `df`; `pd`, `np` and `px` (plotly.express) are "
    "available. Do NOT import anything, read/write files or use the network. Put the final answer in a variable named "
    "`result` (DataFrame/Series/number) or a Plotly figure in `fig`. Return one ```python fenced block, then a short "
    "explanation. Use ONLY the listed columns."
)


class AIError(Exception):
    pass


def get_env_key() -> str:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


def _request(url: str, key: str, body: dict | None = None, timeout: int = 60) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt == 0:
                time.sleep(3)
                continue
            raise AIError(_http_message(e.code, e.read().decode(errors="ignore"))) from e
        except urllib.error.URLError as e:
            raise AIError(f"Network error: {e.reason}. Check your internet connection.") from e
        except (TimeoutError, json.JSONDecodeError) as e:
            raise AIError(f"Request failed: {e}") from e
    raise AIError("Request failed after retry.")


def _http_message(code: int, body: str) -> str:
    detail = ""
    try:
        detail = json.loads(body).get("error", {}).get("message", "")
    except Exception:
        pass
    msg = {400: "Bad request (check the model name or request).",
           401: "API key rejected.", 403: "API key not allowed for this model/region, or the API is not enabled.",
           404: "Model not found - pick another model from the list.",
           429: "Rate limit or free-tier quota reached. Wait a minute, or pick another model.",
           500: "Google server error - try again.", 503: "Model overloaded - try again shortly."}.get(code, f"HTTP {code}")
    return f"{msg} {detail[:200]}".strip()


def list_models(key: str) -> list[str]:
    if not key:
        raise AIError("No API key provided.")
    names, token = [], ""
    for _ in range(5):
        data = _request(f"{BASE}/models?pageSize=100" + (f"&pageToken={token}" if token else ""), key)
        for m in data.get("models", []):
            n = m.get("name", "").removeprefix("models/")
            if "generateContent" in m.get("supportedGenerationMethods", []) and n.startswith("gemini") \
                    and not any(x in n for x in _SKIP):
                names.append(n)
        token = data.get("nextPageToken", "")
        if not token:
            break
    return sorted(set(names))


def default_model(models: list[str]) -> str | None:
    env = os.getenv("GEMINI_MODEL")
    if env:
        return env
    for pref in ("flash-lite", "flash"):
        for m in models:
            if pref in m and "preview" not in m and "exp" not in m:
                return m
    return models[0] if models else None


def generate(key: str, model: str, system: str, prompt: str, temperature: float = 0.2) -> str:
    if not key or not model:
        raise AIError("Set a Gemini API key and model in the sidebar first.")
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": 4096}}
    data = _request(f"{BASE}/models/{model}:generateContent", key, body)
    fb = data.get("promptFeedback", {})
    if fb.get("blockReason"):
        raise AIError(f"The request was blocked ({fb['blockReason']}).")
    cands = data.get("candidates") or []
    if not cands:
        raise AIError("The model returned no answer.")
    parts = cands[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    if not text:
        raise AIError(f"The model returned an empty answer (finish reason: {cands[0].get('finishReason', 'unknown')}).")
    return text


def extract_code(text: str, lang: str) -> str | None:
    m = re.search(rf"```{lang}\s*\n(.*?)```", text, re.S | re.I) or re.search(r"```\w*\s*\n(.*?)```", text, re.S)
    return m.group(1).strip() if m else None


def schema_text(df, sample_rows: int = 0) -> str:
    cols = "\n".join(f"- {c} ({t})" for c, t in df.dtypes.items())
    s = f"Rows: {len(df)}\nColumns:\n{cols}"
    if sample_rows:
        s += "\nSample rows (user opted in):\n" + df.head(sample_rows).to_csv(index=False)
    return s


def analyst_prompt(question: str, facts: dict, local_answer: str | None = None, history: list | None = None) -> str:
    p = "VERIFIED FACTS (JSON):\n" + json.dumps(facts, default=str)[:24000]
    if local_answer:
        p += "\n\nCALCULATED ANSWER ALREADY SHOWN TO USER:\n" + local_answer
    if history:
        p += "\n\nRECENT CONVERSATION:\n" + "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-6:])
    return p + "\n\nQUESTION:\n" + question
