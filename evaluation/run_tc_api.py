#!/usr/bin/env python3
"""
API tests (QA process group 10) against a running backend.

Status codes, validation, input edge cases, provider handling, LLM failure and
the response schema. Only API_17 and API_18 may call Gemini (2 requests at most).

    python evaluation/run_tc_api.py --base http://127.0.0.1:8001 \
        --output evaluation/results/2026-10-20_flash-lite/api_tests.json
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

FRIENDLY_EMPTY = "vui lòng nhập câu hỏi"
RESPONSE_KEYS = {"query", "answer", "citations", "image_urls", "total_duration_ms", "logs"}
CITATION_KEYS = {"source_id", "file", "page"}


def call(base, method, path, body=None, raw=None, timeout=240):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, text = r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        status, text = e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # connection error, timeout
        status, text = None, f"{type(e).__name__}: {e}"
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    return status, payload, text, round(time.time() - started, 1)


def chat(q, **extra):
    return ("POST", "/api/rag/chat", {"query": q, **extra}, None)


def answer_of(p):
    return ((p or {}).get("answer") or "").lower()


# (id, title, request, check(status, payload) -> (ok, expected text))
CASES = [
    ("API_01", "Health", ("GET", "/health", None, None),
     lambda s, p: (s == 200 and (p or {}).get("status") == "ok", "200, status=ok")),
    ("API_02", "Index status matches the build", ("GET", "/api/pipeline/status", None, None),
     lambda s, p: (s == 200 and p["pdf_documents"] == 5 and p["chunks"] == p["vectors"] > 0,
                   "200, 5 documents, chunks == vectors")),
    ("API_03", "Chat without body", ("POST", "/api/rag/chat", None, b""),
     lambda s, p: (s == 422, "422")),
    ("API_04", "Chat with empty object", ("POST", "/api/rag/chat", {}, None),
     lambda s, p: (s == 422, "422 (query is required)")),
    ("API_05", "Malformed JSON", ("POST", "/api/rag/chat", None, b"{query: "),
     lambda s, p: (s == 422, "422")),
    ("API_06", "query is not a string", chat(123),
     lambda s, p: (s == 422, "422")),
    ("API_07", "top_k = 0", chat("thời giờ làm việc", top_k=0),
     lambda s, p: (s == 422, "422 (top_k >= 1)")),
    ("API_08", "top_k = 31", chat("thời giờ làm việc", top_k=31),
     lambda s, p: (s == 422, "422 (top_k <= 30)")),
    ("API_09", "threshold = 1.5", chat("thời giờ làm việc", threshold=1.5),
     lambda s, p: (s == 422, "422 (0 <= threshold <= 1)")),
    ("API_10", "top_n = 0", chat("thời giờ làm việc", top_n=0),
     lambda s, p: (s == 422, "422 (top_n >= 1)")),
    ("API_11", "Wrong method", ("GET", "/api/rag/chat", None, None),
     lambda s, p: (s == 405, "405")),
    ("API_12", "Empty query", chat(""),
     lambda s, p: (s in (400, 422) or (s == 200 and FRIENDLY_EMPTY in answer_of(p)), "400/422 or 200 with a friendly message")),
    ("API_13", "Whitespace-only query", chat("   \n  "),
     lambda s, p: (s in (400, 422) or (s == 200 and FRIENDLY_EMPTY in answer_of(p)), "400/422 or 200 with a friendly message")),
    ("API_14", "Unknown provider", chat("Người lao động được nghỉ hằng năm bao nhiêu ngày?", provider="foo"),
     lambda s, p: (s in (400, 422), "400/422 - must not silently answer with another provider")),
    ("API_15", "Client selects the mock provider", chat("Người lao động được nghỉ hằng năm bao nhiêu ngày?", provider="mock"),
     lambda s, p: (s in (400, 403, 422), "400/403/422 - production must not let clients pick mock")),
    ("API_16", "LLM backend unavailable (ollama not running)", chat("Người lao động được nghỉ hằng năm bao nhiêu ngày?", provider="ollama"),
     lambda s, p: (s in (502, 503), "502/503 with a clear message, not 500")),
    ("API_17", "5,000-character query", chat(("Người lao động được nghỉ hằng năm bao nhiêu ngày? " * 100)[:5000]),
     lambda s, p: (s in (200, 413, 422), "200, or 413/422 if a length limit is set; never 500")),
    ("API_18", "Response schema on a normal question", chat("Người lao động làm việc trong điều kiện bình thường được nghỉ hằng năm bao nhiêu ngày?"),
     lambda s, p: (s == 200 and RESPONSE_KEYS <= set(p or {}) and all(CITATION_KEYS <= set(c) for c in p["citations"])
                   and bool(p["citations"]), "200, keys " + ", ".join(sorted(RESPONSE_KEYS)) + "; >= 1 citation with source_id/file/page")),
    ("API_19", "Emoji / special characters", chat("😀😀 <script>alert(1)</script> ' OR 1=1 --"),
     lambda s, p: (s == 200, "200, no 500")),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ids", default="", help="comma-separated subset, e.g. API_01,API_02")
    args = ap.parse_args()
    only = {i.strip() for i in args.ids.split(",") if i.strip()}

    results = []
    for cid, title, (method, path, body, raw), check in CASES:
        if only and cid not in only:
            continue
        status, payload, text, secs = call(args.base, method, path, body, raw)
        try:
            ok, expected = check(status, payload)
        except Exception as e:
            ok, expected = False, f"check failed: {type(e).__name__}: {e}"
        provider = next((l.get("provider") for l in (payload or {}).get("logs") or [] if l.get("stage") == "llm_generation"), None) \
            if isinstance(payload, dict) else None
        results.append({"id": cid, "title": title, "request": f"{method} {path} {json.dumps(body, ensure_ascii=False)[:120] if body is not None else (raw or b'').decode()}",
                        "expected": expected, "status": status, "seconds": secs, "result": "PASS" if ok else "FAIL",
                        "llm_provider": provider, "body": text[:600]})
        r = results[-1]
        print(f"{cid} {r['result']:4} {status} {secs:5}s {title} | {provider or ''} | {text[:140]!r}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"base": args.base, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{sum(r['result'] == 'PASS' for r in results)}/{len(results)} PASS -> {args.output}")


if __name__ == "__main__":
    main()
