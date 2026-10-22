#!/usr/bin/env python3
"""
RAGAS scoring of a run_tc_excel.py run, on the exact context the LLM received.

Contexts are rebuilt from debug.context_sources[].chunk_id in <run-dir>/raw.jsonl
and the chunk text in data/processed/chunked.jsonl, so no chat call is repeated.
Metrics: Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference,
LLMContextRecall (judge = Gemini via langchain-google-genai).

Needs the eval environment (ragas, langchain-google-genai, openpyxl):
    .venv-eval/bin/python evaluation/run_tc_ragas.py --input ~/Downloads/TC_LuatLD.xlsx \
        --run-dir evaluation/results/2026-10-20_flash-lite --judge-model gemini-3.5-flash-lite
Resumable: re-running skips cases already in <run-dir>/ragas.jsonl.
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from src.env import load_env  # noqa: E402
from tc_common import load_cases  # noqa: E402

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
QUOTA_MARKERS = ("429", "RESOURCE_EXHAUSTED", "quota")


def interleave(cases):
    """A1, B1, C1, D1, E1, A2, ... so a partial run stays balanced across documents."""
    groups = defaultdict(list)
    for c in cases:
        groups[c["group"]].append(c)
    order = []
    for i in range(max(len(v) for v in groups.values())):
        for g in sorted(groups):
            if i < len(groups[g]):
                order.append(groups[g][i])
    return order


def make_judge(args):
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set (.env)")
    config = RunConfig(timeout=args.timeout, max_retries=3, max_wait=60, max_workers=1)
    limiter = InMemoryRateLimiter(requests_per_second=args.rps, check_every_n_seconds=0.2, max_bucket_size=1)
    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model=args.judge_model, google_api_key=key, temperature=0, max_output_tokens=8192,
        timeout=args.timeout, max_retries=2, rate_limiter=limiter), run_config=config)
    emb = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
        model=args.embedding_model, google_api_key=key, task_type="SEMANTIC_SIMILARITY"), run_config=config)
    return llm, emb, config


def score(case, contexts, answer, judge):
    from langchain_core.callbacks import BaseCallbackHandler
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    class Errors(BaseCallbackHandler):
        def __init__(self):
            self.messages = []

        def on_chain_error(self, error, **kwargs):
            self.messages.append(f"{type(error).__name__}: {str(error)[:400]}")

        def on_llm_error(self, error, **kwargs):
            self.messages.append(f"{type(error).__name__}: {str(error)[:400]}")

    llm, emb, config = judge
    metrics = [Faithfulness(), ResponseRelevancy(strictness=1), LLMContextPrecisionWithReference(), LLMContextRecall()]
    data = EvaluationDataset.from_list([{"user_input": case["query"], "reference": case["expected"],
                                         "response": answer, "retrieved_contexts": contexts}])
    errors = Errors()
    result = evaluate(data, metrics=metrics, llm=llm, embeddings=emb, run_config=config,
                      callbacks=[errors], raise_exceptions=False, show_progress=False)
    raw = result.scores[0]
    values = {}
    for name, metric in zip(METRICS, metrics):
        v = raw.get(metric.name)
        values[name] = float(v) if v is not None and math.isfinite(float(v)) else None
    return values, list(dict.fromkeys(errors.messages))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True, help="run_tc_excel.py output dir (raw.jsonl)")
    ap.add_argument("--chunked", type=Path, default=ROOT / "data" / "processed" / "chunked.jsonl")
    ap.add_argument("--judge-model", default="gemini-3.5-flash-lite")
    ap.add_argument("--embedding-model", default="models/gemini-embedding-001")
    ap.add_argument("--rps", type=float, default=0.2, help="judge requests per second")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    load_env()

    _, cases = load_cases(args.input.expanduser())
    cases = interleave(cases)
    if args.limit:
        cases = cases[: args.limit]
    raw = {}
    for line in (args.run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            raw[rec["id"]] = rec
    chunks = {}
    with args.chunked.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                chunks[rec["chunk_id"]] = rec["chunk_content"]

    out_path = args.run_dir / "ragas.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec["status"] in ("OK", "NO_CONTEXT", "PARTIAL_METRICS"):
                    done.add(rec["id"])
    todo = [c for c in cases if c["id"] not in done]
    print(f"{len(done)} scored, {len(todo)} to score (judge {args.judge_model})", flush=True)

    judge = None
    streak = 0
    with out_path.open("a", encoding="utf-8") as out:
        for i, case in enumerate(todo, 1):
            resp = (raw.get(case["id"]) or {}).get("response")
            rec = {"id": case["id"], "group": case["group"], "judge_model": args.judge_model,
                   "category": case["category"], "difficulty": case["difficulty"]}
            started = time.time()
            if not resp:
                rec.update(status="NO_RESPONSE", **dict.fromkeys(METRICS))
            else:
                sources = (resp.get("debug") or {}).get("context_sources") or []
                contexts = [chunks[s["chunk_id"]] for s in sources if s.get("chunk_id") in chunks]
                missing = [s.get("chunk_id") for s in sources if s.get("chunk_id") not in chunks]
                rec["n_contexts"] = len(contexts)
                if missing:
                    rec["missing_chunks"] = missing
                if not contexts:
                    # Early exit: nothing reached the LLM, so nothing relevant was retrieved for it.
                    rec.update(status="NO_CONTEXT", faithfulness=None, answer_relevancy=None,
                               context_precision=0.0, context_recall=0.0)
                else:
                    judge = judge or make_judge(args)
                    values, errors = score(case, contexts, resp.get("answer") or "", judge)
                    rec.update(values)
                    got = [v for v in values.values() if v is not None]
                    rec["status"] = "OK" if len(got) == len(METRICS) else ("PARTIAL_METRICS" if got else "ERROR")
                    if errors:
                        rec["errors"] = errors[:5]
            rec["seconds"] = round(time.time() - started, 1)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{i}/{len(todo)}] {case['id']} {rec['status']} " +
                  " ".join(f"{m[:5]}={rec.get(m) if rec.get(m) is None else round(rec[m], 2)}" for m in METRICS) +
                  f" ({rec['seconds']}s)", flush=True)
            quota_hit = any(q.lower() in " ".join(rec.get("errors", [])).lower() for q in QUOTA_MARKERS)
            streak = streak + 1 if (rec["status"] in ("ERROR", "PARTIAL_METRICS") and quota_hit) else 0
            if streak >= 3:
                print("Stopping: judge quota exhausted 3 times in a row. Re-run later to resume.", flush=True)
                break


if __name__ == "__main__":
    main()
