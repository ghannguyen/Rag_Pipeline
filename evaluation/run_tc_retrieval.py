#!/usr/bin/env python3
"""
Retrieval-only test of TC_LuatLD.xlsx (no LLM / no Gemini quota).

Runs the same steps as /api/rag/chat up to the LLM context:
hybrid search -> rerank -> RERANK_THRESHOLD -> cross-document dedupe -> top_n.
Loads bge-m3 + the reranker in this process: stop the API server first on
machines with 8 GB RAM.

    python evaluation/run_tc_retrieval.py --input ~/Downloads/TC_LuatLD.xlsx \
        --output evaluation/results/retrieval_new.json
    # A previous index, for regression:
    python evaluation/run_tc_retrieval.py --input ... --label old \
        --vector-dir <backup>/chroma --chunked <backup>/processed/chunked.jsonl \
        --output evaluation/results/retrieval_old.json
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.env import load_env  # noqa: E402

load_env()

from tc_common import keyword_coverage, load_cases, same_doc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--label", default="new")
    ap.add_argument("--vector-dir", type=Path, default=ROOT / "data" / "vectorstore" / "chroma")
    ap.add_argument("--chunked", type=Path, default=ROOT / "data" / "processed" / "chunked.jsonl")
    args = ap.parse_args()

    from src.rag.context_builder import dedupe_with_source_preference, preferred_sources_from_env
    from src.rag.hybrid_retriever import hybrid_search
    from src.rag.reranker import rerank_documents

    top_k = int(os.getenv("RETRIEVAL_TOP_K", "15"))
    top_n = int(os.getenv("RERANK_TOP_N", "5"))
    threshold = float(os.getenv("RERANK_THRESHOLD", "0.35"))
    _, cases = load_cases(args.input.expanduser())

    results, started = [], time.time()
    for case in cases:
        with contextlib.redirect_stdout(io.StringIO()):
            candidates = hybrid_search(case["query"], top_k_dense=top_k, top_k_sparse=top_k,
                                       vector_dir=args.vector_dir, chunked_file=args.chunked)
            # threshold=0 keeps the real best score even when the case would be refused.
            ranked, _ = rerank_documents(case["query"], candidates, top_n=max(len(candidates), 1), threshold=0.0)
            ranked = dedupe_with_source_preference(ranked, preferred_sources_from_env())
        best = ranked[0]["rerank_score"] if ranked else 0.0
        top = [d for d in ranked if d["rerank_score"] >= threshold][:top_n]

        def src(d):
            return (d.get("metadata") or {}).get("source") or ""

        def pages(d):
            m = d.get("metadata") or {}
            return int(m.get("page_start") or m.get("page") or 0), int(m.get("page_end") or m.get("page") or 0)

        in_doc = [d for d in top if same_doc(src(d), case["expected_doc"])]
        page_hit = None if case["page"] is None else any(p0 - 1 <= case["page"] <= p1 + 1 for p0, p1 in map(pages, in_doc))
        results.append({
            "id": case["id"], "group": case["group"], "category": case["category"], "difficulty": case["difficulty"],
            "query": case["query"], "expected_doc": case["expected_doc"], "expected_page": case["page"],
            "passed_threshold": bool(top), "best_rerank": round(best, 4),
            "doc_hit_at1": bool(top) and same_doc(src(top[0]), case["expected_doc"]),
            "doc_hit_at5": bool(in_doc), "page_hit_at5": page_hit,
            "keyword_coverage": keyword_coverage(case["keywords"], " ".join(d.get("content", "") for d in top)),
            "top": [{"source": src(d), "pages": pages(d), "rerank": d.get("rerank_score"), "chunk_id": d.get("chunk_id")} for d in top],
            # Ranking before the threshold cut: tells "not retrieved" apart from "cut by the threshold".
            "ranked": [{"source": src(d), "pages": pages(d), "rerank": d.get("rerank_score"), "chunk_id": d.get("chunk_id")} for d in ranked[:15]],
            "candidates": [d.get("chunk_id") for d in candidates],
        })
        r = results[-1]
        print(f"[{args.label}] {len(results):3d}/{len(cases)} {case['id']} doc@5={int(r['doc_hit_at5'])} "
              f"best={best:.2f} ({time.time() - started:.0f}s)", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "label": args.label, "vector_dir": str(args.vector_dir), "chunked": str(args.chunked),
        "top_k": top_k, "top_n": top_n, "threshold": threshold, "results": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{args.label}] DONE {len(results)} cases -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
