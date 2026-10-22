#!/usr/bin/env python3
"""
Recall@5 / Hit@5 / MRR@5 on gold chunks, plus a per-case failure stage.

Gold chunks are derived per test case (no manual labels exist): among chunks of
the expected document, score = keyword coverage + overlap with the Evidence and
Expected Answer text (+ a bonus near Source Page); the best chunks (<= 3) above
GOLD_MIN are gold. Cases with no chunk above GOLD_MIN are "no gold" and are kept
out of the recall averages. Gold labels are written for a tester to confirm.

    python evaluation/run_tc_recall.py --input ~/Downloads/TC_LuatLD.xlsx \
        --retrieval evaluation/results/2026-10-20/retrieval_new.json \
        --run-dir evaluation/results/2026-10-20_flash-lite
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tc_common import answer_recall, keyword_coverage, load_cases, same_doc  # noqa: E402

GOLD_MIN = 0.5
GOLD_MARGIN = 0.1
GOLD_MAX = 3
K = 5

STAGES = {
    "OK": "Đạt",
    "NO_GOLD": "Không gán được đoạn đúng (câu hỏi ngoài khả năng retrieval / cần xem lại test)",
    "RETRIEVAL_MISS": "Không tìm thấy: đoạn đúng không có trong ứng viên hybrid",
    "RERANK_DROP": "Có trong ứng viên nhưng rerank đẩy khỏi top 5",
    "THRESHOLD_CUT": "Có trong top 5 nhưng điểm rerank dưới ngưỡng 0,35",
    "LLM_REFUSAL": "LLM từ chối dù đoạn đúng có trong ngữ cảnh",
    "LLM_ANSWER": "Đoạn đúng có trong ngữ cảnh nhưng câu trả lời sai/thiếu",
    "WRONG_SOURCE_ANSWER": "Không có đoạn đúng trong ngữ cảnh nhưng LLM vẫn trả lời (từ nguồn khác)",
    "CITATION": "Trả lời đạt nhưng mất trích dẫn",
}


def evidence_text(row):
    text = str(row.get("Evidence") or "")
    m = re.search(r"xác nhận\s*:\s*(.*)", text, re.S | re.I)
    return (m.group(1) if m else text).strip()


def load_status(run_dir):
    """Effective status per case from the results workbook (Final Status, else Suggested)."""
    import openpyxl
    ws = openpyxl.load_workbook(run_dir / "TC_LuatLD_results.xlsx", read_only=True)["Results"]
    rows = ws.iter_rows(values_only=True)
    header = list(next(rows))
    out = {}
    for r in rows:
        d = dict(zip(header, r))
        out[str(d["Test Case ID"])] = (d.get("Final Status") or d.get("Suggested Status") or "", d.get("Cited Expected Doc"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--retrieval", type=Path, required=True, help="run_tc_retrieval.py JSON with ranked lists")
    ap.add_argument("--run-dir", type=Path, required=True, help="run_tc_excel.py output dir")
    ap.add_argument("--chunked", type=Path, default=ROOT / "data" / "processed" / "chunked.jsonl")
    args = ap.parse_args()

    _, cases = load_cases(args.input.expanduser())
    retrieval = {r["id"]: r for r in json.loads(args.retrieval.read_text(encoding="utf-8"))["results"]}
    raw = {}
    for line in (args.run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            raw[rec["id"]] = rec
    status = load_status(args.run_dir)

    by_doc = defaultdict(list)
    with args.chunked.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                by_doc[c["metadata"]["source"]].append(c)

    results = []
    for case in cases:
        target = " ".join([evidence_text(case["row"]), case["expected"]])
        scored = []
        for c in next((v for k, v in by_doc.items() if same_doc(k, case["expected_doc"])), []):
            body = c["chunk_content"].split("\n---\n", 1)[-1]
            kw = keyword_coverage(case["keywords"], body)
            ov = answer_recall(target, body) or 0.0
            s = (0.5 * kw + 0.5 * ov) if kw is not None else ov
            m = c["metadata"]
            if case["page"] is not None and int(m.get("page_start", 0)) - 1 <= case["page"] <= int(m.get("page_end", 0)) + 1:
                s += 0.1
            scored.append((s, c["chunk_id"]))
        scored.sort(reverse=True)
        best = scored[0][0] if scored else 0.0
        gold = [cid for s, cid in scored[:GOLD_MAX] if s >= GOLD_MIN and s >= best - GOLD_MARGIN]

        r = retrieval.get(case["id"], {})
        ranked = [t["chunk_id"] for t in r.get("ranked", [])][:K]
        candidates = r.get("candidates") or []
        resp = (raw.get(case["id"]) or {}).get("response") or {}
        context = [s.get("chunk_id") for s in (resp.get("debug") or {}).get("context_sources") or []]
        answer = resp.get("answer") or ""
        refused = "không tìm thấy thông tin để trả lời" in answer.lower()
        st, cited_doc = status.get(case["id"], ("", None))

        def recall(ids):
            return len(set(gold) & set(ids)) / len(gold) if gold else None

        first = next((i for i, cid in enumerate(ranked, 1) if cid in gold), None)
        gold_in_context = bool(set(gold) & set(context))

        if st == "PASS" and cited_doc == "Yes":
            stage = "OK"
        elif st == "PASS":
            stage = "CITATION"
        elif not gold:
            stage = "NO_GOLD"
        elif gold_in_context:
            stage = "LLM_REFUSAL" if refused else "LLM_ANSWER"
        elif not refused and context:
            stage = "WRONG_SOURCE_ANSWER"
        elif not set(gold) & set(candidates):
            stage = "RETRIEVAL_MISS"
        elif not set(gold) & set(ranked):
            stage = "RERANK_DROP"
        else:
            stage = "THRESHOLD_CUT"

        results.append({
            "id": case["id"], "group": case["group"], "category": case["category"], "difficulty": case["difficulty"],
            "status": st, "stage": stage, "gold": gold, "gold_score": round(best, 3),
            "recall_at5_context": recall(context), "recall_at5_ranked": recall(ranked),
            "hit_at5_ranked": None if not gold else bool(first), "rr_at5": None if not gold else (1 / first if first else 0.0),
            "gold_in_candidates": None if not gold else bool(set(gold) & set(candidates)),
            "context": context, "ranked_top5": ranked,
        })

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    def agg(rs):
        g = [r for r in rs if r["gold"]]
        return {"n": len(rs), "with_gold": len(g),
                "recall_at5_context": mean(r["recall_at5_context"] for r in g),
                "recall_at5_ranked": mean(r["recall_at5_ranked"] for r in g),
                "hit_at5_ranked": mean(float(r["hit_at5_ranked"]) for r in g),
                "mrr_at5": mean(r["rr_at5"] for r in g),
                "gold_in_candidates": mean(float(r["gold_in_candidates"]) for r in g)}

    summary = {"all": agg(results),
               "by_group": {k: agg([r for r in results if r["group"] == k]) for k in sorted({r["group"] for r in results})},
               "by_difficulty": {k: agg([r for r in results if r["difficulty"] == k]) for k in ("Easy", "Medium", "Hard")},
               "stages": dict(Counter(r["stage"] for r in results).most_common()),
               "stage_labels": STAGES,
               "method": {"gold_min": GOLD_MIN, "gold_margin": GOLD_MARGIN, "gold_max": GOLD_MAX, "k": K}}
    out = args.run_dir / "recall_at5.json"
    out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")

    a = summary["all"]
    print(f"cases with gold: {a['with_gold']}/{a['n']}")
    print(f"Recall@5 (context sent to LLM) {a['recall_at5_context']:.1%} | Recall@5 (ranking before threshold) {a['recall_at5_ranked']:.1%} | "
          f"Hit@5 {a['hit_at5_ranked']:.1%} | MRR@5 {a['mrr_at5']:.3f} | gold in hybrid candidates {a['gold_in_candidates']:.1%}")
    for k, v in summary["by_group"].items():
        print(f"  {k}: gold {v['with_gold']}/{v['n']} R@5ctx {v['recall_at5_context'] or 0:.0%} R@5rank {v['recall_at5_ranked'] or 0:.0%} MRR {v['mrr_at5'] or 0:.2f}")
    for k, v in summary["by_difficulty"].items():
        print(f"  {k}: gold {v['with_gold']}/{v['n']} R@5ctx {v['recall_at5_context'] or 0:.0%} R@5rank {v['recall_at5_ranked'] or 0:.0%}")
    print("stages:", summary["stages"])
    print("->", out)


if __name__ == "__main__":
    main()
