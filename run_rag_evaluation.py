"""
================================================================================
RAG PIPELINE — TỰ ĐỘNG CHẠY TEST CASE VÀ XUẤT DASHBOARD ĐÁNH GIÁ
================================================================================

Script này đọc bộ test case (.xlsx bạn đã có sẵn), gửi từng câu hỏi vào hệ
thống RAG của bạn, so sánh câu trả lời thực tế với "Expected Answer", rồi
tổng hợp thành 4 tiêu chí giống dashboard bạn đang có:

  1. Similarity Score      — độ khớp ngữ nghĩa giữa câu trả lời thực tế và
                              câu trả lời kỳ vọng
  2. Pass / Fail rate      — % test case đạt ngưỡng similarity
  3. Latency               — thời gian phản hồi mỗi câu hỏi
  4. Early Exit detection  — hệ thống có từ chối trả lời đúng lúc khi tài
                              liệu không có thông tin không (chống hallucination)

────────────────────────────────────────────────────────────────────────────
BẠN CHỈ CẦN SỬA 1 CHỖ DUY NHẤT: hàm `call_rag_system()` bên dưới.
Đó là nơi gọi vào pipeline RAG thật của bạn (API, hoặc gọi thẳng hàm Python).
Phần còn lại của script không cần đụng vào.
────────────────────────────────────────────────────────────────────────────

Cách chạy:
    pip install -r requirements.txt
    python run_rag_evaluation.py --input RAG_TestCase.xlsx

Kết quả:
    - results_raw.csv              : chi tiết từng test case (để soi lỗi)
    - results_raw.json              : dữ liệu thô dạng JSON
    - dashboard.html                 : dashboard trực quan, mở bằng trình duyệt
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

# ==============================================================================
# 1. CẤU HÌNH — chỉnh các thông số này cho phù hợp với hệ thống của bạn
# ==============================================================================

# Ngưỡng similarity để tính Pass/Fail (0.0 - 1.0). Dashboard cũ của bạn có vẻ
# đang dùng khoảng 0.6-0.65. Hãy tự canh chỉnh theo yêu cầu chất lượng thực tế.
SIMILARITY_THRESHOLD = 0.65

# Phương pháp tính similarity:
#   "embedding" -> dùng model embedding đa ngôn ngữ (chính xác hơn, cần tải
#                  model từ HuggingFace lần đầu chạy, cần internet)
#   "tfidf"     -> dùng TF-IDF + cosine (chạy offline hoàn toàn, nhẹ, độ
#                  chính xác thấp hơn embedding nhưng đủ dùng để sàng lọc)
SIMILARITY_METHOD = "tfidf"  # đổi thành "tfidf" nếu máy không có mạng

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Các cụm từ cho thấy hệ thống đang "từ chối trả lời vì thiếu thông tin"
# (Early Exit). Bổ sung thêm các câu mà hệ thống của BẠN hay dùng.
EARLY_EXIT_PHRASES = [
    "không đủ thông tin",
    "không có thông tin",
    "tài liệu không đề cập",
    "tài liệu không cung cấp",
    "không tìm thấy thông tin",
    "không thể trả lời",
    "xin lỗi, tôi không biết",
    "không có trong tài liệu",
    "insufficient information",
    "i don't know",
    "i cannot find",
]


# ==============================================================================
# 2. NƠI DUY NHẤT BẠN CẦN SỬA — kết nối vào RAG pipeline thật của bạn
# ==============================================================================
def call_rag_system(query: str) -> str:
    import requests

    resp = requests.post(
        "http://127.0.0.1:8001/api/rag/chat",
        json={
            "query": query,
            "top_k": 15,
            "top_n": 5,
            "threshold": 0.35,
            "provider": "gemini",
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()

# ==============================================================================
# 3. ĐỌC BỘ TEST CASE TỪ FILE EXCEL
# ==============================================================================
def load_test_cases(xlsx_path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path)
    frames = []
    for sheet_name in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=sheet_name, header=0)
        # Dòng đầu tiên của mỗi sheet là header thật (do file gốc merge cell)
        raw.columns = raw.iloc[0]
        df = raw[1:].reset_index(drop=True)
        df = df.dropna(subset=["Test Case ID"])
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Chuẩn hoá cột "Is Answerable" — sheet 2 không có cột này, suy luận từ Category
    if "Is Answerable" not in combined.columns:
        combined["Is Answerable"] = "Yes"
    else:
        combined["Is Answerable"] = combined["Is Answerable"].fillna(
            combined["Category"].astype(str).apply(
                lambda c: "No" if "unanswerable" in c.lower() else "Yes"
            )
        )
    # Với các dòng đến từ sheet không có sẵn cột này, suy luận theo Category
    mask_unknown = combined["Is Answerable"].isna()
    combined.loc[mask_unknown, "Is Answerable"] = combined.loc[mask_unknown, "Category"].astype(
        str
    ).apply(lambda c: "No" if "unanswerable" in c.lower() else "Yes")

    return combined


# ==============================================================================
# 4. TÍNH SIMILARITY
# ==============================================================================
class SimilarityScorer:
    def __init__(self, method: str):
        self.method = method
        if method == "embedding":
            from sentence_transformers import SentenceTransformer, util

            print(f"Đang tải embedding model: {EMBEDDING_MODEL_NAME} ...")
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            self.util = util
        elif method == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            self.TfidfVectorizer = TfidfVectorizer
            self.cosine_similarity = cosine_similarity
        else:
            raise ValueError(f"SIMILARITY_METHOD không hợp lệ: {method}")

    def score(self, expected: str, actual: str) -> float:
        expected = str(expected or "").strip()
        actual = str(actual or "").strip()
        if not expected or not actual:
            return 0.0

        if self.method == "embedding":
            emb = self.model.encode([expected, actual], convert_to_tensor=True)
            sim = self.util.cos_sim(emb[0], emb[1]).item()
            return max(0.0, min(1.0, sim))
        else:
            vec = self.TfidfVectorizer().fit_transform([expected, actual])
            sim = self.cosine_similarity(vec[0], vec[1])[0][0]
            return max(0.0, min(1.0, float(sim)))


def is_early_exit(answer: str) -> bool:
    answer_lower = str(answer or "").lower()
    return any(phrase in answer_lower for phrase in EARLY_EXIT_PHRASES)


# ==============================================================================
# 5. CHẠY TOÀN BỘ TEST CASE
# ==============================================================================
def run_evaluation(df: pd.DataFrame, scorer: SimilarityScorer) -> list[dict]:
    results = []
    total = len(df)

    for i, row in df.iterrows():
        tc_id = row["Test Case ID"]
        query = str(row["Query"])
        expected = str(row.get("Expected Answer", ""))
        should_be_answerable = str(row.get("Is Answerable", "Yes")).strip().lower() == "yes"

        print(f"[{i + 1}/{total}] {tc_id} ... ", end="", flush=True)

        t0 = time.time()
        try:
            actual = call_rag_system(query)
            error = None
        except Exception as e:  # noqa: BLE001
            actual = ""
            error = str(e)
        latency = time.time() - t0

        sim_score = scorer.score(expected, actual) if not error else 0.0
        exited_early = is_early_exit(actual)

        # Logic Pass/Fail:
        #  - Câu hỏi KHÔNG có câu trả lời trong tài liệu (Is Answerable = No):
        #       Pass nếu hệ thống early-exit đúng; Fail nếu hệ thống "bịa" câu trả lời.
        #  - Câu hỏi CÓ câu trả lời:
        #       Fail nếu hệ thống early-exit nhầm (bỏ sót context có sẵn),
        #       ngược lại Pass nếu similarity >= threshold.
        if error:
            passed = False
        elif not should_be_answerable:
            passed = exited_early
        else:
            passed = (not exited_early) and (sim_score >= SIMILARITY_THRESHOLD)

        results.append(
            {
                "test_case_id": tc_id,
                "category": row.get("Category", ""),
                "query": query,
                "expected_answer": expected,
                "actual_answer": actual,
                "is_answerable": should_be_answerable,
                "early_exit": exited_early,
                "similarity_score": round(sim_score, 4),
                "latency_sec": round(latency, 3),
                "passed": passed,
                "error": error,
            }
        )
        print(f"sim={sim_score:.2f}  latency={latency:.1f}s  {'PASS' if passed else 'FAIL'}")

    return results


# ==============================================================================
# 6. TỔNG HỢP + XUẤT DASHBOARD HTML
# ==============================================================================
def build_dashboard(results: list[dict], out_path: str):
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    pass_rate = round(100 * passed / total, 2) if total else 0

    avg_sim = round(sum(r["similarity_score"] for r in results) / total, 4) if total else 0
    avg_latency = round(sum(r["latency_sec"] for r in results) / total, 2) if total else 0

    early_exit_cases = [r for r in results if r["early_exit"]]
    early_exit_count = len(early_exit_cases)
    early_exit_rate = round(100 * early_exit_count / total, 2) if total else 0

    # true vs false early exit (chỉ có ý nghĩa khi should-be-unanswerable đã biết)
    true_early_exit = sum(1 for r in results if r["early_exit"] and not r["is_answerable"])
    false_early_exit = sum(1 for r in results if r["early_exit"] and r["is_answerable"])

    # Histogram similarity theo 5 khoảng, giống dashboard gốc
    bins = ["0 - 0.2", "0.2 - 0.4", "0.4 - 0.6", "0.6 - 0.8", "0.8 - 1.0"]
    hist = [0, 0, 0, 0, 0]
    for r in results:
        s = r["similarity_score"]
        idx = min(int(s // 0.2), 4)
        hist[idx] += 1

    latency_labels = [r["test_case_id"] for r in results]
    latency_values = [r["latency_sec"] for r in results]

    fail_rows = "".join(
        f"""<tr>
            <td>{r['test_case_id']}</td>
            <td>{r['category']}</td>
            <td>{r['query']}</td>
            <td>{r['similarity_score']}</td>
            <td>{'Có' if r['early_exit'] else 'Không'}</td>
            <td>{(r['error'] or '')}</td>
        </tr>"""
        for r in results
        if not r["passed"]
    )

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Dashboard Kết Quả Đánh Giá RAG Pipeline</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#f4f6f8; margin:0; padding:24px; color:#1f2933; }}
  h1 {{ text-align:center; color:#1e2a3a; }}
  .summary {{ display:flex; gap:16px; flex-wrap:wrap; justify-content:center; margin:24px 0; }}
  .card {{ background:#fff; border-radius:8px; padding:16px 24px; box-shadow:0 1px 3px rgba(0,0,0,.1); text-align:center; min-width:150px; border-top:4px solid #2b6cb0; }}
  .card h3 {{ margin:0; font-size:12px; letter-spacing:.05em; color:#718096; text-transform:uppercase; }}
  .card p {{ margin:8px 0 0; font-size:26px; font-weight:700; color:#1e2a3a; }}
  .section-header {{ background:#243044; color:#fff; padding:14px 20px; border-radius:6px; font-size:20px; font-weight:700; margin:28px 0 12px; }}
  .panel {{ background:#fff; border-radius:8px; padding:20px; box-shadow:0 1px 3px rgba(0,0,0,.08); display:flex; gap:24px; flex-wrap:wrap; }}
  .panel .text {{ flex:1; min-width:280px; }}
  .panel .chart {{ flex:1; min-width:280px; max-width:420px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:12px; font-size:14px; }}
  th, td {{ border:1px solid #e2e8f0; padding:8px; text-align:left; vertical-align:top; }}
  th {{ background:#edf2f7; }}
  .red {{ color:#c53030; font-weight:700; }}
</style>
</head>
<body>
<h1>Dashboard Kết Quả Đánh Giá RAG Pipeline</h1>

<div class="summary">
  <div class="card"><h3>Tổng test cases</h3><p>{total}</p></div>
  <div class="card"><h3>Tỷ lệ đạt</h3><p>{pass_rate}%</p></div>
  <div class="card"><h3>Similarity Score</h3><p>{avg_sim}</p></div>
  <div class="card"><h3>Độ trễ trung bình</h3><p>{avg_latency}s</p></div>
  <div class="card"><h3>Số lần thoát sớm</h3><p>{early_exit_count}</p></div>
</div>

<div class="section-header">Tiêu chí 1: Điểm tương đồng (Similarity Score)</div>
<div class="panel">
  <div class="text">
    <b>Điểm trung bình toàn hệ thống:</b> <span class="red">{avg_sim}</span> (thang 0-1)<br><br>
    Phân bố theo 5 khoảng điểm được thể hiện ở biểu đồ bên cạnh.
  </div>
  <div class="chart"><canvas id="simHist"></canvas></div>
</div>

<div class="section-header">Tiêu chí 2: Tỷ lệ Đạt / Không đạt (Pass / Fail)</div>
<div class="panel">
  <div class="text">
    <b>Tổng số test cases:</b> {total}<br>
    <b>Số câu Đạt (Pass):</b> {passed}<br>
    <b>Số câu Không đạt (Fail):</b> {failed}<br>
    <b>Tỷ lệ thành công:</b> <span class="red">{pass_rate}%</span>
  </div>
  <div class="chart"><canvas id="passFail"></canvas></div>
</div>

<div class="section-header">Tiêu chí 3: Thời gian phản hồi (Latency)</div>
<div class="panel">
  <div class="text">
    <b>Thời gian trung bình:</b> <span class="red">{avg_latency}s</span> / test case
  </div>
  <div class="chart" style="max-width:100%;flex:2;"><canvas id="latencyChart"></canvas></div>
</div>

<div class="section-header">Tiêu chí 4: Nhận diện thiếu thông tin (Early Exit)</div>
<div class="panel">
  <div class="text">
    <b>Số lần nhận diện thiếu thông tin (Early Exit):</b> {early_exit_count} / {total} ({early_exit_rate}%)<br>
    <b>Early Exit đúng (câu hỏi thực sự không có trong tài liệu):</b> <span class="red">{true_early_exit}</span><br>
    <b>Early Exit sai (bỏ sót context dù tài liệu có đủ thông tin — false early exit):</b> <span class="red">{false_early_exit}</span>
  </div>
  <div class="chart"><canvas id="earlyExit"></canvas></div>
</div>

<div class="section-header">Chi tiết các test case Fail (để điều tra nguyên nhân)</div>
<table>
  <tr><th>ID</th><th>Category</th><th>Câu hỏi</th><th>Similarity</th><th>Early Exit</th><th>Lỗi</th></tr>
  {fail_rows if fail_rows else '<tr><td colspan="6">Không có test case fail 🎉</td></tr>'}
</table>

<script>
new Chart(document.getElementById('simHist'), {{
  type: 'bar',
  data: {{ labels: {json.dumps(bins)}, datasets: [{{ label: 'Số lượng câu trả lời', data: {json.dumps(hist)}, backgroundColor: '#9f7aea' }}] }},
  options: {{ plugins: {{ legend: {{ display: true }} }} }}
}});
new Chart(document.getElementById('passFail'), {{
  type: 'doughnut',
  data: {{ labels: ['Đạt (Pass)', 'Không đạt (Fail)'], datasets: [{{ data: [{passed}, {failed}], backgroundColor: ['#38a169', '#e53e3e'] }}] }}
}});
new Chart(document.getElementById('latencyChart'), {{
  type: 'line',
  data: {{ labels: {json.dumps(latency_labels)}, datasets: [{{ label: 'Độ trễ (giây)', data: {json.dumps(latency_values)}, borderColor: '#dd6b20', backgroundColor: 'rgba(221,107,32,.15)', fill: true, tension: .3 }}] }}
}});
new Chart(document.getElementById('earlyExit'), {{
  type: 'pie',
  data: {{ labels: ['Từ chối trả lời (Early Exit)', 'Tiến hành sinh câu trả lời'], datasets: [{{ data: [{early_exit_count}, {total - early_exit_count}], backgroundColor: ['#ecc94b', '#4299e1'] }}] }}
}});
</script>
</body>
</html>
"""
    Path(out_path).write_text(html, encoding="utf-8")


# ==============================================================================
# 7. MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Đường dẫn file test case .xlsx")
    parser.add_argument("--out-dir", default=".", help="Thư mục xuất kết quả")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Đang đọc bộ test case ...")
    df = load_test_cases(args.input)
    print(f"Đã đọc {len(df)} test case.")

    scorer = SimilarityScorer(SIMILARITY_METHOD)

    results = run_evaluation(df, scorer)

    pd.DataFrame(results).to_csv(out_dir / "results_raw.csv", index=False, encoding="utf-8-sig")
    (out_dir / "results_raw.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_dashboard(results, str(out_dir / "dashboard.html"))

    print(f"\nXong! Xem kết quả tại: {out_dir / 'dashboard.html'}")


if __name__ == "__main__":
    main()
