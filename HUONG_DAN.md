# Hướng dẫn chạy test tự động cho RAG Pipeline

## 1. Cài đặt

```bash
pip install -r requirements.txt
```

## 2. Việc DUY NHẤT bạn cần sửa

Mở `run_rag_evaluation.py`, tìm hàm `call_rag_system(query)` (khoảng dòng 85).
Đây là nơi kết nối vào RAG pipeline **thật** của bạn. Có 2 tình huống:

- **Nếu pipeline của bạn chạy như một API** (FastAPI/Flask, hoặc bạn gõ
  `python src/rag/retriever.py --query "..."` và có thêm bước sinh câu trả
  lời): dựng nó thành một API nội bộ (`localhost:8000/chat` chẳng hạn), rồi
  dùng đoạn code mẫu "VÍ DỤ A" trong file (gọi `requests.post`).

- **Nếu bạn chạy trực tiếp bằng Python** (import hàm `retrieve()` / hàm sinh
  câu trả lời từ trong repo `RagPipeline`): dùng "VÍ DỤ B" — import thẳng
  các module đó và gọi tuần tự retrieve → generate.

Chỉ cần hàm này trả về **string** là câu trả lời của hệ thống, phần còn lại
của script tự lo hết (đo thời gian, tính similarity, tổng hợp, vẽ dashboard).

## 3. Chạy

```bash
python run_rag_evaluation.py --input RAG_TestCase.xlsx --out-dir ./ket_qua
```

Script sẽ tự động:
- Đọc cả 2 sheet (`RAG_TC` 60 câu cơ bản + `RAG_TC2` 57 câu nâng cao/edge case)
- Gửi từng câu hỏi vào `call_rag_system()`
- Đo latency, tính similarity, phát hiện Early Exit
- Xuất 3 file trong thư mục `--out-dir`:
  - `dashboard.html` — mở bằng trình duyệt, giống dashboard bạn đã có
  - `results_raw.csv` — chi tiết từng test case để soi trong Excel
  - `results_raw.json` — dữ liệu thô

## 4. Các thông số nên chỉnh theo hệ thống của bạn

Ở đầu file `run_rag_evaluation.py`:

| Biến | Ý nghĩa |
|---|---|
| `SIMILARITY_THRESHOLD` | Ngưỡng similarity để tính Pass/Fail (mặc định 0.65) |
| `SIMILARITY_METHOD` | `"embedding"` (chính xác, cần tải model lần đầu) hoặc `"tfidf"` (offline, nhẹ hơn nhưng kém chính xác hơn) |
| `EARLY_EXIT_PHRASES` | Danh sách câu cho thấy hệ thống đang "từ chối trả lời" — bổ sung đúng câu mà prompt hệ thống của bạn hay dùng, ví dụ "Xin lỗi, tài liệu không đề cập..." |

## 5. Logic Pass/Fail đang dùng

- Câu hỏi **có** đáp án trong tài liệu (`Is Answerable = Yes`): Pass nếu
  `similarity >= threshold` **và** hệ thống không early-exit nhầm.
- Câu hỏi **không có** đáp án trong tài liệu (`Is Answerable = No`, 6 câu
  trong sheet `RAG_TC`): Pass nếu hệ thống **early-exit đúng** (từ chối trả
  lời), Fail nếu hệ thống bịa ra câu trả lời (hallucination).

Bạn có thể sửa lại logic này trong hàm `run_evaluation()` nếu quy tắc chấm
điểm của bạn khác.

## 6. Lưu ý về sheet `RAG_TC2`

Sheet này (57 câu, RAG_061 trở đi) là các câu edge-case: false premise, câu
hỏi mơ hồ, typo, so sánh chéo section... và **không có cột `Is Answerable`**.
Script tự suy luận: nếu `Category` chứa chữ "Unanswerable" thì coi là câu
không có đáp án, còn lại coi là có đáp án. Bạn nên xem lại danh sách category
"Unanswerable..." trong file Excel để chắc chắn suy luận này đúng ý bạn —
nếu không, có thể thêm cột `Is Answerable` thủ công vào sheet đó trước khi
chạy.
