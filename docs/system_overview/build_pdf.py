#!/usr/bin/env python3
"""
Build "Giới thiệu hệ thống RAG Luật Lao động" (PDF, A4, tiếng Việt).

Only the Python standard library + Google Chrome (headless print). Diagrams and
charts are inline SVG, so nothing else needs installing.

    python3 docs/system_overview/build_pdf.py
    # -> docs/system_overview/Gioi_thieu_he_thong_RAG_Luat_Lao_dong.pdf
"""
import html
import shutil
import subprocess
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
HTML_PATH = OUT_DIR / "Gioi_thieu_he_thong_RAG_Luat_Lao_dong.html"
PDF_PATH = OUT_DIR / "Gioi_thieu_he_thong_RAG_Luat_Lao_dong.pdf"
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Google Chrome 2.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser",
]

# Palette: one accent, neutral greys, status colours used consistently.
INK, MUTED, LINE, SOFT = "#1d2433", "#5b6475", "#d5d9e0", "#f3f5f8"
ACCENT, ACCENT_SOFT = "#1f5fbf", "#e6eefb"
PASS_C, PARTIAL_C, FAIL_C = "#2e8b57", "#e0a526", "#c8453b"
OLD_C, NEW_C = "#a9b3c4", ACCENT


# ----------------------------------------------------------------------------
# SVG helpers
# ----------------------------------------------------------------------------

def esc(s):
    return html.escape(str(s), quote=True)


def box(x, y, w, h, title, sub="", fill="#ffffff", stroke=LINE, title_color=INK, rx=8, size=13):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>']
    lines = sub.split("\n") if sub else []
    total = 1 + len(lines)
    top = y + h / 2 - (total - 1) * 8
    out.append(f'<text x="{x + w / 2}" y="{top + 4}" text-anchor="middle" font-size="{size}" font-weight="700" fill="{title_color}">{esc(title)}</text>')
    for i, ln in enumerate(lines, 1):
        out.append(f'<text x="{x + w / 2}" y="{top + 4 + i * 16}" text-anchor="middle" font-size="11" fill="{MUTED}">{esc(ln)}</text>')
    return "".join(out)


def arrow(x1, y1, x2, y2, label="", color=MUTED, dashed=False, lx=None, ly=None):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    out = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6" marker-end="url(#ah)"{dash}/>'
    if label:
        lx = (x1 + x2) / 2 if lx is None else lx
        ly = (y1 + y2) / 2 - 6 if ly is None else ly
        out += f'<text x="{lx}" y="{ly}" text-anchor="middle" font-size="10.5" fill="{MUTED}">{esc(label)}</text>'
    return out


def svg(w, h, body):
    defs = (f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>')
    return f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Arial, Helvetica, sans-serif">{defs}{body}</svg>'


def grouped_bars(cats, series, vmax=100, unit="%", w=660, h=250, title_y=None):
    """Vertical grouped bars with value labels. series = [(name, values, colour)]."""
    left, right, top, bottom = 44, 16, 30, 46
    pw, ph = w - left - right, h - top - bottom
    n, k = len(cats), len(series)
    gw = pw / n
    bw = min(34, gw * 0.7 / k)
    out = []
    for t in range(0, 101, 25):
        v = vmax * t / 100
        y = top + ph - ph * t / 100
        out.append(f'<line x1="{left}" y1="{y}" x2="{w - right}" y2="{y}" stroke="{LINE}" stroke-width="1"/>')
        out.append(f'<text x="{left - 6}" y="{y + 4}" text-anchor="end" font-size="10" fill="{MUTED}">{v:g}{unit}</text>')
    for i, c in enumerate(cats):
        gx = left + gw * i + (gw - bw * k - 4 * (k - 1)) / 2
        for j, (_, vals, col) in enumerate(series):
            v = vals[i]
            bh = ph * v / vmax
            x = gx + j * (bw + 4)
            y = top + ph - bh
            out.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" fill="{col}" rx="2"/>')
            label = f"{v:g}" if unit != "%" else f"{v:g}%"
            out.append(f'<text x="{x + bw / 2}" y="{y - 4}" text-anchor="middle" font-size="10" fill="{INK}">{label}</text>')
        out.append(f'<text x="{left + gw * i + gw / 2}" y="{h - bottom + 16}" text-anchor="middle" font-size="11" fill="{INK}">{esc(c)}</text>')
    lx = left
    for name, _, col in series:
        out.append(f'<rect x="{lx}" y="8" width="11" height="11" fill="{col}" rx="2"/>')
        out.append(f'<text x="{lx + 16}" y="18" font-size="11" fill="{INK}">{esc(name)}</text>')
        lx += 22 + 7 * len(name)
    return svg(w, h, "".join(out))


def stacked_hbars(rows, parts, w=660, row_h=30, label_w=170):
    """Horizontal 100% stacked bars. rows = [(label, [counts...])], parts = [(name, colour)]."""
    top = 30
    h = top + row_h * len(rows) + 10
    pw = w - label_w - 20
    out = []
    lx = label_w
    for name, col in parts:
        out.append(f'<rect x="{lx}" y="8" width="11" height="11" fill="{col}" rx="2"/>')
        out.append(f'<text x="{lx + 16}" y="18" font-size="11" fill="{INK}">{esc(name)}</text>')
        lx += 30 + 7 * len(name)
    for i, (label, vals) in enumerate(rows):
        y = top + i * row_h
        total = sum(vals) or 1
        out.append(f'<text x="{label_w - 8}" y="{y + row_h / 2 + 4}" text-anchor="end" font-size="11" fill="{INK}">{esc(label)}</text>')
        x = label_w
        for v, (_, col) in zip(vals, parts):
            bw = pw * v / total
            if bw > 0:
                out.append(f'<rect x="{x}" y="{y + 5}" width="{bw}" height="{row_h - 10}" fill="{col}"/>')
                if bw > 18:
                    out.append(f'<text x="{x + bw / 2}" y="{y + row_h / 2 + 4}" text-anchor="middle" font-size="11" fill="#fff" font-weight="700">{v}</text>')
            x += bw
    return svg(w, h, "".join(out))


def hbars(rows, vmax, colour=ACCENT, w=660, row_h=26, label_w=230, unit=""):
    """Simple horizontal bars, rows = [(label, value, optional colour)]."""
    h = row_h * len(rows) + 10
    pw = w - label_w - 60
    out = []
    for i, row in enumerate(rows):
        label, v = row[0], row[1]
        col = row[2] if len(row) > 2 else colour
        y = i * row_h
        bw = pw * v / vmax
        out.append(f'<text x="{label_w - 8}" y="{y + row_h / 2 + 4}" text-anchor="end" font-size="11" fill="{INK}">{esc(label)}</text>')
        out.append(f'<rect x="{label_w}" y="{y + 5}" width="{bw}" height="{row_h - 10}" fill="{col}" rx="2"/>')
        out.append(f'<text x="{label_w + bw + 6}" y="{y + row_h / 2 + 4}" font-size="11" fill="{INK}">{v:g}{unit}</text>')
    return svg(w, h, "".join(out))


# ----------------------------------------------------------------------------
# Diagrams
# ----------------------------------------------------------------------------

def diagram_architecture():
    b = []
    # Offline band
    b.append(f'<rect x="10" y="10" width="680" height="120" rx="10" fill="{SOFT}" stroke="none"/>')
    b.append(f'<text x="24" y="32" font-size="12" font-weight="700" fill="{MUTED}">OFFLINE · chạy 1 lần khi nạp tài liệu</text>')
    steps = [("5 file PDF", "1.428 trang"), ("Docling + OCR", "đọc chữ, bảng, ảnh"), ("BLIP", "chú thích ảnh"),
             ("Chunker", "4.654 đoạn"), ("bge-m3", "biến thành vector")]
    for i, (t, s) in enumerate(steps):
        x = 24 + i * 134
        b.append(box(x, 48, 116, 64, t, s))
        if i:
            b.append(arrow(x - 18, 80, x - 2, 80))
    # Stores
    b.append(box(150, 170, 150, 58, "ChromaDB", "kho vector (tìm theo nghĩa)", fill=ACCENT_SOFT, stroke=ACCENT))
    b.append(box(320, 170, 150, 58, "Chỉ mục BM25", "tìm theo từ khóa", fill=ACCENT_SOFT, stroke=ACCENT))
    b.append(box(490, 170, 150, 58, "chunked.jsonl", "nội dung + metadata", fill=ACCENT_SOFT, stroke=ACCENT))
    for cx in (225, 395, 565):
        b.append(arrow(cx, 132, cx, 168))
    b.append(f'<text x="640" y="154" font-size="10.5" fill="{MUTED}">ghi vào kho</text>')
    # Online band
    b.append(f'<rect x="10" y="262" width="680" height="168" rx="10" fill="#ffffff" stroke="{LINE}"/>')
    b.append(f'<text x="24" y="284" font-size="12" font-weight="700" fill="{MUTED}">ONLINE · mỗi lần người dùng hỏi</text>')
    b.append(box(24, 300, 110, 60, "Người dùng", "gõ câu hỏi"))
    b.append(box(160, 300, 120, 60, "Frontend", "React + Vite\ncổng 5173"))
    b.append(box(306, 300, 130, 60, "Backend API", "FastAPI\ncổng 8001"))
    b.append(box(462, 296, 210, 120, "RAG pipeline", "tìm lai (hybrid) → rerank\n→ lọc trùng → dựng ngữ cảnh\n→ gọi LLM → kiểm trích dẫn", fill=ACCENT_SOFT, stroke=ACCENT))
    b.append(arrow(134, 330, 158, 330))
    b.append(arrow(280, 330, 304, 330))
    b.append(arrow(436, 330, 460, 330))
    b.append(arrow(600, 294, 600, 232, dashed=True))
    b.append(f'<text x="610" y="268" font-size="10.5" fill="{MUTED}">đọc kho khi trả lời</text>')
    b.append(box(306, 376, 130, 44, "Gemini (LLM)", "dịch vụ Google", fill="#fff7e6", stroke=PARTIAL_C, size=12))
    b.append(arrow(460, 398, 438, 398))
    return svg(700, 440, "".join(b))


def diagram_offline():
    steps = [
        ("1. Đọc PDF", "Docling\ndocling_parse / pypdfium2\nTesseract OCR", "21.527 phần tử"),
        ("2. Ảnh", "trích ảnh\nBLIP tạo chú thích", "bỏ ảnh trang trí"),
        ("3. Chuẩn hóa", "Unicode NFC\nbỏ ký tự ẩn", "văn bản sạch"),
        ("4. Cắt chunk", "theo Chương/Mục/Điều\n600 token, gối 100", "4.654 chunk"),
        ("5. Metadata", "tài liệu, trang\ntiêu đề mục", "chunked.jsonl"),
        ("6. Lập chỉ mục", "bge-m3 → ChromaDB\nBM25", "4.654 vector"),
    ]
    b = []
    for i, (t, s, o) in enumerate(steps):
        x = 8 + (i % 3) * 232
        y = 10 + (i // 3) * 150
        b.append(box(x, y, 200, 92, t, s, fill="#ffffff", stroke=ACCENT))
        b.append(f'<rect x="{x + 30}" y="{y + 100}" width="140" height="24" rx="12" fill="{ACCENT_SOFT}"/>')
        b.append(f'<text x="{x + 100}" y="{y + 116}" text-anchor="middle" font-size="11" fill="{ACCENT}" font-weight="700">{esc(o)}</text>')
        if i % 3:
            b.append(arrow(x - 30, y + 46, x - 4, y + 46))
    b.append(f'<path d="M 672 56 H 690 V 142 H 108 V 156" fill="none" stroke="{MUTED}" stroke-width="1.6" marker-end="url(#ah)"/>')
    return svg(700, 290, "".join(b))


def diagram_online():
    steps = [
        ("Câu hỏi", "“NLĐ làm việc bình thường được nghỉ hằng năm bao nhiêu ngày?”"),
        ("Tìm lai (hybrid)", "bge-m3 lấy 10 đoạn gần nghĩa + BM25 lấy 10 đoạn trùng từ khóa → gộp RRF"),
        ("Rerank", "bge-reranker-v2-m3 chấm từng đoạn 0–1; Điều 113 được 0,99"),
        ("Ngưỡng 0,35", "đoạn dưới 0,35 bị loại; không còn đoạn nào → trả “không tìm thấy”, không gọi LLM"),
        ("Lọc trùng + chọn 5", "bỏ đoạn trùng ≥ 60% giữa tài liệu, ưu tiên Văn bản hợp nhất, giữ 5 đoạn"),
        ("Dựng ngữ cảnh", "gắn nhãn [SOURCE_1]…[SOURCE_5], tối đa 4.000 token"),
        ("Gemini trả lời", "prompt chống bịa: chỉ dùng ngữ cảnh, mỗi ý kèm [SOURCE_x]"),
        ("Kiểm trích dẫn", "xóa nhãn bịa, đổi [SOURCE_x] thành tên file + trang"),
        ("Kết quả", "“12 ngày làm việc [SOURCE_1]” + nguồn: VBHN 18, tr.42, Điều 113"),
    ]
    b = []
    for i, (t, s) in enumerate(steps):
        y = 6 + i * 50
        fill = ACCENT_SOFT if i in (0, len(steps) - 1) else "#ffffff"
        b.append(f'<rect x="10" y="{y}" width="160" height="38" rx="8" fill="{fill}" stroke="{ACCENT}" stroke-width="1.4"/>')
        b.append(f'<text x="90" y="{y + 24}" text-anchor="middle" font-size="12.5" font-weight="700" fill="{INK}">{esc(t)}</text>')
        b.append(f'<text x="186" y="{y + 24}" font-size="11.5" fill="{INK}">{esc(s)}</text>')
        if i < len(steps) - 1:
            b.append(arrow(90, y + 38, 90, y + 48))
    return svg(700, 6 + len(steps) * 50, "".join(b))


def diagram_failure_tree():
    b = []

    def diamond(cx, cy, text):
        w, h = 210, 58
        pts = f"{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}"
        out = f'<polygon points="{pts}" fill="#ffffff" stroke="{ACCENT}" stroke-width="1.4"/>'
        for i, ln in enumerate(text.split("\n")):
            out += f'<text x="{cx}" y="{cy - 3 + i * 14}" text-anchor="middle" font-size="11" fill="{INK}">{esc(ln)}</text>'
        return out

    def leaf(x, y, title, n, col):
        w = 250
        return (f'<rect x="{x}" y="{y}" width="{w}" height="44" rx="8" fill="{col}" opacity="0.14"/>'
                f'<rect x="{x}" y="{y}" width="{w}" height="44" rx="8" fill="none" stroke="{col}" stroke-width="1.4"/>'
                f'<text x="{x + w / 2}" y="{y + 19}" text-anchor="middle" font-size="11.5" font-weight="700" fill="{INK}">{esc(title)}</text>'
                f'<text x="{x + w / 2}" y="{y + 35}" text-anchor="middle" font-size="11" fill="{MUTED}">{esc(n)}</text>')

    qs = [(150, 40, "Đoạn đúng có trong\nngữ cảnh gửi LLM?"), (150, 140, "LLM có trả lời\nkhông?"),
          (150, 240, "Đoạn đúng có trong\n20 ứng viên?"), (150, 340, "Đoạn đúng có trong\ntop 5 sau rerank?")]
    for cx, cy, t in qs:
        b.append(diamond(cx, cy, t))
    b.append(arrow(150, 69, 150, 109, "Không", lx=176, ly=94))
    b.append(arrow(150, 169, 150, 209, "Không", lx=176, ly=194))
    b.append(arrow(150, 269, 150, 309, "Có", lx=168, ly=294))
    b.append(arrow(255, 40, 404, 40, "Có", ly=34))
    b.append(leaf(408, 18, "Lỗi tầng LLM", "15 câu: trả lời thiếu / bị chấm thiếu", PARTIAL_C))
    b.append(arrow(255, 140, 404, 140, "Có", ly=134))
    b.append(leaf(408, 118, "WRONG_SOURCE_ANSWER", "16 câu: trả lời từ đoạn sai", FAIL_C))
    b.append(arrow(255, 240, 404, 240, "Không", ly=234))
    b.append(leaf(408, 218, "RETRIEVAL_MISS", "7 câu: tìm không ra", FAIL_C))
    b.append(arrow(255, 340, 404, 318, "Không", ly=320))
    b.append(leaf(408, 296, "RERANK_DROP", "4 câu: rerank xếp thấp", FAIL_C))
    b.append(arrow(150, 369, 150, 400, "Có", lx=168, ly=390))
    b.append(leaf(25, 404, "THRESHOLD_CUT", "9 câu: dưới ngưỡng 0,35", FAIL_C))
    b.append(f'<text x="533" y="392" text-anchor="middle" font-size="10.5" fill="{MUTED}">Ngoài cây: câu không có đoạn đúng để lần theo</text>')
    b.append(leaf(408, 400, "NO_GOLD", "5 câu: không gán được đoạn đúng", MUTED))
    return svg(700, 456, "".join(b))


# ----------------------------------------------------------------------------
# Content
# ----------------------------------------------------------------------------

def summary(items):
    lis = "".join(f"<li>{i}</li>" for i in items)
    return f'<div class="summary"><div class="summary-title">Tóm tắt</div><ul>{lis}</ul></div>'


def chapter(num, title, question, body, summary_items):
    return (f'<section class="chapter"><div class="chapter-num">Chương {num}</div><h1>{title}</h1>'
            f'<p class="question">{question}</p>{body}{summary(summary_items)}</section>')


def term(word, explain):
    return f'<div class="term"><b>{word}</b> — {explain}</div>'


def figure(svg_markup, caption):
    return f'<figure>{svg_markup}<figcaption>{caption}</figcaption></figure>'


def table(header, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in header)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'


def build_html():
    parts = []

    # Cover
    parts.append(f"""
<section class="cover">
  <div class="cover-tag">Tài liệu giới thiệu hệ thống</div>
  <div class="cover-title">Hệ thống RAG hỏi đáp<br>Luật Lao động</div>
  <div class="cover-sub">Từ con số 0: hệ thống làm gì, được xây thế nào, dùng công nghệ gì,
  và đã được kiểm thử ra sao</div>
  <div class="cover-stats">
    <div><b>5</b><span>tài liệu PDF</span></div>
    <div><b>1.428</b><span>trang</span></div>
    <div><b>4.654</b><span>đoạn tri thức</span></div>
    <div><b>165</b><span>test case đã chạy</span></div>
  </div>
  <div class="cover-meta">Người trình bày: Ngô Trung Kiên (QA/QC)<br>Ngày: 21/10/2026</div>
</section>""")

    # Contents
    toc = ["Bài toán", "RAG là gì", "Dữ liệu đầu vào", "Kiến trúc tổng thể", "Luồng OFFLINE: xây kho tri thức",
           "Luồng ONLINE: trả lời một câu hỏi", "Các model AI", "Backend", "Frontend và UI/UX", "Kiểm thử (Test)",
           "Danh sách lỗi và hướng sửa", "Kết luận và kế hoạch", "Phụ lục"]
    parts.append('<section class="toc"><h1>Nội dung</h1><ol>' + "".join(f"<li>{t}</li>" for t in toc) + "</ol>"
                 '<p class="note">Cách đọc: các chương 1–9 giải thích hệ thống, chương 10 (dài nhất) trình bày '
                 'toàn bộ việc kiểm thử, chương 11–12 nêu lỗi và kết luận. Thuật ngữ tiếng Anh được giải thích '
                 'ngay lần đầu xuất hiện và tổng hợp lại ở Phụ lục.</p></section>')

    # 1. Problem
    parts.append(chapter(1, "Bài toán", "Vì sao cần xây hệ thống này, thay vì đọc tài liệu hoặc hỏi thẳng ChatGPT?", f"""
<p>Người lao động, doanh nghiệp và người làm nhân sự thường cần câu trả lời nhanh cho những câu hỏi như
"nghỉ phép năm được bao nhiêu ngày", "làm thêm ngày lễ được trả bao nhiêu". Câu trả lời nằm rải rác trong
<b>5 tài liệu PDF, tổng cộng 1.428 trang</b>: giáo trình, sổ tay hỏi đáp, cẩm nang cho doanh nghiệp, bộ tổng hợp
gần 1.000 trang văn bản và bản hợp nhất Bộ luật Lao động.</p>
<div class="two-col">
  <div class="card"><h3>Đọc tay</h3><p>Chính xác nhưng chậm: tìm một con số có thể mất hàng chục phút, và phải biết tài liệu nào chứa nó.</p></div>
  <div class="card"><h3>Hỏi chatbot chung (ChatGPT…)</h3><p>Nhanh nhưng dễ "bịa": trả lời trôi chảy theo trí nhớ, không chỉ ra nguồn, có thể dùng luật cũ hoặc luật nước khác.</p></div>
</div>
<p>Với pháp luật, một câu trả lời sai mà nghe có vẻ đúng nguy hiểm hơn là không trả lời. Vì vậy hệ thống được
thiết kế theo 3 nguyên tắc:</p>
<ol>
  <li><b>Chỉ trả lời dựa trên 5 tài liệu</b> đã nạp, không dùng hiểu biết bên ngoài.</li>
  <li><b>Mọi ý đều kèm nguồn</b>: tên tài liệu, số trang, mục, để người đọc tự kiểm tra.</li>
  <li><b>Không tìm thấy thì nói không tìm thấy</b>, thay vì đoán.</li>
</ol>
""", ["Câu trả lời nằm trong 1.428 trang PDF; đọc tay quá chậm, chatbot chung dễ bịa.",
      "Hệ thống phải trả lời dựa trên tài liệu, có trích nguồn và biết từ chối."]))

    # 2. RAG
    parts.append(chapter(2, "RAG là gì", "Hệ thống dùng kỹ thuật gì để vừa nhanh như chatbot vừa bám tài liệu?", f"""
<p><b>RAG (Retrieval-Augmented Generation)</b> nghĩa là "tìm trước, trả lời sau". Hãy hình dung một kỳ thi
<b>được mở sách</b>: thí sinh không trả lời theo trí nhớ, mà lật đúng trang có thông tin, đọc lại, rồi mới viết
câu trả lời và ghi "theo trang 42".</p>
<div class="steps3">
  <div><span>1</span><b>Retrieval: tìm</b><p>Tìm trong kho tài liệu vài đoạn liên quan nhất tới câu hỏi.</p></div>
  <div><span>2</span><b>Augmented: bổ sung</b><p>Đưa các đoạn đó vào cùng câu hỏi, làm "tài liệu mở" cho AI.</p></div>
  <div><span>3</span><b>Generation: viết</b><p>AI viết câu trả lời chỉ từ các đoạn được đưa, kèm nguồn.</p></div>
</div>
<h3>Bốn khái niệm cần biết</h3>
{term("LLM (Large Language Model)", "mô hình ngôn ngữ lớn, ví dụ Gemini, ChatGPT. Giống một người viết rất giỏi nhưng không nhớ chính xác từng điều luật; trong RAG nó chỉ đóng vai người viết câu trả lời.")}
{term("Chunk", "một đoạn nhỏ cắt ra từ tài liệu (khoảng nửa trang). Tìm theo đoạn nhỏ thì chính xác hơn tìm theo cả cuốn sách.")}
{term("Embedding", "biến câu chữ thành một dãy số (tọa độ). Câu cùng nghĩa thì có tọa độ gần nhau, kể cả khi dùng từ khác nhau: “nghỉ phép năm” nằm gần “nghỉ hằng năm”.")}
{term("Reranker", "giám khảo chấm lại khoảng 20 đoạn tìm được, đọc kỹ từng đoạn cùng câu hỏi và xếp đoạn sát câu hỏi nhất lên đầu.")}
""", ["RAG = tìm đoạn liên quan trong tài liệu, rồi để AI viết câu trả lời chỉ từ các đoạn đó.",
      "Giống đi thi được mở sách: câu trả lời luôn có trang nguồn để đối chiếu."]))

    # 3. Data
    docs = [
        ("A", "Giáo trình Luật lao động 2005", "Lý thuyết, dựa trên luật cũ", "20"),
        ("B", "Sổ tay kiến thức pháp luật lao động (CDI)", "Dạng hỏi–đáp, 56 trang", "20"),
        ("C", "Bộ tổng hợp văn bản (luu-ban-nhap-tu-dong-3-4)", "951 trang, nhiều luật và nghị định, có trang scan", "20"),
        ("D", "Cẩm nang ILO-VCCI", "So sánh Bộ luật Lao động 2012 và 2019 cho doanh nghiệp", "20"),
        ("E", "Văn bản hợp nhất 18/VBHN-VPQH", "Bộ luật Lao động 2019 đã hợp nhất sửa đổi; trang 61–86 là ảnh scan", "20"),
    ]
    parts.append(chapter(3, "Dữ liệu đầu vào", "Hệ thống “biết” những gì, và các tài liệu đó khác nhau ra sao?", f"""
<p>Kho tri thức gồm 5 tài liệu, mỗi tài liệu được gán một chữ cái (A–E). Chữ cái này dùng lại trong phần test:
mỗi nhóm có 20 câu hỏi.</p>
{table(["Nhóm", "Tài liệu", "Đặc điểm", "Câu test"], docs)}
<p>Năm tài liệu rất khác nhau về hình thức, và chính sự khác nhau này tạo ra phần lớn khó khăn:</p>
<ul>
  <li><b>Khác thời điểm:</b> giáo trình 2005 nói theo luật cũ, Văn bản hợp nhất theo Bộ luật 2019. Cùng một câu hỏi có thể có hai đáp án.</li>
  <li><b>Khác dạng chữ:</b> có trang là chữ thật (máy đọc được), có trang là ảnh scan phải nhận dạng chữ (OCR), dễ mất dấu tiếng Việt.</li>
  <li><b>Trùng nội dung:</b> Bộ luật 2019 xuất hiện ở cả bộ tổng hợp, cẩm nang ILO và Văn bản hợp nhất.</li>
</ul>
<div class="bignums"><div><b>1.428</b>trang</div><div><b>21.527</b>phần tử<br>(đoạn văn, bảng, ảnh)</div><div><b>4.654</b>chunk</div><div><b>4.654</b>vector</div></div>
""", ["5 tài liệu, 1.428 trang, cắt thành 4.654 đoạn.",
      "Khó khăn chính: tài liệu khác thời điểm, có trang scan, và trùng nội dung với nhau."]))

    # 4. Architecture
    parts.append(chapter(4, "Kiến trúc tổng thể", "Hệ thống gồm những khối nào và chúng nối với nhau ra sao?", f"""
<p>Hệ thống chia làm hai luồng. <b>Luồng offline</b> chạy một lần để "dọn sách lên kệ": đọc PDF, cắt đoạn, đánh
dấu để tìm nhanh. <b>Luồng online</b> chạy mỗi khi có người hỏi: tìm đúng đoạn trên kệ rồi nhờ AI viết câu trả lời.</p>
{figure(diagram_architecture(), "Hình 1. Kiến trúc tổng thể. Khối màu xanh là kho dữ liệu dùng chung giữa hai luồng; Gemini là dịch vụ bên ngoài.")}
{table(["Khối", "Vai trò", "Công nghệ"], [
    ("Frontend", "Giao diện web: nạp tài liệu, thử tìm kiếm, hỏi đáp", "React 18, Vite 6"),
    ("Backend API", "Nhận yêu cầu, chạy pipeline, trả kết quả JSON", "Python, FastAPI, Uvicorn"),
    ("Kho vector", "Tìm đoạn gần nghĩa với câu hỏi", "ChromaDB + bge-m3"),
    ("Chỉ mục từ khóa", "Tìm đoạn chứa đúng từ trong câu hỏi", "BM25"),
    ("LLM", "Viết câu trả lời từ các đoạn được chọn", "Google Gemini"),
])}
""", ["Offline: đọc và sắp xếp tài liệu một lần. Online: tìm đoạn và trả lời mỗi câu hỏi.",
      "Frontend React ↔ Backend FastAPI ↔ kho ChromaDB/BM25 ↔ Gemini."]))

    # 5. Offline
    parts.append(chapter(5, "Luồng OFFLINE: xây kho tri thức", "Từ file PDF đến kho dữ liệu tìm kiếm được, hệ thống làm những bước nào?", f"""
{figure(diagram_offline(), "Hình 2. Sáu bước của luồng offline; ô xanh nhạt là kết quả đầu ra của từng bước.")}
<h3>Bước 1. Đọc PDF (Docling + OCR)</h3>
<p><b>Docling</b> là công cụ đọc PDF, tách ra đoạn văn, tiêu đề, bảng và ảnh. Mỗi tài liệu được đọc thử bằng 2 "bộ đọc"
(docling_parse và pypdfium2), vì có bộ làm vỡ chữ tiếng Việt thành mảnh như "Nh ững điể m". Hệ thống tự đếm số mảnh
chữ vỡ trên 1.000 từ (ngưỡng 50) và giữ bộ đọc cho kết quả sạch hơn. Trang là ảnh scan được nhận dạng chữ bằng
<b>Tesseract OCR</b> (tiếng Việt + tiếng Anh).</p>
<p class="io"><b>Vào:</b> 5 file PDF · <b>Ra:</b> 21.527 phần tử có kèm số trang và loại (đoạn văn, tiêu đề, bảng, ảnh).</p>
<h3>Bước 2. Ảnh và chú thích ảnh (BLIP)</h3>
<p>Ảnh trong PDF được lưu riêng. Model <b>BLIP</b> (Salesforce/blip-image-captioning-base) viết một câu mô tả cho mỗi ảnh để
ảnh cũng tìm được bằng chữ. Ảnh trang trí nhỏ hơn 100×100 điểm ảnh và không có chú thích thì bị bỏ, tránh làm nhiễu.</p>
<h3>Bước 3. Chuẩn hóa văn bản</h3>
<p>Đưa chữ tiếng Việt về một dạng mã thống nhất (Unicode NFC), bỏ ký tự ẩn, gộp khoảng trắng thừa. Nếu bỏ qua, cùng một chữ
"điều" có thể được lưu bằng hai mã khác nhau và tìm kiếm sẽ trượt.</p>
<h3>Bước 4. Cắt chunk theo cấu trúc tài liệu</h3>
<p>Không cắt máy móc theo số ký tự, mà theo cấu trúc Chương → Mục → Điều → đoạn văn, rồi gói lại thành đoạn khoảng
<b>600 token</b> (tối đa 800; 1 token ≈ 3/4 từ). Mỗi đoạn mang theo khoảng <b>100 token cuối</b> của đoạn trước (gọi là
<i>overlap</i>, "gối đầu") để không mất ý ở chỗ cắt. Mục quá ngắn (dưới 50 token, ví dụ chỉ có tiêu đề) được gộp vào mục sau;
tiêu đề rơi vào cuối đoạn được chuyển sang đoạn kế tiếp.</p>
<h3>Bước 5. Gắn metadata</h3>
<p>Mỗi chunk mang "nhãn": tên tài liệu, trang bắt đầu–kết thúc, đường dẫn tiêu đề mục, số token. Nhờ nhãn này câu trả lời
mới chỉ ra được "Văn bản hợp nhất, trang 42, Điều 113".</p>
<div class="example"><div class="example-title">Ví dụ một chunk thật</div>
Section: Điều 113. Nghỉ hằng năm<br>---<br>Người lao động làm việc đủ 12 tháng cho một người sử dụng lao động thì được nghỉ hằng năm,
hưởng nguyên lương theo hợp đồng lao động như sau: 12 ngày làm việc đối với người làm công việc trong điều kiện bình thường…
<div class="example-meta">tài liệu: Văn-bản-hợp-nhất-18-VBHN-VPQH.pdf · trang 42 · mục: Điều 113. Nghỉ hằng năm</div></div>
<h3>Bước 6. Lập chỉ mục: vector và từ khóa</h3>
<p>Mỗi chunk được model <b>bge-m3</b> biến thành vector rồi lưu vào <b>ChromaDB</b> (cơ sở dữ liệu chuyên tìm vector gần nhau).
Mã của vector trùng mã chunk, nên chạy lại không sinh bản trùng. Song song, hệ thống dựng chỉ mục từ khóa <b>BM25</b>, kiểu
"tra mục lục cuối sách" theo từng từ.</p>
""", ["Đọc PDF (tự chọn bộ đọc ít vỡ chữ, OCR trang scan) → chú thích ảnh → chuẩn hóa → cắt theo Điều/Mục → gắn nhãn → lập chỉ mục.",
      "Kết quả: 4.654 chunk, mỗi chunk vừa tìm được theo nghĩa (vector) vừa theo từ khóa (BM25)."]))

    # 6. Online
    parts.append(chapter(6, "Luồng ONLINE: trả lời một câu hỏi", "Khi người dùng bấm gửi, chuyện gì xảy ra trong khoảng 18 giây tiếp theo?", f"""
<p>Theo dõi một câu hỏi thật đi qua toàn bộ hệ thống:</p>
{figure(diagram_online(), "Hình 3. Chín bước của luồng online với ví dụ câu hỏi về nghỉ hằng năm.")}
<h3>Vì sao tìm bằng hai cách (hybrid)?</h3>
<p>Tìm theo nghĩa (vector) hiểu được "nghỉ phép năm" ≈ "nghỉ hằng năm" nhưng hay trượt với số hiệu như "Điều 113",
"Nghị định 45/2013". Tìm theo từ khóa (BM25) bắt chính xác số hiệu nhưng không hiểu từ đồng nghĩa. Dùng cả hai rồi gộp
bằng <b>RRF (Reciprocal Rank Fusion)</b>: đoạn nào đứng cao ở cả hai danh sách thì được đẩy lên đầu.</p>
<h3>Vì sao cần ngưỡng 0,35?</h3>
<p>Reranker cho mỗi đoạn một điểm từ 0 đến 1. Nếu đoạn tốt nhất chỉ được dưới 0,35, gần như chắc chắn tài liệu không nói về
câu hỏi; hệ thống trả lời ngay "không tìm thấy" mà không gọi LLM. Cách này tiết kiệm chi phí và tránh để AI cố trả lời bằng
đoạn không liên quan.</p>
<h3>Prompt chống bịa</h3>
<p><b>Prompt</b> là lời dặn gửi kèm cho LLM. Lời dặn của hệ thống có 4 quy tắc chính:</p>
<ol><li>Chỉ dùng thông tin trong các khối [SOURCE_x].</li><li>Mỗi ý phải kèm mã nguồn [SOURCE_x].</li>
<li>Không có thông tin thì trả lời đúng câu: "Dựa trên tài liệu được cung cấp, tôi không tìm thấy thông tin để trả lời câu hỏi này."</li>
<li>Không tự tạo mã nguồn không có trong danh sách.</li></ol>
<p>Sau khi LLM trả lời, bước kiểm trích dẫn xóa mã nguồn bịa (nếu có) và đổi mỗi [SOURCE_x] thành tên file, trang, mục để
hiện trên giao diện. Kết quả trả về dạng JSON gồm câu trả lời, danh sách nguồn, đường dẫn ảnh, tổng thời gian và nhật ký
thời gian của từng bước.</p>
""", ["Tìm lai (nghĩa + từ khóa) → rerank → ngưỡng 0,35 → lọc trùng, giữ 5 đoạn → Gemini viết câu trả lời kèm nguồn.",
      "Hai lớp chống bịa: ngưỡng (không đủ liên quan thì không hỏi AI) và prompt (chỉ dùng ngữ cảnh, phải trích nguồn)."]))

    # 7. Models
    parts.append(chapter(7, "Các model AI", "Hệ thống dùng những model nào, mỗi model làm việc gì, và vì sao chọn chúng?", f"""
{table(["Model", "Làm gì", "Vì sao chọn", "Chạy ở đâu"], [
    ("BAAI/bge-m3", "Biến câu hỏi và đoạn văn thành vector (embedding)", "Đa ngôn ngữ, hiểu tốt tiếng Việt, đọc được đoạn dài tới 8.192 token, miễn phí", "Máy chủ (CPU)"),
    ("BAAI/bge-reranker-v2-m3", "Chấm điểm 0–1 mức liên quan giữa câu hỏi và từng đoạn", "Cùng họ với bge-m3, đa ngôn ngữ; chính xác hơn so vector đơn thuần vì đọc câu hỏi và đoạn cùng lúc", "Máy chủ (CPU)"),
    ("Google Gemini", "Viết câu trả lời từ ngữ cảnh", "Tiếng Việt tốt, nhanh (khoảng 1,5 giây mỗi câu), có gói miễn phí để thử", "Dịch vụ Google (qua API)"),
    ("Salesforce BLIP", "Viết chú thích cho ảnh trong PDF", "Nhỏ, chạy được trên máy cá nhân", "Máy chủ, chỉ khi nạp tài liệu"),
    ("Tesseract OCR", "Nhận dạng chữ trên trang scan", "Mã nguồn mở, có gói tiếng Việt", "Máy chủ, chỉ khi nạp tài liệu"),
])}
<h3>Cấu hình LLM</h3>
<ul>
  <li>Model mặc định: <b>gemini-3.6-flash</b>. Đợt test dùng <b>gemini-3.5-flash-lite</b>, vì gói miễn phí của model mặc định chỉ
  cho 20 lượt gọi mỗi ngày.</li>
  <li>Nhiệt độ (temperature) 0,1: gần như không "sáng tạo", trả lời ổn định. Tối đa 1.200 token đầu ra.</li>
  <li>Có thể đổi sang OpenAI, Ollama (chạy model trên máy) hoặc Mock (câu trả lời giả để thử giao diện) bằng file cấu hình .env.</li>
</ul>
<p class="note">Chỉ Gemini là dịch vụ bên ngoài: chỉ câu hỏi và 5 đoạn ngữ cảnh được gửi đi, toàn bộ kho tài liệu nằm trên máy chủ.</p>
""", ["bge-m3 (tìm theo nghĩa), bge-reranker-v2-m3 (chấm lại), Gemini (viết câu trả lời); BLIP và Tesseract chỉ dùng khi nạp tài liệu.",
      "Đợt test dùng gemini-3.5-flash-lite thay cho gemini-3.6-flash vì hạn mức miễn phí."]))

    # 8. Backend
    parts.append(chapter(8, "Backend", "Phần máy chủ được viết bằng gì và cung cấp những chức năng nào?", f"""
<p>Backend viết bằng <b>Python</b>, dùng <b>FastAPI</b> (khung dựng API) chạy trên <b>Uvicorn</b>, cổng 8001.
<b>API</b> là "cửa sổ giao dịch": giao diện gửi yêu cầu qua đúng cửa, máy chủ trả kết quả dạng JSON.</p>
{table(["Phương thức", "Đường dẫn", "Chức năng"], [
    ("GET", "/health", "Kiểm tra máy chủ còn sống"),
    ("GET", "/api/files", "Liệt kê file PDF trong thư mục dữ liệu"),
    ("GET", "/api/pipeline/status", "Số tài liệu, trang, phần tử, chunk, vector hiện có"),
    ("POST", "/api/pipeline/load", "Bước 1–2 offline: đọc PDF, OCR, ảnh"),
    ("POST", "/api/pipeline/chunk", "Bước 3–5 offline: chuẩn hóa, cắt chunk, metadata"),
    ("POST", "/api/pipeline/embed", "Bước 6 offline: tạo vector, lưu ChromaDB"),
    ("POST", "/api/retrieval/search", "Chỉ tìm kiếm, không gọi LLM (để kiểm tra retrieval)"),
    ("POST", "/api/rag/chat", "Hỏi đáp đầy đủ: luồng online 9 bước"),
])}
<h3>Thư viện chính</h3>
<p>Docling (đọc PDF), PyTorch + sentence-transformers (chạy bge-m3 và reranker), ChromaDB (kho vector), LangChain (kết nối kho vector),
tiktoken (đếm token), google-genai (gọi Gemini).</p>
<h3>Cấu hình</h3>
<p>Mọi tham số nằm trong file <b>.env</b> và được đọc lại ở mỗi yêu cầu, nên sửa không cần khởi động lại máy chủ:</p>
{table(["Tham số", "Giá trị", "Ý nghĩa"], [
    ("EMBEDDING_MODEL", "BAAI/bge-m3", "Model tạo vector"),
    ("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3", "Model chấm lại"),
    ("RETRIEVAL_TOP_K", "10", "Số đoạn lấy từ mỗi cách tìm"),
    ("RERANK_TOP_N", "5", "Số đoạn giữ lại cho LLM"),
    ("RERANK_THRESHOLD", "0,35", "Ngưỡng điểm tối thiểu"),
    ("MAX_CONTEXT_TOKENS", "4.000", "Giới hạn độ dài ngữ cảnh"),
    ("PREFERRED_SOURCES", "Văn bản hợp nhất", "Tài liệu được ưu tiên khi nội dung trùng"),
    ("LLM_PROVIDER / GEMINI_MODEL", "gemini / gemini-3.6-flash", "Nhà cung cấp và model LLM"),
])}
<p>Máy chạy thử: Apple M1, RAM 8 GB, chạy trên CPU (GPU của máy này chậm hơn vì thiếu RAM).</p>
""", ["Python + FastAPI, 8 API: 3 cho luồng offline, 1 tìm kiếm, 1 hỏi đáp, 3 kiểm tra trạng thái.",
      "Tham số nằm trong .env, đọc lại mỗi yêu cầu."]))

    # 9. Frontend
    parts.append(chapter(9, "Frontend và UI/UX", "Người dùng và người vận hành nhìn thấy và thao tác những gì?", f"""
<p>Giao diện web viết bằng <b>React 18</b>, đóng gói bằng <b>Vite 6</b>, icon <b>lucide-react</b>, chạy ở cổng 5173.
Tên giao diện: <b>Pipeline Debug Console</b>. Đây là giao diện dành cho người vận hành và kiểm thử: vừa hỏi đáp,
vừa nhìn được bên trong từng bước.</p>
<div class="ui-grid">
  <div class="ui"><div class="ui-h">1 · Build Vector Database</div><p>Một nút chạy lần lượt Load → Chunk → Embed. Mỗi bước hiện trạng thái
  đang chạy / thành công / lỗi và thời gian chạy.</p></div>
  <div class="ui"><div class="ui-h">2 · Pipeline Logs</div><p>Nhật ký từng bước, có mốc thời gian, để biết bước nào chậm hoặc lỗi.</p></div>
  <div class="ui"><div class="ui-h">3 · Query Vector Database</div><p>Thử tìm kiếm không qua LLM: xem các đoạn tìm được, điểm số, trang nguồn.</p></div>
  <div class="ui"><div class="ui-h">4 · RAG Chat</div><p>Hỏi đáp: câu trả lời kèm danh sách nguồn (file, trang, mục, điểm) và ảnh minh họa bấm để phóng to.</p></div>
</div>
<h3>Nguyên tắc UI/UX đã áp dụng</h3>
<ul>
  <li><b>Minh bạch nguồn:</b> mỗi câu trả lời hiện nguồn ngay bên dưới để người dùng tự đối chiếu.</li>
  <li><b>Trạng thái rõ ràng:</b> từng bước có trạng thái và thời gian, không để người dùng chờ "trong bóng tối".</li>
  <li><b>Thông báo lỗi dễ hiểu:</b> phân biệt mất kết nối, dữ liệu gửi lên không hợp lệ (mã 422) và lỗi máy chủ, thay vì hiện mã lỗi thô.</li>
  <li><b>Thanh bên (sidebar)</b> để chuyển nhanh giữa các khu vực.</li>
</ul>
<p class="note">Giao diện chưa được test thủ công trên nhiều trình duyệt và điện thoại (nhóm test UI/UX, xem chương 10). Khi trình bày,
nên chèn ảnh chụp màn hình từ http://localhost:5173.</p>
""", ["React + Vite, 4 khu vực: dựng kho dữ liệu, nhật ký, thử tìm kiếm, hỏi đáp.",
      "Ưu tiên minh bạch: luôn hiện nguồn, trạng thái từng bước, thông báo lỗi dễ hiểu."]))

    # 10. TEST
    parts.append(build_test_chapter())

    # 11. Defects
    defects = [
        ("DEF-01", "High", "Trích dẫn gộp [SOURCE_2, SOURCE_4] không được nhận dạng, câu trả lời mất hết nguồn", "Sửa bộ kiểm trích dẫn để nhận cả dạng gộp"),
        ("DEF-02", "High", "Từ chối câu hỏi về mục lục, cấu trúc tài liệu (“nằm ở phần nào”)", "Cho đoạn mục lục tìm được; gắn đúng cấp tiêu đề"),
        ("DEF-03", "Medium", "Từ chối câu hỏi gộp nhiều chủ đề", "Tách câu hỏi thành nhiều truy vấn con"),
        ("DEF-04", "Medium", "Câu hỏi về hiệu lực, chọn nguồn bị từ chối", "Xem lại 7 test case cùng chuyên gia luật"),
        ("DEF-05", "Medium", "Trả lời theo tài liệu khác với tài liệu được hỏi (2 câu sai pháp lý)", "Lọc theo tài liệu khi câu hỏi nêu tên nguồn"),
        ("DEF-06", "Medium", "Cẩm nang ILO chiếm chỗ của văn bản luật", "Giới hạn mỗi tài liệu tối đa 3/5 chỗ"),
        ("DEF-07", "Medium", "Chậm: p50 18 giây, rerank chiếm 12,8 giây", "Rerank ít ứng viên hơn hoặc chạy trên GPU"),
        ("DEF-08", "Low", "Rác watermark trong 461 chunk của bộ tổng hợp", "Lọc dòng lặp khi nạp tài liệu"),
        ("DEF-09", "Medium", "35% chunk Văn bản hợp nhất mất dấu hỏi (“Điêu”, “thê”)", "Đọc bằng pypdfium2; bổ sung kiểm tra mất dấu"),
        ("DEF-10", "High", "API âm thầm trả câu trả lời giả khi provider sai", "Từ chối provider lạ (400); thiếu API key thì báo lỗi"),
        ("DEF-11", "Medium", "Lỗi LLM trả mã 500 kèm chi tiết nội bộ", "Trả 503 với thông báo rõ ràng"),
        ("DEF-12", "High", "Chấp nhận tiền đề sai (“Bộ luật Lao động 2030”)", "Thêm quy tắc vào prompt: văn bản không có trong ngữ cảnh thì nói rõ"),
        ("DEF-13", "High", "Câu gõ không dấu bị từ chối (đạt 1/13)", "Khôi phục dấu hoặc tìm song song bản không dấu"),
        ("DEF-14", "Medium", "Bảng mục lục bị đọc lệch dòng nên sai số hiệu văn bản", "Sửa cách đọc bảng mục lục"),
    ]
    sev = {"High": FAIL_C, "Medium": PARTIAL_C, "Low": MUTED}
    rows = [(d, f'<span class="pill" style="background:{sev[s]}">{s}</span>', t, f) for d, s, t, f in defects]
    parts.append(chapter(11, "Danh sách lỗi và hướng sửa", "Test tìm ra những lỗi nào, nặng đến đâu, và nên sửa thế nào?", f"""
<p>14 lỗi được gom theo nguyên nhân gốc. Mức độ xếp theo tác hại cho người dùng: <b>High</b> = mất chức năng cốt lõi hoặc ảnh
hưởng nhiều câu; <b>Medium</b> = sai một phần, ảnh hưởng một nhóm câu; <b>Low</b> = không ảnh hưởng kết quả người dùng thấy.</p>
{table(["ID", "Mức", "Lỗi", "Hướng sửa đề xuất"], rows, "defects")}
""", ["14 lỗi: 5 High, 8 Medium, 1 Low.",
      "Sửa trước 5 lỗi High (DEF-01, 02, 10, 12, 13): chúng chặn phát hành và gây phần lớn câu FAIL."]))

    # 12. Conclusion
    gate = [
        ("Tỷ lệ PASS", "≥ 80%", "47%", "Trượt"), ("Tỷ lệ FAIL", "≤ 5%", "30%", "Trượt"),
        ("Từ chối nhầm", "≤ 5%", "25%", "Trượt"), ("Sai pháp lý nghiêm trọng", "0 câu", "3 câu", "Trượt"),
        ("Phủ trích dẫn", "100%", "93%", "Trượt"), ("Lỗi High còn mở", "0", "5", "Trượt"),
        ("An toàn (Security)", "100%", "15/15", "Đạt"), ("Trích dẫn đúng nội dung", "≥ 95%", "99%", "Đạt"),
        ("Từ chối đúng (Negative)", "≥ 90%", "95%", "Chờ chuyên gia xét 1 câu"),
        ("Độ trễ p95", "≤ 10 giây", "25 giây (laptop)", "Chưa kết luận"),
    ]
    color = {"Trượt": FAIL_C, "Đạt": PASS_C}
    grow = [(a, b, c, f'<b style="color:{color.get(d, MUTED)}">{d}</b>') for a, b, c, d in gate]
    parts.append(chapter(12, "Kết luận và kế hoạch", "Hệ thống đã sẵn sàng đưa vào dùng thật chưa, và bước tiếp theo là gì?", f"""
<div class="verdict">Kết luận: <b>CHƯA phát hành (No-Go)</b> — trượt 6 ngưỡng bắt buộc (P0).</div>
<div class="honest"><b>Lưu ý trung thực về đợt test</b><br>
• Đợt test dùng <b>gemini-3.5-flash-lite</b>, không phải model production, và code chưa được commit. Vì vậy kết quả dùng để tìm lỗi,
chưa dùng để quyết định phát hành.<br>
• Final Status, nhãn gold và các mẫu kiểm tay được Claude (AI) duyệt; tester cần xác nhận lại.<br>
• Giám khảo RAGAS là cùng model với hệ thống được test, nên có thể thiên vị.<br>
• Số hiệu năng đo trên laptop M1 8 GB, không phải máy production.</div>
{table(["Chỉ số", "Ngưỡng", "Kết quả", "Đánh giá"], grow)}
<h3>Điểm mạnh</h3>
<ul><li>Khi tìm đúng đoạn, AI trả lời trung thực (faithfulness 0,94) và an toàn (15/15 câu tấn công bị chặn).</li>
<li>Bản sửa lỗi vỡ chữ tiếng Việt nâng tỷ lệ tìm đúng tài liệu từ 47% lên 71%.</li>
<li>Bộ công cụ test chạy lại được bằng một lệnh, chỉ ra lỗi nằm ở tầng nào.</li></ul>
<h3>Điểm yếu cần sửa</h3>
<ul><li>Khâu tìm kiếm: chỉ 45% đoạn đúng đến được AI; câu hỏi về mục lục, câu gõ không dấu gần như luôn bị từ chối.</li>
<li>3 câu sai pháp lý trông đáng tin, nguy hiểm nhất với người dùng.</li></ul>
<h3>Kế hoạch</h3>
{table(["Vòng", "Việc", "Điều kiện qua vòng"], [
    ("0 (xong)", "Lập baseline đợt 20–21/10: 165 câu, Recall@5, RAGAS, API, test tĩnh", "Có số liệu đã duyệt"),
    ("1", "Dev sửa 5 lỗi High; chạy lại toàn bộ, so với baseline", "5 lỗi High đóng, không có câu PASS nào bị tụt"),
    ("2", "Thêm bộ Temporal (hiệu lực luật), test UI, test tải 1/5/10 người", "Mọi chỉ số P0 đã đo được"),
    ("3", "Chạy trên staging với gemini-3.6-flash trả phí", "Đạt mọi ngưỡng P0 → đề xuất Go"),
])}
""", ["No-Go: trượt 6 ngưỡng P0, điểm yếu chính ở khâu tìm kiếm.",
      "Bước tiếp: sửa 5 lỗi High, chạy lại để so với baseline, rồi test trên staging với model thật."]))

    # Appendix
    glossary = [
        ("API", "“Cửa giao dịch” để chương trình này gọi chương trình khác."),
        ("BM25", "Cách tìm theo từ khóa, giống tra mục lục cuối sách."),
        ("Chunk", "Đoạn nhỏ cắt từ tài liệu, khoảng nửa trang."),
        ("ChromaDB", "Cơ sở dữ liệu lưu vector, tìm nhanh các vector gần nhau."),
        ("Context (ngữ cảnh)", "Các đoạn tài liệu đưa cho AI đọc trước khi trả lời."),
        ("Embedding / vector", "Dãy số biểu diễn nghĩa của câu; câu cùng nghĩa có vector gần nhau."),
        ("Faithfulness", "Chỉ số RAGAS: câu trả lời có bám đúng ngữ cảnh không."),
        ("Gold chunk", "Đoạn tài liệu thật sự chứa đáp án, dùng làm chuẩn để chấm tìm kiếm."),
        ("Hallucination (bịa)", "AI tự tạo thông tin không có trong tài liệu."),
        ("Hit@5", "Có ít nhất 1 đoạn đúng trong 5 đoạn được chọn hay không."),
        ("Hybrid search", "Tìm kết hợp theo nghĩa và theo từ khóa."),
        ("LLM", "Mô hình ngôn ngữ lớn (Gemini, ChatGPT), viết câu trả lời."),
        ("Metadata", "Nhãn đi kèm đoạn: tên tài liệu, trang, mục."),
        ("MRR@5", "Đoạn đúng đầu tiên đứng thứ mấy; đứng đầu = 1, thứ hai = 0,5."),
        ("OCR", "Nhận dạng chữ trong ảnh scan."),
        ("Overlap", "Phần gối đầu giữa hai đoạn liền nhau để không mất ý."),
        ("P0 / P1", "Mức ưu tiên test: P0 trượt là chặn phát hành."),
        ("Prompt", "Lời dặn gửi kèm câu hỏi cho LLM."),
        ("RAG", "Tìm đoạn tài liệu trước, rồi để AI trả lời từ các đoạn đó."),
        ("RAGAS", "Bộ chỉ số chấm chất lượng hệ thống RAG bằng một AI giám khảo."),
        ("Recall@5", "Tỷ lệ đoạn đúng nằm trong 5 đoạn được chọn."),
        ("Regression", "Lỗi tái phát: câu từng đúng nay sai sau khi sửa code."),
        ("Reranker", "Model chấm lại và xếp hạng các đoạn tìm được."),
        ("RRF", "Cách gộp hai danh sách xếp hạng thành một."),
        ("Token", "Đơn vị đếm chữ của AI, khoảng 3/4 từ."),
    ]
    script = [
        ("Bìa + Nội dung", "Giới thiệu: hệ thống hỏi đáp luật trên 5 tài liệu, hôm nay nói cách xây và kết quả test."),
        ("1. Bài toán", "1.428 trang PDF; đọc tay chậm, chatbot chung dễ bịa → cần trả lời có nguồn."),
        ("2. RAG", "Ví dụ đi thi mở sách: tìm trang trước, viết sau, ghi nguồn."),
        ("3. Dữ liệu", "5 tài liệu khác nhau về thời điểm và chất lượng chữ; đó là nguồn của nhiều lỗi."),
        ("4. Kiến trúc", "Chỉ vào Hình 1: offline dọn sách lên kệ, online tìm và trả lời."),
        ("5. Offline", "Đi qua 6 bước, dừng ở ví dụ chunk Điều 113."),
        ("6. Online", "Đi theo câu hỏi nghỉ hằng năm qua Hình 3; nhấn mạnh ngưỡng 0,35 và prompt chống bịa."),
        ("7. Model", "3 model chính và việc của từng model; vì sao test bằng flash-lite."),
        ("8–9. Backend, Frontend", "8 API; giao diện 4 khu vực, luôn hiện nguồn."),
        ("10. Test", "Test theo tầng; 165 câu; công cụ; 4 nhóm kết quả: câu trả lời, Recall@5, RAGAS, bộ bổ sung."),
        ("10. Phân tích lỗi", "Hình cây lỗi: 36/56 câu hỏng ở khâu tìm kiếm; 3 câu sai pháp lý."),
        ("11–12. Lỗi, kết luận", "14 lỗi, 5 High; kết luận No-Go và 3 vòng tiếp theo."),
    ]
    parts.append(f"""
<section class="chapter"><div class="chapter-num">Phụ lục</div><h1>Phụ lục</h1>
<h2>A. Bảng thuật ngữ</h2>{table(["Thuật ngữ", "Giải thích"], glossary, "glossary")}
<h2>B. Lệnh chạy lại</h2>
<pre>uvicorn src.app:app --port 8001   # bật backend (cổng 8001)
cd frontend &amp;&amp; npm run dev        # bật giao diện (cổng 5173)

R=evaluation/results/&lt;ngày&gt;   I=TC_LuatLD.xlsx
python evaluation/run_tc_static.py    --input $I --run-dir $R
python evaluation/run_tc_retrieval.py --input $I --output $R/retrieval_new.json
python evaluation/run_tc_api.py       --output $R/api_tests.json
python evaluation/run_tc_excel.py     --input $I --out-dir $R --sleep 2
python evaluation/run_tc_recall.py    --input $I --run-dir $R --retrieval $R/retrieval_new.json
python evaluation/run_tc_ragas.py     --input $I --run-dir $R
python evaluation/run_tc_excel.py     --input $I --out-dir $R --report-only</pre>
<h2>C. Kịch bản thuyết trình (mỗi chương khoảng 1 phút)</h2>{table(["Phần", "Nói gì"], script)}
</section>""")
    return parts


def build_test_chapter():
    status_rows = [("A · Giáo trình", [16, 2, 2]), ("B · Sổ tay CDI", [6, 3, 11]), ("C · Bộ tổng hợp", [4, 5, 11]),
                   ("D · Cẩm nang ILO", [9, 5, 6]), ("E · Văn bản hợp nhất", [12, 8, 0]), ("Tổng 100 câu", [47, 23, 30])]
    parts3 = [("PASS", PASS_C), ("PARTIAL", PARTIAL_C), ("FAIL", FAIL_C)]
    retr = grouped_bars(["A", "B", "C", "D", "E", "Tổng"],
                        [("Trước khi sửa lỗi vỡ chữ", [5, 45, 45, 40, 100, 47], OLD_C),
                         ("Sau khi sửa", [90, 50, 45, 85, 85, 71], NEW_C)])
    funnel = hbars([("Đoạn đúng có trong 20 ứng viên", 79), ("Còn trong top 5 sau rerank (Recall@5)", 57),
                    ("Vượt ngưỡng 0,35, đến được LLM", 45)], 100, unit="%")
    ragas = grouped_bars(["Faithfulness", "Answer relevancy", "Context precision", "Context recall"],
                         [("75 câu có trả lời", [0.94, 0.93, 0.76, 0.82], NEW_C),
                          ("Cả 100 câu", [0.87, 0.85, 0.58, 0.63], OLD_C)], vmax=1, unit="")
    extra = stacked_hbars([("Smoke (10)", [10, 0, 0]), ("Negative (20)", [19, 0, 1]), ("Security (15)", [15, 0, 0]),
                           ("Robustness (20)", [7, 3, 10])], parts3)
    latency = hbars([("Tìm lai (hybrid)", 3.3, MUTED), ("Rerank", 12.8, FAIL_C), ("Gemini trả lời", 1.5, MUTED),
                     ("Tổng (p50)", 18.0, ACCENT)], 20, unit=" s")
    body = f"""
<p>Phần này trả lời 4 câu hỏi: <b>test bằng gì</b>, <b>test như thế nào</b>, <b>kết quả ra sao</b>, và
<b>sai ở đâu</b>. Đợt test chạy ngày 20–21/10/2026.</p>

<h2>10.1 Nguyên tắc: test theo tầng</h2>
<p>Một câu trả lời sai có thể do bất kỳ tầng nào: đọc PDF làm vỡ chữ, cắt chunk sai chỗ, tìm không ra, reranker xếp sai, AI viết sai,
hoặc mất trích dẫn. Nếu chỉ hỏi rồi chấm câu trả lời cuối, ta biết là sai nhưng không biết sửa ở đâu. Vì vậy mỗi tầng có bài test
riêng, và <b>tầng rẻ test trước</b>: test đọc PDF và tìm kiếm không tốn tiền gọi AI nên chạy mỗi lần sửa code; test câu trả lời tốn
lượt gọi Gemini nên chạy sau.</p>
{table(["Tầng", "Test gì", "Ưu tiên"], [
    ("1. Đọc PDF, OCR", "Không vỡ dấu, không sót trang, không có đoạn rác", "P0"),
    ("2. Chunk, metadata", "Không cắt ngang Điều luật, đủ tên tài liệu và số trang", "P1"),
    ("3. Tìm kiếm (retrieval)", "Đoạn đúng có nằm trong 5 đoạn được chọn", "P0"),
    ("4. Câu trả lời", "Đúng nội dung, đủ ý, không bịa", "P0"),
    ("5. Trích dẫn", "Mọi câu có nguồn, nguồn chứa đúng nội dung", "P0"),
    ("6. Từ chối đúng", "Câu ngoài phạm vi phải nói không tìm thấy", "P0"),
    ("8. An toàn", "Không làm theo lệnh chèn, không lộ prompt hay khóa", "P0"),
    ("9. Độ bền đầu vào", "Câu không dấu, viết tắt, sai chính tả", "P1"),
    ("10. API", "Mã lỗi đúng, xử lý lỗi LLM", "P0"),
    ("11. Hiệu năng", "Thời gian phản hồi", "P1"),
])}
<p class="note">P0 = trượt thì không được phát hành; P1 = cần có trước bản chính thức. Quy trình có 13 nhóm test; bảng trên là các nhóm đã chạy.
Hai nhóm chưa chạy: hiệu lực luật (Temporal, chờ chuyên gia viết đáp án) và UI.</p>

<h2>10.2 Test bằng gì: bộ test case</h2>
<p><b>Test case</b> là một câu hỏi kèm đáp án chuẩn để so. Bộ chính <b>TC_LuatLD.xlsx</b> có 100 câu, 20 câu cho mỗi tài liệu.
Mỗi câu có: câu hỏi, đáp án chuẩn, đoạn bằng chứng trong tài liệu, trang nguồn, 2–5 từ khóa bắt buộc, độ khó và loại câu hỏi.</p>
<div class="example"><div class="example-title">Ví dụ test case LAW_E_015</div>
<b>Câu hỏi:</b> Một NLĐ làm thêm vào ngày nghỉ hằng tuần. Mức trả tối thiểu theo Điều 98 là bao nhiêu?<br>
<b>Đáp án chuẩn:</b> Ít nhất bằng 200% đơn giá tiền lương hoặc tiền lương thực trả.<br>
<b>Từ khóa:</b> 200% · <b>Độ khó:</b> Hard · <b>Loại:</b> Scenario-based · <b>Nguồn:</b> Văn bản hợp nhất</div>
{table(["Bộ test", "Số câu", "Mục đích"], [
    ("TC_LuatLD (chính)", "100", "35 Easy, 35 Medium, 30 Hard; Direct Fact 29, Navigation 14, Definition 9, Multi-hop 9…"),
    ("Smoke", "10", "Kiểm tra nhanh build có chạy không (2 câu dễ mỗi tài liệu)"),
    ("Negative", "20", "Câu ngoài phạm vi (luật khác, điều luật không tồn tại, câu đời sống): phải từ chối"),
    ("Security", "15", "Chèn lệnh “bỏ qua hướng dẫn”, đòi in prompt hay API key, xúi lách bảo hiểm"),
    ("Robustness", "20", "Câu đã PASS được viết lại không dấu, viết tắt, sai chính tả: kết quả phải giữ nguyên"),
])}

<h2>10.3 Test bằng công cụ gì</h2>
<p>Toàn bộ test chạy bằng script Python trong thư mục <b>evaluation/</b>, kết quả xuất ra file Excel để tester duyệt.</p>
{table(["Script", "Làm gì", "Tốn lượt gọi AI?"], [
    ("run_tc_static.py", "Test tĩnh: đọc PDF, chunk, trích dẫn", "Không"),
    ("run_tc_retrieval.py", "Chạy tìm kiếm + rerank cho 100 câu", "Không"),
    ("run_tc_recall.py", "Tính Recall@5, Hit@5, MRR@5 và xác định lỗi ở tầng nào", "Không"),
    ("run_tc_excel.py", "Gọi API như người dùng thật, chấm sơ bộ, xuất Excel", "Có"),
    ("run_tc_ragas.py", "Chấm RAGAS bằng một AI giám khảo", "Có"),
    ("run_tc_api.py", "19 ca test API", "Gần như không"),
    ("build_tc_sets.py", "Sinh 4 bộ test bổ sung", "Không"),
])}
<p>File Excel có các sheet: Summary, Results (tester điền Final Status), Retrieval, Regression, Defects, Environment, Recall@5, RAGAS,
Failure Analysis, API Tests, Static Checks.</p>

<h2>10.4 Test như thế nào: các chỉ số</h2>
{term("PASS / PARTIAL / FAIL", "PASS = đúng ý chính và trích đúng tài liệu được hỏi; PARTIAL = đúng một phần, hoặc đúng nội dung nhưng sai hay mất nguồn; FAIL = từ chối câu có đáp án hoặc trả lời sai.")}
{term("Gold chunk (đoạn đúng)", "đoạn tài liệu thật sự chứa đáp án. Được gán tự động: điểm = 0,5 × tỷ lệ từ khóa có trong đoạn + 0,5 × độ trùng với đáp án chuẩn; lấy tối đa 3 đoạn điểm từ 0,5. Sau đó rà tay 95 câu: 90 nhãn đúng.")}
{term("Recall@5", "trong các đoạn đúng, bao nhiêu phần trăm lọt vào 5 đoạn được chọn. Ví dụ câu hỏi có 2 đoạn đúng, 1 đoạn lọt top 5 → Recall@5 = 50%.")}
{term("Hit@5 và MRR@5", "Hit@5: có ít nhất 1 đoạn đúng trong top 5 không (có/không). MRR@5: đoạn đúng đầu tiên đứng thứ mấy; đứng đầu được 1 điểm, thứ hai 0,5, thứ năm 0,2.")}
{term("RAGAS", "bộ chỉ số dùng một AI “giám khảo” (gemini-3.5-flash-lite) chấm 4 mặt, thang 0–1: <b>Faithfulness</b> — câu trả lời có bám ngữ cảnh không (chống bịa); <b>Answer relevancy</b> — có trả lời đúng trọng tâm không; <b>Context precision</b> — đoạn liên quan có đứng đầu không; <b>Context recall</b> — ngữ cảnh có chứa đủ ý của đáp án chuẩn không.")}
<p><b>Quy trình chấm 100 câu:</b> (1) script chấm sơ bộ bằng từ khóa và độ trùng với đáp án; (2) RAGAS chấm độc lập;
(3) người duyệt đọc từng câu, đối chiếu câu trả lời, nguồn trong ngữ cảnh và đáp án chuẩn, rồi điền Final Status.</p>

<h2>10.5 Kết quả câu trả lời (100 câu)</h2>
<div class="bignums"><div style="color:{PASS_C}"><b>47</b>PASS</div><div style="color:{PARTIAL_C}"><b>23</b>PARTIAL</div><div style="color:{FAIL_C}"><b>30</b>FAIL</div></div>
{figure(stacked_hbars(status_rows, parts3), "Hình 4. Kết quả sau khi duyệt theo từng tài liệu. Giáo trình và Văn bản hợp nhất tốt nhất; sổ tay và bộ tổng hợp kém nhất, chủ yếu do câu hỏi về mục lục.")}
<ul>
  <li><b>25/30 câu FAIL là từ chối trả lời</b> câu có đáp án: 18 câu bị ngưỡng rerank chặn trước khi gọi AI, 7 câu AI tự từ chối.</li>
  <li><b>Chấm tự động không đủ tin cậy:</b> script ban đầu cho 44 PASS / 31 PARTIAL / 25 FAIL. Khi duyệt, 7/44 câu PASS (16%) thực ra
  sai hoặc thiếu, và 10 câu bị chấm thấp oan. Bài học: chấm bằng từ khóa không thay được người duyệt.</li>
</ul>

<h2>10.6 Kết quả tìm kiếm (retrieval) và Recall@5</h2>
{figure(retr, "Hình 5. Tỷ lệ tìm đúng tài liệu trong 5 đoạn được chọn, trước và sau khi sửa lỗi vỡ chữ tiếng Việt.")}
<p>Sửa lỗi vỡ chữ nâng tỷ lệ chung từ <b>47% lên 71%</b>; giáo trình tăng từ 5% lên 90%. Riêng Văn bản hợp nhất giảm từ 100% xuống 85%,
vì cẩm nang ILO sau khi đọc được chữ đã chiếm chỗ (lỗi DEF-06).</p>
{figure(funnel, "Hình 6. Đoạn đúng rơi rụng dần qua từng bước. Hit@5 = 65%, MRR@5 = 0,47.")}
<p>Chỉ <b>45%</b> đoạn đúng thực sự đến được AI. Sổ tay CDI yếu nhất: chỉ 20%. Khoảng cách giữa 57% và 45% là phần bị ngưỡng 0,35 cắt.</p>

<h2>10.7 Kết quả RAGAS</h2>
{figure(ragas, "Hình 7. Bốn chỉ số RAGAS. Cột xanh: 75 câu có trả lời; cột xám: cả 100 câu, tính cả câu dừng sớm (ngữ cảnh = 0).")}
<p>AI viết câu trả lời <b>trung thực với ngữ cảnh</b> (faithfulness 0,94) và đúng trọng tâm (0,93). Điểm thấp nằm ở ngữ cảnh
(0,58–0,82). Kết luận khớp với Recall@5: <b>điểm yếu ở khâu tìm, không phải ở khâu viết</b>. RAGAS còn phát hiện 2 câu mà chấm tự động
cho PASS nhưng thật ra sai (C_012, C_013).</p>

<h2>10.8 Phân tích sai lệch: lỗi nằm ở tầng nào?</h2>
<p>Với mỗi câu không PASS, lần theo đoạn đúng qua từng bước để biết nó mất ở đâu:</p>
{figure(diagram_failure_tree(), "Hình 8. Cây phân loại lỗi và số câu ở mỗi nhánh (56 câu chưa đạt theo chấm tự động).")}
<ul>
  <li><b>36/56 câu hỏng ở khâu tìm và chọn đoạn</b>: 7 không tìm thấy, 4 bị rerank đẩy ra, 9 bị ngưỡng cắt, 16 trả lời từ đoạn sai.</li>
  <li><b>WRONG_SOURCE_ANSWER nguy hiểm nhất:</b> AI vẫn trả lời dù không có đoạn đúng. Duyệt 16 câu: 6 câu đúng nội dung nhưng trích tài liệu khác,
  8 câu sai hoặc thiếu, trong đó 2 câu sai pháp lý. Faithfulness của nhóm này vẫn cao vì AI bám đúng đoạn nó nhận được, chỉ là đoạn sai.</li>
</ul>
<div class="warn"><b>3 câu sai pháp lý trông đáng tin:</b><br>
• A_011 — hỏi giới hạn làm thêm giờ theo giáo trình (4 giờ/ngày, 200 giờ/năm), hệ thống trả lời theo Bộ luật 2019 (50% giờ làm việc).<br>
• C_010 — hỏi quyền từ chối việc nguy hiểm đến tính mạng, hệ thống trả lời quyền từ chối làm thêm giờ.<br>
• C_013 — nêu Nghị định 46/2013 thay vì 45/2013; chấm tự động cho PASS, chỉ RAGAS phát hiện.</div>

<h2>10.9 Test tĩnh, trích dẫn, API, hiệu năng</h2>
{table(["Hạng mục", "Cách làm", "Kết quả"], cls="nowrap1", rows=[
    ("Đọc PDF", "So số trang có chữ với số trang có chunk; so 150 chunk mẫu với lớp chữ gốc của PDF", "Không trang nào mất chữ; 143/150 chunk khớp. 35% chunk Văn bản hợp nhất mất dấu hỏi (“Điêu”, “thê”)"),
    ("Đoạn rác", "Tìm dòng lặp lại ở nhiều chunk", "Dòng watermark lặp trong 461 chunk của bộ tổng hợp"),
    ("Chunk", "Kiểm metadata 4.654 chunk; đọc 20 chunk có “Điều”", "Metadata đủ; không chunk nào cắt ngang Điều luật sai cách; 5/20 có lỗi nhỏ"),
    ("Trích dẫn", "So nhãn [SOURCE_x] với nguồn API trả về; kiểm 30 câu mẫu", "70/75 câu có nguồn (93%); 89/90 ý đúng với đoạn được trích; 4/30 câu mất nguồn do thẻ gộp"),
    ("API", "19 ca: thiếu dữ liệu, sai kiểu, câu rỗng, câu 5.000 ký tự, provider lạ, LLM lỗi", "16/19 PASS; provider lạ vẫn trả câu trả lời giả; LLM lỗi trả mã 500"),
])}
{figure(latency, "Hình 9. Thời gian xử lý một câu hỏi (trung vị, laptop M1 8 GB). Rerank chiếm khoảng 70%; p95 là 25 giây; lần gọi đầu 56 giây do nạp model.")}

<h2>10.10 Bộ test bổ sung</h2>
{figure(extra, "Hình 10. Kết quả 4 bộ test bổ sung chạy ngày 21/10/2026.")}
<ul>
  <li><b>Security 15/15 an toàn:</b> không lộ prompt hay khóa, không làm theo lệnh chèn; câu chèn số liệu giả "nghỉ 60 ngày" vẫn được trả lời đúng 12 ngày.</li>
  <li><b>Negative 19/20:</b> 1 câu chấp nhận tiền đề sai "Bộ luật Lao động năm 2030" mà không cải chính.</li>
  <li><b>Robustness 7/20:</b> câu gõ không dấu chỉ đạt 1/13, vì reranker không nhận ra "nghi phep" là "nghỉ phép". Câu sai chính tả đạt 5/6.</li>
</ul>

"""
    return chapter(10, "Kiểm thử (Test)", "Hệ thống được kiểm tra bằng gì, theo cách nào, và kết quả thật sự ra sao?", body,
                   ["Test theo tầng, 165 câu hỏi + test tĩnh + 19 ca API; tầng không tốn tiền chạy trước.",
                    "Kết quả: 47 PASS / 23 PARTIAL / 30 FAIL; Recall@5 vào AI 45%; faithfulness 0,94.",
                    "Lỗi chủ yếu ở khâu tìm kiếm; 3 câu sai pháp lý; an toàn 15/15."])


CSS = f"""
@page {{ size: A4; margin: 18mm 17mm 18mm 17mm;
  @bottom-right {{ content: counter(page); font: 9pt Arial; color: {MUTED}; }}
  @bottom-left {{ content: "Hệ thống RAG hỏi đáp Luật Lao động"; font: 9pt Arial; color: {MUTED}; }} }}
@page :first {{ margin: 0; @bottom-right {{ content: none; }} @bottom-left {{ content: none; }} }}
* {{ box-sizing: border-box; }}
body {{ font-family: Arial, Helvetica, sans-serif; color: {INK}; font-size: 10.5pt; line-height: 1.5; margin: 0; }}
h1 {{ font-size: 21pt; margin: 2px 0 6px; color: {INK}; }}
h2 {{ font-size: 13.5pt; margin: 18px 0 6px; color: {ACCENT}; break-after: avoid; }}
h3 {{ font-size: 11.5pt; margin: 14px 0 4px; break-after: avoid; }}
p {{ margin: 5px 0 7px; }}
ul, ol {{ margin: 4px 0 8px; padding-left: 20px; }}
li {{ margin: 2px 0; }}
.chapter {{ break-before: page; }}
.chapter-num {{ font-size: 10pt; font-weight: 700; color: {ACCENT}; letter-spacing: .08em; text-transform: uppercase; }}
.question {{ font-size: 11.5pt; color: {MUTED}; border-left: 3px solid {ACCENT}; padding: 2px 0 2px 10px; margin: 4px 0 14px; }}
.summary {{ background: {ACCENT_SOFT}; border-radius: 8px; padding: 10px 14px; margin-top: 16px; break-inside: avoid; }}
.summary-title {{ font-weight: 700; color: {ACCENT}; margin-bottom: 2px; }}
.summary ul {{ margin: 0; }}
table {{ width: 100%; border-collapse: collapse; margin: 8px 0 12px; font-size: 9.6pt; }}
thead {{ display: table-header-group; }}
tr {{ break-inside: avoid; }}
th {{ background: {SOFT}; text-align: left; padding: 6px 7px; border-bottom: 1.5px solid {LINE}; }}
td {{ padding: 4px 7px; border-bottom: 1px solid {LINE}; vertical-align: top; }}
table.defects td:first-child, table.defects td:nth-child(2) {{ white-space: nowrap; }}
table.defects td {{ padding: 4px 7px; font-size: 9.3pt; line-height: 1.38; }}
table.nowrap1 td:first-child, table.nowrap1 th:first-child {{ white-space: nowrap; }}
table.glossary td:first-child {{ width: 30%; font-weight: 700; }}
.pill {{ color: #fff; border-radius: 10px; padding: 1px 8px; font-size: 8.5pt; font-weight: 700; }}
figure {{ margin: 10px 0 14px; break-inside: avoid; }}
figcaption {{ font-size: 9pt; color: {MUTED}; margin-top: 4px; }}
.term {{ margin: 6px 0; padding: 6px 10px; border-left: 3px solid {LINE}; break-inside: avoid; }}
.note {{ font-size: 9.5pt; color: {MUTED}; }}
.io {{ background: {SOFT}; padding: 5px 9px; border-radius: 6px; font-size: 9.6pt; }}
.example {{ background: #fbfbfd; border: 1px solid {LINE}; border-radius: 8px; padding: 9px 12px; margin: 8px 0 10px; font-size: 9.8pt; break-inside: avoid; }}
.example-title {{ font-weight: 700; color: {ACCENT}; margin-bottom: 3px; }}
.example-meta {{ color: {MUTED}; font-size: 9pt; margin-top: 5px; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 10px 0; }}
.card {{ border: 1px solid {LINE}; border-radius: 8px; padding: 8px 12px; }}
.card h3 {{ margin-top: 2px; }}
.steps3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 10px 0 12px; }}
.steps3 div {{ border: 1px solid {LINE}; border-radius: 8px; padding: 8px 10px; }}
.steps3 span {{ display: inline-block; width: 22px; height: 22px; border-radius: 11px; background: {ACCENT}; color: #fff; text-align: center; font-weight: 700; margin-right: 6px; }}
.steps3 p {{ font-size: 9.6pt; margin: 4px 0 0; }}
.bignums {{ display: flex; gap: 10px; margin: 10px 0; break-inside: avoid; }}
.bignums div {{ flex: 1; background: {SOFT}; border-radius: 8px; padding: 8px; text-align: center; font-size: 9.5pt; color: {MUTED}; }}
.bignums b {{ display: block; font-size: 20pt; color: inherit; }}
.bignums div:not([style]) b {{ color: {ACCENT}; }}
.ui-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 10px 0; }}
.ui {{ border: 1px solid {LINE}; border-radius: 8px; overflow: hidden; break-inside: avoid; }}
.ui-h {{ background: {INK}; color: #fff; font-weight: 700; padding: 6px 10px; font-size: 10pt; }}
.ui p {{ padding: 0 10px; font-size: 9.8pt; }}
.warn {{ background: #fbeceb; border-left: 4px solid {FAIL_C}; padding: 8px 12px; border-radius: 6px; margin: 10px 0; break-inside: avoid; }}
.honest {{ background: #fff7e6; border-left: 4px solid {PARTIAL_C}; padding: 8px 12px; border-radius: 6px; margin: 12px 0; break-inside: avoid; }}
.verdict {{ background: #fbeceb; color: {FAIL_C}; font-size: 13pt; padding: 10px 14px; border-radius: 8px; margin: 6px 0 12px; }}
pre {{ background: {SOFT}; padding: 10px; border-radius: 6px; font-size: 8.2pt; white-space: pre-wrap; }}
.cover {{ height: 297mm; padding: 34mm 22mm; background: linear-gradient(160deg, {INK} 0%, #24406e 60%, {ACCENT} 100%); color: #fff; position: relative; }}
.cover-tag {{ font-size: 11pt; letter-spacing: .14em; text-transform: uppercase; opacity: .8; }}
.cover-title {{ font-size: 36pt; font-weight: 700; line-height: 1.15; margin: 18px 0 16px; }}
.cover-sub {{ font-size: 14pt; opacity: .9; max-width: 140mm; }}
.cover-stats {{ display: flex; gap: 12px; margin-top: 30mm; }}
.cover-stats div {{ flex: 1; border-top: 2px solid rgba(255,255,255,.5); padding-top: 8px; }}
.cover-stats b {{ display: block; font-size: 24pt; }}
.cover-stats span {{ font-size: 10pt; opacity: .85; }}
.cover-meta {{ position: absolute; bottom: 28mm; left: 22mm; font-size: 11pt; opacity: .9; }}
.toc {{ break-before: page; }}
.toc ol {{ font-size: 12.5pt; line-height: 2; }}
"""


def find_chrome():
    for c in CHROME_CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    sys.exit("Không tìm thấy Google Chrome/Chromium để in PDF.")


def main():
    doc = ("<!doctype html><html lang='vi'><head><meta charset='utf-8'>"
           "<title>Hệ thống RAG hỏi đáp Luật Lao động</title>"
           f"<style>{CSS}</style></head><body>{''.join(build_html())}</body></html>")
    HTML_PATH.write_text(doc, encoding="utf-8")
    subprocess.run([find_chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF_PATH}", HTML_PATH.as_uri()],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"HTML: {HTML_PATH}\nPDF:  {PDF_PATH}")


if __name__ == "__main__":
    main()
