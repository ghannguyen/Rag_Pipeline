import json
import re
from collections import defaultdict
from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path("data/staging/5pdf_tesseract/processed")
INPUT = ROOT / "elements.jsonl"
OUTPUT = ROOT / "elements_hybrid_candidate.jsonl"
REPORT = ROOT / "hybrid_report.json"

PDF_ROOT = Path("data/raw")

# These documents have shown substantial text-layer advantages.
PREFER_PDF = {
    "ILO-VCCI_Labour-Law-Guide-Book-V04-Final",
    "giao trinh Luat lao dong CB 2005 (1)",
    "Văn-bản-hợp-nhất-18-VBHN-VPQH",
    "luu-ban-nhap-tu-dong-3-4",
}

# CDI has significant image/layout content; retain Docling for now.
PREFER_DOCLING = {
    "so-tay-kien-thuc-phap-luat-lao-dong-cdi",
}

def normalize_text(text):
    return re.sub(r"\s+", " ", text or "").strip()

elements = []
with INPUT.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            elements.append(json.loads(line))

by_page = defaultdict(list)
for e in elements:
    by_page[(e["document_id"], e["page"])].append(e)

documents = sorted({e["document_id"] for e in elements})
output = []
report = []

for doc_id in documents:
    path = PDF_ROOT / f"{doc_id}.pdf"
    pdf = pdfium.PdfDocument(str(path))

    for page_number in range(1, len(pdf) + 1):
        originals = by_page.get((doc_id, page_number), [])

        page = pdf[page_number - 1]
        raw_text = normalize_text(
            page.get_textpage().get_text_range()
        )
        page.close()

        docling_text = normalize_text(
            " ".join(
                e.get("text") or ""
                for e in originals
                if e.get("type") in ("paragraph", "heading")
            )
        )

        special = [
            e for e in originals
            if e.get("type") in ("table", "image", "picture", "figure")
        ]

        use_pdf = (
            doc_id in PREFER_PDF
            and doc_id not in PREFER_DOCLING
            and len(raw_text) >= 100
        )

        if use_pdf:
            # One text element per page. This is a candidate only:
            # original heading/paragraph structure is not retained.
            output.append({
                "document_id": doc_id,
                "element_id": f"hybrid_page_{page_number:04d}",
                "type": "paragraph",
                "text": raw_text,
                "page": page_number,
                "level": None,
                "table_data": None,
            })

            # Keep images for later metadata and visual inspection.
            # Tables are intentionally not duplicated in this candidate.
            output.extend(
                e for e in special
                if e.get("type") in ("image", "picture", "figure")
            )

            method = "pdf_text"
        else:
            output.extend(originals)
            method = "docling"

        report.append({
            "document_id": doc_id,
            "page": page_number,
            "method": method,
            "pdf_chars": len(raw_text),
            "docling_text_chars": len(docling_text),
            "special_elements": len(special),
            "tables_not_copied": (
                sum(e.get("type") == "table" for e in special)
                if use_pdf else 0
            ),
        })

    pdf.close()

with OUTPUT.open("w", encoding="utf-8") as f:
    for e in output:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

REPORT.write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("Hybrid candidate:", OUTPUT)
print("Elements:", len(output))
print("Report:", REPORT)

for doc_id in documents:
    rows = [r for r in report if r["document_id"] == doc_id]
    pdf_pages = sum(r["method"] == "pdf_text" for r in rows)
    docling_pages = sum(r["method"] == "docling" for r in rows)
    omitted_tables = sum(r["tables_not_copied"] for r in rows)

    print(f"\n{doc_id}")
    print("PDF text pages:", pdf_pages)
    print("Docling pages:", docling_pages)
    print("Tables requiring review:", omitted_tables)
