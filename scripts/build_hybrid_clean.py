import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("data/staging/5pdf_tesseract/processed")
ORIGINAL = ROOT / "elements.jsonl"
REPORT = ROOT / "hybrid_report.json"
OUTPUT = ROOT / "elements_clean.jsonl"
QA = ROOT / "hybrid_clean_qa.json"

def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

original = read_jsonl(ORIGINAL)
report = json.loads(REPORT.read_text(encoding="utf-8"))

by_page = defaultdict(list)
for e in original:
    by_page[(e["document_id"], int(e["page"]))].append(e)

# Pages where the structured table is important.
# TOC pages are excluded from this override.
important_table_pages = set()

toc_pages = {
    "ILO-VCCI_Labour-Law-Guide-Book-V04-Final": {5, 6, 7, 8},
    "giao trinh Luat lao dong CB 2005 (1)": {1},
    "luu-ban-nhap-tu-dong-3-4": {2, 3, 4, 5},
}

for key, elements in by_page.items():
    doc, page = key
    if page in toc_pages.get(doc, set()):
        continue
    if any(e.get("type") == "table" for e in elements):
        important_table_pages.add(key)

# Reuse the first hybrid candidate's PDF text elements.
candidate_path = ROOT / "elements_hybrid_candidate.jsonl"
candidate = read_jsonl(candidate_path)
candidate_by_page = defaultdict(list)

for e in candidate:
    candidate_by_page[(e["document_id"], int(e["page"]))].append(e)

output = []
page_sources = []
missing = []
seen_ids = set()

for r in report:
    key = (r["document_id"], int(r["page"]))
    original_page = by_page.get(key, [])
    candidate_page = candidate_by_page.get(key, [])

    if key in important_table_pages:
        selected = original_page
        source = "docling_structured"
    elif r["method"] == "pdf_text":
        selected = candidate_page
        source = "pdf_text"
    else:
        selected = original_page
        source = "docling_ocr"

    if not selected:
        missing.append({"document_id": key[0], "page": key[1]})
        continue

    for e in selected:
        item = dict(e)
        item["extraction_source"] = source

        # Avoid collisions between generated and original element IDs.
        identity = (item["document_id"], item.get("element_id"))
        if identity in seen_ids:
            item["element_id"] = f"{item.get('element_id', 'element')}_p{item['page']}"
            identity = (item["document_id"], item["element_id"])
        seen_ids.add(identity)

        output.append(item)

    page_sources.append({
        "document_id": key[0],
        "page": key[1],
        "source": source,
        "elements": len(selected),
        "tables": sum(e.get("type") == "table" for e in selected),
    })

with OUTPUT.open("w", encoding="utf-8") as f:
    for e in output:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

source_counts = Counter(r["source"] for r in page_sources)
table_counts = Counter(
    e["document_id"] for e in output if e.get("type") == "table"
)

qa = {
    "output": str(OUTPUT),
    "elements": len(output),
    "pages": len(page_sources),
    "source_counts": dict(source_counts),
    "table_counts": dict(table_counts),
    "missing_pages": missing,
    "important_table_pages": len(important_table_pages),
    "warning": (
        "Conservative page-level hybrid. Docling table pages "
        "may still contain fragmented Vietnamese text. "
        "Requires validation before embedding."
    ),
}

QA.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")

print("=" * 60)
print("HYBRID CLEAN GENERATED")
print("Elements:", len(output))
print("Pages:", len(page_sources))
print("Sources:", dict(source_counts))
print("Tables:", dict(table_counts))
print("Important table pages:", len(important_table_pages))
print("Missing pages:", len(missing))
print("Output:", OUTPUT)
print("QA report:", QA)
print("=" * 60)
