"""
Shared helpers for the TC_LuatLD test runs (run_tc_excel.py, run_tc_retrieval.py).
"""
import hashlib
import re
import unicodedata
from pathlib import Path

# Test case ID prefix (LAW_<group>_NNN) -> source PDF in data/raw.
GROUP_DOCS = {
    "A": "giao trinh Luat lao dong CB 2005 (1).pdf",
    "B": "so-tay-kien-thuc-phap-luat-lao-dong-cdi.pdf",
    "C": "luu-ban-nhap-tu-dong-3-4.pdf",
    "D": "ILO-VCCI_Labour-Law-Guide-Book-V04-Final.pdf",
    "E": "Văn-bản-hợp-nhất-18-VBHN-VPQH.pdf",
}

# Abbreviations used in the test file and the documents; either form counts.
ABBREVIATIONS = {
    "nlđ": "người lao động",
    "nsdlđ": "người sử dụng lao động",
    "hđlđ": "hợp đồng lao động",
    "bhxh": "bảo hiểm xã hội",
    "bllđ": "bộ luật lao động",
    "atvslđ": "an toàn, vệ sinh lao động",
    "tnlđ": "tai nạn lao động",
    "qhlđ": "quan hệ lao động",
}

STOPWORDS = {
    "là", "và", "của", "các", "những", "được", "có", "không", "cho", "với", "theo", "trong",
    "một", "này", "đó", "để", "khi", "thì", "hoặc", "từ", "đến", "về", "tại", "bởi", "như",
    "gồm", "ra", "nào", "gì", "sao", "đã", "đang", "sẽ", "cũng", "nếu", "mà", "phải", "tài", "liệu",
}

FIELDS = {
    "id": "Test Case ID",
    "category": "Category",
    "query": "Query",
    "expected": "Expected Answer",
    "page": "Source Page",
    "section": "Source Section",
    "keywords": "Required Keywords / Key Concepts",
    "difficulty": "Retrieval Difficulty",
    "retrieval_type": "Expected Retrieval Type",
    "answerable": "Is Answerable",
    "behavior": "Expected Behavior",
    # Optional (TC_Security.xlsx): text that must never appear in the answer.
    "forbidden": "Forbidden Keywords",
}


def norm(text) -> str:
    return unicodedata.normalize("NFC", str(text or "")).lower().strip()


def words(text) -> list:
    return re.findall(r"\w+", norm(text))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases(path: Path, sheet: str = None):
    """Return (header, cases). Each case keeps the original row plus normalized fields."""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet] if sheet else wb.worksheets[0]
    rows = [r for r in ws.iter_rows(values_only=True) if any(c not in (None, "") for c in r)]
    wb.close()

    # The evidence column has no header in TC_LuatLD.xlsx.
    header = [str(c).strip() if c not in (None, "") else ("Evidence" if i == 4 else f"Column {i + 1}")
              for i, c in enumerate(rows[0])]
    missing = [v for v in (FIELDS["id"], FIELDS["query"], FIELDS["expected"]) if v not in header]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    cases = []
    for raw in rows[1:]:
        row = dict(zip(header, raw))
        case_id = str(row[FIELDS["id"]]).strip()
        group = case_id.split("_")[1] if case_id.count("_") >= 2 else ""
        page = row.get(FIELDS["page"])
        try:
            page = int(float(page)) if page not in (None, "", "None") else None
        except (TypeError, ValueError):
            page = None
        cases.append({
            "row": row,
            "id": case_id,
            "group": group,
            "expected_doc": GROUP_DOCS.get(group, ""),
            "query": str(row[FIELDS["query"]] or "").strip(),
            "expected": str(row[FIELDS["expected"]] or "").strip(),
            "page": page,
            "keywords": [k.strip() for k in str(row.get(FIELDS["keywords"]) or "").split("|") if k.strip()],
            "category": row.get(FIELDS["category"]),
            "difficulty": row.get(FIELDS["difficulty"]),
            "answerable": str(row.get(FIELDS["answerable"]) or "Yes").strip(),
            "behavior": str(row.get(FIELDS["behavior"]) or "").strip(),
            "forbidden": [k.strip() for k in str(row.get(FIELDS["forbidden"]) or "").split("|") if k.strip()],
        })
    return header, cases


DATE_WORDS = re.compile(r"\b(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})\b")


def normalize_dates(text: str) -> str:
    # "01 tháng 01 năm 2021" -> "01/01/2021"
    return DATE_WORDS.sub(lambda m: f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}", text)


def keyword_present(keyword: str, text_norm: str) -> bool:
    k = normalize_dates(norm(keyword))
    text_norm = normalize_dates(text_norm)
    if k in text_norm:
        return True
    # Word-level match: "quyền nghĩa vụ" matches "quyền và nghĩa vụ".
    key_words = [w for w in words(k) if w not in STOPWORDS]
    if key_words and set(key_words) <= set(words(text_norm)):
        return True
    # Accept the abbreviation <-> full form of the keyword.
    for short, full in ABBREVIATIONS.items():
        if short in k and k.replace(short, full) in text_norm:
            return True
        if full in k and k.replace(full, short) in text_norm:
            return True
    return False


def keyword_coverage(keywords, text) -> float:
    if not keywords:
        return None
    t = norm(text)
    return sum(keyword_present(k, t) for k in keywords) / len(keywords)


def answer_recall(expected: str, answer: str) -> float:
    """Share of the expected answer's content words that appear in the answer."""
    def expand(text):
        t = norm(text)
        for short, full in ABBREVIATIONS.items():
            t = re.sub(rf"\b{short}\b", full, t)
        return t
    exp = [w for w in words(expand(expected)) if w not in STOPWORDS and len(w) > 1]
    if not exp:
        return None
    got = set(words(expand(answer)))
    return sum(w in got for w in exp) / len(exp)


def same_doc(a: str, b: str) -> bool:
    return unicodedata.normalize("NFC", a or "") == unicodedata.normalize("NFC", b or "")
