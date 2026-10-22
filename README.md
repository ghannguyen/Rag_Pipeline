<div align="center">

# Vietnamese Labour Law RAG Pipeline

### A Retrieval-Augmented Generation system for Vietnamese PDF documents using Docling, BGE-M3, ChromaDB, BM25, and Gemini

This project builds a complete RAG pipeline for question answering over Vietnamese PDF documents, mainly labour law materials. It covers PDF parsing with OCR, structure-aware chunking, local embedding, hybrid retrieval (Dense + BM25), Cross-Encoder reranking, grounded answer generation with source citations, a React console for running each stage, and a test suite with RAGAS scoring.

<br>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge\&logo=fastapi\&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge\&logo=react\&logoColor=black)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F00?style=for-the-badge)
![Hugging Face](https://img.shields.io/badge/BGE--M3-Embedding%20%7C%20Reranker-FFD21E?style=for-the-badge\&logo=huggingface\&logoColor=black)
![Gemini](https://img.shields.io/badge/Gemini-LLM-8E75B2?style=for-the-badge\&logo=googlegemini\&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-Team%20Collaboration-181717?style=for-the-badge\&logo=github\&logoColor=white)

<br>

[Source Code](https://github.com/ghannguyen/Rag_Pipeline)

</div>

---

## Table of Contents

* [Project Overview](#project-overview)
* [Problem Understanding](#problem-understanding)
* [Dataset](#dataset)
* [Data Pipeline](#data-pipeline)
* [System Architecture](#system-architecture)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Storage Design](#storage-design)
* [Installation and Environment Setup](#installation-and-environment-setup)
* [How to Run the Project](#how-to-run-the-project)
* [API Endpoints](#api-endpoints)
* [Retrieval and Generation](#retrieval-and-generation)
* [Evaluation](#evaluation)
* [Evaluation Results](#evaluation-results)
* [GitHub Collaboration](#github-collaboration)
* [Important Notes](#important-notes)
* [Team Members and Contributions](#team-members-and-contributions)
* [Project Status](#project-status)
* [Future Improvements](#future-improvements)
* [Lessons Learned](#lessons-learned)
* [Acknowledgements](#acknowledgements)
* [License](#license)

---

## Project Overview

Vietnamese Labour Law RAG Pipeline is a question answering system that answers questions in Vietnamese from PDF documents and cites the source file, page, and section for each answer.

The project does not train or fine-tune any model. Instead, it builds a searchable knowledge base from the PDFs and, for each question, selects the most relevant passages and gives them to an LLM as evidence.

The project covers the full RAG lifecycle:

* Parsing PDFs with Docling and Tesseract OCR (Vietnamese + English)
* Extracting headings, paragraphs, tables, and images
* Generating image captions with BLIP
* Structure-aware chunking that keeps sections and table rows together
* Embedding chunks locally with `BAAI/bge-m3`
* Storing vectors in a persistent ChromaDB collection
* Hybrid retrieval: Dense search + BM25, fused with Reciprocal Rank Fusion
* Cross-Encoder reranking with `BAAI/bge-reranker-v2-m3` and a relevance threshold
* Grounded answer generation with `[SOURCE_X]` citations
* A React console to run and inspect every pipeline stage
* Automated test runs, RAGAS scoring, and retrieval metrics

The current knowledge base contains **7 PDF files** and **4,713 chunks**.

---

## Problem Understanding

Legal and technical documents in Vietnamese are long, mix text with tables and images, and are often scanned. A general LLM can give answers that sound right but have no evidence behind them.

This project answers questions such as:

1. How can Vietnamese PDFs, including scanned pages and tables, be turned into searchable text?
2. How should documents be split so that a section heading, its paragraphs, and table rows stay together?
3. How can semantic search and keyword search be combined for legal terms, article numbers, and abbreviations?
4. How can the system refuse to answer when the documents do not contain the answer?
5. How can every answer point back to the exact file and page it came from?
6. How can retrieval quality and answer quality be measured separately?

---

## Dataset

The knowledge base is stored in `data/raw/` (not pushed to GitHub, see [Important Notes](#important-notes)).

### Source Documents

| Group | File                                           | Description                                     | Chunks |
| ----- | ---------------------------------------------- | ----------------------------------------------- | -----: |
| A     | `giao trinh Luat lao dong CB 2005 (1).pdf`     | Labour law textbook                             |    623 |
| B     | `so-tay-kien-thuc-phap-luat-lao-dong-cdi.pdf`  | Labour law handbook (CDI)                       |    101 |
| C     | `luu-ban-nhap-tu-dong-3-4.pdf`                 | Compiled labour law legal documents             |  3,286 |
| D     | `ILO-VCCI_Labour-Law-Guide-Book-V04-Final.pdf` | ILO–VCCI labour law guide book                  |    354 |
| E     | `Văn-bản-hợp-nhất-18-VBHN-VPQH.pdf`            | Consolidated Labour Code (18/VBHN-VPQH)         |    290 |
| –     | `cafe.pdf`                                     | Coffee document (out-of-domain content)         |     30 |
| –     | `sách chế biến cà phê.pdf`                     | Coffee processing book (out-of-domain content)  |     29 |

The groups A–E match the test case ID prefixes (`LAW_A_001` … `LAW_E_020`).

### Processed Files

| File                                    |  Records | Description                                         |
| --------------------------------------- | -------: | --------------------------------------------------- |
| `data/processed/elements.jsonl`         |   22,040 | Structured elements: heading, paragraph, table, image |
| `data/processed/pdf_extract.jsonl`      |    1,459 | Text grouped by page (legacy format)                |
| `data/processed/image_metadata.jsonl`   |    1,044 | Extracted image metadata                            |
| `data/processed/image_captions.jsonl`   |       59 | BLIP captions                                       |
| `data/processed/chunked.jsonl`          |    4,713 | Final chunks used by ChromaDB and BM25              |

### Chunk Record

```text
chunk_id, document_id, chunk_index, chunk_content
metrics:  char_count, token_count
metadata: source, title, section, section_path, page_start, page_end,
          chunk_strategy, element_ids, table_ids, image_ids, image_paths
```

---

## Data Pipeline

The pipeline has an **offline** part that builds the knowledge base and an **online** part that answers each question.

```text
PDF Files (data/raw)
     |
     v
Docling + Tesseract OCR (vie, eng)
     |
     v
elements.jsonl + PNG images
     |
     v
BLIP Image Captioning
     |
     v
Structure-Aware Chunking
     |
     v
chunked.jsonl
     |
     v
BGE-M3 Embedding
     |
     v
ChromaDB (rag_documents)
```

### Pipeline Steps

| Step | Script                         | Input                       | Output                                       |
| ---- | ------------------------------ | --------------------------- | -------------------------------------------- |
| 1    | `src/data/pdf_loader.py`       | PDF file or folder          | `elements.jsonl`, `pdf_extract.jsonl`, images |
| 2    | `src/data/image_captioner.py`  | `elements.jsonl` + images   | `ai_caption` field, `image_captions.jsonl`   |
| 3    | `src/data/chunker.py`          | `elements.jsonl`            | `chunked.jsonl`                              |
| 4    | `src/rag/indexer.py`           | `chunked.jsonl`             | `data/vectorstore/chroma/`                   |
| 5    | `src/rag/rag_pipeline.py`      | Question                    | Answer, citations, image URLs                |
| 6    | `evaluation/run_tc_*.py`       | Test case Excel + API       | Results Excel, JSON metrics                  |

### Chunking Modules

| Module                  | Responsibility                                                         |
| ----------------------- | ---------------------------------------------------------------------- |
| `section_builder.py`    | Assigns a `section_path` to each element using a heading stack         |
| `block_normalizer.py`   | Converts elements into paragraph, table, and image blocks              |
| `paragraph_splitter.py` | Splits long paragraphs recursively with token overlap                  |
| `table_splitter.py`     | Splits long tables by row and repeats the header in each part          |
| `block_packer.py`       | Packs blocks within the same document and section up to the token limit |
| `metadata_builder.py`   | Builds the final chunk record with page range and element IDs          |
| `token_counter.py`      | Counts tokens with `tiktoken` (`cl100k_base`)                          |

Default chunking parameters: `target_tokens=600`, `token_overlap=100`, `hard_max_tokens=800`.

---

## System Architecture

```text
                  React Pipeline Console (Vite, port 5173)
                                 |
                                 v
                     FastAPI Backend (port 8001)
                                 |
          +----------------------+----------------------+
          |                                             |
          v                                             v
   Offline Pipeline                              Online Pipeline
   Load -> Chunk -> Embed                        Question
          |                                             |
          v                                             v
   ChromaDB  +  chunked.jsonl  ---------->   Dense (BGE-M3) + BM25
                                                        |
                                                        v
                                               RRF Fusion (k = 60)
                                                        |
                                                        v
                                          Cross-Encoder Reranker + Threshold
                                                        |
                                     +------------------+------------------+
                                     |                                     |
                                     v                                     v
                           No chunk above threshold             Context Builder + Prompt
                           -> refuse, skip LLM                             |
                                                                           v
                                                         Gemini / OpenAI / Ollama / Mock
                                                                           |
                                                                           v
                                                         Citation Resolver -> Answer
```

### Component Responsibilities

| Component           | Responsibility                                                          |
| ------------------- | ----------------------------------------------------------------------- |
| Docling + Tesseract | Parses PDFs, runs OCR, detects tables and images                        |
| BLIP                | Turns images into text captions so they can be searched                 |
| Chunker             | Splits documents while keeping sections and table rows together         |
| BGE-M3              | Encodes chunks and questions into normalized vectors                    |
| ChromaDB            | Stores vectors on disk in the `rag_documents` collection                |
| BM25                | Keyword search over `chunked.jsonl`, built in memory                    |
| RRF                 | Merges Dense and BM25 rankings without mixing their score scales        |
| BGE Reranker        | Scores each question–chunk pair and filters by threshold                |
| Context Builder     | Removes near-duplicate chunks and assigns `SOURCE_X` IDs                |
| LLM Provider        | Generates the Vietnamese answer from the context only                   |
| Citation Resolver   | Maps `[SOURCE_X]` tags back to file, page, section, and images          |
| FastAPI             | Exposes the pipeline as a REST API and serves images                    |
| React Console       | Runs each stage, shows chunks, scores, citations, and images            |

---

## Technology Stack

### Document Processing

* Docling
* Tesseract OCR (`vie` + `eng`)
* pypdfium2
* Pillow
* BLIP (`Salesforce/blip-image-captioning-base`)

### Retrieval and Generation

* Sentence Transformers / Transformers / PyTorch
* `BAAI/bge-m3` embedding model
* `BAAI/bge-reranker-v2-m3` Cross-Encoder
* ChromaDB, LangChain (`langchain-chroma`, `langchain-huggingface`, `langchain-text-splitters`)
* tiktoken
* BM25 and RRF (custom implementation)
* Google Gemini (`google-genai`), OpenAI, Ollama

### Backend and Frontend

* Python 3.11
* FastAPI, Uvicorn, Pydantic
* React 18, Vite 6, Lucide icons

### Evaluation

* RAGAS + `langchain-google-genai` (Gemini as judge)
* pandas, openpyxl
* scikit-learn

### Development Tools

* Git
* GitHub
* Visual Studio Code
* macOS development environment (Apple Silicon)

---

## Project Structure

```text
Rag_Pipeline/
├── data/
│   ├── raw/                    # Source PDFs (not in git)
│   ├── processed/              # elements, chunks, captions (JSONL)
│   ├── images/                 # Extracted figures per document
│   └── vectorstore/chroma/     # ChromaDB (not in git)
├── docs/
│   ├── RAG_PROJECT_TECHNICAL_OVERVIEW.md
│   └── system_overview/
├── evaluation/
│   ├── tc_common.py
│   ├── run_tc_excel.py
│   ├── run_tc_retrieval.py
│   ├── run_tc_recall.py
│   ├── run_tc_ragas.py
│   ├── run_tc_static.py
│   ├── run_tc_api.py
│   ├── build_tc_sets.py
│   ├── evaluate_rag.py
│   ├── testsets/               # Smoke, Negative, Robustness, Security
│   └── results/                # Test run outputs
├── frontend/
│   ├── index.html
│   ├── package.json
│   └── src/
│       ├── main.jsx
│       └── styles.css
├── scripts/
│   ├── hybrid_extract.py
│   └── build_hybrid_clean.py
├── src/
│   ├── app.py                  # FastAPI application
│   ├── env.py                  # .env loader
│   ├── data/
│   │   ├── pdf_loader.py
│   │   ├── image_captioner.py
│   │   ├── chunker.py
│   │   ├── section_builder.py
│   │   ├── block_normalizer.py
│   │   ├── block_packer.py
│   │   ├── paragraph_splitter.py
│   │   ├── table_splitter.py
│   │   ├── metadata_builder.py
│   │   └── token_counter.py
│   ├── rag/
│   │   ├── indexer.py
│   │   ├── retriever.py
│   │   ├── bm25_retriever.py
│   │   ├── hybrid_retriever.py
│   │   ├── reranker.py
│   │   ├── context_builder.py
│   │   ├── prompt_builder.py
│   │   ├── citation.py
│   │   ├── device.py
│   │   ├── rag_pipeline.py
│   │   └── llm/                # gemini, openai, ollama, mock providers
│   └── eval/
├── .env.example
├── requirements.txt
├── requirements_eval.txt
└── README.md
```

---

## Storage Design

| Data Type                    | Path                                    |
| ---------------------------- | --------------------------------------- |
| Source PDFs                  | `data/raw/`                             |
| Structured elements          | `data/processed/elements.jsonl`         |
| Page-level text (legacy)     | `data/processed/pdf_extract.jsonl`      |
| Extracted images             | `data/images/<document>/fig_XXX.png`    |
| Image captions               | `data/processed/image_captions.jsonl`   |
| Chunks                       | `data/processed/chunked.jsonl`          |
| Vector database              | `data/vectorstore/chroma/` (`rag_documents`) |
| BM25 index                   | In memory, built from `chunked.jsonl`   |
| Test sets                    | `evaluation/testsets/`                  |
| Test results                 | `evaluation/results/<date>/`            |

---

## Installation and Environment Setup

### Prerequisites

* Python 3.11
* Node.js 18 or higher and npm
* Tesseract OCR with Vietnamese and English language data
* A Gemini API key (or OpenAI key, or a local Ollama server)
* Git

Install Tesseract on macOS:

```bash
brew install tesseract tesseract-lang
```

### Python Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the evaluation scripts (RAGAS), use a separate environment:

```bash
python3.11 -m venv .venv-eval
.venv-eval/bin/pip install -r requirements.txt -r requirements_eval.txt
```

### Configuration

```bash
cp .env.example .env
```

Main settings in `.env`:

| Variable              | Default                    | Meaning                                         |
| --------------------- | -------------------------- | ----------------------------------------------- |
| `EMBEDDING_MODEL`     | `BAAI/bge-m3`              | Embedding model for indexing and search         |
| `RERANKER_MODEL`      | `BAAI/bge-reranker-v2-m3`  | Cross-Encoder reranker                          |
| `TORCH_DEVICE`        | auto (CUDA → CPU)          | Set `mps` on Apple Silicon with ≥ 16 GB RAM     |
| `RETRIEVAL_TOP_K`     | `10`                       | Candidates from each of Dense and BM25          |
| `RERANK_TOP_N`        | `5`                        | Chunks kept after reranking                     |
| `RERANK_THRESHOLD`    | `0.35`                     | Minimum rerank score; below it the system refuses |
| `MAX_CONTEXT_TOKENS`  | `4000`                     | Context budget for the LLM                      |
| `PREFERRED_SOURCES`   | –                          | Source kept when two documents contain the same article |
| `LLM_PROVIDER`        | `gemini`                   | `gemini` / `openai` / `ollama` / `mock`         |
| `LLM_MODEL`           | `gemini-3.6-flash`         | Model name for the selected provider            |
| `GEMINI_API_KEY`      | –                          | Required for Gemini                             |

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone git@github.com:ghannguyen/Rag_Pipeline.git
cd Rag_Pipeline
```

---

### 2. Add the Source PDFs

Copy the PDF files into:

```text
data/raw/
```

---

### 3. Parse PDFs

```bash
python src/data/pdf_loader.py --input data/raw
```

This step performs:

* OCR with Tesseract (Vietnamese + English)
* Table structure detection
* Image extraction to `data/images/`
* Unicode NFC normalization and whitespace cleanup

---

### 4. Generate Image Captions

```bash
python src/data/image_captioner.py
```

Images smaller than 100 × 100 pixels are skipped.

---

### 5. Build Chunks

```bash
python src/data/chunker.py \
  --input data/processed/elements.jsonl \
  --output data/processed/chunked.jsonl \
  --target-tokens 600 --token-overlap 100 --hard-max-tokens 800
```

---

### 6. Build the Vector Index

```bash
python src/rag/indexer.py
```

---

### 7. Ask a Question from the Command Line

```bash
python src/rag/rag_pipeline.py --query "Thời gian thử việc tối đa là bao lâu?"
```

---

### 8. Start the Backend

```bash
uvicorn src.app:app --port 8001
```

Check the server:

```bash
curl http://127.0.0.1:8001/health
```

---

### 9. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The console can run Load → Chunk → Embed for one PDF, test retrieval, and ask questions with **RAG Answer**.

Set `VITE_API_URL` if the backend does not run on `http://127.0.0.1:8001`.

---

## API Endpoints

| Method | Endpoint                 | Description                                              |
| ------ | ------------------------ | -------------------------------------------------------- |
| GET    | `/health`                | Server status and torch device                           |
| GET    | `/api/files`             | Lists PDFs in `data/raw`                                 |
| GET    | `/api/pipeline/status`   | Record counts of JSONL files and number of vectors       |
| POST   | `/api/pipeline/load`     | Parses one PDF and generates captions                    |
| POST   | `/api/pipeline/chunk`    | Builds chunks, returns count and preview                 |
| POST   | `/api/pipeline/embed`    | Rebuilds the ChromaDB index (`reset_db=true` by default) |
| POST   | `/api/retrieval/search`  | Hybrid search only, no reranking and no LLM              |
| POST   | `/api/rag/chat`          | Full pipeline: retrieval → rerank → LLM → citations      |
| GET    | `/images/...`            | Serves extracted images                                  |

Example request:

```bash
curl -X POST http://127.0.0.1:8001/api/rag/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Người lao động được nghỉ phép năm bao nhiêu ngày?"}'
```

Response fields: `query`, `answer`, `citations`, `image_urls`, `total_duration_ms`, `logs`.

---

## Retrieval and Generation

### 1. Dense Retrieval

Questions are encoded with `BAAI/bge-m3` (normalized embeddings) and searched in ChromaDB.

### 2. BM25 Keyword Retrieval

BM25 is implemented in Python over `chunked.jsonl`.

| Parameter    | Value                                                      |
| ------------ | ---------------------------------------------------------- |
| Tokenization | Lowercase + Unicode `\w+`, Vietnamese stopwords removed    |
| `k1`         | 1.5                                                        |
| `b`          | 0.75                                                       |

### 3. Reciprocal Rank Fusion

```text
RRF(d) = Σ 1 / (60 + rank_list(d))
```

RRF uses ranks instead of raw scores, so Dense distances and BM25 scores do not need to be on the same scale.

### 4. Cross-Encoder Reranking

| Step       | Description                                                         |
| ---------- | ------------------------------------------------------------------- |
| Scoring    | `BAAI/bge-reranker-v2-m3` scores each (question, chunk) pair        |
| Threshold  | Chunks below `RERANK_THRESHOLD` (0.35) are removed                  |
| Top-N      | The best `RERANK_TOP_N` (5) chunks are kept                         |
| Early exit | If no chunk passes, the system refuses and does not call the LLM    |

Refusal message:

> Dựa trên tài liệu được cung cấp, tôi không tìm thấy thông tin để trả lời câu hỏi này.

### 5. Context and Prompt

* Near-duplicate chunks are removed, including the same article appearing in two documents (`PREFERRED_SOURCES` decides which one is kept)
* Each chunk gets a `[SOURCE_X]` ID with file, page, and section
* The system prompt asks for a Vietnamese answer based only on the context, with a citation after each point

### 6. LLM Providers

| Provider | Default model       | Notes                                        |
| -------- | ------------------- | -------------------------------------------- |
| Gemini   | `gemini-3.6-flash`  | Default provider                             |
| OpenAI   | `gpt-4o-mini`       | Called over HTTP                             |
| Ollama   | `qwen2.5:7b`        | Local model at `OLLAMA_BASE_URL`             |
| Mock     | –                   | Fixed answer, used for wiring tests          |

### 7. Citation Resolver

The resolver finds `[SOURCE_X]` tags in the answer, maps them to file, page, section, and chunk ID, removes tags that do not exist, and returns the URLs of related images.

---

## Evaluation

The evaluation follows a QA process with a main test set and four extra test sets.

### Test Sets

| Test Set              | Cases | Purpose                                                      |
| --------------------- | ----: | ------------------------------------------------------------ |
| `TC_LuatLD.xlsx`      |   100 | Main set, 20 questions per labour law document (A–E)         |
| `TC_Smoke.xlsx`       |    10 | Easy cases that passed in the baseline, 2 per document       |
| `TC_Negative.xlsx`    |    20 | Out-of-scope questions that must be refused                  |
| `TC_Robustness.xlsx`  |    20 | No diacritics, abbreviations, and typos                      |
| `TC_Security.xlsx`    |    15 | Prompt injection, prompt leak, and abuse                     |

### Evaluation Scripts

| Script                           | Purpose                                                                  |
| -------------------------------- | ------------------------------------------------------------------------ |
| `evaluation/run_tc_excel.py`     | Calls `/api/rag/chat` for each case, auto-scores, writes an Excel report |
| `evaluation/run_tc_retrieval.py` | Retrieval-only run (no LLM), supports regression against an old index    |
| `evaluation/run_tc_recall.py`    | Recall@5, Hit@5, MRR@5 on gold chunks, plus failure stage per case       |
| `evaluation/run_tc_ragas.py`     | RAGAS on the exact context the LLM received                              |
| `evaluation/run_tc_static.py`    | Static checks of ingestion, chunking, and citations                      |
| `evaluation/run_tc_api.py`       | API tests: status codes, validation, edge cases, response schema         |
| `evaluation/build_tc_sets.py`    | Builds the Smoke, Negative, Robustness, and Security sets                |

### Example Commands

```bash
# Backend must be running on port 8001
python evaluation/run_tc_excel.py --input TC_LuatLD.xlsx \
  --out-dir evaluation/results/2026-10-20_flash-lite

python evaluation/run_tc_retrieval.py --input TC_LuatLD.xlsx \
  --output evaluation/results/2026-10-20/retrieval_new.json

.venv-eval/bin/python evaluation/run_tc_ragas.py --input TC_LuatLD.xlsx \
  --run-dir evaluation/results/2026-10-20_flash-lite \
  --judge-model gemini-3.5-flash-lite
```

`run_tc_excel.py` and `run_tc_ragas.py` can be resumed: re-running the same command skips cases that are already done.

---

## Evaluation Results

Results from the run in `evaluation/results/2026-10-20_flash-lite/` (20–21 October 2026).

### Main Test Set (100 cases)

| Status  | Cases |
| ------- | ----: |
| PASS    |    47 |
| PARTIAL |    23 |
| FAIL    |    30 |

These statuses come from an AI-assisted review (`review_ai.json`) and still need confirmation by a tester.

### Retrieval Metrics (95 cases with gold chunks)

| Metric                         | Value |
| ------------------------------ | ----: |
| Gold chunk in candidates       | 0.789 |
| Hit@5 (after rerank)           | 0.653 |
| Recall@5 (after rerank)        | 0.572 |
| Recall@5 (context sent to LLM) | 0.454 |
| MRR@5                          | 0.471 |

Gold chunks are derived automatically from keywords, evidence, and the source page, not labelled by hand.

### RAGAS (judge: Gemini)

| Metric            | Mean  | Scored cases |
| ----------------- | ----: | -----------: |
| Faithfulness      | 0.873 |           80 |
| Answer Relevancy  | 0.854 |           82 |
| Context Precision | 0.584 |           99 |
| Context Recall    | 0.629 |           97 |

### Extra Test Sets (suggested status)

| Test Set   | PASS | PARTIAL | FAIL |
| ---------- | ---: | ------: | ---: |
| Smoke      |   10 |       0 |    0 |
| Negative   |   19 |       0 |    1 |
| Robustness |    7 |       3 |   10 |
| Security   |   13 |       2 |    0 |
| API tests  |   16 |       – |    3 |

### Key Findings

* Answers that are generated stay close to the context (faithfulness 0.87), and out-of-scope and injection questions are handled well.
* The main weakness is retrieval: about 21% of gold chunks never reach the candidate list, and more are lost between reranking and the final context.
* Questions without diacritics or with typos fail often (10/20 FAIL), because both BM25 and the embeddings depend on correct Vietnamese spelling.

---

## GitHub Collaboration

The team uses GitHub for source code management.

| Evidence     | Description                                          |
| ------------ | ---------------------------------------------------- |
| Repository   | `ghannguyen/Rag_Pipeline`                            |
| Commits      | Track pipeline, evaluation, and frontend changes     |
| Contributors | Shows member contribution and commit activity        |

---

## Important Notes

### Data Not Included in Git

The following are excluded by `.gitignore` and must be created locally:

```text
.env
data/raw/            # source PDFs
data/vectorstore/    # ChromaDB index
*.pdf
```

After cloning, add the PDFs to `data/raw/` and run steps 3–6 of [How to Run the Project](#how-to-run-the-project) to rebuild the index.

---

### Keep ChromaDB and BM25 in Sync

Dense search reads ChromaDB, while BM25 reads `chunked.jsonl`. Both must come from the same chunking run. After re-chunking, rebuild the vector index and restart the backend, because the BM25 index is cached in memory.

---

### Loading from the UI Replaces the Knowledge Base

`/api/pipeline/load` overwrites the shared JSONL files, and `/api/pipeline/embed` resets the collection by default. Loading one PDF from the console replaces the knowledge base with that single PDF.

---

### Memory on 8 GB Machines

`bge-m3` and the reranker are loaded in the backend process. On machines with 8 GB RAM, stop the backend before running `run_tc_retrieval.py`, and keep `TORCH_DEVICE` on CPU.

---

### Images Are Searched by Caption

Images are converted into BLIP captions and searched as text. The LLM does not see the image itself; the related images are only returned to the user with the answer.

---

### Citation Check

The citation resolver checks that each `[SOURCE_X]` tag exists. It does not check that the cited chunk actually supports the sentence.

---

## Team Members and Contributions

| Member       | Main Contributions                                                                                                  |
| ------------ | ------------------------------------------------------------------------------------------------------------------- |
| ghannguyen   | Data pipeline, chunking, hybrid retrieval, reranking, LLM integration, backend API, frontend console, evaluation scripts, test runs, documentation |

> The contribution table can be adjusted based on the final team agreement and GitHub commit history.

---

## Project Status

```text
Status: Working end-to-end, evaluation completed
Version: 1.0
Last evaluation: October 2026
```

Completed components:

* PDF parsing with OCR, tables, and images
* BLIP image captioning
* Structure-aware chunking with row-aware table splitting
* BGE-M3 embedding and ChromaDB index
* Hybrid retrieval (Dense + BM25 + RRF)
* Cross-Encoder reranking with threshold and early exit
* Cross-document duplicate removal
* Grounded generation with Gemini / OpenAI / Ollama / Mock
* Citation resolution with file, page, and images
* FastAPI backend and React console
* 100-case test run, 4 extra test sets, RAGAS, Recall@5, static and API checks

---

## Future Improvements

* Improve recall for questions without diacritics and with typos (diacritic restoration or query rewriting)
* Add Vietnamese word segmentation for BM25
* Calibrate the rerank threshold on a labelled set of answerable and unanswerable questions
* Label gold chunks by hand and report Recall@K, MRR, and nDCG on them
* Add a corpus version manifest so ChromaDB, BM25, and JSONL files always match
* Use stable chunk IDs and incremental upsert instead of a full index reset
* Check that each cited chunk actually supports the claim (entailment)
* Add PDF upload, chat history, and streaming answers in the UI
* Add authentication and a background job queue for ingestion
* Add automated tests and a CI workflow

---

## Lessons Learned

Through this project, the team gained practical experience in:

* Parsing Vietnamese PDFs with OCR and keeping table and heading structure
* Designing chunking around document structure instead of fixed sizes
* Combining dense and keyword retrieval with rank fusion
* Using a Cross-Encoder and a threshold to decide when not to answer
* Writing prompts that force answers to stay within the given context
* Linking answers back to source pages with citation IDs
* Building a REST API and a React console around an ML pipeline
* Measuring retrieval and generation separately, and reading each metric by its own definition
* Using RAGAS with an LLM judge and handling quota and timeout errors
* Running regression tests when the index changes
* Collaborating through GitHub

---

## Acknowledgements

The team would like to thank:

* Docling and Tesseract OCR
* BAAI for BGE-M3 and BGE Reranker
* Salesforce for BLIP
* ChromaDB and LangChain
* Google Gemini
* RAGAS
* FastAPI and React
* GitHub

---

## License

This project was developed for educational and academic purposes.

---

<div align="center">

Made with care by the Rag Pipeline team – Vietnamese Labour Law RAG

</div>
