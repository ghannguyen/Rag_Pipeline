from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# This app uses no Pydantic plugins; avoid scanning every installed distribution.
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "__all__")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.env import load_env

# Load .env before reading module-level settings such as EMBEDDING_MODEL.
load_env()

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
IMAGES_DIR = ROOT / "data" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

PDF_EXTRACT = PROCESSED_DIR / "pdf_extract.jsonl"
ELEMENTS_FILE = PROCESSED_DIR / "elements.jsonl"
CHUNKED = PROCESSED_DIR / "chunked.jsonl"
VECTOR_DIR = ROOT / "data" / "vectorstore" / "chroma"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
COLLECTION_NAME = "rag_documents"

app = FastAPI(
    title="Vietnamese PDF RAG Pipeline Demo",
    version="1.0.0",
)

app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


@app.middleware("http")
async def reload_env(request: Request, call_next):
    # Pick up .env edits (thresholds, device, preferred sources) without a restart.
    load_env()
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoadRequest(BaseModel):
    file_name: str


class ChunkRequest(BaseModel):
    strategy: str = "structure_block_aware"
    target_tokens: int = Field(default=600, ge=100, le=4000)
    token_overlap: int = Field(default=100, ge=0, le=1000)


class EmbedRequest(BaseModel):
    # False = incremental (embed only new/changed chunks, drop stale ones).
    # True = wipe the collection and re-embed everything.
    reset_db: bool = False


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def count_documents(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return len({record_document_id(json.loads(line)) for line in f if line.strip()})


def record_document_id(record: dict[str, Any]) -> str | None:
    # elements/image records use document_id; pdf_extract pages use metadata.title.
    return record.get("document_id") or record.get("metadata", {}).get("title")


def upsert_jsonl(target: Path, new_file: Path, document_ids: set[str]) -> None:
    """Replace records of document_ids in target with new_file, keeping other documents."""
    kept = []
    if target.exists():
        with target.open("r", encoding="utf-8") as f:
            kept = [
                line.rstrip("\n")
                for line in f
                if line.strip() and record_document_id(json.loads(line)) not in document_ids
            ]

    added = []
    if new_file.exists():
        with new_file.open("r", encoding="utf-8") as f:
            added = [line.rstrip("\n") for line in f if line.strip()]

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for line in kept + added:
            f.write(line + "\n")
    os.replace(tmp, target)


def read_chunk_preview(limit: int = 3) -> list[dict[str, Any]]:
    if not CHUNKED.exists():
        return []
    result = []
    with CHUNKED.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            result.append(
                {
                    "chunk_id": data.get("chunk_id"),
                    "content": data.get("chunk_content", "")[:500],
                    "metadata": data.get("metadata", {}),
                    "metrics": data.get("metrics", {}),
                }
            )
            if len(result) >= limit:
                break
    return result


def vector_count() -> int:
    if not (VECTOR_DIR / "chroma.sqlite3").exists():
        return 0
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(VECTOR_DIR))
        collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
        return int(collection.count())
    except Exception:
        return 0


def current_device() -> str:
    # Status must not import PyTorch; report the device only once a model loaded it.
    if "torch" not in sys.modules:
        return os.getenv("TORCH_DEVICE", "").strip().lower() or "not_loaded"
    from src.rag.device import pick_device

    return pick_device()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": current_device(),
    }


@app.get("/api/files")
def list_files():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [
            {
                "name": p.name,
                "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
            }
            for p in RAW_DIR.glob("*.pdf")
            if p.is_file()
        ],
        key=lambda x: x["name"].lower(),
    )

    return {
        "directory": str(RAW_DIR),
        "files": files,
    }


@app.get("/api/pipeline/status")
def pipeline_status():
    return {
        "pdf_documents": count_documents(ELEMENTS_FILE),
        "pages": count_jsonl(PDF_EXTRACT),
        "elements": count_jsonl(ELEMENTS_FILE),
        "chunks": count_jsonl(CHUNKED),
        "vectors": vector_count(),
        "embedding_model": EMBEDDING_MODEL,
        "device": current_device(),
        "paths": {
            "raw": str(RAW_DIR),
            "elements": str(ELEMENTS_FILE),
            "pdf_extract": str(PDF_EXTRACT),
            "chunked": str(CHUNKED),
            "vector_db": str(VECTOR_DIR),
        },
    }


@app.post("/api/pipeline/load")
def run_load(request: LoadRequest):
    pdf_path = RAW_DIR / Path(request.file_name).name

    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy file: {pdf_path}",
        )

    document_id = pdf_path.stem
    start = time.perf_counter()

    try:
        from src.data.pdf_loader import convert_pdfs

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        # Parse into a staging dir, then upsert only this document's records so
        # previously loaded PDFs stay in the knowledge base.
        with tempfile.TemporaryDirectory(dir=PROCESSED_DIR, prefix=".load_") as tmp:
            tmp_dir = Path(tmp)
            tmp_elements = tmp_dir / ELEMENTS_FILE.name
            convert_pdfs(pdf_path, tmp_dir / PDF_EXTRACT.name, tmp_elements, images_root=IMAGES_DIR)

            doc_elements = count_jsonl(tmp_elements)
            if doc_elements == 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Loader lỗi: không trích xuất được element nào từ {pdf_path.name}. Dữ liệu cũ được giữ nguyên.",
                )

            try:
                from src.data.image_captioner import caption_elements

                caption_elements(tmp_elements)
            except Exception as cap_err:
                print(f"! Captioner warning: {cap_err}")

            for name in (ELEMENTS_FILE.name, PDF_EXTRACT.name, "image_metadata.jsonl", "image_captions.jsonl"):
                upsert_jsonl(PROCESSED_DIR / name, tmp_dir / name, {document_id})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Loader lỗi: {exc}") from exc

    duration_ms = (time.perf_counter() - start) * 1000
    elements_cnt = count_jsonl(ELEMENTS_FILE)

    return {
        "stage": "loading",
        "status": "success",
        "message": f"Đã load {request.file_name} ({doc_elements} elements & BLIP captions)",
        "duration_ms": round(duration_ms, 2),
        "details": {
            "elements": doc_elements,
            "total_elements": elements_cnt,
            "documents": count_documents(ELEMENTS_FILE),
            "output": str(ELEMENTS_FILE),
        },
    }


@app.post("/api/pipeline/chunk")
def run_chunk(request: ChunkRequest):
    input_file = ELEMENTS_FILE if ELEMENTS_FILE.exists() else PDF_EXTRACT

    if not input_file.exists():
        raise HTTPException(
            status_code=400,
            detail="Chưa có elements.jsonl hoặc pdf_extract.jsonl. Hãy chạy Loading trước.",
        )

    from src.data.chunker_optimized import ChunkConfig, process_chunking

    config = ChunkConfig(
        strategy=request.strategy,
        target_tokens=request.target_tokens,
        token_overlap=request.token_overlap,
    )

    start = time.perf_counter()

    try:
        process_chunking(input_file, CHUNKED, config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chunking lỗi: {exc}") from exc

    duration_ms = (time.perf_counter() - start) * 1000
    chunks = count_jsonl(CHUNKED)

    return {
        "stage": "chunking",
        "status": "success",
        "message": f"Đã tạo {chunks} chunks (Structure & Block-Aware)",
        "duration_ms": round(duration_ms, 2),
        "details": {
            "chunks": chunks,
            "strategy": request.strategy,
            "target_tokens": request.target_tokens,
            "token_overlap": request.token_overlap,
            "output": str(CHUNKED),
            "preview": read_chunk_preview(),
        },
    }


@app.post("/api/pipeline/embed")
def run_embed(request: EmbedRequest):
    if not CHUNKED.exists():
        raise HTTPException(
            status_code=400,
            detail="Chưa có chunked.jsonl. Hãy chạy Chunking trước.",
        )

    from src.rag.indexer import IndexBusyError, acquire_index_lock, build_index

    start = time.perf_counter()

    # Refuse a second run up front, before reset_db could wipe a collection
    # another run is still writing to.
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    try:
        acquire_index_lock(VECTOR_DIR).close()
    except IndexBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        if request.reset_db and VECTOR_DIR.exists():
            import chromadb

            # Keep the database/schema alive while Chroma clients are cached.
            client = chromadb.PersistentClient(path=str(VECTOR_DIR))
            try:
                client.delete_collection(COLLECTION_NAME)
            except chromadb.errors.NotFoundError:
                pass

            retriever = sys.modules.get("src.rag.retriever")
            if retriever is not None:
                cache = retriever._VECTOR_DB_CACHE
                for key in list(cache):
                    if Path(key[0]).resolve() == VECTOR_DIR.resolve():
                        del cache[key]

        db = build_index(
            input_file=CHUNKED,
            persist_dir=VECTOR_DIR,
            model_name=EMBEDDING_MODEL,
        )
        vectors = int(db._collection.count())
    except IndexBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding/Indexing lỗi: {exc}") from exc

    duration_ms = (time.perf_counter() - start) * 1000

    return {
        "stage": "embedding",
        "status": "success",
        "message": f"Đã embedding và lưu {vectors} vectors",
        "duration_ms": round(duration_ms, 2),
        "details": {
            "vectors": vectors,
            "model": EMBEDDING_MODEL,
            "device": current_device(),
            "database": str(VECTOR_DIR),
        },
    }


@app.post("/api/retrieval/search")
def search_top_k(request: SearchRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query không được để trống.")

    if not VECTOR_DIR.exists():
        raise HTTPException(
            status_code=400,
            detail="Chưa có Vector Database. Hãy chạy Embedding trước.",
        )

    logs = []
    total_start = time.perf_counter()

    logs.append(
        {
            "stage": "query",
            "status": "success",
            "message": f'Nhận query: "{query}"',
            "duration_ms": 0,
        }
    )

    load_start = time.perf_counter()
    try:
        from src.rag.retriever import load_vector_db

        db = load_vector_db(VECTOR_DIR, EMBEDDING_MODEL)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không load được ChromaDB: {exc}") from exc

    logs.append(
        {
            "stage": "vector_db",
            "status": "success",
            "message": "Đã load ChromaDB và embedding model",
            "duration_ms": round((time.perf_counter() - load_start) * 1000, 2),
        }
    )

    search_start = time.perf_counter()

    try:
        from src.rag.hybrid_retriever import hybrid_search
        hybrid_results = hybrid_search(
            query=query,
            top_k_dense=request.top_k,
            top_k_sparse=request.top_k,
            rrf_k=60,
            vector_dir=VECTOR_DIR,
            embedding_model=EMBEDDING_MODEL
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Hybrid Retrieval lỗi: {exc}") from exc

    retrieval_ms = (time.perf_counter() - search_start) * 1000

    results = []

    for rank, item in enumerate(hybrid_results[:request.top_k], start=1):
        metadata = item.get("metadata", {})
        display_score = item.get("score", 0.0)

        image_urls = []
        raw_paths = metadata.get("image_paths", "")
        if raw_paths:
            path_list = [p.strip() for p in raw_paths.split(",") if p.strip()]
            for p in path_list:
                clean_p = p.replace("\\", "/")
                if "data/images/" in clean_p:
                    rel = clean_p.split("data/images/")[1]
                    image_urls.append(f"/images/{rel}")
                elif clean_p.startswith("/images/"):
                    image_urls.append(clean_p)

        results.append(
            {
                "rank": rank,
                "score": round(display_score, 6),
                "distance": 0.0,
                "content": item.get("content", ""),
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "chunk_id": item.get("chunk_id") or metadata.get("chunk_id"),
                "image_urls": image_urls,
                "metadata": metadata,
            }
        )

    logs.append(
        {
            "stage": "retrieval",
            "status": "success",
            "message": f"Đã lấy Top-{len(results)} chunks",
            "duration_ms": round(retrieval_ms, 2),
        }
    )

    total_ms = (time.perf_counter() - total_start) * 1000

    logs.append(
        {
            "stage": "done",
            "status": "success",
            "message": "Hoàn thành Retrieval",
            "duration_ms": round(total_ms, 2),
        }
    )

    return {
        "query": query,
        "top_k": request.top_k,
        "total_vectors": int(db._collection.count()),
        "total_duration_ms": round(total_ms, 2),
        "logs": logs,
        "results": results,
    }


class ChatRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi người dùng")
    # None = use RETRIEVAL_TOP_K / RERANK_TOP_N / RERANK_THRESHOLD / LLM_PROVIDER from .env.
    top_k: int | None = Field(default=None, ge=1, le=30, description="Số lượng candidate từ Vector Search")
    top_n: int | None = Field(default=None, ge=1, le=10, description="Số lượng chunk giữ lại sau Re-ranking")
    threshold: float | None = Field(default=None, ge=0.0, le=1.0, description="Ngưỡng điểm RERANK_THRESHOLD")
    provider: str | None = Field(default=None, description="LLM Provider: gemini | openai | ollama | mock")


@app.post("/api/rag/chat")
def run_chat(request: ChatRequest):
    try:
        from src.rag.rag_pipeline import answer_query

        res = answer_query(
            query=request.query,
            top_k=request.top_k,
            top_n=request.top_n,
            threshold=request.threshold,
            provider_name=request.provider
        )
        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG Chat lỗi: {exc}") from exc
