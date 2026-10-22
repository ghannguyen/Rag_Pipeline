"""
RAG Indexer

Input:
    chunked.jsonl

Output:
    Chroma Vector Database

Embedding:
    BAAI/bge-m3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8"
    )


from langchain_core.documents import Document

from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings



# ============================================================
# CONFIG
# ============================================================

DEFAULT_MODEL = "BAAI/bge-m3"

DEFAULT_DB = "data/vectorstore/chroma"

# Chunks per model forward pass (was 1). 8 keeps peak RAM low on 8 GB machines.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))

# Chunks written to Chroma per call; also the progress-log granularity.
UPSERT_BATCH = 64



# ============================================================
# LOAD JSONL
# ============================================================

def load_chunks(
    path: Path
):

    documents = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            data = json.loads(
                line
            )

            raw_meta = {
                **data.get(
                    "metadata",
                    {}
                ),
                "chunk_id":
                    data.get(
                        "chunk_id",
                        ""
                    ),
                "doc_id":
                    data.get(
                        "doc_id",
                        ""
                    )
            }

            clean_meta = {}
            for k, v in raw_meta.items():
                if v is None:
                    continue
                if isinstance(v, list):
                    if not v:
                        continue
                    clean_meta[k] = ",".join(str(x) for x in v)
                else:
                    clean_meta[k] = v

            documents.append(

                Document(

                    page_content=
                        data["chunk_content"],

                    metadata=clean_meta

                )

            )

    return documents



class IndexBusyError(RuntimeError):
    pass


def acquire_index_lock(persist_dir: Path):
    """Exclusive, non-blocking lock on <persist_dir>/.index.lock; released when
    the returned file is closed or the process exits."""
    import fcntl

    handle = open(persist_dir / ".index.lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise IndexBusyError("Đang có một tiến trình Embedding khác chạy, hãy đợi nó xong.")
    return handle



# ============================================================
# BUILD VECTOR DATABASE
# ============================================================

def build_index(

    input_file: Path,

    persist_dir: Path,

    model_name: str

):


    print("=" * 60)

    print("Loading chunks...")

    documents = load_chunks(
        input_file
    )


    print(
        f"Documents loaded: {len(documents)}"
    )


    print("=" * 60)

    print(
        f"Loading embedding model: {model_name}"
    )


    from src.rag.device import pick_device
    device = pick_device()

    embeddings = HuggingFaceEmbeddings(

        model_name=model_name,

        model_kwargs={
            "device": device
        },

        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": EMBED_BATCH_SIZE
        }

    )


    print("Building vector database...")


    persist_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Chroma's on-disk HNSW index is not safe for concurrent writers: two
    # indexing runs at once corrupted it. One writer at a time, across
    # processes (API server, CLI).
    lock = acquire_index_lock(persist_dir)


    # Stable ids make re-indexing an upsert instead of appending duplicates.
    ids = [
        doc.metadata.get("chunk_id") or ""
        for doc in documents
    ]

    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError(
            "Every chunk needs a unique chunk_id before indexing"
        )


    db = Chroma(

        collection_name="rag_documents",

        embedding_function=embeddings,

        persist_directory=str(
            persist_dir
        )

    )


    # Incremental: only chunks that are new or whose text changed are embedded.
    # chunk_id is positional (doc + index), so the stored text is compared too.
    stored = db.get(include=["documents", "metadatas"])
    stored_text = dict(zip(stored["ids"], stored["documents"]))
    stored_meta = dict(zip(stored["ids"], stored["metadatas"]))

    to_embed = [
        doc for doc, cid in zip(documents, ids)
        if stored_text.get(cid) != doc.page_content
    ]
    # Same text, different metadata (page, section): update without re-embedding.
    meta_only = [
        (cid, doc.metadata) for doc, cid in zip(documents, ids)
        if stored_text.get(cid) == doc.page_content
        and stored_meta.get(cid) != doc.metadata
    ]

    print(
        f"Chunks: {len(documents)} | unchanged: {len(documents) - len(to_embed) - len(meta_only)} | "
        f"metadata only: {len(meta_only)} | to embed: {len(to_embed)} (batch {EMBED_BATCH_SIZE}, {device})",
        flush=True
    )

    if meta_only:
        db._collection.update(
            ids=[cid for cid, _ in meta_only],
            metadatas=[meta for _, meta in meta_only]
        )

    started = time.perf_counter()

    for start in range(0, len(to_embed), UPSERT_BATCH):

        batch = to_embed[start:start + UPSERT_BATCH]

        db.add_documents(
            batch,
            ids=[doc.metadata["chunk_id"] for doc in batch]
        )

        done = start + len(batch)
        elapsed = time.perf_counter() - started
        eta = elapsed / done * (len(to_embed) - done)
        print(
            f"Embedded {done}/{len(to_embed)} ({elapsed:.0f}s, ~{eta:.0f}s left)",
            flush=True
        )


    # Drop vectors whose chunk no longer exists in chunked.jsonl.
    stale_ids = sorted(
        set(stored["ids"]) - set(ids)
    )

    if stale_ids:
        db.delete(ids=stale_ids)
        print(f"Removed stale vectors: {len(stale_ids)}")


    lock.close()

    print("=" * 60)

    print("INDEX COMPLETE")

    print(
        f"Vectors stored: {db._collection.count()} (embedded {len(to_embed)})"
    )

    print(
        f"Database: {persist_dir}"
    )

    print("=" * 60)


    return db



# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Build RAG vector index"
    )


    parser.add_argument(

        "--input",

        type=Path,

        default=Path(
            "data/processed/chunked.jsonl"
        )

    )


    parser.add_argument(

        "--db",

        type=Path,

        default=Path(
            DEFAULT_DB
        )

    )


    parser.add_argument(

        "--model",

        type=str,

        default=DEFAULT_MODEL

    )


    args = parser.parse_args()



    if not args.input.exists():

        raise FileNotFoundError(
            args.input
        )


    build_index(

        args.input,

        args.db,

        args.model

    )



if __name__ == "__main__":
    main()