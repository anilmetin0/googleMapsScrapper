import json
import os
import sys
import logging
import argparse
import hashlib
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote

import redis
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import torch
import chromadb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBED_MODEL     = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
CHROMA_PATH     = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_HOST     = os.getenv("CHROMA_HOST", "")
CHROMA_PORT     = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "place_reviews_v2")
REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379")
INDEXER_INPUT   = os.getenv("INDEXER_INPUT", "scraped_data.json")
QUEUE_NAME      = "queue:places:indexer"

if torch.cuda.is_available():
    DEVICE     = "cuda"
    BATCH_SIZE = 256
elif torch.backends.mps.is_available():
    DEVICE     = "mps"
    BATCH_SIZE = 64
else:
    DEVICE     = "cpu"
    BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_place_name(entry: dict) -> str:
    if entry.get("place_name"):
        return entry["place_name"]
    url   = entry.get("url", "")
    match = re.search(r"/place/([^/]+)", url)
    if match:
        return unquote(match.group(1).replace("+", " "))
    return "Bilinmeyen Mekan"


def place_slug(place_name: str) -> str:
    return hashlib.md5(place_name.encode()).hexdigest()[:10]


def review_doc_id(place_name: str, idx: int) -> str:
    return f"{place_slug(place_name)}_{idx}"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def index_place(
    collection,
    model:      SentenceTransformer,
    place_name: str,
    source_url: str,
    reviews:    list,
    upsert:     bool = False,
) -> int:
    existing = collection.get(where={"place_name": place_name}, limit=1)
    if existing["ids"]:
        if not upsert:
            log.info("Zaten indexli, atlanıyor.")
            return 0
        all_existing = collection.get(where={"place_name": place_name})
        if all_existing["ids"]:
            collection.delete(ids=all_existing["ids"])
        log.info(f"Güncelleniyor — {len(all_existing['ids'])} eski yorum silindi.")

    prefixed   = [f"passage: {r}" for r in reviews]
    embeddings = model.encode(
        prefixed,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

    collection.add(
        ids=[review_doc_id(place_name, i) for i in range(len(reviews))],
        embeddings=embeddings,
        documents=reviews,
        metadatas=[{"place_name": place_name, "source_url": source_url}] * len(reviews),
    )
    return len(reviews)


# ---------------------------------------------------------------------------
# Batch pipeline (CLI)
# ---------------------------------------------------------------------------

def run(input_path: str, reset: bool = False):
    input_file = Path(input_path)
    if not input_file.exists():
        log.error(f"Girdi dosyası bulunamadı: {input_file}")
        sys.exit(1)

    log.info(f"Device: {DEVICE} | Batch size: {BATCH_SIZE}")
    log.info(f"Model yükleniyor: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)

    with open(input_file, encoding="utf-8") as f:
        scraped_data = json.load(f)

    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT) if CHROMA_HOST else chromadb.PersistentClient(path=CHROMA_PATH)
    if reset:
        log.warning("--reset: koleksiyon siliniyor ve yeniden oluşturuluyor.")
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total   = len(scraped_data)
    indexed = 0
    skipped = 0

    for i, entry in enumerate(scraped_data, start=1):
        place_name = extract_place_name(entry)
        reviews    = entry.get("reviews", [])
        source_url = entry.get("url", "")
        log.info(f"[{i}/{total}] '{place_name}' — {len(reviews)} yorum")

        if not reviews:
            log.warning("Yorum yok, atlanıyor.")
            skipped += 1
            continue

        added = index_place(collection, model, place_name, source_url, reviews)
        if added:
            indexed += 1
            log.info(f"  ✓ {added} yorum eklendi.")
        else:
            skipped += 1

    log.info(
        f"Tamamlandı — Yeni: {indexed}, Atlanan: {skipped} | "
        f"Koleksiyon toplam: {collection.count()} yorum"
    )


# ---------------------------------------------------------------------------
# FastAPI mikroservis
# ---------------------------------------------------------------------------

_job:          dict = {"status": "idle", "detail": ""}
_lock               = threading.Lock()
_worker_stats: dict = {"processed": 0, "failed": 0, "running": False}


def _worker_loop():
    log.info(f"[worker] Indexer Redis worker başladı — kuyruk: {QUEUE_NAME}")
    model         = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    r             = redis.from_url(REDIS_URL, decode_responses=True)
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT) if CHROMA_HOST else chromadb.PersistentClient(path=CHROMA_PATH)
    collection    = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    while _worker_stats["running"]:
        item = r.brpop(QUEUE_NAME, timeout=5)
        if item is None:
            continue

        _, raw = item
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            log.error("[worker] Geçersiz JSON, atlanıyor.")
            continue

        place_name = entry.get("name") or extract_place_name(entry)
        reviews    = entry.get("reviews", [])
        source_url = entry.get("url", "")

        log.info(f"[worker] İndeksleniyor: '{place_name}' — {len(reviews)} yorum")

        if not reviews:
            _worker_stats["failed"] += 1
            continue

        try:
            added = index_place(collection, model, place_name, source_url, reviews, upsert=True)
            if added:
                _worker_stats["processed"] += 1
                log.info(f"[worker] ✓ '{place_name}' — {added} yorum eklendi/güncellendi.")
            else:
                log.info(f"[worker] '{place_name}' zaten indexli, atlandı.")
        except Exception as e:
            log.error(f"[worker] ✗ '{place_name}': {e}")
            _worker_stats["failed"] += 1

    log.info("[worker] Indexer worker durduruldu.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _worker_stats["running"] = True
    threading.Thread(target=_worker_loop, daemon=True).start()
    log.info("[startup] Indexer Redis worker otomatik başlatıldı.")
    yield
    _worker_stats["running"] = False


fa_app = FastAPI(title="Indexer Mikroservisi", lifespan=lifespan)


@fa_app.get("/health")
def health():
    count = None
    try:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT) if CHROMA_HOST else chromadb.PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        count = collection.count()
    except Exception as e:
        return {
            "status": "degraded",
            "detail": str(e),
            "collection": COLLECTION_NAME,
            "model": EMBED_MODEL,
        }

    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "model": EMBED_MODEL,
        "total_indexed_reviews": count,
        "worker": _worker_stats,
    }


@fa_app.post("/run")
def trigger(reset: bool = False):
    with _lock:
        if _job["status"] == "running":
            return {"status": "already_running"}
        _job.update({"status": "running", "detail": ""})

    def _task():
        try:
            run(INDEXER_INPUT, reset=reset)
            with _lock:
                _job["status"] = "done"
        except Exception as e:
            with _lock:
                _job.update({"status": "error", "detail": str(e)})

    threading.Thread(target=_task, daemon=True).start()
    return {"status": "started", "input": INDEXER_INPUT, "chroma_path": CHROMA_PATH}


@fa_app.get("/status")
def status():
    return _job


@fa_app.post("/worker/stop")
def worker_stop():
    _worker_stats["running"] = False
    return {"status": "stopping"}


@fa_app.get("/worker/status")
def worker_status():
    pending = 0
    try:
        r       = redis.from_url(REDIS_URL, decode_responses=True)
        pending = r.llen(QUEUE_NAME)
    except Exception:
        pass
    return {**_worker_stats, "queue_pending": pending}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChromaDB review indexer")
    parser.add_argument("--input", default=INDEXER_INPUT)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    run(args.input, reset=args.reset)
