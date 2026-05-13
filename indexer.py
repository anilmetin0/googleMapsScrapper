import json
import os
import sys
import logging
import argparse
import hashlib
import math
import re
import threading
from datetime import datetime, UTC
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
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
EMBED_MODEL_SECONDARY = os.getenv("EMBED_MODEL_SECONDARY", "").strip()
CHROMA_PATH     = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_HOST     = os.getenv("CHROMA_HOST", "")
CHROMA_PORT     = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "place_reviews_v2")
COLLECTION_NAME_SECONDARY = os.getenv("CHROMA_COLLECTION_SECONDARY", "").strip()
HASH_EMBED_DIM  = int(os.getenv("HASH_EMBED_DIM", "384"))
HASH_EMBED_DIM_SECONDARY = int(os.getenv("HASH_EMBED_DIM_SECONDARY", str(HASH_EMBED_DIM)))
REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379")
INDEXER_INPUT   = os.getenv("INDEXER_INPUT", "scraped_data.json")
INDEXER_ARCHIVE_ROOT = os.getenv("INDEXER_ARCHIVE_ROOT", "/data/index")
INDEXER_ENABLE_SECONDARY_WORKER = os.getenv("INDEXER_ENABLE_SECONDARY_WORKER", "true").lower() in {"1", "true", "yes", "on"}
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


def use_hash_embeddings(model_name: str = EMBED_MODEL) -> bool:
    return model_name.lower() in {"hash", "hashing", "local-hashing"}


def hash_embedding(text: str, dim: int = HASH_EMBED_DIM) -> list[float]:
    vector = [0.0] * dim
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    if not tokens:
        tokens = [text.lower()]

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def encode_texts(
    model: Optional[SentenceTransformer],
    texts: list[str],
    model_name: str = EMBED_MODEL,
    hash_dim: int = HASH_EMBED_DIM,
) -> list[list[float]]:
    if use_hash_embeddings(model_name):
        return [hash_embedding(text, hash_dim) for text in texts]

    if model is None:
        raise RuntimeError("SentenceTransformer model is not loaded")

    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def index_place(
    collection,
    model:      Optional[SentenceTransformer],
    place_name: str,
    source_url: str,
    reviews:    list,
    model_name:  str = EMBED_MODEL,
    hash_dim:    int = HASH_EMBED_DIM,
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
    embeddings = encode_texts(model, prefixed, model_name=model_name, hash_dim=hash_dim)

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
    model = None if use_hash_embeddings() else SentenceTransformer(EMBED_MODEL, device=DEVICE)

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

        added = index_place(collection, model, place_name, source_url, reviews, model_name=EMBED_MODEL, hash_dim=HASH_EMBED_DIM)
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
_models: dict[str, SentenceTransformer] = {}
_model_lock = threading.Lock()


def get_model(model_name: str) -> Optional[SentenceTransformer]:
    if use_hash_embeddings(model_name):
        return None

    if model_name not in _models:
        with _model_lock:
            if model_name not in _models:
                log.info(f"Model lazy yükleniyor: {model_name} ({DEVICE})")
                _models[model_name] = SentenceTransformer(model_name, device=DEVICE)
    return _models[model_name]


def embedding_targets(chroma_client):
    targets = [
        {
            "name": "primary",
            "collection_name": COLLECTION_NAME,
            "model_name": EMBED_MODEL,
            "hash_dim": HASH_EMBED_DIM,
        }
    ]
    if COLLECTION_NAME_SECONDARY and EMBED_MODEL_SECONDARY:
        targets.append(
            {
                "name": "secondary",
                "collection_name": COLLECTION_NAME_SECONDARY,
                "model_name": EMBED_MODEL_SECONDARY,
                "hash_dim": HASH_EMBED_DIM_SECONDARY,
            }
        )

    for target in targets:
        target["collection"] = chroma_client.get_or_create_collection(
            name=target["collection_name"],
            metadata={"hnsw:space": "cosine"},
        )
    return targets


def archive_index_result(run_id: Optional[str], payload: dict):
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(run_id or "unknown")).strip("-") or "unknown"
    root = Path(INDEXER_ARCHIVE_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": safe_run_id,
        "archived_at": datetime.now(UTC).isoformat(),
        **payload,
    }
    with (root / f"{safe_run_id}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _worker_loop():
    log.info(f"[worker] Indexer Redis worker başladı — kuyruk: {QUEUE_NAME}")
    r             = redis.from_url(REDIS_URL, decode_responses=True)
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT) if CHROMA_HOST else chromadb.PersistentClient(path=CHROMA_PATH)
    targets       = embedding_targets(chroma_client)

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
        run_id     = entry.get("run_id")

        log.info(f"[worker] İndeksleniyor: '{place_name}' — {len(reviews)} yorum")

        if not reviews:
            _worker_stats["failed"] += 1
            continue

        try:
            indexed_targets = []
            total_added = 0
            for target in targets:
                if target["name"] == "secondary" and not INDEXER_ENABLE_SECONDARY_WORKER:
                    indexed_targets.append({
                        "target": target["name"],
                        "collection": target["collection_name"],
                        "model": target["model_name"],
                        "added": 0,
                        "count": target["collection"].count(),
                        "skipped": "secondary_deferred",
                    })
                    continue
                added = index_place(
                    target["collection"],
                    get_model(target["model_name"]),
                    place_name,
                    source_url,
                    reviews,
                    model_name=target["model_name"],
                    hash_dim=target["hash_dim"],
                    upsert=True,
                )
                total_added += added
                indexed_targets.append({
                    "target": target["name"],
                    "collection": target["collection_name"],
                    "model": target["model_name"],
                    "added": added,
                    "count": target["collection"].count(),
                })
            try:
                archive_index_result(run_id, {
                    "place_name": place_name,
                    "source_url": source_url,
                    "raw_archive_path": entry.get("raw_archive_path"),
                    "review_count": len(reviews),
                    "targets": indexed_targets,
                })
            except Exception as archive_error:
                log.warning(f"Indexer arşivi yazılamadı: {archive_error}")
            if total_added:
                _worker_stats["processed"] += 1
                log.info(f"[worker] ✓ '{place_name}' — {total_added} indeks kaydı eklendi/güncellendi.")
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
    target_payload = []
    try:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT) if CHROMA_HOST else chromadb.PersistentClient(path=CHROMA_PATH)
        for target in embedding_targets(chroma_client):
            target_payload.append({
                "target": target["name"],
                "collection": target["collection_name"],
                "model": target["model_name"],
                "hash_embed_dim": target["hash_dim"] if use_hash_embeddings(target["model_name"]) else None,
                "total_indexed_reviews": target["collection"].count(),
            })
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
        "model_loaded": use_hash_embeddings() or EMBED_MODEL in _models,
        "hash_embed_dim": HASH_EMBED_DIM if use_hash_embeddings() else None,
        "total_indexed_reviews": target_payload[0]["total_indexed_reviews"] if target_payload else 0,
        "targets": target_payload,
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
