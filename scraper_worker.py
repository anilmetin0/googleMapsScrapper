import json
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import redis
import uvicorn
from fastapi import FastAPI
from playwright.sync_api import sync_playwright

from main import (
    _launch_page,
    _scrape_place,
    archive_place,
    get_redis,
    ANALYZER_QUEUE,
    INDEXER_QUEUE,
    SCRAPED_URLS_KEY,
    PENDING_URLS_KEY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

REDIS_URL  = os.getenv("REDIS_URL", "redis://localhost:6379")
WORKER_ID  = os.getenv("HOSTNAME", "worker")   # Docker hostname = container id

_stats: dict = {"processed": 0, "failed": 0, "running": False}


def _worker_loop():
    log.info(f"[{WORKER_ID}] Scraper worker başladı — kuyruk: {PENDING_URLS_KEY}")
    r = get_redis()

    with sync_playwright() as playwright:
        page = _launch_page(playwright)

        while _stats["running"]:
            item = r.brpop(PENDING_URLS_KEY, timeout=5)
            if item is None:
                continue

            _, raw = item
            try:
                job = json.loads(raw)
                url         = job["url"]
                max_reviews: Optional[int] = job.get("max_reviews")
                run_id: Optional[str] = job.get("run_id")
            except (json.JSONDecodeError, KeyError):
                log.error(f"[{WORKER_ID}] Geçersiz iş formatı, atlanıyor: {raw[:100]}")
                continue

            log.info(f"[{WORKER_ID}] Scrape ediliyor: {url}")

            try:
                place_data = _scrape_place(page, url, max_reviews)
                place_data["run_id"] = run_id
                raw_archive_path = archive_place(run_id, place_data)
                place_data["raw_archive_path"] = raw_archive_path

                log.info(
                    f"[{WORKER_ID}] ✓ {place_data['name']} | "
                    f"{place_data['total_reviews_scraped']} yorum | {raw_archive_path}"
                )

                payload = json.dumps(place_data, ensure_ascii=False)
                r.lpush(ANALYZER_QUEUE, payload)
                r.lpush(INDEXER_QUEUE, payload)
                r.sadd(SCRAPED_URLS_KEY, url)
                _stats["processed"] += 1

            except Exception as e:
                log.warning(f"[{WORKER_ID}] Hata, tekrar kuyruğa alındı: {e}")
                r.rpush(PENDING_URLS_KEY, raw)
                _stats["failed"] += 1
                time.sleep(2)

    log.info(f"[{WORKER_ID}] Worker durduruldu.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _stats["running"] = True
    threading.Thread(target=_worker_loop, daemon=True).start()
    log.info(f"[{WORKER_ID}] Scraper worker başlatıldı.")
    yield
    _stats["running"] = False


app = FastAPI(title="Scraper Worker", lifespan=lifespan)


@app.get("/health")
def health():
    r = get_redis()
    return {
        "status":          "ok",
        "worker_id":       WORKER_ID,
        "processed":       _stats["processed"],
        "failed":          _stats["failed"],
        "queue_pending":   r.llen(PENDING_URLS_KEY),
    }


@app.get("/status")
def status():
    r = get_redis()
    return {
        **_stats,
        "worker_id":     WORKER_ID,
        "queue_pending": r.llen(PENDING_URLS_KEY),
    }


@app.post("/worker/stop")
def stop():
    _stats["running"] = False
    return {"status": "stopping"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8085)
