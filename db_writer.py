import json
import logging
import os
import re
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime, UTC
from pathlib import Path

import psycopg2
import psycopg2.extras
import psycopg2.pool
import redis
from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

REDIS_URL    = os.getenv("REDIS_URL", "redis://localhost:6379")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://app:changeme@db:5432/neredenevar")
DB_QUEUE     = "queue:places:to_db"
DB_WRITER_ARCHIVE_ROOT = os.getenv("DB_WRITER_ARCHIVE_ROOT", "/data/db-writes")
DB_SCHEMA = (os.getenv("HAIKU_DB_SCHEMA") or os.getenv("DB_SCHEMA") or "public").strip() or "public"

# ---------------------------------------------------------------------------
# Connection pool (min=1, max=5 — worker is single-threaded, pool for health checks)
# ---------------------------------------------------------------------------

_pool: psycopg2.pool.ThreadedConnectionPool = None


def _quote_ident(identifier: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise RuntimeError(f"Invalid Postgres schema identifier: {identifier!r}")
    return '"' + identifier.replace('"', '""') + '"'


def _set_search_path(cur):
    cur.execute(f"SET search_path TO {_quote_ident(DB_SCHEMA)}, public")


def _init_pool():
    global _pool
    _pool = psycopg2.pool.ThreadedConnectionPool(1, 5, DATABASE_URL)
    log.info("DB connection pool oluşturuldu. schema=%s", DB_SCHEMA)


def get_conn():
    return _pool.getconn()


def release_conn(conn):
    _pool.putconn(conn)


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def _migrate():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS postgis")
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {_quote_ident(DB_SCHEMA)}")
            _set_search_path(cur)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS places (
                    id            SERIAL PRIMARY KEY,
                    name          TEXT,
                    source_url    TEXT UNIQUE,
                    place_type    TEXT,
                    location      GEOMETRY(Point, 4326),
                    lat           FLOAT,
                    lng           FLOAT,
                    total_reviews INT,
                    address       TEXT,
                    phone         TEXT,
                    rating        FLOAT,
                    total_ratings INT,
                    website_url   TEXT,
                    website_type  TEXT,
                    images        TEXT[],
                    scraped_at    TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    id       SERIAL PRIMARY KEY,
                    place_id INT REFERENCES places(id) ON DELETE CASCADE,
                    content  TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS place_analysis (
                    id                      SERIAL PRIMARY KEY,
                    place_id                INT UNIQUE REFERENCES places(id) ON DELETE CASCADE,
                    analyzed_at             TIMESTAMPTZ DEFAULT NOW(),
                    overall_score           FLOAT,
                    summary                 TEXT,
                    ideal_for               TEXT,
                    price_level             TEXT,
                    score_service           FLOAT,
                    score_price_performance FLOAT,
                    score_atmosphere        FLOAT,
                    scores_extra            JSONB,
                    highlights              TEXT[],
                    downsides               TEXT[],
                    popular_items           TEXT[],
                    tags                    TEXT[],
                    wifi_priz               TEXT,
                    kalabalik_seviyesi      TEXT
                )
            """)
            cur.execute("ALTER TABLE places ADD COLUMN IF NOT EXISTS location GEOMETRY(Point, 4326)")
            cur.execute("ALTER TABLE places ADD COLUMN IF NOT EXISTS address TEXT")
            cur.execute("ALTER TABLE places ADD COLUMN IF NOT EXISTS phone TEXT")
            cur.execute("ALTER TABLE places ADD COLUMN IF NOT EXISTS rating FLOAT")
            cur.execute("ALTER TABLE places ADD COLUMN IF NOT EXISTS total_ratings INT")
            cur.execute("ALTER TABLE places ADD COLUMN IF NOT EXISTS website_url TEXT")
            cur.execute("ALTER TABLE places ADD COLUMN IF NOT EXISTS website_type TEXT")
            cur.execute("ALTER TABLE places ADD COLUMN IF NOT EXISTS images TEXT[]")
            cur.execute("ALTER TABLE place_analysis ADD COLUMN IF NOT EXISTS analyzed_at TIMESTAMPTZ DEFAULT NOW()")
            cur.execute("ALTER TABLE place_analysis ADD COLUMN IF NOT EXISTS wifi_priz TEXT")
            cur.execute("ALTER TABLE place_analysis ADD COLUMN IF NOT EXISTS kalabalik_seviyesi TEXT")
            cur.execute("""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conrelid = 'place_analysis'::regclass
                        AND contype = 'u'
                        AND conname = 'place_analysis_place_id_key'
                    ) THEN
                        ALTER TABLE place_analysis ADD CONSTRAINT place_analysis_place_id_key UNIQUE (place_id);
                    END IF;
                END $$
            """)
        conn.commit()
        log.info("[migrate] Tablolar güncellendi.")
    finally:
        release_conn(conn)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_to_db(payload: dict):
    place    = payload["place"]
    analysis = payload.get("analysis", {})

    lat = place.get("lat")
    lng = place.get("lng")
    url = place.get("url", "")

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            _set_search_path(cur)
            if lat is not None and lng is not None:
                cur.execute("""
                    INSERT INTO places
                        (name, source_url, place_type, location, total_reviews,
                         address, phone, rating, total_ratings,
                         website_url, website_type, images, scraped_at)
                    VALUES
                        (%s, %s, 'cafe', ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s,
                         %s, %s, %s, %s,
                         %s, %s, %s, NOW())
                    ON CONFLICT (source_url) DO UPDATE SET
                        name          = EXCLUDED.name,
                        location      = EXCLUDED.location,
                        total_reviews = EXCLUDED.total_reviews,
                        address       = EXCLUDED.address,
                        phone         = EXCLUDED.phone,
                        rating        = EXCLUDED.rating,
                        total_ratings = EXCLUDED.total_ratings,
                        website_url   = EXCLUDED.website_url,
                        website_type  = EXCLUDED.website_type,
                        images        = EXCLUDED.images,
                        scraped_at    = NOW()
                    RETURNING id
                """, (
                    place.get("name"), url,
                    lng, lat,
                    place.get("total_ratings") or len(place.get("reviews", [])),
                    place.get("address"), place.get("phone"),
                    place.get("rating"), place.get("total_ratings"),
                    place.get("website_url"), place.get("website_type"),
                    place.get("images") or [],
                ))
            else:
                cur.execute("""
                    INSERT INTO places
                        (name, source_url, place_type, location, total_reviews,
                         address, phone, rating, total_ratings,
                         website_url, website_type, images, scraped_at)
                    VALUES
                        (%s, %s, 'cafe', ST_SetSRID(ST_MakePoint(0,0), 4326), %s,
                         %s, %s, %s, %s,
                         %s, %s, %s, NOW())
                    ON CONFLICT (source_url) DO UPDATE SET
                        name          = EXCLUDED.name,
                        total_reviews = EXCLUDED.total_reviews,
                        address       = EXCLUDED.address,
                        phone         = EXCLUDED.phone,
                        rating        = EXCLUDED.rating,
                        total_ratings = EXCLUDED.total_ratings,
                        website_url   = EXCLUDED.website_url,
                        website_type  = EXCLUDED.website_type,
                        images        = EXCLUDED.images,
                        scraped_at    = NOW()
                    RETURNING id
                """, (
                    place.get("name"), url,
                    place.get("total_ratings") or len(place.get("reviews", [])),
                    place.get("address"), place.get("phone"),
                    place.get("rating"), place.get("total_ratings"),
                    place.get("website_url"), place.get("website_type"),
                    place.get("images") or [],
                ))

            row = cur.fetchone()
            if not row:
                conn.rollback()
                log.warning(f"place upsert dönmedi: {url}")
                return
            place_id = row[0]

            cur.execute("DELETE FROM reviews WHERE place_id = %s", (place_id,))
            reviews = place.get("reviews", [])
            if reviews:
                psycopg2.extras.execute_values(
                    cur,
                    "INSERT INTO reviews (place_id, content) VALUES %s",
                    [(place_id, r) for r in reviews],
                )

            if analysis:
                scores      = analysis.get("scores", {})
                base_keys   = {"hizmet", "fiyat_performans", "atmosfer"}
                scores_extra = {k: v for k, v in scores.items() if k not in base_keys}
                cur.execute("""
                    INSERT INTO place_analysis (
                        place_id, analyzed_at, overall_score, summary, ideal_for, price_level,
                        score_service, score_price_performance, score_atmosphere,
                        scores_extra, highlights, downsides, popular_items, tags,
                        wifi_priz, kalabalik_seviyesi
                    ) VALUES (%s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (place_id) DO UPDATE SET
                        analyzed_at             = NOW(),
                        overall_score           = EXCLUDED.overall_score,
                        summary                 = EXCLUDED.summary,
                        ideal_for               = EXCLUDED.ideal_for,
                        price_level             = EXCLUDED.price_level,
                        score_service           = EXCLUDED.score_service,
                        score_price_performance = EXCLUDED.score_price_performance,
                        score_atmosphere        = EXCLUDED.score_atmosphere,
                        scores_extra            = EXCLUDED.scores_extra,
                        highlights              = EXCLUDED.highlights,
                        downsides               = EXCLUDED.downsides,
                        popular_items           = EXCLUDED.popular_items,
                        tags                    = EXCLUDED.tags,
                        wifi_priz               = EXCLUDED.wifi_priz,
                        kalabalik_seviyesi      = EXCLUDED.kalabalik_seviyesi
                """, (
                    place_id,
                    analysis.get("genel_puan"),
                    analysis.get("ozet"),
                    analysis.get("kim_icin_ideal"),
                    analysis.get("fiyat_seviyesi"),
                    scores.get("hizmet"),
                    scores.get("fiyat_performans"),
                    scores.get("atmosfer"),
                    json.dumps(scores_extra, ensure_ascii=False),
                    analysis.get("one_cikanlar") or [],
                    analysis.get("eksiler") or [],
                    analysis.get("populer_urunler") or [],
                    analysis.get("etiketler") or [],
                    analysis.get("wifi_priz", "belirtilmemiş"),
                    analysis.get("kalabalik_seviyesi", "belirtilmemiş"),
                ))

        conn.commit()
        log.info(f"  ✓ DB yazıldı: '{place.get('name')}' (id={place_id})")
        try:
            _archive_db_write(place, analysis, place_id)
        except Exception as archive_error:
            log.warning(f"DB write arşivi yazılamadı: {archive_error}")
    except Exception:
        conn.rollback()
        raise
    finally:
        release_conn(conn)


def _archive_db_write(place: dict, analysis: dict, place_id: int):
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(place.get("run_id") or "unknown")).strip("-") or "unknown"
    root = Path(DB_WRITER_ARCHIVE_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "archived_at": datetime.now(UTC).isoformat(),
        "place_id": place_id,
        "source_url": place.get("url", ""),
        "place_name": place.get("name"),
        "raw_archive_path": place.get("raw_archive_path"),
        "analysis_present": bool(analysis),
    }
    with (root / f"{run_id}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

_worker_stats: dict = {"processed": 0, "failed": 0, "running": False}


def _worker_loop():
    log.info(f"[worker] DB Writer başladı — kuyruk: {DB_QUEUE}")
    r = redis.from_url(REDIS_URL, decode_responses=True)

    while _worker_stats["running"]:
        item = r.brpop(DB_QUEUE, timeout=5)
        if item is None:
            continue

        _, raw = item
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            log.error("[worker] Geçersiz JSON, atlanıyor.")
            continue

        place_name = payload.get("place", {}).get("name", "?")
        log.info(f"[worker] Yazılıyor: '{place_name}'")
        try:
            write_to_db(payload)
            _worker_stats["processed"] += 1
        except Exception as e:
            log.error(f"[worker] Hata '{place_name}': {e}")
            _worker_stats["failed"] += 1

    log.info("[worker] DB Writer durduruldu.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_pool()
    _migrate()
    _worker_stats["running"] = True
    threading.Thread(target=_worker_loop, daemon=True).start()
    log.info("[startup] DB Writer worker otomatik başlatıldı.")
    yield
    _worker_stats["running"] = False
    if _pool:
        _pool.closeall()


app = FastAPI(title="DB Writer", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "schema": DB_SCHEMA}


@app.get("/worker/status")
def worker_status():
    pending = 0
    try:
        r       = redis.from_url(REDIS_URL, decode_responses=True)
        pending = r.llen(DB_QUEUE)
    except Exception:
        pass
    return {**_worker_stats, "queue_pending": pending}


@app.post("/worker/stop")
def worker_stop():
    _worker_stats["running"] = False
    return {"status": "stopping"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8086)
