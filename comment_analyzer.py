import json
import os
import re
import argparse
import logging
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime, UTC
from pathlib import Path
from urllib.parse import unquote

import time

import redis
from anthropic import Anthropic, RateLimitError, APIStatusError
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL  = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
MAX_RETRIES    = 3
MAX_REVIEWS    = 200

ANALYZER_INPUT  = os.getenv("ANALYZER_INPUT",  "scraped_data.json")
ANALYZER_OUTPUT = os.getenv("ANALYZER_OUTPUT", "analyzed_data.json")
ANALYZER_ARCHIVE_ROOT = os.getenv("ANALYZER_ARCHIVE_ROOT", "/data/analyzed")
REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379")
QUEUE_NAME      = "queue:places:analyzer"
DB_QUEUE        = "queue:places:to_db"

TARGET_SCHEMA = {
    "scores": {
        "atmosfer":                   "float 1-10",
        "urun_kalitesi":              "float 1-10",
        "yiyecek_kalitesi":           "float 1-10",
        "hizmet":                     "float 1-10",
        "sessizlik_calisma_uygunlugu": "float 1-10",
        "fiyat_performans":           "float 1-10",
    },
    "genel_puan":    "float 1-10",
    "ozet":          "string — 2-3 cümle nesnel özet",
    "one_cikanlar":  ["string — en fazla 5 madde"],
    "eksiler":       ["string — yorumlarda geçen somut şikayetler"],
    "populer_urunler": ["string — yorumlarda adı geçen ürünler"],
    "etiketler":     ["string — kısa tanımlayıcı etiketler"],
    "kim_icin_ideal": "string — tek cümle",
    "fiyat_seviyesi": "ucuz | orta | orta-üst | pahalı | belirtilmemiş",
    "wifi_priz":     "var | yok | belirtilmemiş",
    "kalabalik_seviyesi": "sakin | orta | kalabalık | değişken | belirtilmemiş",
}

SYSTEM_PROMPT = (
    "Sen bir mekan analiz asistanısın. "
    "Sana verilen Google Maps yorumlarını analiz edip YALNIZCA geçerli bir JSON objesi döndür. "
    "Başka hiçbir şey yazma. Markdown, açıklama veya kod bloğu kullanma."
)

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


def clean_response(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    return raw


def _max_score_for_review_count(n: int) -> float:
    if n <= 10:  return 7.0
    if n <= 30:  return 8.0
    if n <= 50:  return 9.0
    return 10.0


def build_user_message(place_name: str, reviews: list) -> str:
    truncated    = reviews[:MAX_REVIEWS]
    reviews_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(truncated))
    schema_str   = json.dumps(TARGET_SCHEMA, ensure_ascii=False, indent=2)
    max_score    = _max_score_for_review_count(len(truncated))
    return f"""Mekan adı: {place_name}

Yorumlar ({len(truncated)} adet):
{reviews_text}

Döndürmen gereken JSON formatı:
{schema_str}

Kurallar:
- Tüm puanlar 1-10 arasında float (örn: 8.5)
- Bu mekan için maksimum verebileceğin puan {max_score}'dir çünkü yalnızca {len(truncated)} yorum var
- ozet: 2-3 cümle, nesnel, yalnızca yorumlara dayalı
- one_cikanlar: en fazla 5 madde, yorumlarda geçen güçlü yönler
- eksiler: yorumlarda geçen somut şikayetler; yoksa boş liste []
- populer_urunler: yorumlarda adı geçen yiyecek/içecekler
- etiketler: mekanı tanımlayan kısa kelimeler (örn: "cozy", "bahçeli", "çalışma dostu")
- fiyat_seviyesi: yorumlardaki ipuçlarına göre kategorize et
- wifi_priz: yorumlarda wifi/priz/şarj geçiyorsa "var", açıkça yoktu/çalışmıyor deniyorsa "yok", hiç geçmiyorsa "belirtilmemiş"
- kalabalik_seviyesi: yorumlardaki kalabalık/kuyruk/sessiz/sakin ifadelerine göre kategorize et"""


def fallback_analysis(place_name: str, reviews: list, reason: str) -> dict:
    text = " ".join(str(review) for review in reviews).lower()
    joined_sample = " ".join(str(review).strip() for review in reviews[:3] if str(review).strip())

    positives = []
    if any(word in text for word in ["güzel", "harika", "iyi", "mükemmel", "sevdim"]):
        positives.append("Yorumlarda olumlu genel deneyim vurgusu var.")
    if any(word in text for word in ["lezzet", "kahve", "tatlı", "yemek", "çay"]):
        positives.append("Ürün ve lezzet tarafı yorumlarda öne çıkıyor.")
    if any(word in text for word in ["temiz", "ferah", "atmosfer", "ortam"]):
        positives.append("Ortam/atmosfer hakkında yorum sinyali var.")
    if not positives:
        positives.append("Sınırlı yorum üzerinden temel kullanıcı deneyimi çıkarıldı.")

    negatives = []
    if any(word in text for word in ["pahalı", "fiyat", "ücret"]):
        negatives.append("Fiyat algısı yorumlarda takip edilmeli.")
    if any(word in text for word in ["kalabalık", "sıra", "bekle"]):
        negatives.append("Yoğunluk veya bekleme süresi sinyali var.")
    if any(word in text for word in ["kötü", "yavaş", "soğuk", "kirli"]):
        negatives.append("Olumsuz deneyim belirten yorumlar var.")

    products = [
        product
        for product in ["kahve", "çay", "tatlı", "pasta", "simit", "yemek"]
        if product in text
    ]

    score = min(_max_score_for_review_count(len(reviews)), 7.0)
    if negatives:
        score = max(5.5, score - 0.5)

    return {
        "place_name": place_name,
        "scores": {
            "atmosfer": score,
            "urun_kalitesi": score,
            "yiyecek_kalitesi": score,
            "hizmet": score,
            "sessizlik_calisma_uygunlugu": 6.0,
            "fiyat_performans": 6.0 if negatives else score,
        },
        "genel_puan": score,
        "ozet": (
            f"{place_name} için LLM analizi kullanılamadığı için yorumlara dayalı "
            f"deterministik özet üretildi. {joined_sample[:240]}"
        ).strip(),
        "one_cikanlar": positives[:5],
        "eksiler": negatives,
        "populer_urunler": products,
        "etiketler": ["llm-fallback", "google-yorumlari"],
        "kim_icin_ideal": "Kısa yorum sinyallerine göre hızlı mekan keşfi yapmak isteyen kullanıcılar için.",
        "fiyat_seviyesi": "belirtilmemiş",
        "wifi_priz": "var" if any(word in text for word in ["wifi", "wi-fi", "priz", "şarj"]) else "belirtilmemiş",
        "kalabalik_seviyesi": "kalabalık" if any(word in text for word in ["kalabalık", "sıra", "kuyruk"]) else "belirtilmemiş",
        "fallback_reason": reason,
    }


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------

def analyze_single(
    client:     Anthropic,
    model:      str,
    place_name: str,
    reviews:    list,
    dry_run:    bool = False,
) -> dict:
    if dry_run:
        log.info(f"[dry-run] '{place_name}' için API çağrısı atlanıyor.")
        return {
            "place_name":   place_name,
            "scores":       {k: 0.0 for k in TARGET_SCHEMA["scores"]},
            "genel_puan":   0.0,
            "ozet":         "dry-run modu",
            "one_cikanlar": [],
            "eksiler":      [],
            "populer_urunler": [],
            "etiketler":    ["dry-run"],
            "kim_icin_ideal": "dry-run",
            "fiyat_seviyesi": "belirtilmemiş",
        }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=model,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_message(place_name, reviews)}],
                temperature=0.1,
                max_tokens=2048,
            )
            raw    = clean_response(response.content[0].text or "")
            parsed = json.loads(raw)
            return parsed

        except json.JSONDecodeError as e:
            log.warning(f"JSON parse hatası (deneme {attempt}/{MAX_RETRIES}): {e}")
        except RateLimitError:
            wait = 2 ** attempt
            log.warning(f"Rate limit, {wait}s bekleniyor (deneme {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)
            continue
        except APIStatusError as e:
            log.warning(f"API hatası {e.status_code} (deneme {attempt}/{MAX_RETRIES}): {e.message}")
        except Exception as e:
            log.warning(f"Hata (deneme {attempt}/{MAX_RETRIES}): {e}")

        if attempt < MAX_RETRIES:
            log.info("Tekrar deneniyor...")

    reason = f"llm_failed_after_{MAX_RETRIES}_attempts"
    log.warning(f"'{place_name}' için LLM analizi başarısız; fallback analiz kullanılacak.")
    return fallback_analysis(place_name, reviews, reason)


# ---------------------------------------------------------------------------
# Batch pipeline (CLI)
# ---------------------------------------------------------------------------

def run(input_path: str, output_path: str, model: str, dry_run: bool = False):
    input_file = Path(input_path)
    if not input_file.exists():
        log.error(f"Girdi dosyası bulunamadı: {input_file}")
        sys.exit(1)

    with open(input_file, encoding="utf-8") as f:
        scraped_data = json.load(f)

    if not isinstance(scraped_data, list):
        log.error("scraped_data.json bir liste (array) olmalı.")
        sys.exit(1)

    total  = len(scraped_data)
    client = None if dry_run else Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    log.info(f"{total} mekan bulundu → analiz başlıyor.")

    results = []
    failed  = []

    for idx, entry in enumerate(scraped_data, start=1):
        place_name = extract_place_name(entry)
        reviews    = entry.get("reviews", [])
        log.info(f"[{idx}/{total}] '{place_name}' — {len(reviews)} yorum")

        if not reviews:
            log.warning("Yorum bulunamadı, atlanıyor.")
            failed.append({"place_name": place_name, "reason": "yorum yok"})
            continue

        try:
            result                          = analyze_single(client, model, place_name, reviews, dry_run)
            result["place_name"]            = place_name
            result["source_url"]            = entry.get("url", "")
            result["total_reviews_scraped"] = entry.get("total_reviews_scraped", len(reviews))
            result["analyzed_at"]           = datetime.now(UTC).isoformat()
            results.append(result)
            log.info(f"  ✓ Genel puan: {result.get('genel_puan', '?')}")
        except RuntimeError as e:
            log.error(f"  ✗ {e}")
            failed.append({"place_name": place_name, "reason": str(e)})

    output_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_places": total,
        "successful":   len(results),
        "failed":       len(failed),
        "failed_places": failed,
        "data":         results,
    }
    with open(Path(output_path), "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    log.info(
        f"Tamamlandı → {output_path} | "
        f"Başarılı: {len(results)}/{total} | Başarısız: {len(failed)}/{total}"
    )


# ---------------------------------------------------------------------------
# FastAPI mikroservis
# ---------------------------------------------------------------------------

_job:            dict = {"status": "idle", "detail": ""}
_lock                 = threading.Lock()
_worker_stats:   dict = {"processed": 0, "failed": 0, "running": False}


def _append_to_output(result: dict, output_path: str):
    path = Path(output_path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {"data": [], "successful": 0, "failed": 0}

    payload["data"].append(result)
    payload["successful"]   = len(payload["data"])
    payload["generated_at"] = datetime.now(UTC).isoformat()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _archive_analysis(entry: dict, result: dict):
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(entry.get("run_id") or "unknown")).strip("-") or "unknown"
    root = Path(ANALYZER_ARCHIVE_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "archived_at": datetime.now(UTC).isoformat(),
        "source_url": entry.get("url", ""),
        "place_name": result.get("place_name"),
        "analysis": result,
    }
    with (root / f"{run_id}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _worker_loop(model: str, output_path: str):
    log.info(f"[worker] Redis worker başladı — kuyruk: {QUEUE_NAME}")
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    r      = redis.from_url(REDIS_URL, decode_responses=True)

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

        log.info(f"[worker] İşleniyor: '{place_name}' — {len(reviews)} yorum")

        if not reviews:
            log.warning(f"[worker] '{place_name}' için yorum yok, atlanıyor.")
            _worker_stats["failed"] += 1
            continue

        try:
            result                          = analyze_single(client, model, place_name, reviews)
            result["place_name"]            = place_name
            result["source_url"]            = entry.get("url", "")
            result["total_reviews_scraped"] = entry.get("total_reviews_scraped", len(reviews))
            result["analyzed_at"]           = datetime.now(UTC).isoformat()

            _append_to_output(result, output_path)
            try:
                _archive_analysis(entry, result)
            except Exception as archive_error:
                log.warning(f"Analiz arşivi yazılamadı: {archive_error}")
            _worker_stats["processed"] += 1
            log.info(f"[worker] ✓ '{place_name}' — genel puan: {result.get('genel_puan', '?')}")

            db_payload = json.dumps({
                "place": {
                    "url":           entry.get("url", ""),
                    "name":          place_name,
                    "lat":           entry.get("lat"),
                    "lng":           entry.get("lng"),
                    "address":       entry.get("address"),
                    "phone":         entry.get("phone"),
                    "rating":        entry.get("rating"),
                    "total_ratings": entry.get("total_ratings"),
                    "website_url":   entry.get("website_url"),
                    "website_type":  entry.get("website_type"),
                    "images":        entry.get("images", []),
                    "reviews":       entry.get("reviews", []),
                    "run_id":        entry.get("run_id"),
                    "raw_archive_path": entry.get("raw_archive_path"),
                },
                "analysis": result,
            }, ensure_ascii=False)
            r.lpush(DB_QUEUE, db_payload)

        except RuntimeError as e:
            log.error(f"[worker] ✗ '{place_name}': {e}")
            _worker_stats["failed"] += 1

    log.info("[worker] Worker durduruldu.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _worker_stats["running"] = True
    threading.Thread(
        target=_worker_loop,
        args=(DEFAULT_MODEL, ANALYZER_OUTPUT),
        daemon=True,
    ).start()
    log.info("[startup] Redis worker otomatik başlatıldı.")
    yield
    _worker_stats["running"] = False


fa_app = FastAPI(title="Comment Analyzer Mikroservisi", lifespan=lifespan)


@fa_app.get("/health")
def health():
    return {"status": "ok"}


@fa_app.post("/run")
def trigger(model: str = DEFAULT_MODEL):
    with _lock:
        if _job["status"] == "running":
            return {"status": "already_running"}
        _job.update({"status": "running", "detail": ""})

    def _task():
        try:
            run(ANALYZER_INPUT, ANALYZER_OUTPUT, model)
            with _lock:
                _job["status"] = "done"
        except Exception as e:
            with _lock:
                _job.update({"status": "error", "detail": str(e)})

    threading.Thread(target=_task, daemon=True).start()
    return {"status": "started", "input": ANALYZER_INPUT, "output": ANALYZER_OUTPUT}


@fa_app.get("/status")
def status():
    with _lock:
        return dict(_job)


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
    parser = argparse.ArgumentParser(description="Kafe yorum analiz mikroservisi")
    parser.add_argument("--input",   default=ANALYZER_INPUT)
    parser.add_argument("--output",  default=ANALYZER_OUTPUT)
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(input_path=args.input, output_path=args.output, model=args.model, dry_run=args.dry_run)
