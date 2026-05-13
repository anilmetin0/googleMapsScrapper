import json
import logging
import os
import re
import sys
import time
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

import redis
import uvicorn
from fastapi import BackgroundTasks, FastAPI
from playwright.sync_api import Page, sync_playwright
from pydantic import BaseModel, Field

from URLs import URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

REDIS_URL         = os.getenv("REDIS_URL", "redis://localhost:6379")
SCRAPER_OUTPUT    = os.getenv("SCRAPER_OUTPUT", "scraped_data.json")
RAW_ARCHIVE_ROOT  = os.getenv("RAW_ARCHIVE_ROOT", "/data/raw")
ANALYZER_QUEUE    = "queue:places:analyzer"
INDEXER_QUEUE     = "queue:places:indexer"
SCRAPED_URLS_KEY  = "scraped:urls"
SCRAPED_CELLS_KEY = "scraped:cells"
PENDING_URLS_KEY  = "pending:urls"
SCRAPE_PHASE_KEY  = "scrape:phase"
SCRAPER_DRAIN_QUEUES = [
    item.strip()
    for item in os.getenv(
        "SCRAPER_DRAIN_QUEUES",
        "pending:urls,queue:places:analyzer,queue:places:indexer,queue:places:to_db",
    ).split(",")
    if item.strip()
]

ANKARA_BOUNDS = {
    "lat_min": 39.75, "lat_max": 40.05,
    "lon_min": 32.50, "lon_max": 33.10,
}


def get_redis():
    return redis.from_url(REDIS_URL, decode_responses=True)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class CityScrapRequest(BaseModel):
    keyword:      str            = "kafe"
    divisions:    int            = Field(default=4, ge=1, le=20)
    max_reviews:  Optional[int]  = Field(default=None, ge=1)
    place_limit:  Optional[int]  = Field(default=None, ge=1)
    run_id:       Optional[str]  = None
    resume:       bool           = True
    bounds:       Optional[dict] = None
    callback_url: Optional[str]  = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Scraper Mikroservisi", lifespan=lifespan)


@app.post("/api/googleMaps/city/cafes")
def scrape_city_cafes(background_tasks: BackgroundTasks, req: CityScrapRequest):
    bounds = req.bounds or ANKARA_BOUNDS
    grid   = _build_grid_divisions(bounds, req.divisions)
    run_id = _safe_run_id(req.run_id)
    _write_run_manifest(run_id, {
        "run_id": run_id,
        "keyword": req.keyword,
        "bounds": bounds,
        "divisions": req.divisions,
        "max_reviews": req.max_reviews,
        "place_limit": req.place_limit,
        "started_at": datetime.now(UTC).isoformat(),
    })
    background_tasks.add_task(
        _scrape_city_grid, grid, req.keyword, req.resume, req.max_reviews, req.place_limit, req.callback_url, run_id
    )
    return {
        "status":       "started",
        "grid_cells":   len(grid),
        "divisions":    f"{req.divisions}x{req.divisions}",
        "run_id":       run_id,
        "resume":       req.resume,
        "callback_url": req.callback_url,
    }


@app.get("/api/googleMaps/city/status")
def scrape_city_status():
    r           = get_redis()
    phase       = r.get(SCRAPE_PHASE_KEY) or "idle"
    pending     = r.llen(PENDING_URLS_KEY)
    scraped     = r.scard(SCRAPED_URLS_KEY)
    cells_done  = r.scard(SCRAPED_CELLS_KEY)
    place_count = 0
    if os.path.exists(SCRAPER_OUTPUT):
        try:
            with open(SCRAPER_OUTPUT, encoding="utf-8") as f:
                place_count = len(json.load(f))
        except Exception:
            pass
    return {
        "phase":         phase,
        "cells_completed": cells_done,
        "urls_found":    scraped + pending,
        "urls_scraped":  scraped,
        "urls_pending":  pending,
        "places_in_file": place_count,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Browser helpers
# ---------------------------------------------------------------------------

def _launch_page(playwright) -> Page:
    headless = os.getenv("HEADLESS", "false").lower() == "true"
    browser  = playwright.chromium.launch(
        headless=headless,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
            "--window-size=1920,1080",
        ],
    )
    proxy_url    = os.getenv("PROXY_URL", "")
    context_opts = {
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "viewport":           {"width": 1920, "height": 1080},
        "locale":             "tr-TR",
        "extra_http_headers": {"Accept-Language": "tr-TR,tr;q=0.9"},
    }
    if proxy_url:
        context_opts["proxy"] = {"server": proxy_url}
        log.info(f"Proxy aktif: {proxy_url}")

    context = browser.new_context(**context_opts)
    context.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )

    cookies_file = os.getenv("COOKIES_FILE", "cookies.json")
    if os.path.exists(cookies_file):
        with open(cookies_file, encoding="utf-8") as f:
            cookies = json.load(f)
        context.add_cookies(cookies)
        log.info(f"Cookie yüklendi: {cookies_file} ({len(cookies)} adet)")

    page = context.new_page()
    page.set_default_navigation_timeout(30000)
    page.set_default_timeout(15000)
    return page


def _accept_consent(page: Page):
    for sel in [
        'button[jsname="b3VHJd"]',
        'span[jsname="m9ZlFb"]',
        'form[action*="consent"] button[type="submit"]:last-of-type',
        'button:has-text("Kabul et")',
        'button:has-text("Accept all")',
    ]:
        try:
            el = page.locator(sel).first
            if el.is_visible(timeout=3000):
                el.click()
                page.wait_for_timeout(3000)
                log.info("[consent] Kabul edildi.")
                return
        except Exception:
            continue


def _renew_tor_circuit():
    tor_host = os.getenv("TOR_HOST", "localhost")
    tor_control_password = os.getenv("TOR_CONTROL_PASSWORD", "")
    try:
        with urllib.request.urlopen(f"http://{tor_host}:8118", timeout=3):
            pass
    except Exception:
        pass
    try:
        import socket
        auth = (
            f'AUTHENTICATE "{tor_control_password}"\r\n'
            if tor_control_password
            else "AUTHENTICATE\r\n"
        )
        with socket.create_connection((tor_host, 9051), timeout=5) as s:
            s.sendall((auth + "SIGNAL NEWNYM\r\nQUIT\r\n").encode("utf-8"))
            response = s.recv(4096).decode(errors="replace")
        if "250 OK" not in response:
            raise RuntimeError(response.strip() or "empty Tor control response")
        log.info("[tor] Yeni devre istendi.")
        time.sleep(5)
    except Exception as e:
        log.warning(f"[tor] Devre yenilenemedi: {e}")


def _goto(page: Page, url: str, retries: int = 3):
    for attempt in range(1, retries + 1):
        page.goto(url)
        page.wait_for_timeout(3000)
        if "/sorry/" in page.url:
            log.warning(f"[captcha] Google bot sayfası — yeni Tor devresi isteniyor (deneme {attempt}/{retries})")
            _renew_tor_circuit()
            continue
        _accept_consent(page)
        return
    log.error(f"[captcha] {retries} denemede de bot sayfası aşılamadı: {url}")


# ---------------------------------------------------------------------------
# Faz 1: URL toplama
# ---------------------------------------------------------------------------

def _collect_urls_from_feed(page: Page, seen_urls: set, limit: Optional[int] = None) -> list:
    feed_selector  = 'div[role="feed"]'
    place_selector = 'a[href*="/maps/place/"]'
    collected      = []
    last_total     = 0
    no_new_attempts = 0

    # feed içinde mi yoksa direkt mi — hangisi çalışıyorsa onu kullan
    use_feed = page.locator(f'{feed_selector} {place_selector}').count() > 0
    log.info(
        f"[feed] feed_count={page.locator(feed_selector).count()} "
        f"place_links={page.locator(place_selector).count()} "
        f"use_feed_scope={use_feed}"
    )

    while True:
        scope   = f'{feed_selector} ' if use_feed else ''
        places  = page.locator(f'{scope}{place_selector}').all()
        current_total = len(places)
        for place in places:
            try:
                url = place.get_attribute("href", timeout=1000)
            except Exception:
                continue
            if not url or url in seen_urls or url in collected:
                continue
            collected.append(url)
            if limit and len(collected) >= limit:
                log.info(f"URL toplama limiti — {len(collected)} yeni URL")
                return collected

        end_of_list = page.locator(
            'span:has-text("Bu listenin sonuna geldiniz"), '
            'span:has-text("You\'ve reached the end of the list")'
        )
        if end_of_list.count() > 0:
            log.info(f"Liste sonu — {len(collected)} yeni URL")
            break

        if current_total == last_total:
            no_new_attempts += 1
            if no_new_attempts >= 4:
                log.info(f"Feed durdu — {len(collected)} yeni URL")
                break
        else:
            no_new_attempts = 0

        last_total = current_total
        try:
            if use_feed:
                page.locator(feed_selector).evaluate("node => node.scrollBy(0, 2500)")
            else:
                page.evaluate("window.scrollBy(0, 2500)")
        except Exception:
            no_new_attempts += 1
        page.wait_for_timeout(2000)

    return collected


# ---------------------------------------------------------------------------
# Faz 2: Mekan scraping
# ---------------------------------------------------------------------------

def _detect_link_type(url: str) -> str:
    if not url:
        return "unknown"
    if "instagram.com" in url:
        return "instagram"
    if "facebook.com" in url:
        return "facebook"
    if "tripadvisor.com" in url:
        return "tripadvisor"
    return "website"


def _get_images(page: Page, max_images: int = 10) -> list:
    urls = []
    try:
        photos_btn = page.locator(
            'button[aria-label*="Fotoğraf"], button[aria-label*="Photo"]'
        ).first
        if photos_btn.count() > 0:
            photos_btn.click()
            page.wait_for_timeout(2000)

        for el in page.locator('button[style*="background-image"]').all():
            style = el.get_attribute("style") or ""
            match = re.search(r'url\("?(https?://[^")\s]+)"?\)', style)
            if match:
                img_url = match.group(1)
                if img_url not in urls:
                    urls.append(img_url)
            if len(urls) >= max_images:
                break

        if not urls:
            for img in page.locator('img[src*="googleusercontent"]').all():
                src = img.get_attribute("src") or ""
                if src and src not in urls:
                    urls.append(src)
                if len(urls) >= max_images:
                    break
    except Exception:
        pass
    return urls[:max_images]


def _extract_external_links(page: Page) -> dict:
    result = {"website_url": None, "website_type": None}
    try:
        for sel in [
            'a[data-item-id="authority"]',
            'a[aria-label*="web" i]',
            'a[aria-label*="site" i]',
        ]:
            el = page.locator(sel).first
            if el.count() > 0:
                href = el.get_attribute("href")
                if href and href.startswith("http"):
                    result["website_url"]  = href
                    result["website_type"] = _detect_link_type(href)
                    break
    except Exception:
        pass
    return result


def _extract_coords_from_url(url: str):
    m = re.search(r'!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)', url)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def _extract_place_details(page: Page) -> dict:
    details = {"address": None, "phone": None, "rating": None, "total_ratings": None}
    try:
        el = page.locator('button[data-item-id="address"]').first
        if el.count() > 0:
            details["address"] = el.inner_text(timeout=2000).strip()
    except Exception:
        pass
    try:
        el = page.locator('button[data-item-id^="phone:tel"]').first
        if el.count() > 0:
            details["phone"] = el.inner_text(timeout=2000).strip()
    except Exception:
        pass
    try:
        el = page.locator('div.F7nice span[aria-hidden="true"]').first
        if el.count() > 0:
            details["rating"] = float(
                el.inner_text(timeout=2000).replace(",", ".").strip()
            )
    except Exception:
        pass
    for kw in ["yorum", "review", "Rezension", "avis", "opinión"]:
        try:
            el = page.locator(
                f'div.F7nice span[role="img"][aria-label*="{kw}"]'
            ).first
            if el.count() > 0:
                label = el.get_attribute("aria-label", timeout=2000) or ""
                nums  = re.findall(r'\d+', label.replace(".", "").replace(",", ""))
                if nums:
                    details["total_ratings"] = int(nums[0])
                    break
        except Exception:
            continue
    return details


def _scrape_place(page: Page, url: str, max_reviews: Optional[int]) -> dict:
    _goto(page, url)

    log.info(f"[page] URL: {page.url} | Title: {page.title()}")

    place_name = ""
    try:
        place_name = page.locator('h1').first.inner_text(timeout=3000)
    except Exception:
        pass

    lat, lng  = _extract_coords_from_url(url)
    details   = _extract_place_details(page)
    links     = _extract_external_links(page)

    effective_max = max_reviews if max_reviews else (details["total_ratings"] or 500)
    reviews       = _get_reviews(page, max_reviews=effective_max)
    images        = _get_images(page, max_images=10)

    return {
        "url":                  url,
        "name":                 place_name,
        "lat":                  lat,
        "lng":                  lng,
        "address":              details["address"],
        "phone":                details["phone"],
        "rating":               details["rating"],
        "total_ratings":        details["total_ratings"],
        "website_url":          links["website_url"],
        "website_type":         links["website_type"],
        "images":               images,
        "total_reviews_scraped": len(reviews),
        "reviews":              reviews,
    }


def _get_reviews(page: Page, max_reviews: int = 200) -> list:
    try:
        tab_selector = (
            'button[role="tab"][data-tab-index="2"],'
            'button[role="tab"]:has-text("Yorumlar"),'
            'button[role="tab"]:has-text("Reviews")'
        )
        try:
            page.wait_for_selector(tab_selector, timeout=10000)
        except Exception:
            pass
        reviews_tab = page.locator(tab_selector)
        if reviews_tab.count() > 0:
            reviews_tab.first.click()
            page.wait_for_timeout(2000)
            log.info("Yorumlar sekmesine geçildi.")
        else:
            log.warning("Yorumlar sekmesi bulunamadı.")

        review_panel   = page.locator('div.m6QErb.DxyBCb').first
        collected      = []
        last_count     = 0
        scroll_attempts = 0

        log.info(f"Yorumlar toplanıyor (Hedef: {max_reviews})...")

        while len(collected) < max_reviews:
            more_buttons = page.locator(
                'button:has-text("Tamamını oku"), button:has-text("More")'
            ).all()
            clicked = 0
            for btn in more_buttons:
                try:
                    if btn.is_visible():
                        btn.click(timeout=500)
                        clicked += 1
                except Exception:
                    pass
            if clicked > 0:
                page.wait_for_timeout(800)

            elements = page.locator('span.wiI7pd').all_text_contents()
            for review in elements:
                if review not in collected:
                    collected.append(review)
                    if len(collected) >= max_reviews:
                        break

            log.info(f"  [{len(collected)}/{max_reviews}] yorum toplandı...")

            if len(collected) == last_count:
                scroll_attempts += 1
                limit = 2 if len(collected) == 0 else 8
                if scroll_attempts > limit:
                    log.info("Daha fazla yeni yorum bulunamadı, durduruluyor.")
                    break
            else:
                scroll_attempts = 0

            last_count = len(collected)
            try:
                review_panel.evaluate("node => node.scrollBy(0, 3000)")
            except Exception:
                page.mouse.wheel(0, 3000)
            page.wait_for_timeout(2500)

        log.info(f"Tamamlandı: {len(collected[:max_reviews])} yorum çekildi.")
        return collected[:max_reviews]

    except Exception as e:
        log.error(f"Yorum çekilirken hata: {e}")
        return []


# ---------------------------------------------------------------------------
# Grid tarama
# ---------------------------------------------------------------------------

def _build_grid_divisions(bounds: dict, divisions: int) -> list:
    lat_size = (bounds["lat_max"] - bounds["lat_min"]) / divisions
    lon_size = (bounds["lon_max"] - bounds["lon_min"]) / divisions
    cells    = []
    for row in range(divisions):
        for col in range(divisions):
            lat_min = bounds["lat_min"] + row * lat_size
            lat_max = lat_min + lat_size
            lon_min = bounds["lon_min"] + col * lon_size
            lon_max = lon_min + lon_size
            cells.append({
                "lat": round((lat_min + lat_max) / 2, 6),
                "lon": round((lon_min + lon_max) / 2, 6),
                "key": f"{row},{col}",
            })
    return cells


def _fire_callback(callback_url: str, payload: dict):
    try:
        body = json.dumps(payload, ensure_ascii=False).encode()
        req  = urllib.request.Request(
            callback_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            log.info(f"[callback] {callback_url} → {resp.status}")
    except Exception as e:
        log.error(f"[callback] Hata: {e}")


def _wait_for_queues_to_drain(timeout_s: int = 1800) -> bool:
    r        = get_redis()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pending = sum(r.llen(queue_name) for queue_name in SCRAPER_DRAIN_QUEUES)
        if pending == 0:
            return True
        time.sleep(5)
    return False


def _save_json(data: list, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    log.info(f"Dosyaya kaydedildi: {len(data)} mekan")


def _safe_run_id(run_id: Optional[str]) -> str:
    raw = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-")
    return safe or "run"


def _safe_place_filename(place_data: dict) -> str:
    source = place_data.get("url") or place_data.get("name") or "place"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", source)[-140:].strip("-")
    return f"{safe or 'place'}.json"


def _archive_dir(run_id: str, *parts: str) -> Path:
    path = Path(RAW_ARCHIVE_ROOT) / _safe_run_id(run_id)
    for part in parts:
        path = path / part
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_run_manifest(run_id: str, payload: dict):
    run_dir = _archive_dir(run_id)
    manifest = run_dir / "manifest.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def archive_place(run_id: Optional[str], place_data: dict) -> str:
    safe_run_id = _safe_run_id(run_id)
    places_dir = _archive_dir(safe_run_id, "places")
    path = places_dir / _safe_place_filename(place_data)
    payload = {
        "run_id": safe_run_id,
        "archived_at": datetime.now(UTC).isoformat(),
        "place": place_data,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    jsonl = _archive_dir(safe_run_id) / "places.jsonl"
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return str(path)


def _scrape_city_grid(
    grid:         list,
    keyword:      str,
    resume:       bool,
    max_reviews:  Optional[int] = None,
    place_limit:  Optional[int] = None,
    callback_url: Optional[str] = None,
    run_id:       Optional[str] = None,
):
    r = get_redis()

    # ── Faz 1: URL toplama ──
    r.set(SCRAPE_PHASE_KEY, "faz-1: URL toplanıyor")
    total_cells = len(grid)

    if not resume:
        r.delete(PENDING_URLS_KEY, SCRAPED_CELLS_KEY, SCRAPED_URLS_KEY)

    pending_urls = set()
    for raw_pending in r.lrange(PENDING_URLS_KEY, 0, -1):
        try:
            pending_urls.add(json.loads(raw_pending).get("url", raw_pending))
        except Exception:
            pending_urls.add(raw_pending)
    already_seen: set = set(r.smembers(SCRAPED_URLS_KEY)) | pending_urls
    found_for_run = 0

    for idx, cell in enumerate(grid, 1):
        if place_limit and found_for_run >= place_limit:
            log.info(f"[Faz 1] place_limit={place_limit} sınırına ulaşıldı.")
            break

        lat, lon, cell_key = cell["lat"], cell["lon"], cell["key"]
        if r.sismember(SCRAPED_CELLS_KEY, cell_key):
            log.info(f"[Faz 1] Hücre {idx}/{total_cells} atlandı → ({lat}, {lon})")
            continue

        log.info(f"[Faz 1] Hücre {idx}/{total_cells} → ({lat}, {lon})")
        try:
            with sync_playwright() as playwright:
                page     = _launch_page(playwright)
                _goto(page, URL.GOOGLE_MAPS_BASE_URL.format_url(
                    latitude=lat, longitude=lon, keyword=keyword
                ))
                remaining = max(0, place_limit - found_for_run) if place_limit else None
                new_urls = _collect_urls_from_feed(page, already_seen, remaining)

            if new_urls:
                if place_limit:
                    remaining = max(0, place_limit - found_for_run)
                    new_urls = new_urls[:remaining]

                items = [
                    json.dumps({"url": u, "max_reviews": max_reviews, "run_id": run_id}, ensure_ascii=False)
                    for u in new_urls
                ]
                r.rpush(PENDING_URLS_KEY, *items)
                already_seen.update(new_urls)
                found_for_run += len(new_urls)
                log.info(
                    f"[Faz 1] +{len(new_urls)} URL | "
                    f"Toplam bekleyen: {r.llen(PENDING_URLS_KEY)}"
                )

            r.sadd(SCRAPED_CELLS_KEY, cell_key)
        except Exception as e:
            log.warning(f"[Faz 1] Hücre hatası ({lat},{lon}): {e}")

    total_found = r.llen(PENDING_URLS_KEY)
    log.info(f"[Faz 1 Tamamlandı] Toplam {total_found} benzersiz mekan bulundu")

    # ── Faz 2: Worker'lara devredildi ──
    r.set(SCRAPE_PHASE_KEY, "faz-2: worker'lar scrape ediyor")
    log.info(
        f"[Faz 2] {total_found} URL worker'lara devredildi. "
        f"Kuyruklar boşalana kadar bekleniyor..."
    )

    drained = _wait_for_queues_to_drain()
    scraped = r.scard(SCRAPED_URLS_KEY)
    r.set(SCRAPE_PHASE_KEY, "tamamlandi")
    log.info(f"[Tamamlandı] {scraped} mekan | Kuyruklar boşaldı: {drained}")
    if run_id:
        _write_run_manifest(run_id, {
            "run_id": _safe_run_id(run_id),
            "keyword": keyword,
            "completed_at": datetime.now(UTC).isoformat(),
            "places_scraped": scraped,
            "queues_drained": drained,
        })

    if callback_url:
        _fire_callback(callback_url, {
            "status":        "completed",
            "keyword":       keyword,
            "places_scraped": scraped,
            "queues_drained": drained,
        })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
