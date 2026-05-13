import os
import re
import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from anthropic import Anthropic
import chromadb

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

LLM_MODEL       = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
CHROMA_PATH     = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_HOST     = os.getenv("CHROMA_HOST", "")
CHROMA_PORT     = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "place_reviews_v2")
DEFAULT_TOP_K   = 6

SYSTEM_PROMPT = (
    "Sen bir mekan değerlendirme asistanısın. "
    "Yalnızca verilen Google Maps yorumlarına dayanarak Türkçe yanıtla. "
    "Yorumlarda bilgi yoksa tek cümleyle 'Bu konuda yorumlarda bilgi bulunamadı.' de. "
    "Yorumlarda bilgi varsa 2-3 cümleyle özetle. "
    "Asla düşünce sürecini yazma, doğrudan cevabı ver."
)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Singletons — startup'ta bir kez init edilir
# ---------------------------------------------------------------------------

embed_model: Optional[SentenceTransformer] = None
llm_client:  Optional[Anthropic]           = None
collection                                 = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, llm_client, collection
    log.info(f"Embedding modeli yükleniyor: {EMBED_MODEL} ({DEVICE})")
    embed_model  = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    llm_client   = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT) if CHROMA_HOST else chromadb.PersistentClient(path=CHROMA_PATH)
    collection   = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    log.info(f"ChromaDB koleksiyonu yüklendi: {collection.count()} yorum")
    yield


app = FastAPI(title="Place Q&A Mikroservisi", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    place_name: str
    question:   str
    top_k:      int = Field(default=DEFAULT_TOP_K, ge=1, le=20)


class AskSource(BaseModel):
    snippet_text: str
    score:        Optional[float] = None


class AskResponse(BaseModel):
    place_name:   str
    question:     str
    answer:       str
    sources_used: int
    sources:      list[AskSource] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_user_message(place_name: str, question: str, reviews: list) -> str:
    reviews_text = "\n".join(f"{i+1}. {r[:200]}" for i, r in enumerate(reviews))
    return f"Mekan: {place_name}\nSoru: {question}\n\nYorumlar:\n{reviews_text}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status":                "ok",
        "device":                DEVICE,
        "collection":            COLLECTION_NAME,
        "total_indexed_reviews": collection.count() if collection else 0,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    existing = collection.get(where={"place_name": req.place_name}, limit=1)
    if not existing["ids"]:
        raise HTTPException(
            status_code=404,
            detail=f"'{req.place_name}' için indexli yorum bulunamadı.",
        )

    # Soruyu embed et
    question_embedding = embed_model.encode(
        f"query: {req.question}",
        normalize_embeddings=True,
    ).tolist()

    # O mekana ait tüm yorumları al
    place_docs = collection.get(
        where={"place_name": req.place_name},
        include=["documents"],
    )
    all_docs = place_docs["documents"] or []
    n        = min(req.top_k, len(all_docs))

    reviews = []
    if n > 0:
        try:
            # ChromaDB HNSW bazen where-filtresiyle n_results > filtered_count hatası veriyor
            # safe_n: toplam koleksiyon büyüklüğünü geçmemeli
            safe_n  = min(n, max(1, collection.count()))
            results = collection.query(
                query_embeddings=[question_embedding],
                n_results=safe_n,
                where={"place_name": req.place_name},
                include=["documents", "distances"],
            )
            reviews = results["documents"][0] if results["documents"] else []
            distances = results.get("distances", [[]])[0] if results.get("distances") else []
        except Exception as e:
            log.warning(f"[/ask] ChromaDB query hatası, fallback'e geçildi: {e}")
            reviews = all_docs[:n]
            distances = []
    else:
        distances = []

    if not reviews:
        raise HTTPException(
            status_code=404,
            detail=f"'{req.place_name}' için yorum bulunamadı.",
        )

    response = llm_client.messages.create(
        model=LLM_MODEL,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(req.place_name, req.question, reviews)}],
        max_tokens=512,
        temperature=0.1,
    )
    raw = response.content[0].text or ""
    if "</think>" in raw:
        answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    else:
        answer = re.sub(r"<think>.*", "", raw, flags=re.DOTALL).strip()

    log.info(f"[/ask] '{req.place_name}' | '{req.question}' | {len(reviews)} kaynak")
    return AskResponse(
        place_name=req.place_name,
        question=req.question,
        answer=answer,
        sources_used=len(reviews),
        sources=[
            AskSource(
                snippet_text=review,
                score=(
                    float(1 - distances[index])
                    if index < len(distances) and distances[index] is not None
                    else None
                ),
            )
            for index, review in enumerate(reviews)
        ],
    )
