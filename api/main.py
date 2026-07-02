"""
FastAPI application — Math & Quant AI
Endpoints: /health, /query, /ingest
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from pathlib import Path

from rag.pipeline import query as rag_query
from ingestion.pdf_parser import parse_pdf
from ingestion.chunker import chunk_pages
from ingestion.embedder import embed_and_store
from config.settings import get_settings

settings = get_settings()

app = FastAPI(
    title="Math & Quant AI",
    description="RAG-powered quantitative finance and mathematics assistant",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Depends(api_key_header)):
    if key != settings.api_secret_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


# ── Request / Response schemas ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str

class IngestRequest(BaseModel):
    pdf_path: str

class IngestResponse(BaseModel):
    message: str
    chunks_stored: int
    source: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "Math & Quant AI", "version": "0.1.0"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest, _=Depends(verify_api_key)):
    result = rag_query(body.query, top_k=body.top_k)
    return QueryResponse(**result)


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(body: IngestRequest, _=Depends(verify_api_key)):
    path = Path(body.pdf_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    pages = parse_pdf(path)
    chunks = chunk_pages(pages)
    count = embed_and_store(chunks)

    return IngestResponse(
        message="Ingestion complete",
        chunks_stored=count,
        source=path.name,
    )
