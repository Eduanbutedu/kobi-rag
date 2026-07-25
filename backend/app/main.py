import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.deps import get_store
from rag.extraction import SUPPORTED_EXTENSIONS
from rag.llm import generate_answer
from rag.service import ingest_file, retrieve

app = FastAPI(
    title="KOBİ RAG API",
    description="Local RAG-powered document Q&A assistant",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    k: int = 3


class AskRequest(BaseModel):
    question: str
    k: int = 3


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents")
async def upload_document(file: UploadFile) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    renamed_path = tmp_path.with_name(file.filename)
    try:
        tmp_path.rename(renamed_path)
        chunk_count = ingest_file(get_store(), renamed_path)
    finally:
        renamed_path.unlink(missing_ok=True)

    return {"filename": file.filename, "chunks": chunk_count}


@app.post("/search")
def search(request: SearchRequest) -> dict:
    results = retrieve(get_store(), request.query, k=request.k)
    return {"query": request.query, "results": results}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    chunks = retrieve(get_store(), request.question, k=request.k)
    if not chunks:
        return {
            "question": request.question,
            "answer": "Henüz yüklenmiş doküman yok.",
            "sources": [],
        }
    answer = generate_answer(request.question, chunks)
    return {"question": request.question, "answer": answer, "sources": chunks}

@app.get("/documents")
def list_documents() -> dict:
    return {"documents": get_store().list_documents()}


@app.delete("/documents/{source}")
def delete_document(source: str) -> dict:
    deleted = get_store().delete_document(source)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document not found: {source}")
    return {"source": source, "deleted_chunks": deleted}