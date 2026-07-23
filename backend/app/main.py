import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from app.deps import get_store
from rag.extraction import SUPPORTED_EXTENSIONS
from rag.service import ingest_file, retrieve

app = FastAPI(
    title="KOBİ RAG API",
    description="Local RAG-powered document Q&A assistant",
    version="0.1.0",
)


class SearchRequest(BaseModel):
    query: str
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

    try:
        chunk_count = ingest_file(get_store(), tmp_path.rename(tmp_path.with_name(file.filename)))
    finally:
        tmp_path.with_name(file.filename).unlink(missing_ok=True)

    return {"filename": file.filename, "chunks": chunk_count}


@app.post("/search")
def search(request: SearchRequest) -> dict:
    results = retrieve(get_store(), request.query, k=request.k)
    return {"query": request.query, "results": results}