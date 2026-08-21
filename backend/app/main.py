import json
import shutil
import tempfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.deps import get_chat, get_store
from rag.extraction import SUPPORTED_EXTENSIONS
from rag.llm import LLMUnavailableError, check_llm, generate_answer, generate_title, stream_answer
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
    session_id: int | None = None


NO_DOCUMENTS = "Henüz yüklenmiş doküman yok."


def _resolve_session(session_id: int | None) -> int:
    """Use the session given, or start one. Unknown ids are a client error."""
    chat = get_chat()
    if session_id is None:
        return chat.create_session()
    if not chat.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return session_id


def _record_exchange(session_id: int, question: str, answer: str, sources: list[dict]) -> bool:
    """Store one question and its answer. True if this was the session's first."""
    chat = get_chat()
    first = chat.message_count(session_id) == 0
    chat.add_message(session_id, "user", question)
    chat.add_message(session_id, "assistant", answer, sources)
    return first


def _name_session(session_id: int, question: str) -> None:
    """Title a session from its opening question, after the answer is sent.

    Runs as a background task: it is a second model call and nobody should wait
    for a sidebar label. On failure the title stays empty, which the interface
    already renders as "Yeni sohbet".
    """
    try:
        get_chat().set_title(session_id, generate_title(question))
    except Exception:  # noqa: BLE001 - başlık üretimi cevabı asla düşürmemeli
        pass


@app.get("/health")
def health_check() -> dict:
    """Report whether the model can answer, not merely that the API is up.

    The interface asks on startup so it can warn beforehand, instead of
    leaving the user in front of a caret that never moves.
    """
    return {"status": "ok", "llm": check_llm()}


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
def ask(request: AskRequest, background: BackgroundTasks) -> dict:
    session_id = _resolve_session(request.session_id)
    chunks = retrieve(get_store(), request.question, k=request.k)
    try:
        answer = generate_answer(request.question, chunks) if chunks else NO_DOCUMENTS
    except LLMUnavailableError as exc:
        # Yarım kalan bir turu geçmişe yazmak yerine hatayı açıkça bildir
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if _record_exchange(session_id, request.question, answer, chunks):
        background.add_task(_name_session, session_id, request.question)

    return {
        "question": request.question,
        "answer": answer,
        "sources": chunks,
        "session_id": session_id,
    }


@app.post("/ask/stream")
def ask_stream(request: AskRequest, background: BackgroundTasks) -> StreamingResponse:
    session_id = _resolve_session(request.session_id)
    chunks = retrieve(get_store(), request.question, k=request.k)

    def event_stream():
        failure = None
        meta = json.dumps({"session_id": session_id}, ensure_ascii=False)
        yield f"event: session\ndata: {meta}\n\n"
        yield f"event: sources\ndata: {json.dumps(chunks, ensure_ascii=False)}\n\n"

        if not chunks:
            answer = NO_DOCUMENTS
            yield f"event: delta\ndata: {json.dumps(answer, ensure_ascii=False)}\n\n"
        else:
            pieces = []
            try:
                for piece in stream_answer(request.question, chunks):
                    pieces.append(piece)
                    yield f"event: delta\ndata: {json.dumps(piece, ensure_ascii=False)}\n\n"
            except LLMUnavailableError as exc:
                # Akışın HTTP durumu çoktan 200; hata ancak bir olayla duyurulabilir
                detail = json.dumps({"message": str(exc)}, ensure_ascii=False)
                yield f"event: error\ndata: {detail}\n\n"
                failure = str(exc)
            answer = "".join(pieces)

        # Yarım kalan cevap saklanmaz; sadece tamamlanan turlar geçmişe girer
        if failure is None and _record_exchange(session_id, request.question, answer, chunks):
            background.add_task(_name_session, session_id, request.question)
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/documents")
def list_documents() -> dict:
    return {"documents": get_store().list_documents()}


@app.delete("/documents/{source}")
def delete_document(source: str) -> dict:
    deleted = get_store().delete_document(source)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document not found: {source}")
    return {"source": source, "deleted_chunks": deleted}


@app.post("/sessions")
def create_session() -> dict:
    session_id = get_chat().create_session()
    return {"session_id": session_id, "title": ""}


@app.get("/sessions")
def list_sessions() -> dict:
    return {"sessions": get_chat().list_sessions()}


@app.get("/sessions/{session_id}/messages")
def session_messages(session_id: int) -> dict:
    chat = get_chat()
    if not chat.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return {"session_id": session_id, "messages": chat.get_messages(session_id)}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: int) -> dict:
    if not get_chat().delete_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return {"session_id": session_id, "deleted": True}
