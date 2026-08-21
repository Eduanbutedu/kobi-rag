"""What the API does when the local model cannot answer."""

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.deps import get_chat, get_store
from rag.chat import ChatStore
from rag.llm import LLM_UNAVAILABLE_MESSAGE, LLMUnavailableError

CHUNKS = [{"id": 7, "text": "Bir ay içinde başvurulur.", "source": "is-kanunu.pdf", "score": 2.0}]


def _down(*args, **kwargs):
    raise LLMUnavailableError(LLM_UNAVAILABLE_MESSAGE)


@pytest.fixture
def client(tmp_path, monkeypatch):
    chat = ChatStore(tmp_path / "api.db")
    monkeypatch.setattr(main, "get_chat", lambda: chat)
    monkeypatch.setattr(main, "get_store", lambda: None)
    monkeypatch.setattr(main, "retrieve", lambda store, question, k=3: list(CHUNKS))
    monkeypatch.setattr(main, "generate_title", lambda q: "Başlık")
    monkeypatch.setattr(main, "check_llm", lambda: {"ready": True, "detail": ""})
    yield TestClient(main.app)
    chat.close()
    get_chat.cache_clear()
    get_store.cache_clear()


def _events(body: str) -> list[str]:
    """The event names an SSE body carries, in order."""
    return [
        line.removeprefix("event: ")
        for line in body.splitlines()
        if line.startswith("event: ")
    ]


# --- /health ----------------------------------------------------------------


def test_health_reports_a_ready_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm"] == {"ready": True, "detail": ""}


def test_health_reports_an_unreachable_model(client, monkeypatch):
    monkeypatch.setattr(main, "check_llm", lambda: {"ready": False, "detail": "kapalı"})

    body = client.get("/health").json()

    # Servis ayakta olduğu için durum hâlâ ok; hazır olmayan yalnızca model
    assert body["status"] == "ok"
    assert body["llm"]["ready"] is False
    assert body["llm"]["detail"] == "kapalı"


# --- /ask -------------------------------------------------------------------


def test_ask_answers_with_503_when_the_model_is_down(client, monkeypatch):
    monkeypatch.setattr(main, "generate_answer", _down)

    response = client.post("/ask", json={"question": "soru"})

    assert response.status_code == 503
    assert response.json()["detail"] == LLM_UNAVAILABLE_MESSAGE


def test_a_failed_answer_is_not_written_to_the_history(client, monkeypatch):
    monkeypatch.setattr(main, "generate_answer", _down)
    session_id = client.post("/sessions").json()["session_id"]

    client.post("/ask", json={"question": "soru", "session_id": session_id})

    assert client.get(f"/sessions/{session_id}/messages").json()["messages"] == []


def test_a_failed_answer_leaves_the_session_untitled(client, monkeypatch):
    monkeypatch.setattr(main, "generate_answer", _down)
    session_id = client.post("/sessions").json()["session_id"]

    client.post("/ask", json={"question": "soru", "session_id": session_id})

    assert client.get("/sessions").json()["sessions"][0]["title"] == ""


# --- /ask/stream ------------------------------------------------------------


def test_a_stream_that_cannot_start_reports_an_error_then_ends(client, monkeypatch):
    monkeypatch.setattr(main, "stream_answer", _down)

    with client.stream("POST", "/ask/stream", json={"question": "soru"}) as response:
        body = "".join(response.iter_text())

    # Sessizce bitmiyor: önce hata, sonra kapanış olayı geliyor
    assert _events(body) == ["session", "sources", "error", "done"]
    assert LLM_UNAVAILABLE_MESSAGE in body


def test_a_stream_that_dies_midway_keeps_what_it_sent(client, monkeypatch):
    def half_answer(question, chunks):
        yield "Cevabın ilk yarısı"
        raise LLMUnavailableError(LLM_UNAVAILABLE_MESSAGE)

    monkeypatch.setattr(main, "stream_answer", half_answer)

    with client.stream("POST", "/ask/stream", json={"question": "soru"}) as response:
        body = "".join(response.iter_text())

    assert _events(body) == ["session", "sources", "delta", "error", "done"]
    assert "Cevabın ilk yarısı" in body


def test_a_broken_stream_is_not_written_to_the_history(client, monkeypatch):
    monkeypatch.setattr(main, "stream_answer", _down)
    session_id = client.post("/sessions").json()["session_id"]

    with client.stream(
        "POST", "/ask/stream", json={"question": "soru", "session_id": session_id}
    ) as response:
        "".join(response.iter_text())

    assert client.get(f"/sessions/{session_id}/messages").json()["messages"] == []


def test_a_working_stream_still_records_its_answer(client, monkeypatch):
    monkeypatch.setattr(main, "stream_answer", lambda q, c: iter(["Tam ", "cevap [1]"]))

    with client.stream("POST", "/ask/stream", json={"question": "soru"}) as response:
        body = "".join(response.iter_text())

    assert "event: error" not in body
    session_id = client.get("/sessions").json()["sessions"][0]["id"]
    messages = client.get(f"/sessions/{session_id}/messages").json()["messages"]
    assert messages[1]["content"] == "Tam cevap [1]"
