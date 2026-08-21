"""Session endpoints, with retrieval and the LLM stubbed out."""

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.deps import get_chat, get_store
from rag.chat import ChatStore

CHUNKS = [
    {"id": 41, "text": "İşçi dava açabilir.", "source": "is-kanunu.pdf", "score": 3.2},
    {"id": 87, "text": "Bir ay içinde başvurulur.", "source": "is-kanunu.pdf", "score": 1.1},
]


class FakeStore:
    """A store that only needs to say whether anything has been uploaded."""

    def __init__(self, documents=({"source": "is-kanunu.pdf", "chunks": 2},)):
        self.documents = list(documents)

    def list_documents(self):
        return self.documents


@pytest.fixture
def client(tmp_path, monkeypatch):
    chat = ChatStore(tmp_path / "api.db")
    monkeypatch.setattr(main, "get_chat", lambda: chat)
    monkeypatch.setattr(main, "retrieve", lambda store, question, k=3: list(CHUNKS))
    monkeypatch.setattr(main, "get_store", FakeStore)
    monkeypatch.setattr(main, "generate_answer", lambda q, c: f"{q} cevabı [1]")
    monkeypatch.setattr(main, "generate_title", lambda q: "Üretilmiş başlık")
    yield TestClient(main.app)
    chat.close()
    get_chat.cache_clear()
    get_store.cache_clear()


# --- Oturum yönetimi --------------------------------------------------------


def test_a_new_session_starts_empty(client):
    body = client.post("/sessions").json()
    assert body["title"] == ""
    assert isinstance(body["session_id"], int)


def test_sessions_are_listed_newest_first(client):
    first = client.post("/sessions").json()["session_id"]
    second = client.post("/sessions").json()["session_id"]
    listed = [s["id"] for s in client.get("/sessions").json()["sessions"]]
    assert listed == [second, first]


def test_messages_of_a_missing_session_are_a_404(client):
    assert client.get("/sessions/9999/messages").status_code == 404


def test_deleting_a_missing_session_is_a_404(client):
    assert client.delete("/sessions/9999").status_code == 404


# --- /ask ile bütünleşme ----------------------------------------------------


def test_asking_without_a_session_creates_one(client):
    assert client.get("/sessions").json()["sessions"] == []

    body = client.post("/ask", json={"question": "işçi dava açabilir mi"}).json()

    assert isinstance(body["session_id"], int)
    assert [s["id"] for s in client.get("/sessions").json()["sessions"]] == [body["session_id"]]


def test_asking_stores_the_question_and_the_answer_in_order(client):
    session_id = client.post("/ask", json={"question": "ilk soru"}).json()["session_id"]

    messages = client.get(f"/sessions/{session_id}/messages").json()["messages"]

    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "ilk soru"
    assert messages[1]["content"] == "ilk soru cevabı [1]"


def test_the_answer_keeps_its_sources(client):
    session_id = client.post("/ask", json={"question": "soru"}).json()["session_id"]
    messages = client.get(f"/sessions/{session_id}/messages").json()["messages"]

    assert messages[0]["sources"] == []
    assert messages[1]["sources"] == CHUNKS


def test_a_second_question_continues_the_same_session(client):
    first = client.post("/ask", json={"question": "birinci"}).json()["session_id"]
    second = client.post("/ask", json={"question": "ikinci", "session_id": first}).json()

    assert second["session_id"] == first
    messages = client.get(f"/sessions/{first}/messages").json()["messages"]
    assert [m["content"] for m in messages][::2] == ["birinci", "ikinci"]


def test_asking_against_an_unknown_session_is_a_404(client):
    response = client.post("/ask", json={"question": "soru", "session_id": 4242})
    assert response.status_code == 404


def test_an_explicitly_created_session_can_be_used(client):
    session_id = client.post("/sessions").json()["session_id"]
    body = client.post("/ask", json={"question": "soru", "session_id": session_id}).json()
    assert body["session_id"] == session_id


def test_the_first_exchange_names_the_session(client):
    session_id = client.post("/ask", json={"question": "işçi hakları"}).json()["session_id"]
    # BackgroundTasks TestClient'ta yanıt döndükten sonra çalışır
    [session] = client.get("/sessions").json()["sessions"]
    assert session["id"] == session_id
    assert session["title"] == "Üretilmiş başlık"


def test_a_later_question_does_not_rename_the_session(client, monkeypatch):
    session_id = client.post("/ask", json={"question": "ilk"}).json()["session_id"]
    monkeypatch.setattr(main, "generate_title", lambda q: "İKİNCİ BAŞLIK")
    client.post("/ask", json={"question": "ikinci", "session_id": session_id})

    assert client.get("/sessions").json()["sessions"][0]["title"] == "Üretilmiş başlık"


def test_a_failing_title_never_breaks_the_answer(client, monkeypatch):
    def _boom(question):
        raise RuntimeError("model down")

    monkeypatch.setattr(main, "generate_title", _boom)
    response = client.post("/ask", json={"question": "soru"})

    assert response.status_code == 200
    assert response.json()["answer"] == "soru cevabı [1]"
    assert client.get("/sessions").json()["sessions"][0]["title"] == ""


def test_asking_with_nothing_relevant_still_records_the_exchange(client, monkeypatch):
    monkeypatch.setattr(main, "retrieve", lambda store, question, k=3: [])
    body = client.post("/ask", json={"question": "soru"}).json()

    messages = client.get(f"/sessions/{body['session_id']}/messages").json()["messages"]
    assert messages[1]["content"] == main.NO_ANSWER
    assert messages[1]["sources"] == []


def test_an_empty_library_is_told_apart_from_an_unanswerable_question(client, monkeypatch):
    monkeypatch.setattr(main, "retrieve", lambda store, question, k=3: [])
    monkeypatch.setattr(main, "get_store", lambda: FakeStore([]))

    body = client.post("/ask", json={"question": "soru"}).json()

    assert body["answer"] == main.NO_DOCUMENTS


# --- Silme ------------------------------------------------------------------


def test_deleting_a_session_removes_its_messages(client):
    session_id = client.post("/ask", json={"question": "soru"}).json()["session_id"]

    assert client.delete(f"/sessions/{session_id}").json()["deleted"] is True
    assert client.get("/sessions").json()["sessions"] == []
    assert client.get(f"/sessions/{session_id}/messages").status_code == 404


def test_deleting_one_session_leaves_the_other(client):
    keep = client.post("/ask", json={"question": "kalsın"}).json()["session_id"]
    drop = client.post("/ask", json={"question": "gitsin"}).json()["session_id"]

    client.delete(f"/sessions/{drop}")

    assert [s["id"] for s in client.get("/sessions").json()["sessions"]] == [keep]
    assert len(client.get(f"/sessions/{keep}/messages").json()["messages"]) == 2


# --- Akış -------------------------------------------------------------------


def test_streaming_announces_its_session_and_records_the_answer(client, monkeypatch):
    monkeypatch.setattr(main, "stream_answer", lambda q, c: iter(["Cevap ", "parçası [1]"]))

    with client.stream("POST", "/ask/stream", json={"question": "akan soru"}) as response:
        body = "".join(response.iter_text())

    assert "event: session" in body
    assert "event: done" in body

    session_id = client.get("/sessions").json()["sessions"][0]["id"]
    messages = client.get(f"/sessions/{session_id}/messages").json()["messages"]
    assert messages[0]["content"] == "akan soru"
    # Akan parçalar tek cevapta birleştirilerek saklanıyor
    assert messages[1]["content"] == "Cevap parçası [1]"
    assert messages[1]["sources"] == CHUNKS
