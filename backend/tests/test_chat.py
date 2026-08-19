import pytest

from rag.chat import ChatStore


@pytest.fixture
def chat(tmp_path):
    store = ChatStore(tmp_path / "chat.db")
    yield store
    store.close()


SOURCES = [
    {"id": 41, "text": "İşçi dava açabilir.", "source": "is-kanunu.pdf", "score": 3.2},
    {"id": 87, "text": "Bir ay içinde başvurulur.", "source": "is-kanunu.pdf", "score": 1.1},
]


# --- Oturumlar --------------------------------------------------------------


def test_a_new_session_starts_without_a_title(chat):
    session_id = chat.create_session()
    [session] = chat.list_sessions()
    assert session["id"] == session_id
    assert session["title"] == ""
    assert session["message_count"] == 0


def test_sessions_are_listed_most_recently_used_first(chat):
    first = chat.create_session()
    second = chat.create_session()
    chat.add_message(first, "user", "eski oturuma yeni mesaj")

    assert [s["id"] for s in chat.list_sessions()] == [first, second]


def test_a_title_can_be_set_later(chat):
    session_id = chat.create_session()
    chat.set_title(session_id, "Yıllık izin süreleri")
    assert chat.list_sessions()[0]["title"] == "Yıllık izin süreleri"


def test_naming_a_session_does_not_count_as_activity(chat):
    # Başlık arka planda yazılıyor; sıralamayı bozmamalı
    older = chat.create_session()
    newer = chat.create_session()
    chat.set_title(older, "Sonradan verilen başlık")
    assert [s["id"] for s in chat.list_sessions()] == [newer, older]


def test_session_exists(chat):
    session_id = chat.create_session()
    assert chat.session_exists(session_id) is True
    assert chat.session_exists(9999) is False


def test_an_empty_store_lists_nothing(chat):
    assert chat.list_sessions() == []


# --- Mesajlar ---------------------------------------------------------------


def test_messages_come_back_in_the_order_they_were_added(chat):
    session_id = chat.create_session()
    chat.add_message(session_id, "user", "birinci soru")
    chat.add_message(session_id, "assistant", "birinci cevap")
    chat.add_message(session_id, "user", "ikinci soru")

    messages = chat.get_messages(session_id)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert [m["content"] for m in messages] == ["birinci soru", "birinci cevap", "ikinci soru"]


def test_sources_survive_a_round_trip(chat):
    session_id = chat.create_session()
    chat.add_message(session_id, "assistant", "cevap", SOURCES)

    [message] = chat.get_messages(session_id)
    assert message["sources"] == SOURCES
    assert message["sources"][0]["source"] == "is-kanunu.pdf"


def test_a_message_without_sources_reads_back_as_an_empty_list(chat):
    session_id = chat.create_session()
    chat.add_message(session_id, "user", "soru")
    assert chat.get_messages(session_id)[0]["sources"] == []


def test_turkish_text_survives_a_round_trip(chat):
    session_id = chat.create_session()
    chat.add_message(session_id, "user", "İşçinin yıllık izni kaç gün?")
    assert chat.get_messages(session_id)[0]["content"] == "İşçinin yıllık izni kaç gün?"


def test_adding_a_message_marks_the_session_as_used(chat):
    session_id = chat.create_session()
    before = chat.list_sessions()[0]["updated_at"]
    chat.add_message(session_id, "user", "soru")
    assert chat.list_sessions()[0]["updated_at"] >= before


def test_message_count_tracks_the_session(chat):
    session_id = chat.create_session()
    assert chat.message_count(session_id) == 0
    chat.add_message(session_id, "user", "soru")
    chat.add_message(session_id, "assistant", "cevap")
    assert chat.message_count(session_id) == 2
    assert chat.list_sessions()[0]["message_count"] == 2


def test_messages_stay_in_their_own_session(chat):
    first = chat.create_session()
    second = chat.create_session()
    chat.add_message(first, "user", "birinciye ait")
    chat.add_message(second, "user", "ikinciye ait")

    assert [m["content"] for m in chat.get_messages(first)] == ["birinciye ait"]
    assert [m["content"] for m in chat.get_messages(second)] == ["ikinciye ait"]


def test_an_unknown_role_is_rejected(chat):
    session_id = chat.create_session()
    with pytest.raises(ValueError, match="unknown role"):
        chat.add_message(session_id, "system", "olmaz")


def test_writing_to_a_missing_session_is_rejected(chat):
    with pytest.raises(ValueError, match="no such session"):
        chat.add_message(4242, "user", "olmaz")


# --- Silme ------------------------------------------------------------------


def test_deleting_a_session_removes_its_messages(chat):
    session_id = chat.create_session()
    chat.add_message(session_id, "user", "soru")
    chat.add_message(session_id, "assistant", "cevap")

    assert chat.delete_session(session_id) is True
    assert chat.list_sessions() == []
    assert chat.get_messages(session_id) == []


def test_deleting_one_session_leaves_the_others_alone(chat):
    keep = chat.create_session()
    drop = chat.create_session()
    chat.add_message(keep, "user", "kalsın")
    chat.add_message(drop, "user", "gitsin")

    chat.delete_session(drop)

    assert [s["id"] for s in chat.list_sessions()] == [keep]
    assert [m["content"] for m in chat.get_messages(keep)] == ["kalsın"]


def test_deleting_a_session_that_is_not_there(chat):
    assert chat.delete_session(9999) is False


def test_cascade_survives_reopening_the_database(tmp_path):
    path = tmp_path / "chat.db"
    first = ChatStore(path)
    session_id = first.create_session()
    first.add_message(session_id, "user", "soru")
    first.close()

    # ON DELETE CASCADE bağlantı başına PRAGMA gerektiriyor; yeni bağlantıda da açık olmalı
    second = ChatStore(path)
    try:
        second.delete_session(session_id)
        assert second.get_messages(session_id) == []
        orphans = second.db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert orphans == 0
    finally:
        second.close()


def test_history_lives_beside_the_vector_store_without_touching_it(tmp_path):
    from rag.embedding import EMBEDDING_DIM
    from rag.store import DocumentStore

    path = tmp_path / "shared.db"
    docs = DocumentStore(path)
    docs.add_document("a.pdf", ["metin"], [[0.0] * EMBEDDING_DIM])

    chat = ChatStore(path)
    try:
        session_id = chat.create_session()
        chat.add_message(session_id, "user", "soru")
        # İki tablo grubu yan yana duruyor, biri diğerini bozmuyor
        assert len(docs.all_chunks()) == 1
        assert chat.message_count(session_id) == 1
    finally:
        chat.close()
        docs.close()
