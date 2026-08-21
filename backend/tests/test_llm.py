"""Turning every way the local model can stall into one clear failure."""

import httpx
import openai
import pytest

from rag import llm

REQUEST = httpx.Request("POST", "http://127.0.0.1:5273/v1/chat/completions")


class FakeCompletions:
    """Stands in for client.chat.completions with a scripted outcome."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeClient:
    def __init__(self, outcome):
        self.completions = FakeCompletions(outcome)
        self.chat = self

    @property
    def calls(self):
        return self.completions.calls


def install(monkeypatch, outcome):
    """Replace the cached client with one that always produces `outcome`."""
    client = FakeClient(outcome)
    monkeypatch.setattr(llm, "_get_client", lambda: (client, "qwen3-4b"))
    return client


def reply(text):
    message = type("M", (), {"content": text})()
    choice = type("C", (), {"message": message})()
    return type("R", (), {"choices": [choice]})()


TIMEOUT = openai.APITimeoutError(request=REQUEST)
NO_CONNECTION = openai.APIConnectionError(message="connection refused", request=REQUEST)


# --- Tek seferlik çağrılar --------------------------------------------------


@pytest.mark.parametrize("failure", [TIMEOUT, NO_CONNECTION])
def test_a_stalled_answer_becomes_one_clear_error(monkeypatch, failure):
    install(monkeypatch, failure)

    with pytest.raises(llm.LLMUnavailableError) as caught:
        llm.generate_answer("soru", [{"text": "parça"}])

    assert str(caught.value) == llm.LLM_UNAVAILABLE_MESSAGE


def test_a_stalled_completion_becomes_one_clear_error(monkeypatch):
    install(monkeypatch, TIMEOUT)

    with pytest.raises(llm.LLMUnavailableError):
        llm.complete("sistem", "kullanıcı")


def test_a_working_answer_is_returned_as_usual(monkeypatch):
    install(monkeypatch, reply("Cevap [1]"))
    assert llm.generate_answer("soru", [{"text": "parça"}]) == "Cevap [1]"


def test_a_blocking_call_is_bounded_by_the_total_budget(monkeypatch):
    # Akış dışında cevabın tamamı tek okumada gelir; ilk token bütçesiyle
    # sınırlanırsa normal uzunlukta bir cevap bile zaman aşımına uğrar.
    client = install(monkeypatch, reply("Cevap"))

    llm.generate_answer("soru", [{"text": "parça"}])

    assert client.calls[0]["timeout"].read == llm.TOTAL_TIMEOUT


def test_a_streaming_call_keeps_the_first_token_budget(monkeypatch):
    client = install(monkeypatch, iter([delta("a")]))

    list(llm.stream_answer("soru", [{"text": "parça"}]))

    assert client.calls[0]["timeout"].read == llm.FIRST_TOKEN_TIMEOUT


# --- Akış -------------------------------------------------------------------


def test_a_stream_that_never_starts_becomes_one_clear_error(monkeypatch):
    install(monkeypatch, NO_CONNECTION)

    with pytest.raises(llm.LLMUnavailableError):
        list(llm.stream_answer("soru", [{"text": "parça"}]))


def test_a_stream_that_dies_midway_raises_after_what_it_sent(monkeypatch):
    def pieces():
        yield delta("Yarım ")
        raise TIMEOUT

    install(monkeypatch, pieces())

    stream = llm.stream_answer("soru", [{"text": "parça"}])
    # Görünür metin kırpılarak yayınlanıyor, sondaki boşluk düşüyor
    assert next(stream) == "Yarım"
    with pytest.raises(llm.LLMUnavailableError):
        next(stream)


def delta(text):
    inner = type("D", (), {"content": text})()
    choice = type("C", (), {"delta": inner})()
    return type("R", (), {"choices": [choice]})()


def test_a_stream_that_outlives_the_wall_clock_is_cut_off(monkeypatch):
    clock = iter([0.0, llm.TOTAL_TIMEOUT + 1])
    monkeypatch.setattr(llm.time, "monotonic", lambda: next(clock))

    with pytest.raises(llm.LLMUnavailableError):
        list(llm._iter_stream(iter([delta("a"), delta("b")]), started=0.0))


# --- Hazırlık yoklaması -----------------------------------------------------


def test_the_probe_asks_for_a_single_token(monkeypatch):
    client = install(monkeypatch, reply("ok"))

    assert llm.check_llm() == {"ready": True, "detail": ""}
    # Yoklama kullanıcıyı bekletmemeli: tek token ve kısa zaman aşımı
    assert client.calls[0]["max_tokens"] == 1
    assert client.calls[0]["timeout"] == llm.PROBE_TIMEOUT


def test_the_probe_reports_an_unreachable_service(monkeypatch):
    def boom():
        raise llm.LLMUnavailableError(llm.LLM_UNAVAILABLE_MESSAGE)

    monkeypatch.setattr(llm, "_get_client", boom)

    assert llm.check_llm() == {"ready": False, "detail": llm.LLM_UNAVAILABLE_MESSAGE}


def test_the_probe_never_raises(monkeypatch):
    install(monkeypatch, ValueError("beklenmedik"))

    result = llm.check_llm()

    assert result["ready"] is False
    assert result["detail"] == llm.LLM_UNAVAILABLE_MESSAGE


# --- Sistem promptu ---------------------------------------------------------
#
# Prompt'un metnini test etmek tuhaf görünebilir, ama bu kurallar ölçülerek
# yazıldı ve biri sessizce düşerse model yine sayı uydurmaya başlıyor.
# Testler kuralın varlığını korur; davranışın kendisi rag/llm.py'daki notta.


NO_ANSWER_SENTENCE = "Bu bilgi yüklü dokümanlarda bulunmuyor."


def test_the_prompt_offers_a_way_out_when_the_answer_is_missing():
    assert NO_ANSWER_SENTENCE in llm.SYSTEM_PROMPT


def test_the_prompt_ties_numbers_to_the_chunks():
    # "200 saat" uydurmasının önüne geçen kural
    assert "SAYILAR:" in llm.SYSTEM_PROMPT
    assert "başka sayıların bulunması" in llm.SYSTEM_PROMPT


def test_the_prompt_covers_a_partly_answerable_question():
    assert "EKSİK BİLGİ:" in llm.SYSTEM_PROMPT


def test_the_prompt_ties_a_citation_to_having_seen_the_chunk():
    assert "KAYNAK GÖSTERME:" in llm.SYSTEM_PROMPT
    assert "gerçekten gördüğün parça" in llm.SYSTEM_PROMPT


def test_the_prompt_keeps_the_answer_in_the_language_of_the_question():
    assert "CEVAP DİLİ:" in llm.SYSTEM_PROMPT


def test_the_examples_are_fenced_off_from_the_real_question():
    # Küçük model örnekleri cevabına kopyalıyordu; sınır açıkça çizili olmalı
    assert "yalnızca cevabın biçimini gösterir" in llm.SYSTEM_PROMPT
    assert "Örnekler bitti" in llm.SYSTEM_PROMPT


def test_one_example_answers_and_one_declines():
    _, examples = llm.SYSTEM_PROMPT.split("### Örnek 1", 1)
    found, missing = examples.split("### Örnek 2", 1)
    assert "[1]." in found and NO_ANSWER_SENTENCE not in found
    assert NO_ANSWER_SENTENCE in missing
