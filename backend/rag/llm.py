"""Answer generation via a local LLM served by Foundry Local."""

import re
from collections.abc import Iterator
from functools import lru_cache

import openai
from foundry_local import FoundryLocalManager

MODEL_ALIAS = "qwen3-4b"

SYSTEM_PROMPT = """Sen bir kurumsal doküman asistanısın. Görevin, sana verilen \
doküman parçalarına dayanarak kullanıcının sorusunu cevaplamaktır.

Kurallar:
- SADECE verilen parçalardaki bilgileri kullan.
- Cevap parçalarda yoksa bunu açıkça söyle: "Bu bilgi yüklü dokümanlarda bulunmuyor."
- Asla bilgi uydurma veya tahmin etme.
- Kullanıcı hangi dilde sorduysa o dilde cevap ver.
- Kısa ve net cevap ver.
- Aynı cümleyi veya listeyi asla tekrarlama; cevabını bir kez ver ve bitir."""


def _strip_thinking(text: str) -> str:
    """Remove Qwen3 <think> blocks; keep only the final answer."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def _dedupe_paragraphs(text: str) -> str:
    """Remove repeated lines a small model may emit."""
    seen: set[str] = set()
    result = []
    for block in text.split("\n"):
        key = block.strip().lower()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        result.append(block)
    return "\n".join(result).strip()


@lru_cache(maxsize=1)
def _get_client() -> tuple[openai.OpenAI, str]:
    """Start/attach to the Foundry Local service; return an OpenAI client + model id."""
    manager = FoundryLocalManager(MODEL_ALIAS)
    client = openai.OpenAI(base_url=manager.endpoint, api_key=manager.api_key)
    model_id = manager.get_model_info(MODEL_ALIAS).id
    return client, model_id


def _build_messages(question: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    user_message = f"Doküman parçaları:\n\n{context}\n\nSoru: {question} /no_think"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Generate a grounded answer from retrieved chunks."""
    client, model_id = _get_client()

    response = client.chat.completions.create(
        model=model_id,
        messages=_build_messages(question, chunks),
        temperature=0.2,
        max_tokens=300,
    )
    return _dedupe_paragraphs(_strip_thinking(response.choices[0].message.content or ""))


def stream_answer(question: str, chunks: list[dict]) -> Iterator[str]:
    """Yield answer text incrementally as the model generates it."""
    client, model_id = _get_client()

    stream = client.chat.completions.create(
        model=model_id,
        messages=_build_messages(question, chunks),
        temperature=0.2,
        max_tokens=300,
        stream=True,
    )

    raw = ""
    emitted = 0
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue
        raw += delta
        # Açık kalmış <think> bloğu varsa görünür kısım onun öncesidir;
        # kapanmışsa normal temizlik uygulanır.
        if "<think>" in raw and "</think>" not in raw:
            visible = raw.split("<think>")[0]
        else:
            visible = _strip_thinking(raw)
        if len(visible) > emitted:
            yield visible[emitted:]
            emitted = len(visible)