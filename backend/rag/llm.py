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
- CEVAP DİLİ KURALI (en önemli kural): Cevabını HER ZAMAN kullanıcının SORUSUNUN dilinde yaz. Doküman parçaları farklı dilde olsa bile soruyla aynı dilde cevap ver. Soru Türkçe ise cevap Türkçe olmak zorundadır.
- Kısa ve net cevap ver.
- Aynı cümleyi veya listeyi asla tekrarlama; cevabını bir kez ver ve bitir.
- KAYNAK GÖSTERME: Her bilgiyi hangi parçadan aldıysan, o parçanın numarasını \
cümlenin sonuna [1], [2] gibi köşeli parantez içinde yaz. Sadece sana verilen \
numaraları kullan, olmayan numara uydurma. Bir cümle birden çok parçaya \
dayanıyorsa [1][3] şeklinde arka arkaya yaz. İşaret bir cümlenin sonuna \
eklenir, tek başına cümle olmaz: önce bilgiyi yaz, sonra işareti koy.

Örnek:
Parça [1] "Yıllık izin süresi en az on dört gündür." ve parça [2] "İzin ücreti \
peşin ödenir." ise cevabın şöyle olur:
Yıllık izin en az on dört gündür [1]. İzin ücreti peşin ödenir [2]."""


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
    # Parçalar numaralandırılıyor: model cevabında [1], [2] diye atıf yapabilsin
    # ve numaralar arayüzdeki kaynak kartlarının sırasıyla birebir örtüşsün
    context = "\n\n---\n\n".join(
        f"[{number}] {chunk['text']}" for number, chunk in enumerate(chunks, start=1)
    )
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


def complete(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 200,
) -> str:
    """Single-turn completion against the local model, with thinking stripped.

    Used by offline tooling (such as eval question generation) that needs the
    model but not the RAG answer prompt.
    """
    client, model_id = _get_client()

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _strip_thinking(response.choices[0].message.content or "")


TITLE_SYSTEM_PROMPT = """Sana bir soru verilecek. Görevin, o soruyu bir sohbet \
listesinde temsil edecek çok kısa bir başlık yazmak.

Kurallar:
- En fazla 5 kelime.
- Soruyla aynı dilde yaz.
- Tırnak, nokta, soru işareti veya başka noktalama kullanma.
- Cevap verme, açıklama yapma; sadece başlığı yaz.

Örnek soru: işten çıkardığım işçi dava açmak isterse ne kadar süresi var
Örnek başlık: İşe iade davası süresi"""

TITLE_MAX_WORDS = 6
TITLE_MAX_CHARS = 60


def clean_title(raw: str, fallback: str = "") -> str:
    """Reduce a model reply to a usable session title.

    A small model tends to add quotes, a trailing full stop, or a "Başlık:"
    prefix. Anything left unusable falls back to the question itself, which is
    always better than an empty row in the sidebar.
    """
    line = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    for prefix in ("Başlık:", "BAŞLIK:", "Title:"):
        if line.startswith(prefix):
            line = line[len(prefix) :].strip()
    line = line.strip("\"'“”«»").strip()
    line = re.sub(r"[.!?;:,]+$", "", line).strip()
    line = " ".join(line.split()[:TITLE_MAX_WORDS])

    if not line:
        line = " ".join(fallback.split()[:TITLE_MAX_WORDS])
    return line[:TITLE_MAX_CHARS].strip()


def generate_title(question: str) -> str:
    """Name a chat session after its first question."""
    reply = complete(
        TITLE_SYSTEM_PROMPT,
        f"Soru: {question}\n\nBaşlık: /no_think",
        temperature=0.3,
        max_tokens=32,
    )
    return clean_title(reply, fallback=question)


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