"""Answer generation via a local LLM served by Foundry Local."""

import re
import time
from collections.abc import Iterator
from functools import lru_cache

import httpx
import openai
from foundry_local import FoundryLocalManager

MODEL_ALIAS = "qwen3-4b"

# Servis ayakta ama model yüklü değilse bağlantı kuruluyor ve hiç token
# gelmiyor. Akışta okuma zaman aşımı iki token arasındaki sessizliği ölçer,
# yani ilk token için de geçerlidir; TOTAL_TIMEOUT üretim uzarsa devreye girer.
CONNECT_TIMEOUT = 10.0
FIRST_TOKEN_TIMEOUT = 20.0
TOTAL_TIMEOUT = 120.0
# Hazırlık yoklaması kullanıcıyı bekletmemeli
PROBE_TIMEOUT = 8.0

# Akış dışı çağrılarda cevabın tamamı tek okumada geliyor: orada okuma zaman
# aşımı token arası sessizliği değil, üretimin tamamını ölçer. Bu makinede
# üç parçalık bir soru 12-16 saniye sürüyor, yani 20 saniyelik ilk token
# bütçesi /ask'i olduğu gibi kesiyordu. Sınır toplam süre olmalı.
BLOCKING_TIMEOUT = httpx.Timeout(TOTAL_TIMEOUT, connect=CONNECT_TIMEOUT, read=TOTAL_TIMEOUT)
# Akışta ilk token bütçesi anlamını koruyor: sessizlik gerçekten sessizliktir
STREAM_TIMEOUT = httpx.Timeout(TOTAL_TIMEOUT, connect=CONNECT_TIMEOUT, read=FIRST_TOKEN_TIMEOUT)

LLM_UNAVAILABLE_MESSAGE = (
    "Dil modeline ulaşılamıyor. Foundry servisinin çalıştığından ve modelin "
    "yüklü olduğundan emin olun."
)


class LLMUnavailableError(RuntimeError):
    """The local model could not be reached, or did not answer in time."""


def _unavailable(exc: Exception) -> LLMUnavailableError:
    """One message for every way the model can fail to answer."""
    return LLMUnavailableError(LLM_UNAVAILABLE_MESSAGE)


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
    """Start/attach to the Foundry Local service; return an OpenAI client + model id.

    Raises LLMUnavailableError if the service cannot be started or reached, so a
    dead backend surfaces as a message instead of a hang.
    """
    try:
        manager = FoundryLocalManager(MODEL_ALIAS)
        client = openai.OpenAI(
            base_url=manager.endpoint,
            api_key=manager.api_key,
            timeout=BLOCKING_TIMEOUT,
            max_retries=0,
        )
        model_id = manager.get_model_info(MODEL_ALIAS).id
    except LLMUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001 - SDK türü ne olursa olsun sonuç aynı
        raise _unavailable(exc) from exc
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

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=_build_messages(question, chunks),
            temperature=0.2,
            max_tokens=300,
            timeout=BLOCKING_TIMEOUT,
        )
    except (openai.APITimeoutError, openai.APIConnectionError, openai.APIStatusError) as exc:
        raise _unavailable(exc) from exc
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

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=BLOCKING_TIMEOUT,
        )
    except (openai.APITimeoutError, openai.APIConnectionError, openai.APIStatusError) as exc:
        raise _unavailable(exc) from exc
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


def _iter_stream(stream, started: float) -> Iterator:
    """Iterate a streaming response, converting stalls into LLMUnavailableError.

    The read timeout on the client covers silence between tokens; this adds a
    wall-clock cap so a model that trickles forever still ends.
    """
    try:
        for chunk in stream:
            if time.monotonic() - started > TOTAL_TIMEOUT:
                raise LLMUnavailableError(
                    "Dil modeli yanıtı zaman aşımına uğradı. Lütfen yeniden deneyin."
                )
            yield chunk
    except LLMUnavailableError:
        raise
    except (openai.APITimeoutError, openai.APIConnectionError, openai.APIStatusError) as exc:
        raise _unavailable(exc) from exc


def check_llm() -> dict:
    """Ask the model for one token to see whether it can actually answer.

    Reaching the service is not enough: it answers on its port while the model
    is still unloaded, which is the state that used to hang the interface.
    """
    try:
        client, model_id = _get_client()
        client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "ping /no_think"}],
            max_tokens=1,
            timeout=PROBE_TIMEOUT,
        )
    except LLMUnavailableError as exc:
        return {"ready": False, "detail": str(exc)}
    except Exception:  # noqa: BLE001 - hazırlık yoklaması asla patlamamalı
        return {"ready": False, "detail": LLM_UNAVAILABLE_MESSAGE}
    return {"ready": True, "detail": ""}


def stream_answer(question: str, chunks: list[dict]) -> Iterator[str]:
    """Yield answer text incrementally as the model generates it."""
    client, model_id = _get_client()

    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=_build_messages(question, chunks),
            temperature=0.2,
            max_tokens=300,
            stream=True,
            timeout=STREAM_TIMEOUT,
        )
    except (openai.APITimeoutError, openai.APIConnectionError, openai.APIStatusError) as exc:
        raise _unavailable(exc) from exc

    raw = ""
    emitted = 0
    started = time.monotonic()
    for chunk in _iter_stream(stream, started):
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