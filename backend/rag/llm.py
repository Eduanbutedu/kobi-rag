"""Answer generation via a local LLM served by Foundry Local."""

from functools import lru_cache

from foundry_local_sdk import Configuration, FoundryLocalManager

MODEL_ALIAS = "qwen2.5-1.5b"

SYSTEM_PROMPT = """Sen bir kurumsal doküman asistanısın. Görevin, sana verilen \
doküman parçalarına dayanarak kullanıcının sorusunu cevaplamaktır.

Kurallar:
- SADECE verilen parçalardaki bilgileri kullan.
- Cevap parçalarda yoksa bunu açıkça söyle: "Bu bilgi yüklü dokümanlarda bulunmuyor."
- Asla bilgi uydurma veya tahmin etme.
- Kullanıcı hangi dilde sorduysa o dilde cevap ver.
- Kısa ve net cevap ver."""


@lru_cache(maxsize=1)
def _get_chat_client():
    """Initialize Foundry Local once, load the model, return its chat client."""
    FoundryLocalManager.initialize(Configuration(app_name="kobi-rag"))
    manager = FoundryLocalManager.instance
    manager.download_and_register_eps()
    model = manager.catalog.get_model(MODEL_ALIAS)
    model.download()
    model.load()
    return model.get_chat_client()


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Generate a grounded answer from retrieved chunks."""
    client = _get_chat_client()

    context = "\n\n".join(
        f"[Parça {i} — kaynak: {c['source']}]\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )
    user_message = f"Doküman parçaları:\n\n{context}\n\nSoru: {question}"

    response = client.complete_chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )
    return response.choices[0].message.content or ""