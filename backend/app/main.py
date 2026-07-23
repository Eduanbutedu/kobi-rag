from fastapi import FastAPI

app = FastAPI(
    title="KOBİ RAG API",
    description="Local RAG-powered document Q&A assistant",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}