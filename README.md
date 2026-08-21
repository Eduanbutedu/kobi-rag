# KOBİ RAG — Fully Local Document Q&A Assistant

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![Backend tests](https://img.shields.io/badge/backend%20tests-539%20passed-brightgreen)
![Frontend tests](https://img.shields.io/badge/frontend%20tests-37%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

Upload your documents and ask questions — everything runs **100% locally**.
PDF/TXT files are chunked and embedded, questions are answered by hybrid
retrieval with cross-encoder reranking, and a local LLM writes the answer
grounded strictly in your document content. No data ever leaves your machine.

Built for Turkish SMEs (KOBİ), works in any language your documents are in.

> **Screenshots pending.** `docs/screenshot.png` and `docs/demo.gif` still show
> the v1 interface and no longer match the current design. They will be
> recaptured; until then the interface is described under
> [Interface](#interface).

## Features

- **100% local & private** — the LLM (Qwen3-4B on GPU via Foundry Local), the
  embedding model, the reranker and the databases all run on your own machine;
  no internet connection or API keys required
- **Hybrid retrieval with reranking** — embedding search and BM25 keyword
  search are fused by weighted Reciprocal Rank Fusion, then a multilingual
  cross-encoder re-scores the shortlist. Measured on a 59-question golden set
  this took Hit@1 from 0.475 to 0.729 and MRR@10 from 0.628 to 0.816
- **Relevance threshold** — chunks the reranker scores below −2.5 are dropped,
  so a question your documents cannot answer (or small talk like "design me a
  logo") never reaches the LLM at all and comes back as a plain "not found"
  instead of an invented answer. The cut was chosen by measurement, not by feel
- **Inline citations** — the answer carries `[1]`, `[2]` markers; clicking one
  scrolls to the matching source card and highlights it, so every claim can be
  traced back to the chunk it came from
- **Chat history** — sessions persist in SQLite, appear in the sidebar with
  relative timestamps, and are titled automatically from their first question
  by a background model call. Deletion is two-stage, and searching filters
  titles with Turkish-correct case folding
- **Streaming answers** — responses arrive word by word over SSE
  (Server-Sent Events)
- **Resilient to a dead model** — every LLM call has a deadline, `/health`
  reports whether the model can actually answer rather than just that the API
  is up, and a stream that fails surfaces an error instead of leaving the
  interface waiting forever. A failed turn is not written to the history
- **Smart chunking** — sentence-boundary-aware splitting plus automatic
  references-section filtering for academic PDFs
- **Cross-lingual** — thanks to a multilingual embedding model, you can ask in
  Turkish about English documents (and vice versa) and get answers in the
  language of your question

## Retrieval quality

Three stages, each measured against the same 59-question golden set. Every run
is reproducible and kept under `backend/eval/results/`.

| | Hit@1 | Hit@3 | Hit@10 | MRR@10 | avg ms |
|---|---:|---:|---:|---:|---:|
| dense only | 0.475 | 0.746 | 0.932 | 0.628 | 40 |
| + hybrid (w=0.95) | 0.576 | 0.763 | **0.983** | 0.695 | 42 |
| **+ rerank (top-10)** | **0.729** | **0.864** | **0.983** | **0.816** | 275 |

Reranking is by far the largest single gain, at roughly seven times the
latency. Hybrid search on its own mostly moved results that were already in
the top ten into better positions.

The shipped configuration also applies the relevance threshold on top, which
trades Hit@10 0.983 → 0.966 for silence on questions the corpus cannot answer.

Every parameter here was chosen by measurement rather than by feel — the
keyword weight, the shortlist size, the threshold — and
[backend/eval/README.md](backend/eval/README.md) records the numbers, the
alternatives that were rejected and the trade-offs that were accepted.

## Corpus and evaluation set

The evaluation corpus is **9,258 chunks across 13 documents**:

| Group | Documents |
|---|---|
| Turkish legislation (9) | Türk Ticaret Kanunu, Sosyal Sigortalar Kanunu, Vergi Usul Kanunu, Borçlar Kanunu, Planlı Alanlar İmar Yönetmeliği, İş Kanunu, Tüketicinin Korunması, İSG Kanunu, KVKK |
| KVKK guidance (3) | Uygulama rehberi, örneklerle rehber, veri güvenliği rehberi |
| Academic PDF (1) | Turbofan research paper (for cross-lingual testing) |

The golden set is **59 hand-labelled questions** in
[`backend/eval/dataset.jsonl`](backend/eval/dataset.jsonl) — a mix of questions
phrased the way a user would ask them and questions drawn from document
wording. `backend/eval/` also holds the scripts that download the corpus,
ingest it, generate question drafts, annotate them by hand and run the
measurements.

Legislation texts are freely reproducible under Turkish copyright law (FSEK
art. 31); the KVKK guides are **not** redistributable, which is why
`backend/data/corpus/` is gitignored and fetched by script instead.

## Interface

A dark-first design in the style of LobeChat/NextChat: a four-column layout
(icon rail, session list, chat, documents), frosted-glass surfaces over ambient
light blobs, and a light/dark toggle. Purple marks anything the user does;
teal is reserved for system evidence — citations, source cards and uploads —
so the two never blur together. The mark is **Semrük**, a double-headed eagle.

## Architecture

```
Question ─┬→ embedding search (sqlite-vec)  ─┐
          └→ BM25 keyword search (FTS5)     ─┴→ weighted RRF → top 10
         → cross-encoder reranking → top-k chunks
         → relevance threshold (drops chunks scoring below -2.5)
         → chunks + question to the local LLM (Foundry Local, Qwen3-4B)
         → answer streamed to the browser over SSE, with clickable citations
```

Retrieval runs both searches for every question and merges them by Reciprocal
Rank Fusion, which combines rankings by position and so avoids comparing a
cosine similarity against a BM25 score. Keyword results carry slightly less
weight (0.95) than embedding results, so they add to the ranking rather than
overrule it. A multilingual cross-encoder then re-scores the shortlist, which
is where most of the accuracy comes from: it reads the question and the chunk
together instead of comparing two independently built vectors.

If nothing clears the relevance threshold the list is empty, the LLM is never
called, and the API answers that the documents do not cover the question.

| Layer | Technology |
|---|---|
| Frontend | React 19, Vite, Tailwind CSS v4 |
| API | FastAPI, SSE streaming |
| Text extraction | PyMuPDF |
| Embeddings | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) |
| Vector search | SQLite + sqlite-vec |
| Keyword search | SQLite FTS5 (BM25, Turkish stopword filtering) |
| Reranking | sentence-transformers CrossEncoder (mmarco-mMiniLMv2-L12-H384-v1) |
| LLM | Qwen3-4B — Foundry Local (OpenAI-compatible API) |
| Chat history | SQLite (sessions + messages) |
| Testing / Linting | pytest, ruff, vitest, eslint |

Repository layout:

```
kobi-rag/
├── backend/
│   ├── app/      # FastAPI: endpoints, CORS, dependencies
│   ├── rag/      # extraction, chunking, embedding, store, retrieval, llm, chat
│   ├── eval/     # corpus scripts, golden set, metrics, evaluation harness
│   └── tests/
├── frontend/     # React + Vite + Tailwind UI
└── docs/
```

## Getting started

Requirements: Python 3.12+ (developed on 3.14), Node 20+, Windows with
[Foundry Local](https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-local/)

**Backend:**

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows (Linux/macOS: source .venv/bin/activate)
pip install -e . --group dev
uvicorn app.main:app --reload
```

On first run the embedding model, the reranker and the LLM are downloaded
automatically; the first question takes a little longer while the models load
into memory. `GET /health` reports when the LLM is actually ready.

**Frontend:**

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`, upload a PDF and start asking.

## Tests

```bash
cd backend
pytest              # 539 tests
ruff check .

cd ../frontend
npm test            # 37 tests
npm run lint
```

Retrieval changes should also be measured, not just tested:

```bash
cd backend
python -m eval.run --label my-change
python -m eval.compare <baseline.json> <my-change.json>
```

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness plus whether the LLM can actually answer |
| `POST /documents` | Upload a PDF/TXT (chunk + embed) |
| `GET /documents` | List uploaded documents |
| `DELETE /documents/{source}` | Delete a document and its chunks |
| `POST /search` | Hybrid search + reranking (no LLM) |
| `POST /ask` | Ask a question — full answer at once |
| `POST /ask/stream` | Ask a question — answer streamed over SSE |
| `POST /sessions` | Start a chat session |
| `GET /sessions` | List sessions, newest first |
| `GET /sessions/{id}/messages` | Replay a session's history |
| `DELETE /sessions/{id}` | Delete a session and its messages |

`/ask` and `/ask/stream` accept an optional `session_id`; without one a new
session is opened and returned.

## Known gaps and planned work

- **CI is not set up.** There is no GitHub Actions workflow yet; `pytest`,
  `ruff`, `vitest` and `eslint` are run locally. A workflow running all four on
  push, plus a build-status badge, is planned
- **Screenshots are outdated** — `docs/screenshot.png` and `docs/demo.gif` show
  the v1 interface and need recapturing against the current design
- **Linux support.** The project has moved to Fedora for development, but
  `foundry-local-sdk` is Windows-only, so running the LLM still requires
  Windows. Swapping the LLM layer for Ollama would lift that; this has not been
  done yet and the code targets Foundry Local
- **The model still will not decline.** When retrieval returns topically
  related chunks that do not contain the answer, Qwen3-4B picks a nearby figure
  rather than saying the documents do not cover it. The relevance threshold
  catches the unrelated case, not this one — see the note beside
  `SYSTEM_PROMPT` in `backend/rag/llm.py` for what was tried

## License

MIT
