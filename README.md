# KOBİ RAG — Yerel Doküman Soru-Cevap Asistanı

Dokümanlarınızı yükleyin, sorularınızı sorun — her şey **tamamen yerel** çalışır.
PDF/TXT dosyaları parçalanıp vektörlenir, sorular anlamsal aramayla eşleştirilir ve
cevaplar yerel bir LLM tarafından yalnızca doküman içeriğine dayanarak üretilir.
Hiçbir veri makinenizden çıkmaz.

![KOBİ RAG arayüzü](docs/screenshot.png)

## Özellikler

- **%100 yerel** — LLM (Qwen3-4B, Foundry Local üzerinden GPU'da), embedding modeli ve
  vektör veritabanı tamamen kendi makinenizde çalışır; internet bağlantısı ve API
  anahtarı gerekmez
- **Canlı akan cevaplar** — SSE (Server-Sent Events) ile cevap kelime kelime ekrana gelir
- **Kaynak gösterimi** — her cevabın altında hangi dokümanın hangi parçasından
  yararlanıldığı, benzerlik skorlarıyla birlikte listelenir
- **Halüsinasyon koruması** — cevap dokümanlarda yoksa model bunu açıkça söyler,
  bilgi uydurmaz
- **Akıllı parçalama** — cümle sınırına saygılı chunking ve akademik PDF'lerde
  kaynakça filtreleme ile daha isabetli arama sonuçları
- **Çok dilli** — çok dilli embedding modeli sayesinde Türkçe ve İngilizce
  dokümanlarla ve sorularla çalışır

## Mimari

```
Soru → embedding → sqlite-vec ile en yakın k parça
     → parçalar + soru yerel LLM'e (Foundry Local, Qwen3-4B)
     → cevap SSE ile tarayıcıya akar, kaynaklar skorlarıyla gösterilir
```

| Katman | Teknoloji |
|---|---|
| Frontend | React 19, Vite, Tailwind CSS v4 |
| API | FastAPI, SSE streaming |
| Metin çıkarma | PyMuPDF |
| Embedding | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) |
| Vektör arama | SQLite + sqlite-vec |
| LLM | Qwen3-4B — Foundry Local (OpenAI uyumlu API) |
| Test / Lint | pytest, ruff |

Depo yapısı:

```
kobi-rag/
├── backend/
│   ├── app/      # FastAPI: endpoint'ler, CORS, bağımlılıklar
│   ├── rag/      # extraction, chunking, embedding, store, retrieval, llm
│   └── tests/
├── frontend/     # React + Vite + Tailwind arayüzü
└── docs/
```

## Kurulum

Gereksinimler: Python 3.12+, Node 20+,
[Foundry Local](https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-local/)

**Backend:**

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows (Linux/macOS: source .venv/bin/activate)
pip install -e . --group dev
uvicorn app.main:app --reload
```

İlk çalıştırmada embedding modeli ve LLM otomatik indirilir; ilk soru, modelin
belleğe yüklenmesi nedeniyle biraz uzun sürebilir.

**Frontend:**

```bash
cd frontend
npm install
npm run dev
```

Tarayıcıda `http://localhost:5173` — bir PDF yükleyin ve sorun.

## Testler

```bash
cd backend
pytest          # 22 test
ruff check .
```

## API

| Endpoint | Açıklama |
|---|---|
| `POST /documents` | PDF/TXT yükle (parçala + vektörle) |
| `GET /documents` | Yüklü dokümanları listele |
| `DELETE /documents/{source}` | Dokümanı ve parçalarını sil |
| `POST /search` | Anlamsal arama (LLM'siz) |
| `POST /ask` | Soru sor — tam cevap tek seferde |
| `POST /ask/stream` | Soru sor — cevap SSE ile akar |

## Lisans

MIT