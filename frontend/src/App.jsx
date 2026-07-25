import { useEffect, useRef, useState } from "react";
import { api } from "./api";

const SUGGESTED_QUESTIONS = [
  "Bu doküman ne hakkında?",
  "En önemli bulgular neler?",
  "Which model performed best on FD001?",
];

function SealLogo({ size = 36 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
      <circle cx="32" cy="32" r="29" fill="none" stroke="#C9A050" strokeWidth="3" />
      <circle
        cx="32"
        cy="32"
        r="22"
        fill="none"
        stroke="#EFE7D5"
        strokeWidth="1"
        strokeDasharray="3 4"
      />
      <rect x="24" y="22" width="16" height="20" rx="2" fill="none" stroke="#EFE7D5" strokeWidth="2" />
      <line x1="27.5" y1="28" x2="36.5" y2="28" stroke="#C9A050" strokeWidth="2" strokeLinecap="round" />
      <line x1="27.5" y1="32.5" x2="36.5" y2="32.5" stroke="#EFE7D5" strokeWidth="1.6" strokeLinecap="round" />
      <line x1="27.5" y1="37" x2="33" y2="37" stroke="#EFE7D5" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(null);
  const chatEndRef = useRef(null);

  async function refreshDocuments() {
    try {
      const data = await api.listDocuments();
      setDocuments(data.documents);
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    refreshDocuments();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking]);

  async function handleFileSelected(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      await api.uploadDocument(file);
      await refreshDocuments();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  async function handleDelete(source) {
    if (confirmDelete !== source) {
      setConfirmDelete(source);
      return;
    }
    setConfirmDelete(null);
    setError(null);
    try {
      await api.deleteDocument(source);
      await refreshDocuments();
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleAsk(preset) {
    const q = (preset ?? question).trim();
    if (!q || asking) return;
    setQuestion("");
    setMessages((prev) => [
      ...prev,
      { role: "user", content: q },
      { role: "assistant", content: "", sources: [], streaming: true },
    ]);
    setAsking(true);

    const updateLast = (updater) =>
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = updater(next[next.length - 1]);
        return next;
      });

    try {
      await api.askStream(q, {
        onSources: (sources) => updateLast((m) => ({ ...m, sources })),
        onDelta: (piece) =>
          updateLast((m) => ({ ...m, content: m.content + piece })),
      });
      updateLast((m) => ({ ...m, streaming: false }));
    } catch (err) {
      updateLast((m) => ({
        ...m,
        content: err.message,
        isError: true,
        streaming: false,
      }));
    } finally {
      setAsking(false);
    }
  }

  function handleClearChat() {
    if (asking) return;
    setMessages([]);
  }

  return (
    <div className="flex h-screen bg-ink-950 text-paper">
      {/* Sol panel — doküman yönetimi */}
      <aside className="w-80 shrink-0 border-r border-ink-800 bg-ink-900 p-4 flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <SealLogo size={38} />
          <div>
            <h1 className="font-display text-xl font-semibold tracking-tight">
              KOBİ RAG
            </h1>
            <p className="text-xs text-mist">Yerel doküman asistanı</p>
          </div>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.txt"
          className="hidden"
          onChange={handleFileSelected}
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
          className="rounded-lg border border-dashed border-ink-700 p-6 text-sm text-mist transition hover:border-brass-500 hover:text-brass-400 disabled:opacity-50"
        >
          {uploading ? "İşleniyor... (parçalanıyor ve vektörleniyor)" : "PDF / TXT yükle"}
        </button>

        {error && (
          <div className="rounded-lg bg-red-950/60 border border-red-900 px-3 py-2 text-xs text-red-300">
            {error}
          </div>
        )}

        <div className="flex-1 overflow-y-auto">
          {documents.length === 0 ? (
            <p className="text-sm text-faint">Henüz doküman yok</p>
          ) : (
            <ul className="flex flex-col gap-2">
              {documents.map((doc) => (
                <li
                  key={doc.source}
                  className="group flex items-center justify-between rounded-lg bg-ink-800/60 px-3 py-2"
                >
                  <div className="min-w-0">
                    <p className="truncate text-sm">{doc.source}</p>
                    <p className="text-xs text-faint">{doc.chunks} parça</p>
                  </div>
                  <button
                    onClick={() => handleDelete(doc.source)}
                    onMouseLeave={() =>
                      setConfirmDelete((c) => (c === doc.source ? null : c))
                    }
                    className={`ml-2 shrink-0 rounded px-2 py-1 text-xs transition ${
                      confirmDelete === doc.source
                        ? "bg-red-950 text-red-300 opacity-100"
                        : "text-faint opacity-0 group-hover:opacity-100 hover:bg-red-950 hover:text-red-300"
                    }`}
                  >
                    {confirmDelete === doc.source ? "Emin misin?" : "Sil"}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <p className="text-[11px] leading-relaxed text-faint">
          Dokümanlarınız bu makineden çıkmaz — arama ve cevaplama tamamen yerel
          çalışır.
        </p>
      </aside>

      {/* Sağ panel — sohbet */}
      <main className="flex flex-1 flex-col">
        {messages.length > 0 && (
          <div className="flex justify-end border-b border-ink-800 px-4 py-2">
            <button
              onClick={handleClearChat}
              disabled={asking}
              className="rounded px-2.5 py-1 text-xs text-faint transition hover:bg-ink-800 hover:text-mist disabled:opacity-50"
            >
              Yeni sohbet
            </button>
          </div>
        )}
        <div className="flex-1 overflow-y-auto p-6">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center gap-5">
              <SealLogo size={64} />
              <div className="text-center">
                <p className="font-display text-lg text-paper">
                  Dokümanlarınıza sorun
                </p>
                <p className="mt-1 text-sm text-faint">
                  Bir doküman yükleyin, cevap kaynaklarıyla birlikte gelsin.
                </p>
              </div>
              {documents.length > 0 && (
                <div className="flex flex-wrap justify-center gap-2">
                  {SUGGESTED_QUESTIONS.map((q) => (
                    <button
                      key={q}
                      onClick={() => handleAsk(q)}
                      className="rounded-full border border-ink-700 bg-ink-900 px-4 py-2 text-xs text-mist transition hover:border-brass-500 hover:text-brass-400"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="mx-auto flex max-w-3xl flex-col gap-4">
              {messages.map((msg, i) =>
                msg.role === "user" ? (
                  <div key={i} className="msg-enter self-end max-w-[80%]">
                    <div className="rounded-2xl rounded-br-sm bg-brass-500 px-4 py-2.5 text-sm font-medium text-ink-950">
                      {msg.content}
                    </div>
                  </div>
                ) : (
                  <div key={i} className="msg-enter self-start w-full max-w-[90%]">
                    <div
                      className={`rounded-2xl rounded-bl-sm px-4 py-2.5 text-sm whitespace-pre-wrap ${
                        msg.isError
                          ? "bg-red-950/60 border border-red-900 text-red-300"
                          : "bg-ink-800/80"
                      }`}
                    >
                      {msg.content || (msg.streaming ? "Düşünüyor..." : "")}
                      {msg.streaming && msg.content && (
                        <span className="caret" aria-hidden="true" />
                      )}
                    </div>
                    {msg.sources?.length > 0 && (
                      <div className="mt-2 flex flex-col gap-2">
                        <p className="text-xs font-medium text-faint">
                          Kaynaklar
                        </p>
                        {msg.sources.map((src, j) => (
                          <div
                            key={j}
                            className="flex items-start gap-3 rounded-lg border border-ink-800 bg-ink-900/60 px-3 py-2.5"
                          >
                            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-dashed border-brass-500/70 text-[10px] font-medium text-brass-400">
                              %{Math.round(src.score * 100)}
                            </span>
                            <div className="min-w-0">
                              <p className="truncate text-xs font-medium text-brass-400">
                                {src.source}
                              </p>
                              <p className="mt-1 line-clamp-3 text-xs text-faint">
                                {src.text}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )
              )}
              <div ref={chatEndRef} />
            </div>
          )}
        </div>
        <div className="border-t border-ink-800 p-4">
          <div className="mx-auto flex max-w-3xl gap-2">
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAsk()}
              disabled={asking}
              className="flex-1 rounded-lg bg-ink-900 border border-ink-800 px-4 py-2.5 text-sm outline-none transition focus:border-brass-500 disabled:opacity-50"
              placeholder="Dokümanlarınıza bir soru sorun..."
            />
            <button
              onClick={() => handleAsk()}
              disabled={asking || !question.trim()}
              className="rounded-lg bg-brass-500 px-4 py-2.5 text-sm font-medium text-ink-950 transition hover:bg-brass-400 disabled:opacity-50 disabled:hover:bg-brass-500"
            >
              {asking ? "..." : "Sor"}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}