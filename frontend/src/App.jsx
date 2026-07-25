import { useEffect, useRef, useState } from "react";
import { api } from "./api";

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
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
    setError(null);
    try {
      await api.deleteDocument(source);
      await refreshDocuments();
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleAsk() {
    const q = question.trim();
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

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100">
      {/* Sol panel — doküman yönetimi */}
      <aside className="w-80 shrink-0 border-r border-zinc-800 bg-zinc-900 p-4 flex flex-col gap-4">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">KOBİ RAG</h1>
          <p className="text-xs text-zinc-400">Yerel doküman asistanı</p>
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
          className="rounded-lg border border-dashed border-zinc-700 p-6 text-sm text-zinc-400 hover:border-emerald-600 hover:text-emerald-500 disabled:opacity-50"
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
            <p className="text-sm text-zinc-500">Henüz doküman yok</p>
          ) : (
            <ul className="flex flex-col gap-2">
              {documents.map((doc) => (
                <li
                  key={doc.source}
                  className="group flex items-center justify-between rounded-lg bg-zinc-800/60 px-3 py-2"
                >
                  <div className="min-w-0">
                    <p className="truncate text-sm">{doc.source}</p>
                    <p className="text-xs text-zinc-500">{doc.chunks} parça</p>
                  </div>
                  <button
                    onClick={() => handleDelete(doc.source)}
                    className="ml-2 rounded px-2 py-1 text-xs text-zinc-500 opacity-0 transition group-hover:opacity-100 hover:bg-red-950 hover:text-red-300"
                  >
                    Sil
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>

      {/* Sağ panel — sohbet */}
      <main className="flex flex-1 flex-col">
        <div className="flex-1 overflow-y-auto p-6">
          {messages.length === 0 ? (
            <p className="text-sm text-zinc-500">
              Bir doküman yükleyin ve soru sorun.
            </p>
          ) : (
            <div className="mx-auto flex max-w-3xl flex-col gap-4">
              {messages.map((msg, i) =>
                msg.role === "user" ? (
                  <div key={i} className="self-end max-w-[80%]">
                    <div className="rounded-2xl rounded-br-sm bg-emerald-700 px-4 py-2.5 text-sm">
                      {msg.content}
                    </div>
                  </div>
                ) : (
                  <div key={i} className="self-start w-full max-w-[90%]">
                    <div
                      className={`rounded-2xl rounded-bl-sm px-4 py-2.5 text-sm whitespace-pre-wrap ${
                        msg.isError
                          ? "bg-red-950/60 border border-red-900 text-red-300"
                          : "bg-zinc-800/80"
                      }`}
                    >
                      {msg.content || (msg.streaming ? "Düşünüyor..." : "")}
                    </div>
                    {msg.sources?.length > 0 && (
                      <div className="mt-2 flex flex-col gap-2">
                        <p className="text-xs font-medium text-zinc-500">
                          Kaynaklar
                        </p>
                        {msg.sources.map((src, j) => (
                          <div
                            key={j}
                            className="rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2"
                          >
                            <div className="flex items-center justify-between gap-2">
                              <p className="truncate text-xs font-medium text-emerald-500">
                                {src.source}
                              </p>
                              <span className="shrink-0 rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">
                                %{Math.round(src.score * 100)} benzerlik
                              </span>
                            </div>
                            <p className="mt-1 line-clamp-3 text-xs text-zinc-500">
                              {src.text}
                            </p>
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
        <div className="border-t border-zinc-800 p-4">
          <div className="mx-auto flex max-w-3xl gap-2">
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAsk()}
              disabled={asking}
              className="flex-1 rounded-lg bg-zinc-900 border border-zinc-800 px-4 py-2.5 text-sm outline-none focus:border-emerald-600 disabled:opacity-50"
              placeholder="Dokümanlarınıza bir soru sorun..."
            />
            <button
              onClick={handleAsk}
              disabled={asking || !question.trim()}
              className="rounded-lg bg-emerald-600 px-4 py-2.5 text-sm font-medium hover:bg-emerald-500 disabled:opacity-50 disabled:hover:bg-emerald-600"
            >
              {asking ? "..." : "Sor"}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}