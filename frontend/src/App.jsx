import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import { parseCitations } from "./citations";
import { relativeTime, sessionLabel, toChatMessages } from "./sessions";

const SUGGESTED_QUESTIONS = [
  "İşten çıkarılan işçi kaç gün içinde dava açabilir?",
  "KVKK'da açık rıza nedir?",
  "Limited şirket kurmak için en az kaç ortak gerekir?",
];

// Başlık cevaptan sonra arka planda üretiliyor; ilk tazeleme onu henüz
// yakalayamaz, bu yüzden kısa bir gecikmeyle bir kez daha bakılıyor
const TITLE_SETTLE_MS = 2000;

function AnswerText({ text, sourceCount, onCite }) {
  const parts = parseCitations(text, sourceCount);
  return (
    <>
      {parts.map((part, i) =>
        part.type === "citation" ? (
          <button
            key={i}
            type="button"
            onClick={() => onCite(part.index)}
            title={`Kaynak ${part.value}`}
            aria-label={`Kaynak ${part.value}'e git`}
            className="citation-mark"
          >
            {part.value}
          </button>
        ) : (
          <span key={i}>{part.value}</span>
        )
      )}
    </>
  );
}

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
  // Kaynak kartlarına "mesajIndeksi:kaynakIndeksi" anahtarıyla erişiliyor
  const sourceRefs = useRef({});
  const [highlighted, setHighlighted] = useState(null);

  const [sessions, setSessions] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [loadingSession, setLoadingSession] = useState(false);
  const [confirmDeleteSession, setConfirmDeleteSession] = useState(null);

  const focusSource = (messageIndex, sourceIndex) => {
    const key = `${messageIndex}:${sourceIndex}`;
    const card = sourceRefs.current[key];
    if (!card) return;
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    setHighlighted(key);
    window.setTimeout(() => setHighlighted((k) => (k === key ? null : k)), 1600);
  };

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
    refreshSessions();
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
      let activeSession = sessionId;
      await api.askStream(
        q,
        {
          // Oturum ilk soruda sunucuda açılıyor; id metinden önce geliyor
          onSession: (id) => {
            activeSession = id;
            setSessionId(id);
          },
          onSources: (sources) => updateLast((m) => ({ ...m, sources })),
          onDelta: (piece) =>
            updateLast((m) => ({ ...m, content: m.content + piece })),
        },
        3,
        sessionId
      );
      updateLast((m) => ({ ...m, streaming: false }));
      if (activeSession) {
        // İlk tazeleme mesaj sayısını ve sırayı günceller, ikincisi başlığı
        refreshSessions();
        window.setTimeout(refreshSessions, TITLE_SETTLE_MS);
      }
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

  async function refreshSessions() {
    try {
      const { sessions: rows } = await api.listSessions();
      setSessions(rows);
    } catch {
      // Geçmiş listesi çekilemezse sohbet çalışmaya devam etsin
    }
  }

  function handleNewChat() {
    if (asking) return;
    // Yeni oturum ilk soruda sunucuda açılıyor; burada yalnızca bağ koparılıyor
    setSessionId(null);
    setMessages([]);
    setConfirmDeleteSession(null);
  }

  async function handleOpenSession(id) {
    if (asking || id === sessionId) return;
    setLoadingSession(true);
    setConfirmDeleteSession(null);
    try {
      const { messages: stored } = await api.sessionMessages(id);
      setMessages(toChatMessages(stored));
      setSessionId(id);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingSession(false);
    }
  }

  async function handleDeleteSession(id) {
    if (confirmDeleteSession !== id) {
      setConfirmDeleteSession(id);
      return;
    }
    try {
      await api.deleteSession(id);
      if (id === sessionId) {
        setSessionId(null);
        setMessages([]);
      }
      await refreshSessions();
    } catch (err) {
      setError(err.message);
    } finally {
      setConfirmDeleteSession(null);
    }
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

        {/* Sohbet geçmişi */}
        <div className="flex min-h-0 flex-col gap-2">
          <div className="flex items-center justify-between">
            <p className="text-xs font-medium uppercase tracking-wide text-faint">
              Sohbetler
            </p>
            <button
              onClick={handleNewChat}
              disabled={asking}
              className="rounded px-2 py-0.5 text-xs text-brass-400 transition hover:bg-ink-800 disabled:opacity-50"
            >
              + Yeni
            </button>
          </div>

          {sessions.length === 0 ? (
            <p className="text-xs text-faint">Henüz sohbet yok.</p>
          ) : (
            <div className="flex max-h-56 flex-col gap-1 overflow-y-auto pr-1">
              {sessions.map((session) => (
                <div
                  key={session.id}
                  className={`group flex items-center gap-2 rounded-lg border px-2.5 py-1.5 transition ${
                    session.id === sessionId
                      ? "border-brass-500/70 bg-ink-800/80"
                      : "border-transparent hover:bg-ink-800/50"
                  }`}
                >
                  <button
                    onClick={() => handleOpenSession(session.id)}
                    disabled={asking || loadingSession}
                    className="min-w-0 flex-1 text-left disabled:opacity-50"
                  >
                    <p
                      className={`truncate text-xs ${
                        session.id === sessionId ? "text-brass-400" : "text-paper"
                      }`}
                    >
                      {sessionLabel(session)}
                    </p>
                    <p className="mt-0.5 text-[10px] text-faint">
                      {relativeTime(session.updated_at)}
                    </p>
                  </button>
                  <button
                    onClick={() => handleDeleteSession(session.id)}
                    className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] transition ${
                      confirmDeleteSession === session.id
                        ? "bg-red-950/70 text-red-300"
                        : "text-faint opacity-0 hover:text-red-300 group-hover:opacity-100"
                    }`}
                  >
                    {confirmDeleteSession === session.id ? "Emin misin?" : "Sil"}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="border-t border-ink-800" />

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
              onClick={handleNewChat}
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
                      {msg.content ? (
                        <AnswerText
                          text={msg.content}
                          sourceCount={msg.sources?.length ?? 0}
                          onCite={(sourceIndex) => focusSource(i, sourceIndex)}
                        />
                      ) : (
                        msg.streaming ? "Düşünüyor..." : ""
                      )}
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
                            ref={(el) => {
                              sourceRefs.current[`${i}:${j}`] = el;
                            }}
                            className={`flex items-start gap-3 rounded-lg border bg-ink-900/60 px-3 py-2.5 transition-colors duration-500 ${
                              highlighted === `${i}:${j}`
                                ? "source-flash border-brass-500"
                                : "border-ink-800"
                            }`}
                          >
                            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-dashed border-brass-500/70 text-xs font-medium text-brass-400">
                              {j + 1}
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