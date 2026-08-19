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

function IconChat({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M20 12a7.5 7.5 0 0 1-7.5 7.5H8l-4 3v-3.6A7.5 7.5 0 1 1 20 12Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function IconLibrary({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M5 4h5.5v16H5zM13 4h3.2l3 16H16z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <path d="M5 8.5h5.5M13.6 8.8h3.6" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}

function IconPlus({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

/** One button in the left rail. */
function RailButton({ label, active, onClick, disabled, children }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={label}
      aria-label={label}
      aria-pressed={active}
      className={`relative flex h-11 w-11 items-center justify-center rounded-xl transition-all duration-200 disabled:opacity-40 ${
        active
          ? "bg-ink-800 text-brass-400 shadow-[inset_0_1px_0_rgb(255_255_255/0.06)]"
          : "text-mist hover:bg-ink-800/60 hover:text-paper"
      }`}
    >
      {active && (
        <span className="absolute -left-3 h-5 w-[3px] rounded-r-full bg-brass-500" />
      )}
      {children}
    </button>
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
  // Sol panel iki görünüm arasında geçiş yapıyor: sohbetler ve dokümanlar
  const [panel, setPanel] = useState("chats");

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
      {/* Sol şerit — marka ve panel seçimi */}
      <nav className="z-20 flex w-[68px] shrink-0 flex-col items-center gap-6 border-r border-ink-800 bg-ink-950 py-5 shadow-[var(--shadow-rail)]">
        <SealLogo size={34} />

        <button
          onClick={handleNewChat}
          disabled={asking}
          title="Yeni sohbet"
          aria-label="Yeni sohbet"
          className="flex h-11 w-11 items-center justify-center rounded-xl border border-brass-500/40 text-brass-400 transition-all duration-200 hover:border-brass-500 hover:bg-brass-500/10 disabled:opacity-40"
        >
          <IconPlus />
        </button>

        <div className="flex flex-col items-center gap-2">
          <RailButton
            label="Sohbetler"
            active={panel === "chats"}
            onClick={() => setPanel("chats")}
          >
            <IconChat />
          </RailButton>
          <RailButton
            label="Dokümanlar"
            active={panel === "docs"}
            onClick={() => setPanel("docs")}
          >
            <IconLibrary />
          </RailButton>
        </div>

        <p className="mt-auto text-[9px] uppercase tracking-[0.2em] text-faint [writing-mode:vertical-rl]">
          yerel
        </p>
      </nav>

      {/* Orta panel — seçili görünümün içeriği */}
      <aside className="z-10 flex w-[19rem] shrink-0 flex-col border-r border-ink-800 bg-ink-900 shadow-[var(--shadow-panel)]">
        <header className="flex items-baseline justify-between border-b border-ink-800/80 px-5 py-5">
          <h2 className="panel-title text-[13px] text-paper">
            {panel === "chats" ? "Sohbetler" : "Dokümanlar"}
          </h2>
          <span className="text-[11px] text-faint">
            {panel === "chats" ? sessions.length : documents.length}
          </span>
        </header>

        {error && (
          <div className="mx-5 mt-4 rounded-lg border border-red-900/80 bg-red-950/50 px-3 py-2 text-xs text-red-300">
            {error}
          </div>
        )}

        {panel === "chats" ? (
          <div className="flex-1 overflow-y-auto px-3 py-4">
            {sessions.length === 0 ? (
              <p className="px-2 text-xs leading-relaxed text-faint">
                Henüz sohbet yok. Bir soru sorduğunuzda burada birikir.
              </p>
            ) : (
              <ul className="flex flex-col gap-1.5">
                {sessions.map((session) => (
                  <li key={session.id}>
                    <div
                      className={`group relative flex items-center gap-2 rounded-xl border px-3 py-2.5 transition-all duration-200 ${
                        session.id === sessionId
                          ? "border-brass-500/50 bg-ink-800 shadow-[inset_0_1px_0_rgb(255_255_255/0.05)]"
                          : "border-transparent hover:border-ink-700/70 hover:bg-ink-800/50"
                      }`}
                    >
                      {session.id === sessionId && (
                        <span className="absolute left-0 top-1/2 h-6 w-[3px] -translate-y-1/2 rounded-r-full bg-brass-500" />
                      )}
                      <button
                        onClick={() => handleOpenSession(session.id)}
                        disabled={asking || loadingSession}
                        className="min-w-0 flex-1 text-left disabled:opacity-50"
                      >
                        <p
                          className={`truncate text-[13px] font-medium leading-snug ${
                            session.id === sessionId ? "text-brass-400" : "text-paper"
                          }`}
                        >
                          {sessionLabel(session)}
                        </p>
                        <p className="mt-1 text-[11px] text-faint">
                          {relativeTime(session.updated_at)}
                        </p>
                      </button>
                      <button
                        onClick={() => handleDeleteSession(session.id)}
                        onMouseLeave={() =>
                          setConfirmDeleteSession((c) => (c === session.id ? null : c))
                        }
                        className={`shrink-0 rounded-md px-2 py-1 text-[10px] transition-all duration-200 ${
                          confirmDeleteSession === session.id
                            ? "bg-red-950 text-red-300 opacity-100"
                            : "text-faint opacity-0 hover:bg-red-950 hover:text-red-300 group-hover:opacity-100"
                        }`}
                      >
                        {confirmDeleteSession === session.id ? "Emin misin?" : "Sil"}
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        ) : (
          <div className="flex flex-1 flex-col overflow-hidden">
            <div className="px-5 pt-4">
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
                className="w-full rounded-xl border border-dashed border-ink-700 px-4 py-5 text-xs leading-relaxed text-mist transition-all duration-200 hover:border-brass-500 hover:bg-brass-500/5 hover:text-brass-400 disabled:opacity-50"
              >
                {uploading ? "İşleniyor..." : "PDF / TXT yükle"}
              </button>
            </div>

            <div className="flex-1 overflow-y-auto px-3 py-4">
              {documents.length === 0 ? (
                <p className="px-2 text-xs text-faint">Henüz doküman yok.</p>
              ) : (
                <ul className="flex flex-col gap-1.5">
                  {documents.map((doc) => (
                    <li
                      key={doc.source}
                      className="group flex items-center justify-between rounded-xl border border-transparent px-3 py-2.5 transition-all duration-200 hover:border-ink-700/70 hover:bg-ink-800/50"
                    >
                      <div className="min-w-0">
                        <p className="truncate text-[13px] leading-snug text-paper">
                          {doc.source}
                        </p>
                        <p className="mt-1 text-[11px] text-faint">{doc.chunks} parça</p>
                      </div>
                      <button
                        onClick={() => handleDelete(doc.source)}
                        onMouseLeave={() =>
                          setConfirmDelete((c) => (c === doc.source ? null : c))
                        }
                        className={`ml-2 shrink-0 rounded-md px-2 py-1 text-[10px] transition-all duration-200 ${
                          confirmDelete === doc.source
                            ? "bg-red-950 text-red-300 opacity-100"
                            : "text-faint opacity-0 hover:bg-red-950 hover:text-red-300 group-hover:opacity-100"
                        }`}
                      >
                        {confirmDelete === doc.source ? "Emin misin?" : "Sil"}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <p className="border-t border-ink-800/80 px-5 py-4 text-[11px] leading-relaxed text-faint">
              Dokümanlarınız bu makineden çıkmaz — arama ve cevaplama tamamen
              yerel çalışır.
            </p>
          </div>
        )}
      </aside>

      {/* Sağ panel — sohbet */}
      <main className="flex flex-1 flex-col bg-ink-950">
        {messages.length > 0 && (
          <header className="flex items-center justify-between gap-4 border-b border-ink-800 px-8 py-4">
            <div className="min-w-0">
              <h2 className="truncate font-display text-base font-semibold text-paper">
                {sessionLabel(sessions.find((s) => s.id === sessionId))}
              </h2>
              <p className="mt-0.5 text-[11px] text-faint">
                {documents.length} doküman üzerinde arama yapılıyor
              </p>
            </div>
            {loadingSession && (
              <span className="shrink-0 text-[11px] text-faint">Yükleniyor...</span>
            )}
          </header>
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