import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import { parseCitations } from "./citations";
import {
  IconBook,
  IconChat,
  IconCheck,
  IconChevronLeft,
  IconCopy,
  IconLock,
  IconMoon,
  IconPlus,
  IconRetry,
  IconSearch,
  IconSun,
  IconUpload,
  Semruk,
} from "./icons";
import { matchesSearch, relativeTime, sessionLabel, toChatMessages } from "./sessions";

const SUGGESTED_QUESTIONS = [
  "İşten çıkarılan işçi kaç gün içinde dava açabilir?",
  "KVKK'da açık rıza nedir?",
  "Limited şirket kurmak için en az kaç ortak gerekir?",
];

// Başlık cevaptan sonra arka planda üretiliyor ve yerel model yavaş; tek bir
// gecikme yetmediği için oturum listesi birkaç kez yoklanıyor.
const TITLE_POLL_MS = 1500;
const TITLE_POLL_TRIES = 5;

/** The answer text with [n] markers turned into clickable superscripts. */
function AnswerText({ text, sourceCount, onCite }) {
  return parseCitations(text, sourceCount).map((part, i) =>
    part.type === "citation" ? (
      <button
        key={i}
        type="button"
        className="cite-mark"
        title={`Kaynak ${part.value}`}
        aria-label={`Kaynak ${part.value}'e git`}
        onClick={() => onCite(part.index)}
      >
        [{part.value}]
      </button>
    ) : (
      <span key={i}>{part.value}</span>
    )
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
  const sourceRefs = useRef({});
  const [flashed, setFlashed] = useState(null);

  const [sessions, setSessions] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [loadingSession, setLoadingSession] = useState(false);
  const [confirmDeleteSession, setConfirmDeleteSession] = useState(null);
  const [search, setSearch] = useState("");

  const [theme, setTheme] = useState("dark");
  const [sessionsOpen, setSessionsOpen] = useState(true);
  const [docsOpen, setDocsOpen] = useState(true);
  const [copied, setCopied] = useState(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const focusSource = (messageIndex, sourceIndex) => {
    const key = `${messageIndex}:${sourceIndex}`;
    const card = sourceRefs.current[key];
    if (!card) return;
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    setFlashed(key);
    window.setTimeout(() => setFlashed((k) => (k === key ? null : k)), 1400);
  };

  async function refreshDocuments() {
    try {
      setDocuments((await api.listDocuments()).documents);
    } catch (err) {
      setError(err.message);
    }
  }

  async function refreshSessions() {
    try {
      const { sessions: rows } = await api.listSessions();
      setSessions(rows);
      return rows;
    } catch {
      // Geçmiş çekilemezse sohbet çalışmaya devam etsin
      return [];
    }
  }

  /** Poll until the background-written title lands, then stop. */
  async function waitForTitle(id) {
    for (let attempt = 0; attempt < TITLE_POLL_TRIES; attempt += 1) {
      await new Promise((resolve) => window.setTimeout(resolve, TITLE_POLL_MS));
      const rows = await refreshSessions();
      if (rows.find((s) => s.id === id)?.title?.trim()) return;
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

  function handleNewChat() {
    if (asking) return;
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
          onSession: (id) => {
            activeSession = id;
            setSessionId(id);
          },
          onSources: (sources) => updateLast((m) => ({ ...m, sources })),
          onDelta: (piece) => updateLast((m) => ({ ...m, content: m.content + piece })),
        },
        3,
        sessionId
      );
      updateLast((m) => ({ ...m, streaming: false }));

      if (activeSession) {
        const rows = await refreshSessions();
        // Başlığı zaten dolu olan oturum için boşuna yoklama yapılmıyor
        if (!rows.find((s) => s.id === activeSession)?.title?.trim()) {
          waitForTitle(activeSession);
        }
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

  async function handleCopy(index, text) {
    try {
      await navigator.clipboard?.writeText(text);
      setCopied(index);
      window.setTimeout(() => setCopied((c) => (c === index ? null : c)), 1600);
    } catch {
      // Pano yoksa sessizce geç
    }
  }

  const visibleSessions = sessions.filter((s) => matchesSearch(sessionLabel(s), search));
  const activeTitle = sessionLabel(sessions.find((s) => s.id === sessionId));
  const totalChunks = documents.reduce((sum, d) => sum + d.chunks, 0);
  const emptyNote = { padding: "4px 2px", fontSize: 12, color: "var(--text-3)" };

  return (
    <>
      <div className="ambient" />

      <div className="app">
        {/* 1. İkon şeridi */}
        <nav className="rail">
          <div className="brand-mark" title="KOBİ RAG — Semrük">
            <Semruk />
          </div>

          <button
            className="rail-new"
            onClick={handleNewChat}
            disabled={asking}
            title="Yeni sohbet"
            aria-label="Yeni sohbet"
          >
            <IconPlus />
          </button>

          <button
            className={`rail-btn${sessionsOpen ? " active" : ""}`}
            onClick={() => setSessionsOpen((open) => !open)}
            title="Sohbetler"
            aria-label="Sohbetler"
            aria-pressed={sessionsOpen}
          >
            <IconChat />
          </button>

          <div className="rail-spacer" />

          <button
            className="rail-btn"
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
            title="Açık / koyu mod"
            aria-label="Açık / koyu mod"
          >
            {theme === "dark" ? <IconMoon /> : <IconSun />}
          </button>
        </nav>

        {/* 2. Sohbetler */}
        <aside className={`sessions${sessionsOpen ? "" : " collapsed"}`}>
          <div className="panel-head">
            <span className="panel-title">Sohbetler</span>
            <span className="panel-count">{sessions.length}</span>
          </div>

          <div className="search-wrap">
            <IconSearch />
            <input
              className="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Sohbetlerde ara…"
              aria-label="Sohbetlerde ara"
            />
          </div>

          {error && <div className="panel-error">{error}</div>}

          <div className="session-list">
            {visibleSessions.length === 0 ? (
              <p style={emptyNote}>
                {sessions.length === 0
                  ? "Henüz sohbet yok. Bir soru sorduğunuzda burada birikir."
                  : "Aramanızla eşleşen sohbet yok."}
              </p>
            ) : (
              visibleSessions.map((session) => (
                <div
                  key={session.id}
                  className={`session${session.id === sessionId ? " active" : ""}`}
                  role="button"
                  tabIndex={0}
                  onClick={() => handleOpenSession(session.id)}
                  onKeyDown={(e) => e.key === "Enter" && handleOpenSession(session.id)}
                >
                  <div className="session-name">{sessionLabel(session)}</div>
                  <div className="session-meta">
                    {session.message_count} mesaj
                    <span className="dot" />
                    {relativeTime(session.updated_at)}
                  </div>
                  <button
                    className={`session-del${
                      confirmDeleteSession === session.id ? " confirm" : ""
                    }`}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteSession(session.id);
                    }}
                    onMouseLeave={() =>
                      setConfirmDeleteSession((c) => (c === session.id ? null : c))
                    }
                  >
                    {confirmDeleteSession === session.id ? "Emin misin?" : "Sil"}
                  </button>
                </div>
              ))
            )}
          </div>

          <div className="panel-foot">
            <div className="local-note">
              <IconLock />
              <span>
                Dokümanlar bu makineden çıkmaz. Arama ve cevaplama tamamen yerel
                çalışır.
              </span>
            </div>
          </div>
        </aside>

        {/* Panel katlama düğmesi — panelin dışında, hep görünür */}
        <button
          className={`edge-toggle${sessionsOpen ? "" : " flipped"}`}
          onClick={() => setSessionsOpen((open) => !open)}
          title={sessionsOpen ? "Paneli daralt" : "Paneli genişlet"}
          aria-label={sessionsOpen ? "Paneli daralt" : "Paneli genişlet"}
        >
          <IconChevronLeft />
        </button>

        {/* 3. Sohbet */}
        <main className="chat">
          <header className="chat-head">
            <div style={{ minWidth: 0 }}>
              <div className="chat-title">
                {messages.length > 0 ? activeTitle : "KOBİ RAG"}
              </div>
              <div className="chat-sub">
                {loadingSession
                  ? "Yükleniyor…"
                  : `${documents.length} doküman · ${totalChunks.toLocaleString(
                      "tr-TR"
                    )} parça`}
              </div>
            </div>
            <div className="head-actions">
              <button
                className={`icon-btn${docsOpen ? " active" : ""}`}
                onClick={() => setDocsOpen((open) => !open)}
                title="Doküman panelini aç/kapat"
                aria-label="Doküman panelini aç/kapat"
                aria-pressed={docsOpen}
              >
                <IconBook />
              </button>
            </div>
          </header>

          <div className="thread">
            <div className="thread-inner">
              {messages.length === 0 ? (
                <div className="empty">
                  <div className="empty-mark">
                    <Semruk />
                  </div>
                  <div>
                    <div className="empty-title">Dokümanlarınıza sorun</div>
                    <div className="empty-sub">
                      Cevaplar, dayandıkları kaynaklarla birlikte gelir.
                    </div>
                  </div>
                  {documents.length > 0 && (
                    <div className="chips">
                      {SUGGESTED_QUESTIONS.map((q) => (
                        <button key={q} className="chip" onClick={() => handleAsk(q)}>
                          {q}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                messages.map((msg, i) =>
                  msg.role === "user" ? (
                    <div key={i} className="turn user">
                      <div className="avatar me">S</div>
                      <div className="bubble">{msg.content}</div>
                    </div>
                  ) : (
                    <div key={i} className="turn assistant">
                      <div className="avatar bot">
                        <Semruk />
                      </div>
                      <div className="answer-col">
                        <div className={`bubble${msg.isError ? " is-error" : ""}`}>
                          {msg.content ? (
                            <AnswerText
                              text={msg.content}
                              sourceCount={msg.sources?.length ?? 0}
                              onCite={(sourceIndex) => focusSource(i, sourceIndex)}
                            />
                          ) : (
                            msg.streaming && "Düşünüyor…"
                          )}
                          {msg.streaming && msg.content && (
                            <span className="caret" aria-hidden="true" />
                          )}
                        </div>

                        {!msg.streaming && !msg.isError && msg.content && (
                          <div className="msg-tools">
                            <button
                              className="tool"
                              onClick={() => handleCopy(i, msg.content)}
                            >
                              {copied === i ? <IconCheck /> : <IconCopy />}
                              {copied === i ? "Kopyalandı" : "Kopyala"}
                            </button>
                            <button
                              className="tool"
                              onClick={() => handleAsk(messages[i - 1]?.content)}
                              disabled={asking}
                            >
                              <IconRetry />
                              Yeniden sor
                            </button>
                          </div>
                        )}

                        {msg.sources?.length > 0 && (
                          <div className="sources">
                            <div className="sources-label">
                              {msg.sources.length} kaynak
                            </div>
                            {msg.sources.map((src, j) => (
                              <div
                                key={j}
                                ref={(el) => {
                                  sourceRefs.current[`${i}:${j}`] = el;
                                }}
                                className={`source${
                                  flashed === `${i}:${j}` ? " flash" : ""
                                }`}
                              >
                                <div className="source-num">{j + 1}</div>
                                <div style={{ minWidth: 0 }}>
                                  <div className="source-file">{src.source}</div>
                                  <div className="source-text">{src.text}</div>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  )
                )
              )}
              <div ref={chatEndRef} />
            </div>
          </div>

          <div className="composer">
            <div className="composer-inner">
              <div className="composer-row">
                <input
                  className="field"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleAsk()}
                  disabled={asking}
                  placeholder="Dokümanlarınıza bir soru sorun…"
                  aria-label="Soru"
                />
                <button
                  className="send"
                  onClick={() => handleAsk()}
                  disabled={asking || !question.trim()}
                >
                  {asking ? "…" : "Sor"}
                </button>
              </div>
              <div className="hint">
                <span className="key">Enter</span> ile gönderin ·{" "}
                <span className="key">Shift</span> + <span className="key">Enter</span> ile
                satır ekleyin
              </div>
            </div>
          </div>
        </main>

        {/* 4. Dokümanlar */}
        <aside className={`docs${docsOpen ? "" : " hidden"}`}>
          <div className="panel-head">
            <span className="panel-title">Dokümanlar</span>
            <span className="panel-count">{documents.length}</span>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt"
            style={{ display: "none" }}
            onChange={handleFileSelected}
          />
          <button
            className="upload"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
          >
            <IconUpload />
            {uploading ? "İşleniyor…" : "PDF veya TXT yükleyin"}
          </button>

          <div className="doc-list">
            {documents.length === 0 ? (
              <p style={emptyNote}>Henüz doküman yok.</p>
            ) : (
              documents.map((doc) => (
                <div key={doc.source} className="doc">
                  <div className="doc-name">{doc.source}</div>
                  <div className="doc-meta">
                    {doc.chunks.toLocaleString("tr-TR")} parça
                  </div>
                  <button
                    className={`doc-del${confirmDelete === doc.source ? " confirm" : ""}`}
                    onClick={() => handleDelete(doc.source)}
                    onMouseLeave={() =>
                      setConfirmDelete((c) => (c === doc.source ? null : c))
                    }
                  >
                    {confirmDelete === doc.source ? "Emin misin?" : "Sil"}
                  </button>
                </div>
              ))
            )}
          </div>
        </aside>
      </div>
    </>
  );
}
