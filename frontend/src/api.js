const BASE = "http://127.0.0.1:8000";

// Yerel model çökerse ya da hiç yüklenmemişse akış tek bir parça göndermeden
// kapanabilir. Bunu sessiz bir bitiş değil, başarısızlık olarak sayıyoruz.
export const STREAM_FAILED_MESSAGE =
  "Model cevap üretemedi. Foundry servisinin çalıştığından emin olup yeniden deneyin.";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `İstek başarısız (${res.status})`);
  }
  return res.json();
}

export const api = {
  health: () => request("/health"),
  listDocuments: () => request("/documents"),
  uploadDocument: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/documents", { method: "POST", body: form });
  },
  deleteDocument: (source) =>
    request(`/documents/${encodeURIComponent(source)}`, { method: "DELETE" }),
  ask: (question, k = 3, sessionId = null) =>
    request("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, k, session_id: sessionId }),
    }),
  listSessions: () => request("/sessions"),
  createSession: () => request("/sessions", { method: "POST" }),
  sessionMessages: (id) => request(`/sessions/${id}/messages`),
  deleteSession: (id) => request(`/sessions/${id}`, { method: "DELETE" }),
  askStream: async (question, { onSession, onSources, onDelta }, k = 3, sessionId = null) => {
    const res = await fetch(`${BASE}/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, k, session_id: sessionId }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `İstek başarısız (${res.status})`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let failure = null;
    let deltas = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop();
      for (const block of blocks) {
        let event = "message";
        let data = "";
        for (const line of block.split("\n")) {
          if (line.startsWith("event: ")) event = line.slice(7);
          else if (line.startsWith("data: ")) data += line.slice(6);
        }
        if (event === "session") onSession?.(JSON.parse(data).session_id);
        else if (event === "sources") onSources?.(JSON.parse(data));
        else if (event === "error") failure = JSON.parse(data).message || STREAM_FAILED_MESSAGE;
        else if (event === "delta") {
          deltas += 1;
          onDelta?.(JSON.parse(data));
        }
      }
    }
    // Akışın HTTP durumu 200 olduğu için hata ancak burada yüzeye çıkabilir
    if (failure) throw new Error(failure);
    if (deltas === 0) throw new Error(STREAM_FAILED_MESSAGE);
  },
};