const BASE = "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `İstek başarısız (${res.status})`);
  }
  return res.json();
}

export const api = {
  listDocuments: () => request("/documents"),
  uploadDocument: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/documents", { method: "POST", body: form });
  },
  deleteDocument: (source) =>
    request(`/documents/${encodeURIComponent(source)}`, { method: "DELETE" }),
  ask: (question, k = 3) =>
    request("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, k }),
    }),
};