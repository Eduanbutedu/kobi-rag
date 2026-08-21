import { describe, expect, it, vi } from "vitest";
import { api, STREAM_FAILED_MESSAGE } from "./api";

/** A fetch that replies with the given SSE text, split at the given points. */
function sseFetch(parts, { ok = true, status = 200, body = null } = {}) {
  const encoder = new TextEncoder();
  const queue = parts.map((part) => encoder.encode(part));
  let i = 0;
  return vi.fn(async () => ({
    ok,
    status,
    json: async () => body ?? {},
    body: {
      getReader: () => ({
        read: async () =>
          i < queue.length ? { done: false, value: queue[i++] } : { done: true },
      }),
    },
  }));
}

const event = (name, data) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;

function collect() {
  const seen = { session: null, sources: null, text: "" };
  return {
    seen,
    handlers: {
      onSession: (id) => (seen.session = id),
      onSources: (rows) => (seen.sources = rows),
      onDelta: (piece) => (seen.text += piece),
    },
  };
}

describe("askStream", () => {
  it("delivers the session, the sources and the answer", async () => {
    globalThis.fetch = sseFetch([
      event("session", { session_id: 12 }),
      event("sources", [{ id: 3, text: "parça" }]),
      event("delta", "Cevabın "),
      event("delta", "devamı [1]"),
      "event: done\ndata: {}\n\n",
    ]);
    const { seen, handlers } = collect();

    await api.askStream("soru", handlers);

    expect(seen.session).toBe(12);
    expect(seen.sources).toEqual([{ id: 3, text: "parça" }]);
    expect(seen.text).toBe("Cevabın devamı [1]");
  });

  it("reassembles events split across network chunks", async () => {
    globalThis.fetch = sseFetch([
      'event: delta\ndata: "yar',
      'ım"\n\nevent: done\ndata: {}\n\n',
]);
    const { seen, handlers } = collect();

    await api.askStream("soru", handlers);

    expect(seen.text).toBe("yarım");
  });

  it("rejects with the message an error event carries", async () => {
    globalThis.fetch = sseFetch([
      event("session", { session_id: 1 }),
      event("sources", []),
      event("error", { message: "Dil modeline ulaşılamıyor." }),
      "event: done\ndata: {}\n\n",
    ]);
    const { handlers } = collect();

    await expect(api.askStream("soru", handlers)).rejects.toThrow(
      "Dil modeline ulaşılamıyor."
    );
  });

  it("keeps whatever arrived before the error", async () => {
    globalThis.fetch = sseFetch([
      event("delta", "Yarım cevap"),
      event("error", { message: "kesildi" }),
      "event: done\ndata: {}\n\n",
    ]);
    const { seen, handlers } = collect();

    await expect(api.askStream("soru", handlers)).rejects.toThrow("kesildi");
    expect(seen.text).toBe("Yarım cevap");
  });

  it("treats a stream that ends without a single delta as a failure", async () => {
    // Model yüklü değilken akış açılıyor ama hiç token gelmiyordu; arayüz
    // bu durumda "Düşünüyor…" imlecinde takılı kalıyordu.
    globalThis.fetch = sseFetch([
      event("session", { session_id: 4 }),
      event("sources", []),
      "event: done\ndata: {}\n\n",
    ]);
    const { handlers } = collect();

    await expect(api.askStream("soru", handlers)).rejects.toThrow(STREAM_FAILED_MESSAGE);
  });

  it("surfaces the detail of a rejected request", async () => {
    globalThis.fetch = sseFetch([], { ok: false, status: 503, body: { detail: "model kapalı" } });

    await expect(api.askStream("soru", collect().handlers)).rejects.toThrow("model kapalı");
  });
});
