import { describe, expect, it } from "vitest";

import { UNTITLED, relativeTime, sessionLabel, toChatMessages } from "./sessions";

describe("sessionLabel", () => {
  it("uses the generated title once it arrives", () => {
    expect(sessionLabel({ title: "İşe iade davası süresi" })).toBe("İşe iade davası süresi");
  });

  it("falls back while the title is still being written", () => {
    // Başlık arka planda üretiliyor; ilk cevaptan önce boş
    expect(sessionLabel({ title: "" })).toBe(UNTITLED);
    expect(sessionLabel({ title: "   " })).toBe(UNTITLED);
    expect(sessionLabel({})).toBe(UNTITLED);
    expect(sessionLabel(undefined)).toBe(UNTITLED);
  });
});

describe("relativeTime", () => {
  const now = Date.parse("2026-08-19T12:00:00.000000+00:00");
  const ago = (seconds) =>
    new Date(now - seconds * 1000).toISOString().replace("Z", "+00:00");

  it("reads as just now within the first minute", () => {
    expect(relativeTime(ago(5), now)).toBe("az önce");
    expect(relativeTime(ago(59), now)).toBe("az önce");
  });

  it("counts minutes, hours and days", () => {
    expect(relativeTime(ago(60), now)).toBe("1 dakika önce");
    expect(relativeTime(ago(45 * 60), now)).toBe("45 dakika önce");
    expect(relativeTime(ago(3 * 3600), now)).toBe("3 saat önce");
    expect(relativeTime(ago(26 * 3600), now)).toBe("1 gün önce");
    expect(relativeTime(ago(6 * 24 * 3600), now)).toBe("6 gün önce");
  });

  it("switches to a date once a week has passed", () => {
    const older = relativeTime(ago(30 * 24 * 3600), now);
    expect(older).not.toContain("önce");
    expect(older).toMatch(/2026/);
  });

  it("treats a clock-skewed future timestamp as just now", () => {
    expect(relativeTime(ago(-120), now)).toBe("az önce");
  });

  it("stays quiet on a missing or unparseable timestamp", () => {
    expect(relativeTime(undefined, now)).toBe("");
    expect(relativeTime("", now)).toBe("");
    expect(relativeTime("bir zamanlar", now)).toBe("");
  });

  it("reads the microsecond timestamps the backend writes", () => {
    expect(relativeTime("2026-08-19T09:00:00.123456+00:00", now)).toBe("3 saat önce");
  });
});

describe("toChatMessages", () => {
  it("restores a stored exchange, sources included", () => {
    const stored = [
      { role: "user", content: "soru", sources: [], created_at: "x" },
      {
        role: "assistant",
        content: "cevap [1]",
        sources: [{ id: 41, text: "metin", source: "is-kanunu.pdf", score: 3.2 }],
        created_at: "y",
      },
    ];

    expect(toChatMessages(stored)).toEqual([
      { role: "user", content: "soru", sources: [], streaming: false },
      {
        role: "assistant",
        content: "cevap [1]",
        sources: [{ id: 41, text: "metin", source: "is-kanunu.pdf", score: 3.2 }],
        streaming: false,
      },
    ]);
  });

  it("never leaves a restored message streaming", () => {
    // Aksi hâlde geçmiş yüklenince imleç yanıp sönmeye başlar
    const restored = toChatMessages([{ role: "assistant", content: "eski cevap" }]);
    expect(restored[0].streaming).toBe(false);
    expect(restored[0].sources).toEqual([]);
  });

  it("handles an empty or missing history", () => {
    expect(toChatMessages([])).toEqual([]);
    expect(toChatMessages()).toEqual([]);
  });
});
