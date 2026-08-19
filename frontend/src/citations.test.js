import { describe, expect, it } from "vitest";

import { citedSourceNumbers, parseCitations } from "./citations";

const text = (parts) =>
  parts
    .filter((p) => p.type === "text")
    .map((p) => p.value)
    .join("");
const citations = (parts) => parts.filter((p) => p.type === "citation");

describe("parseCitations", () => {
  it("splits a marker out of the surrounding text", () => {
    const parts = parseCitations("Yıllık izin on dört gündür [1].", 3);
    expect(parts).toEqual([
      { type: "text", value: "Yıllık izin on dört gündür " },
      { type: "citation", value: 1, index: 0 },
      { type: "text", value: "." },
    ]);
  });

  it("maps a marker to a zero-based source index", () => {
    const [, marker] = parseCitations("Bilgi [3].", 3);
    expect(marker).toEqual({ type: "citation", value: 3, index: 2 });
  });

  it("handles several markers in one answer", () => {
    const parts = parseCitations("Önce bu [1]. Sonra şu [2].", 2);
    expect(citations(parts).map((c) => c.value)).toEqual([1, 2]);
  });

  it("handles markers written back to back", () => {
    const parts = parseCitations("İki kaynağa dayanır [1][3].", 3);
    expect(citations(parts).map((c) => c.value)).toEqual([1, 3]);
  });

  it("keeps the visible text intact", () => {
    const answer = "Bir [1] ve iki [2] son.";
    expect(text(parseCitations(answer, 2))).toBe("Bir  ve iki  son.");
  });

  it("returns plain text when the model cites nothing", () => {
    const parts = parseCitations("Hiç kaynak işareti yok.", 3);
    expect(parts).toEqual([{ type: "text", value: "Hiç kaynak işareti yok." }]);
  });

  // Küçük model bazen olmayan bir numara uyduruyor; kırık bağlantı yerine
  // yazdığı şey olduğu gibi görünsün
  it("leaves a number with no source behind it as plain text", () => {
    const parts = parseCitations("Uydurma kaynak [7].", 3);
    expect(citations(parts)).toEqual([]);
    expect(text(parts)).toBe("Uydurma kaynak [7].");
  });

  it("leaves markers alone when there are no sources at all", () => {
    expect(citations(parseCitations("Cevap [1].", 0))).toEqual([]);
    expect(citations(parseCitations("Cevap [1].", undefined))).toEqual([]);
  });

  it("rejects zero and keeps it readable", () => {
    const parts = parseCitations("Sıfır [0] olmaz.", 3);
    expect(citations(parts)).toEqual([]);
    expect(text(parts)).toBe("Sıfır [0] olmaz.");
  });

  it("merges text around a rejected marker instead of fragmenting it", () => {
    const parts = parseCitations("bir [9] iki", 2);
    expect(parts).toHaveLength(1);
    expect(parts[0].value).toBe("bir [9] iki");
  });

  it("ignores brackets that are not citations", () => {
    const answer = "Madde [a] ve dizi [1,2] ile [ 1 ] geçerli değil.";
    expect(citations(parseCitations(answer, 5))).toEqual([]);
  });

  it("handles an empty or missing answer", () => {
    expect(parseCitations("", 3)).toEqual([]);
    expect(parseCitations(undefined, 3)).toEqual([]);
  });

  it("handles a marker at the very start and very end", () => {
    expect(citations(parseCitations("[1] başta", 1)).length).toBe(1);
    expect(citations(parseCitations("sonda [1]", 1)).length).toBe(1);
  });

  it("works on a partially streamed answer", () => {
    // Akış sırasında metin yarım gelebilir; tamamlanmamış işaret metin kalır
    const parts = parseCitations("Yıllık izin [1]. Fazla mesai [", 2);
    expect(citations(parts).map((c) => c.value)).toEqual([1]);
    expect(text(parts).endsWith("[")).toBe(true);
  });
});

describe("citedSourceNumbers", () => {
  it("lists the distinct sources an answer cites, in order", () => {
    expect(citedSourceNumbers("Şu [3] ve bu [1] ve yine [3].", 3)).toEqual([1, 3]);
  });

  it("is empty when nothing valid is cited", () => {
    expect(citedSourceNumbers("Hiçbir şey", 3)).toEqual([]);
    expect(citedSourceNumbers("Uydurma [9]", 3)).toEqual([]);
  });
});
