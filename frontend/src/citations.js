/**
 * Splitting an answer into text and the [1] style citation markers in it.
 *
 * The model is asked to mark each claim with the number of the chunk it came
 * from, and those numbers match the order the sources are listed in. It is a
 * small model, so it sometimes omits markers or invents a number that has no
 * source behind it. Anything that does not point at a real source is left as
 * plain text rather than rendered as a broken link.
 */

const MARKER = /\[(\d{1,3})\]/g;

/**
 * Break `text` into parts for rendering.
 *
 * Returns a flat list of `{ type: "text", value }` and
 * `{ type: "citation", value, index }` where `index` is the zero-based
 * position of the source it points at. A marker outside 1..sourceCount is
 * returned as text, so an invented number simply reads as what the model
 * wrote.
 */
export function parseCitations(text, sourceCount = 0) {
  if (!text) return [];

  const parts = [];
  let cursor = 0;

  const pushText = (value) => {
    if (!value) return;
    const last = parts[parts.length - 1];
    // Bitişik metin parçalarını birleştir: geçersiz işaret düz metne düştüğünde
    // araya sahte bir sınır girmesin
    if (last && last.type === "text") last.value += value;
    else parts.push({ type: "text", value });
  };

  for (const match of text.matchAll(MARKER)) {
    const number = Number(match[1]);
    pushText(text.slice(cursor, match.index));

    if (number >= 1 && number <= sourceCount) {
      parts.push({ type: "citation", value: number, index: number - 1 });
    } else {
      pushText(match[0]);
    }
    cursor = match.index + match[0].length;
  }

  pushText(text.slice(cursor));
  return parts;
}

/** The distinct source numbers an answer actually cites, in ascending order. */
export function citedSourceNumbers(text, sourceCount = 0) {
  const numbers = parseCitations(text, sourceCount)
    .filter((part) => part.type === "citation")
    .map((part) => part.value);
  return [...new Set(numbers)].sort((a, b) => a - b);
}
