/**
 * Presentation logic for the chat session list.
 *
 * Kept out of the component so it can be tested directly: a session's label
 * depends on whether the background title has landed yet, and its timestamp
 * has to read as Turkish rather than as an ISO string.
 */

export const UNTITLED = "Yeni sohbet";

const MINUTE = 60;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;
const WEEK = 7 * DAY;

/**
 * What to show for a session in the sidebar.
 *
 * The title is written by a background task after the first answer, so it is
 * empty for a moment on every new chat — and stays empty if that call failed.
 */
export function sessionLabel(session) {
  const title = session?.title?.trim();
  return title || UNTITLED;
}

/**
 * "3 saat önce" for an ISO timestamp.
 *
 * Anything unparseable returns an empty string: a missing timestamp should
 * leave the row quiet rather than print "Invalid Date".
 */
export function relativeTime(iso, now = Date.now()) {
  if (!iso) return "";
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return "";

  const seconds = Math.round((now - then) / 1000);
  // Saat farkı yüzünden gelecekte görünen bir damga "az önce" olsun
  if (seconds < MINUTE) return "az önce";
  if (seconds < HOUR) return `${Math.floor(seconds / MINUTE)} dakika önce`;
  if (seconds < DAY) return `${Math.floor(seconds / HOUR)} saat önce`;
  if (seconds < WEEK) return `${Math.floor(seconds / DAY)} gün önce`;

  return new Date(then).toLocaleDateString("tr-TR", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

/** Turn stored messages into the shape the chat panel renders. */
export function toChatMessages(messages = []) {
  return messages.map((message) => ({
    role: message.role,
    content: message.content,
    sources: message.sources ?? [],
    streaming: false,
  }));
}
