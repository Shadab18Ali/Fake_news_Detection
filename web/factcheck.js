// Fact-check search: sends a short query (never the full article) to this
// site's /api/factcheck function, which asks Google's Fact Check Tools API.

const MAX_QUERY_WORDS = 14;
const MIN_QUERY_CHARS = 3;
const MAX_QUERY_CHARS = 300;
const TIMEOUT_MS = 15000;

const FALSE_RATING = /\b(false|fake|pants on fire|incorrect|inaccurate|misleading|wrong|fabricated|hoax|scam|no evidence|unsupported|baseless|debunked|distorted|satire)\b/i;
const TRUE_RATING = /^(true|correct|accurate|mostly (true|correct|accurate))\b/i;

/** Suggest a search phrase from the article: its first sentence, minus datelines and "BREAKING:" style prefixes. */
export function suggestQuery(text) {
  const line = text.split(/\n+/u).map((s) => s.trim()).find((s) => /\p{L}/u.test(s)) || "";
  // First sentence: stop at . ! ? after a lowercase letter, digit or quote, so "U.S." doesn't end it.
  const first = line.match(/^.*?[\p{Ll}\d)"'\u201d][.!?](?=\s|$)/u)?.[0] || line;
  const cleaned = first
    .replace(/^[^\n]{0,120}?\((?:Reuters|REUTERS|AP|AFP)\)\s*[-–—]\s*/u, "")
    .replace(/^[A-Z][A-Z /,.'-]{1,40}\s[-–—]\s/u, "")
    .replace(/^(breaking|watch|just in|update|exclusive)\s*[:!-]\s*/iu, "")
    .replace(/[.!?]+$/u, "");
  return cleaned.split(/\s+/u).filter(Boolean).slice(0, MAX_QUERY_WORDS).join(" ");
}

export function ratingTone(rating) {
  if (!rating) return "neutral";
  if (TRUE_RATING.test(rating.trim())) return "real";
  if (FALSE_RATING.test(rating)) return "fake";
  return "neutral";
}

function formatDate(value) {
  const date = value ? new Date(value) : null;
  return date && !Number.isNaN(date.getTime())
    ? date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" })
    : null;
}

function renderResult(item) {
  const li = document.createElement("li");
  li.className = "fc-item";

  const rating = document.createElement("p");
  rating.className = "fc-rating";
  rating.dataset.tone = ratingTone(item.rating);
  rating.textContent = item.rating || "Rating not given";

  const claim = document.createElement("p");
  claim.className = "fc-claim";
  claim.textContent = `“${item.claim}”`;

  const meta = document.createElement("p");
  meta.className = "fc-meta";
  meta.textContent = [
    item.claimant ? `Claim by ${item.claimant}` : null,
    `Reviewed by ${item.publisher}`,
    formatDate(item.date),
  ].filter(Boolean).join(" · ");

  const link = document.createElement("a");
  link.className = "fc-link";
  link.href = item.url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = "Read the fact-check";
  const hidden = document.createElement("span");
  hidden.className = "visually-hidden";
  hidden.textContent = ` by ${item.publisher} (opens in a new tab)`;
  link.append(hidden);

  li.append(rating, claim, meta, link);
  return li;
}

export function initFactCheck(article) {
  const form = document.getElementById("fc-form");
  const input = document.getElementById("fc-query");
  const submit = document.getElementById("fc-submit");
  const message = document.getElementById("fc-message");
  const results = document.getElementById("fc-results");
  const note = document.getElementById("fc-note");
  let editedByUser = false;
  let pending = null;

  const setMessage = (text, level = "info") => {
    message.textContent = text;
    message.dataset.level = level;
  };

  // Keep the query in sync with the article until the visitor types their own.
  const syncSuggestion = () => {
    if (!editedByUser) input.value = suggestQuery(article.value);
  };
  article.addEventListener("input", syncSuggestion);
  input.addEventListener("input", () => {
    editedByUser = input.value.trim() !== "";
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const query = input.value.replace(/\s+/g, " ").trim();
    if (query.length < MIN_QUERY_CHARS) {
      setMessage("Enter a claim, headline or key phrase to search.", "error");
      input.focus();
      return;
    }
    if (query.length > MAX_QUERY_CHARS) {
      setMessage(`Shorten the search to ${MAX_QUERY_CHARS} characters or fewer.`, "error");
      input.focus();
      return;
    }

    pending?.abort();
    const controller = new AbortController();
    pending = controller;
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
    submit.disabled = true;
    results.setAttribute("aria-busy", "true");
    setMessage("Searching fact-checks…");

    try {
      const response = await fetch(`api/factcheck?query=${encodeURIComponent(query)}`, { signal: controller.signal });
      const data = await response.json().catch(() => ({}));
      if (response.status === 503 && data.error === "not_configured") {
        console.error("Fact-check search needs the GOOGLE_FACT_CHECK_API_KEY environment variable on Vercel.");
        throw Object.assign(new Error("not configured"), { userMessage: "Fact-check search isn't set up on this site yet." });
      }
      if (!response.ok || !Array.isArray(data.results)) throw new Error(`HTTP ${response.status}`);

      results.replaceChildren(...data.results.map(renderResult));
      note.hidden = data.results.length === 0;
      setMessage(data.results.length
        ? `Found ${data.results.length} published fact-check${data.results.length === 1 ? "" : "s"} for “${query}”.`
        : `No published fact-checks matched “${query}”. That doesn't mean the claim is true or false. Try a shorter phrase with the key names, places or numbers.`);
    } catch (err) {
      if (controller !== pending) return; // superseded by a newer search
      results.replaceChildren();
      note.hidden = true;
      setMessage(err.userMessage || "Fact-check search is unavailable right now. Please try again in a moment.", "error");
    } finally {
      clearTimeout(timer);
      if (controller === pending) {
        pending = null;
        submit.disabled = false;
        results.removeAttribute("aria-busy");
      }
    }
  });

  return { syncSuggestion, reset() { editedByUser = false; syncSuggestion(); results.replaceChildren(); note.hidden = true; setMessage(""); } };
}
