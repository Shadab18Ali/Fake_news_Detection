import { loadModel, predict } from "./classifier.js";
import { initFactCheck } from "./factcheck.js";

const SUPPORTED_FORMAT = 1;
const MODEL_TIMEOUT_MS = 30000;
const MIN_WORDS = 10;
const SHORT_TEXT_WORDS = 60;
const MAX_CHARS = 100000;
const TOP_WORDS = 8;
// Below this share of words the model knows, the text is probably off-topic or not English.
const LOW_COVERAGE = 0.3;
// Real-news probabilities between these bounds are reported as "uncertain".
const UNSURE_LOW = 0.35;
const UNSURE_HIGH = 0.65;

const SAMPLES = {
  sensational: `BREAKING: You won't BELIEVE what they are hiding from you! Insiders have finally exposed the shocking truth that the mainstream media refuses to report. Thousands of patriots are sharing this video before it gets deleted. The corrupt elites are absolutely FURIOUS that this secret is out, and they are scrambling to cover it up. Watch the footage for yourself and decide. Folks, this is the scandal of the century and nobody is talking about it. If you care about this country, share this everywhere right now before they silence us for good!`,
  wire: `BRUSSELS - European Union finance ministers agreed on Tuesday to extend a program of low-interest loans for small businesses by another year, according to a statement released after the meeting. The ministers said the program had helped firms in several member states maintain lending during a period of tighter credit conditions. Officials told reporters the extension would be funded from the existing budget and would not require new contributions from national governments. The European Commission is expected to publish detailed guidelines next month, a spokeswoman said. Some lawmakers in the European Parliament have called for stricter reporting requirements on how the funds are distributed.`,
};

const VERDICTS = {
  real: {
    label: "Likely real",
    text: "This article's writing patterns resemble the real-news examples the model was trained on.",
  },
  fake: {
    label: "Likely fake",
    text: "This article's writing patterns resemble the fake-news examples the model was trained on.",
  },
  unsure: {
    label: "Uncertain",
    text: "The writing patterns don't clearly resemble either the real or the fake examples.",
  },
  none: {
    label: "Can't analyze",
    text: "None of the words in this text are in the model's vocabulary. Paste a longer English-language news article.",
  },
};

const $ = (id) => document.getElementById(id);
const els = {
  status: $("model-status"),
  statusText: $("model-status-text"),
  article: $("article"),
  wordCount: $("word-count"),
  charCount: $("char-count"),
  message: $("input-message"),
  analyze: $("analyze"),
  paste: $("paste"),
  clear: $("clear"),
  shortcutMod: $("shortcut-mod"),
  announcer: $("announcer"),
  result: $("result"),
  resultTitle: $("result-title"),
  verdict: $("verdict"),
  verdictLabel: $("verdict-label"),
  verdictText: $("verdict-text"),
  confidence: $("confidence"),
  confidenceValue: $("confidence-value"),
  probabilities: $("probabilities"),
  fakeBar: $("fake-bar"),
  fakeValue: $("fake-value"),
  realBar: $("real-bar"),
  realValue: $("real-value"),
  notes: $("result-notes"),
  explain: $("explain"),
  influence: $("influence"),
  wordsFake: $("words-fake"),
  wordsReal: $("words-real"),
  stats: $("model-stats"),
};

let model = null;
const factCheck = initFactCheck(els.article);
let analyzedText = null;
let countsFrame = 0;

const numberFormat = new Intl.NumberFormat();
const plural = (n, word) => `${numberFormat.format(n)} ${word}${n === 1 ? "" : "s"}`;
const percent = (p) => `${Math.round(p * 100)}%`;

function countWords(text) {
  const words = text.match(/\S+/g);
  return words ? words.length : 0;
}

// ---------- Model status ----------

function setModelStatus(state, text) {
  els.status.dataset.state = state;
  els.statusText.textContent = text;
  els.analyze.disabled = state !== "ready";
}

async function fetchModel() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), MODEL_TIMEOUT_MS);
  try {
    const response = await fetch("model.json", { signal: controller.signal });
    if (!response.ok) throw new Error(`model.json: HTTP ${response.status}`);
    const json = await response.json();
    if (json?.format_version !== SUPPORTED_FORMAT) {
      throw new Error(`model.json: unsupported format_version ${json?.format_version}`);
    }
    return loadModel(json);
  } finally {
    clearTimeout(timer);
  }
}

async function init() {
  try {
    model = await fetchModel();
  } catch (err) {
    // Details are for the site owner; visitors get a plain-language message.
    console.error("Could not load the analysis model. If web/model.json is missing, run `python -m fake_news.export`.", err);
    setModelStatus("error", "Unable to load the analysis model. Please refresh and try again.");
    return;
  }
  setModelStatus("ready", "Model ready");
  renderStats(model);
}

function renderStats(m) {
  const items = [];
  if (Number.isFinite(m.metrics?.accuracy)) items.push(["Test accuracy", `${(m.metrics.accuracy * 100).toFixed(1)}%`]);
  if (Number.isFinite(m.metrics?.train_articles)) items.push(["Training articles", numberFormat.format(m.metrics.train_articles)]);
  items.push(["Vocabulary", plural(m.vocabulary.length, "word")]);
  const created = m.created ? new Date(m.created) : null;
  if (created && !Number.isNaN(created.getTime())) {
    items.push(["Model updated", created.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" })]);
  }
  els.stats.replaceChildren(...items.map(([term, value]) => {
    const div = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = term;
    dd.textContent = value;
    div.append(dt, dd);
    return div;
  }));
  els.stats.hidden = false;
}

// ---------- Input ----------

function updateCounts() {
  countsFrame = 0;
  const text = els.article.value;
  const chars = text.length;
  els.wordCount.textContent = plural(countWords(text), "word");
  els.charCount.textContent = `${numberFormat.format(chars)} / ${numberFormat.format(MAX_CHARS)} characters`;
  els.charCount.classList.toggle("over", chars > MAX_CHARS);
}

function scheduleCounts() {
  if (!countsFrame) countsFrame = requestAnimationFrame(updateCounts);
}

function setMessage(text, level = "error") {
  els.message.textContent = text;
  els.message.dataset.level = level;
  if (level === "error" && text) els.article.setAttribute("aria-invalid", "true");
  else els.article.removeAttribute("aria-invalid");
}

function validate(text) {
  const words = countWords(text);
  if (words === 0) return "Paste or type an article to analyze.";
  if (text.length > MAX_CHARS) {
    return `This text is ${numberFormat.format(text.length)} characters long. Paste a single article of up to ${numberFormat.format(MAX_CHARS)} characters.`;
  }
  if (words < MIN_WORDS) {
    return `This text is too short to analyze (${plural(words, "word")}). Paste at least ${MIN_WORDS} words, ideally the full article.`;
  }
  return null;
}

function onInput() {
  scheduleCounts();
  if (els.article.getAttribute("aria-invalid")) setMessage("");
  const stale = analyzedText !== null && els.article.value !== analyzedText && !els.result.hidden;
  els.result.classList.toggle("is-stale", stale);
  setNote("stale", stale ? "You've edited the text since this analysis. Analyze again to update the result." : null);
}

// ---------- Results ----------

function setNote(key, text, level = "warn") {
  let li = els.notes.querySelector(`[data-note="${key}"]`);
  if (!text) {
    li?.remove();
    return;
  }
  if (!li) {
    li = document.createElement("li");
    li.dataset.note = key;
    els.notes.append(li);
  }
  li.dataset.level = level;
  li.textContent = text;
}

function renderWords(list, items, maxAbs) {
  if (items.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No words in this article pushed the prediction this way.";
    list.replaceChildren(li);
    return;
  }
  list.replaceChildren(...items.map(({ term, count, contribution }) => {
    const li = document.createElement("li");
    const chip = document.createElement("span");
    chip.className = "word";
    chip.title = term;
    const text = document.createElement("span");
    text.className = "word-text";
    text.textContent = term;
    chip.append(text);
    if (count > 1) {
      const times = document.createElement("span");
      times.className = "word-count";
      times.textContent = `×${count}`;
      times.setAttribute("aria-label", `appears ${count} times`);
      chip.append(times);
    }
    const bar = document.createElement("span");
    bar.className = "word-bar";
    bar.setAttribute("aria-hidden", "true");
    bar.style.setProperty("--value", `${((Math.abs(contribution) / maxAbs) * 100).toFixed(1)}%`);
    li.append(chip, bar);
    return li;
  }));
}

function showVerdict(kind, probabilityReal) {
  const verdict = VERDICTS[kind];
  els.verdict.dataset.kind = kind;
  els.verdictLabel.textContent = verdict.label;
  els.verdictText.textContent = verdict.text;

  const hasScore = probabilityReal !== null;
  els.confidence.hidden = !hasScore;
  els.probabilities.hidden = !hasScore;
  els.explain.hidden = !hasScore;
  els.influence.hidden = !hasScore;
  if (!hasScore) return;

  // Round once so the two displayed values always add up to 100%.
  const realPct = Math.round(probabilityReal * 100);
  const fakePct = 100 - realPct;
  els.realValue.textContent = `${realPct}%`;
  els.fakeValue.textContent = `${fakePct}%`;
  els.realBar.style.setProperty("--value", `${realPct}%`);
  els.fakeBar.style.setProperty("--value", `${fakePct}%`);
  els.confidenceValue.textContent = `${Math.max(realPct, fakePct)}%`;
}

function render(text, result) {
  els.notes.replaceChildren();
  els.result.classList.remove("is-stale");
  els.result.hidden = false;

  if (!result) {
    showVerdict("none", null);
    els.announcer.textContent = `Analysis complete. ${VERDICTS.none.label}: ${VERDICTS.none.text}`;
    return;
  }

  const p = result.probabilityReal;
  const kind = p >= UNSURE_HIGH ? "real" : p <= UNSURE_LOW ? "fake" : "unsure";
  showVerdict(kind, p);

  const words = countWords(text);
  if (words < SHORT_TEXT_WORDS) {
    setNote("short", `This text is only ${plural(words, "word")} long. Results for fewer than ${SHORT_TEXT_WORDS} words are much less reliable, so paste the full article if you can.`);
  }
  if (result.tokenCount > 0 && result.knownTokenCount / result.tokenCount < LOW_COVERAGE) {
    setNote("coverage", "Most words in this text aren't in the model's vocabulary, so the result is less reliable. The model was trained on English-language political and world news.");
  }

  const fake = result.contributions.filter((c) => c.contribution < 0).slice(0, TOP_WORDS);
  const real = result.contributions.filter((c) => c.contribution > 0).slice(0, TOP_WORDS);
  const maxAbs = Math.max(1e-9, ...[...fake, ...real].map((c) => Math.abs(c.contribution)));
  renderWords(els.wordsFake, fake, maxAbs);
  renderWords(els.wordsReal, real, maxAbs);

  const realPct = Math.round(p * 100);
  els.announcer.textContent = `Analysis complete. ${VERDICTS[kind].label} based on writing patterns. Fake ${100 - realPct}%, real ${realPct}%.`;
}

function showAnalysisError() {
  els.notes.replaceChildren();
  els.result.hidden = false;
  els.result.classList.remove("is-stale");
  showVerdict("none", null);
  els.verdictLabel.textContent = "Analysis failed";
  els.verdictText.textContent = "Something went wrong while analyzing this text. Please try again, or refresh the page.";
  els.announcer.textContent = "Analysis failed. Please try again, or refresh the page.";
}

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function revealResult(moveFocus) {
  const rect = els.result.getBoundingClientRect();
  if (moveFocus) els.resultTitle.focus({ preventScroll: true });
  if (rect.top < 0 || rect.top > window.innerHeight * 0.6) {
    els.result.scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth", block: "start" });
  }
}

function analyze({ moveFocus = true } = {}) {
  if (!model) return;
  const text = els.article.value;
  const problem = validate(text);
  if (problem) {
    setMessage(problem, "error");
    els.article.focus();
    return;
  }
  setMessage("");

  let result;
  try {
    result = predict(model, text);
    if (result && !Number.isFinite(result.probabilityReal)) throw new Error("Non-finite probability");
  } catch (err) {
    console.error("Analysis failed", err);
    analyzedText = null;
    showAnalysisError();
    revealResult(moveFocus);
    return;
  }
  analyzedText = text;
  render(text, result);
  revealResult(moveFocus);
}

// ---------- Wiring ----------

els.article.addEventListener("input", onInput);
els.article.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    if (model) analyze({ moveFocus: false });
  }
});
els.analyze.addEventListener("click", () => analyze());

els.clear.addEventListener("click", () => {
  els.article.value = "";
  analyzedText = null;
  els.result.hidden = true;
  els.influence.hidden = true;
  els.notes.replaceChildren();
  setMessage("");
  updateCounts();
  factCheck.reset();
  els.announcer.textContent = "Cleared.";
  els.article.focus();
});

document.querySelectorAll("[data-sample]").forEach((button) => {
  button.addEventListener("click", () => {
    els.article.value = SAMPLES[button.dataset.sample];
    updateCounts();
    factCheck.syncSuggestion();
    setMessage("");
    if (model) analyze();
    else if (els.status.dataset.state === "error") setMessage("Example loaded, but the analysis model isn't available right now.", "info");
    else setMessage("Example loaded. You can analyze it as soon as the model finishes loading.", "info");
  });
});

if (navigator.clipboard?.readText) {
  els.paste.hidden = false;
  els.paste.addEventListener("click", async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (!text.trim()) {
        setMessage("Your clipboard is empty. Copy an article first.", "info");
        return;
      }
      els.article.value = text;
      updateCounts();
      onInput();
      factCheck.syncSuggestion();
      setMessage("");
      els.article.focus();
    } catch {
      setMessage(`Couldn't read the clipboard. Click in the text box and press ${els.shortcutMod.textContent}+V instead.`, "info");
    }
  });
}

const platform = navigator.userAgentData?.platform || navigator.platform || "";
if (/mac|iphone|ipad/i.test(platform)) els.shortcutMod.textContent = "⌘";

updateCounts();
init();
