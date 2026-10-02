import { loadModel, predict } from "./classifier.js";

const SUPPORTED_FORMAT = 1;
const SHORT_TEXT_WORDS = 60;
const TOP_WORDS = 8;
// Probabilities between these bounds are reported as "uncertain".
const UNSURE_LOW = 0.35;
const UNSURE_HIGH = 0.65;

const SAMPLES = {
  sensational: `BREAKING: You won't BELIEVE what they are hiding from you! Insiders have finally exposed the shocking truth that the mainstream media refuses to report. Thousands of patriots are sharing this video before it gets deleted. The corrupt elites are absolutely FURIOUS that this secret is out, and they are scrambling to cover it up. Watch the footage for yourself and decide. Folks, this is the scandal of the century and nobody is talking about it. If you care about this country, share this everywhere right now before they silence us for good!`,
  wire: `BRUSSELS - European Union finance ministers agreed on Tuesday to extend a program of low-interest loans for small businesses by another year, according to a statement released after the meeting. The ministers said the program had helped firms in several member states maintain lending during a period of tighter credit conditions. Officials told reporters the extension would be funded from the existing budget and would not require new contributions from national governments. The European Commission is expected to publish detailed guidelines next month, a spokeswoman said. Some lawmakers in the European Parliament have called for stricter reporting requirements on how the funds are distributed.`,
};

const $ = (id) => document.getElementById(id);
const els = {
  article: $("article"),
  wordCount: $("word-count"),
  analyze: $("analyze"),
  analyzeLabel: document.querySelector("#analyze .btn-label"),
  clear: $("clear"),
  modelError: $("model-error"),
  result: $("result"),
  badge: $("verdict-badge"),
  icon: $("verdict-icon"),
  label: $("verdict-label"),
  sub: $("verdict-sub"),
  marker: $("meter-marker"),
  shortWarning: $("short-warning"),
  wordsFake: $("words-fake"),
  wordsReal: $("words-real"),
  stats: $("model-stats"),
};

let model = null;

function countWords(text) {
  const words = text.trim().match(/\S+/g);
  return words ? words.length : 0;
}

function updateWordCount() {
  const n = countWords(els.article.value);
  els.wordCount.textContent = `${n.toLocaleString()} word${n === 1 ? "" : "s"}`;
  els.analyze.disabled = !model || n === 0;
}

function showModelError(html) {
  els.modelError.innerHTML = html;
  els.modelError.hidden = false;
  els.analyzeLabel.textContent = "Model unavailable";
  els.analyze.disabled = true;
}

function renderStats(m) {
  const items = [];
  if (m.metrics?.accuracy != null) items.push(["Test accuracy", `${(m.metrics.accuracy * 100).toFixed(1)}%`]);
  if (m.metrics?.train_articles) items.push(["Training articles", m.metrics.train_articles.toLocaleString()]);
  items.push(["Vocabulary", `${m.vocabulary.length.toLocaleString()} words`]);
  if (m.created) {
    const date = new Date(m.created);
    if (!Number.isNaN(date.getTime())) {
      items.push(["Model updated", date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" })]);
    }
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
}

async function init() {
  try {
    const response = await fetch("model.json");
    if (response.status === 404) {
      showModelError(
        "The trained model hasn't been published yet. Download the dataset into <code>data/</code>, run " +
        "<code>python -m fake_news.export</code>, then commit <code>web/model.json</code> and redeploy.",
      );
      return;
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const json = await response.json();
    if (json.format_version !== SUPPORTED_FORMAT) {
      showModelError("<code>model.json</code> was exported by a different version of this project. Re-run <code>python -m fake_news.export</code>.");
      return;
    }
    model = loadModel(json);
    renderStats(model);
    els.analyzeLabel.textContent = "Analyze article";
    updateWordCount();
  } catch (err) {
    console.error(err);
    showModelError("Couldn't load the model. Check your connection and reload the page.");
  }
}

function renderWords(list, items, maxAbs, kind) {
  if (items.length === 0) {
    const li = document.createElement("li");
    li.innerHTML = '<span class="empty">None</span>';
    list.replaceChildren(li);
    return;
  }
  list.className = `word-list ${kind}`;
  list.replaceChildren(...items.map(({ term, count, contribution }) => {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.className = "term";
    name.textContent = term;
    name.title = `"${term}" appears ${count}×`;
    if (count > 1) {
      const small = document.createElement("small");
      small.textContent = ` ×${count}`;
      name.append(small);
    }
    const bar = document.createElement("span");
    bar.className = "bar";
    bar.style.width = `${Math.max(3, (Math.abs(contribution) / maxAbs) * 100)}%`;
    li.append(name, bar);
    return li;
  }));
}

function analyze() {
  if (!model) return;
  const text = els.article.value;
  const result = predict(model, text);
  els.result.hidden = false;

  if (!result) {
    els.badge.className = "verdict-badge is-unsure";
    els.icon.textContent = "?";
    els.label.textContent = "Not enough to go on";
    els.sub.textContent = "None of these words are in the model's vocabulary. Paste a longer English news article.";
    els.marker.style.left = "50%";
    els.shortWarning.hidden = true;
    renderWords(els.wordsFake, [], 1, "fake");
    renderWords(els.wordsReal, [], 1, "real");
    return;
  }

  const p = result.probabilityReal;
  let kind;
  if (p >= UNSURE_HIGH) kind = "real";
  else if (p <= UNSURE_LOW) kind = "fake";
  else kind = "unsure";

  els.badge.className = `verdict-badge is-${kind}`;
  els.icon.textContent = { real: "✓", fake: "!", unsure: "?" }[kind];
  els.label.textContent = { real: "Likely real", fake: "Likely fake", unsure: "Uncertain" }[kind];
  els.sub.textContent = kind === "unsure"
    ? `The model leans ${p >= 0.5 ? "real" : "fake"} (${Math.round(Math.max(p, 1 - p) * 100)}%), but not strongly either way.`
    : `The model estimates a ${Math.round((kind === "real" ? p : 1 - p) * 100)}% chance this is ${kind} news.`;
  els.marker.style.left = `${(p * 100).toFixed(1)}%`;

  const words = countWords(text);
  els.shortWarning.hidden = words >= SHORT_TEXT_WORDS;
  els.shortWarning.textContent = `This text is only ${words} word${words === 1 ? "" : "s"} long. Results on fewer than ${SHORT_TEXT_WORDS} words are much less reliable, so paste the full article if you can.`;

  const fake = result.contributions.filter((c) => c.contribution < 0).slice(0, TOP_WORDS);
  const real = result.contributions.filter((c) => c.contribution > 0).slice(0, TOP_WORDS);
  const maxAbs = Math.max(1e-9, ...[...fake, ...real].map((c) => Math.abs(c.contribution)));
  renderWords(els.wordsFake, fake, maxAbs, "fake");
  renderWords(els.wordsReal, real, maxAbs, "real");

  if (window.matchMedia("(max-width: 760px)").matches) {
    els.result.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

els.article.addEventListener("input", updateWordCount);
els.article.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    if (!els.analyze.disabled) analyze();
  }
});
els.analyze.addEventListener("click", analyze);
els.clear.addEventListener("click", () => {
  els.article.value = "";
  els.result.hidden = true;
  updateWordCount();
  els.article.focus();
});
document.querySelectorAll("[data-sample]").forEach((button) => {
  button.addEventListener("click", () => {
    els.article.value = SAMPLES[button.dataset.sample];
    updateWordCount();
    if (model) analyze();
  });
});

updateWordCount();
init();
