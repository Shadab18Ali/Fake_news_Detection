// Browser port of fake_news.preprocessing.clean_text plus TF-IDF + Logistic
// Regression scoring, using the weights exported by `python -m fake_news.export`.
// Keep in sync with fake_news/preprocessing.py; tests/test_web_parity.py checks
// that both give the same probabilities.

const DATELINE = /^[^\n]{0,120}?\((?:Reuters|REUTERS)\)\s*[-–—]\s*/u;
// Python's "." matches everything except "\n"; JavaScript's "." also stops at
// "\r", "\u2028" and "\u2029", so spell out Python's behaviour with [^\n].
const BRACKETED = /\[[^\n]*?\]/gu;
const URL = /https?:\/\/\S+|www\.\S+/gu;
const HTML_TAG = /<[^\n]*?>+/gu;
// Python's str-pattern \w is Unicode letters, numbers and "_"; \d is Nd.
const WORD_WITH_DIGIT = /[\p{L}\p{N}_]*\p{Nd}[\p{L}\p{N}_]*/gu;
const NON_WORD = /[^\p{L}\p{N}]+/gu;
const WHITESPACE = /\s+/gu;

export function cleanText(text) {
  if (typeof text !== "string") return "";
  return text
    .replace(DATELINE, "")
    .toLowerCase()
    .replace(BRACKETED, " ")
    .replace(URL, " ")
    .replace(HTML_TAG, " ")
    .replace(WORD_WITH_DIGIT, " ")
    .replace(NON_WORD, " ")
    .replace(WHITESPACE, " ")
    .trim();
}

function isNumberArray(value, length) {
  return Array.isArray(value) && value.length === length && value.every(Number.isFinite);
}

/** Build the lookup structures for an exported model; throws if the JSON is malformed. */
export function loadModel(json) {
  const size = Array.isArray(json?.vocabulary) ? json.vocabulary.length : -1;
  if (
    size <= 0 ||
    !json.vocabulary.every((term) => typeof term === "string") ||
    !isNumberArray(json.idf, size) ||
    !isNumberArray(json.coef, size) ||
    !Number.isFinite(json.intercept) ||
    !Array.isArray(json.stop_words)
  ) {
    throw new Error("Invalid model: expected vocabulary, idf and coef arrays of equal length and a numeric intercept");
  }
  return {
    ...json,
    index: new Map(json.vocabulary.map((term, i) => [term, i])),
    stopWords: new Set(json.stop_words),
  };
}

// After cleanText only letters/numbers separated by single spaces remain, so
// scikit-learn's default token pattern (\b\w\w+\b) reduces to "words of 2+
// characters". Length is counted in code points, as Python does.
export function tokenize(model, text) {
  const cleaned = cleanText(text);
  if (!cleaned) return [];
  return cleaned.split(" ").filter((t) => [...t].length >= 2 && !model.stopWords.has(t));
}

/**
 * Score one article. Returns null when none of its words are in the model's
 * vocabulary; otherwise the probability that it is real news, the predicted
 * label (0 = fake, 1 = real) and each known word's contribution to the score
 * (positive pushes towards real, negative towards fake).
 */
export function predict(model, text) {
  const tokens = tokenize(model, text);
  const counts = new Map();
  for (const token of tokens) {
    const i = model.index.get(token);
    if (i !== undefined) counts.set(i, (counts.get(i) || 0) + 1);
  }
  if (counts.size === 0) return null;

  let norm = 0;
  const weights = new Map();
  for (const [i, count] of counts) {
    const w = count * model.idf[i];
    weights.set(i, w);
    norm += w * w;
  }
  norm = Math.sqrt(norm);

  let score = model.intercept;
  const contributions = [];
  for (const [i, w] of weights) {
    const c = model.coef[i] * (w / norm);
    score += c;
    contributions.push({ term: model.vocabulary[i], count: counts.get(i), contribution: c });
  }
  contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  const probabilityReal = 1 / (1 + Math.exp(-score));
  return {
    probabilityReal,
    label: probabilityReal >= 0.5 ? 1 : 0,
    tokenCount: tokens.length,
    knownTokenCount: [...counts.values()].reduce((a, b) => a + b, 0),
    contributions,
  };
}
