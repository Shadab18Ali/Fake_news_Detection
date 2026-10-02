// Vercel serverless function: GET /api/factcheck?query=...
// Searches fact-checks published by organisations such as PolitiFact, Snopes
// and AFP through Google's Fact Check Tools API. The API key stays on the
// server (environment variable GOOGLE_FACT_CHECK_API_KEY); the browser only
// ever sends the short search query, never the full article.

const API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search";
const MIN_QUERY = 3;
const MAX_QUERY = 300;
const MAX_RESULTS = 10;
const TIMEOUT_MS = 8000;

function send(res, status, body, cacheSeconds = 0) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", cacheSeconds ? `public, s-maxage=${cacheSeconds}, max-age=0` : "no-store");
  res.end(JSON.stringify(body));
}

// Keep only the fields the page shows, and only http(s) links.
function simplify(claims) {
  const results = [];
  for (const claim of claims || []) {
    for (const review of claim.claimReview || []) {
      if (!/^https?:\/\//i.test(review.url || "")) continue;
      results.push({
        claim: String(claim.text || "").slice(0, 500),
        claimant: claim.claimant ? String(claim.claimant).slice(0, 200) : null,
        rating: review.textualRating ? String(review.textualRating).slice(0, 200) : null,
        publisher: String(review.publisher?.name || review.publisher?.site || "Fact-checker").slice(0, 200),
        title: review.title ? String(review.title).slice(0, 300) : null,
        url: review.url,
        date: review.reviewDate || claim.claimDate || null,
      });
      if (results.length === MAX_RESULTS) return results;
    }
  }
  return results;
}

async function handler(req, res) {
  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    return send(res, 405, { error: "method_not_allowed" });
  }

  const url = new URL(req.url, "http://localhost");
  const query = (url.searchParams.get("query") || "").replace(/\s+/g, " ").trim();
  if (query.length < MIN_QUERY || query.length > MAX_QUERY) {
    return send(res, 400, { error: "invalid_query" });
  }

  const key = process.env.GOOGLE_FACT_CHECK_API_KEY;
  if (!key) return send(res, 503, { error: "not_configured" });

  const params = new URLSearchParams({ query, languageCode: "en", pageSize: String(MAX_RESULTS), key });
  let response;
  try {
    response = await fetch(`${API_URL}?${params}`, { signal: AbortSignal.timeout(TIMEOUT_MS) });
  } catch (err) {
    console.error("Fact Check API request failed:", err.name);
    return send(res, 502, { error: "upstream_unavailable" });
  }
  if (!response.ok) {
    // Log the status only: Google's error body can echo the request URL, which contains the key.
    console.error("Fact Check API returned HTTP", response.status);
    return send(res, 502, { error: "upstream_error" });
  }

  let data;
  try {
    data = await response.json();
  } catch {
    return send(res, 502, { error: "upstream_error" });
  }
  return send(res, 200, { query, results: simplify(data.claims) }, 3600);
}

module.exports = handler;
module.exports.simplify = simplify;
