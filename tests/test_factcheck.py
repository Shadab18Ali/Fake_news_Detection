"""Fact-check search: the Vercel function (api/factcheck.js) and the page helpers (web/factcheck.js)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
API = ROOT / "api" / "factcheck.js"
WEB = ROOT / "web" / "factcheck.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")

GOOGLE_RESPONSE = {
    "claims": [
        {
            "text": "The moon landing was staged",
            "claimant": "Social media posts",
            "claimDate": "2020-01-01T00:00:00Z",
            "claimReview": [
                {"publisher": {"name": "PolitiFact"}, "url": "https://www.politifact.com/x", "title": "No", "reviewDate": "2020-01-02T00:00:00Z", "textualRating": "Pants on Fire"},
                {"publisher": {"site": "evil.example"}, "url": "javascript:alert(1)", "textualRating": "False"},
            ],
        },
        {"text": "Water is wet", "claimReview": [{"publisher": {"name": "AFP"}, "url": "http://afp.example/y", "textualRating": "True"}]},
    ]
}


def call_api(method="GET", query="moon landing staged", key="secret-key", upstream=None):
    """Invoke the handler with a fake req/res and a stubbed global fetch; return status, headers, body, fetched URL."""
    script = f"""
        const handler = require({json.dumps(str(API))});
        const cfg = JSON.parse(require("node:fs").readFileSync(0, "utf8"));
        if (cfg.key) process.env.GOOGLE_FACT_CHECK_API_KEY = cfg.key; else delete process.env.GOOGLE_FACT_CHECK_API_KEY;
        let fetched = null;
        global.fetch = async (url) => {{
            fetched = url;
            const u = cfg.upstream;
            if (u.throw) throw Object.assign(new Error("boom"), {{ name: u.throw }});
            return {{ ok: u.status === 200, status: u.status, json: async () => {{ if (u.badJson) throw new Error("bad"); return u.body; }} }};
        }};
        console.error = () => {{}};
        const headers = {{}};
        const res = {{ statusCode: 0, setHeader: (k, v) => (headers[k.toLowerCase()] = v), end: (b) => {{
            console.log(JSON.stringify({{ status: res.statusCode, headers, body: JSON.parse(b), fetched }}));
        }} }};
        const q = cfg.query === null ? "" : "?query=" + encodeURIComponent(cfg.query);
        handler({{ method: cfg.method, url: "/api/factcheck" + q }}, res);
    """
    cfg = {"method": method, "query": query, "key": key, "upstream": upstream or {"status": 200, "body": GOOGLE_RESPONSE}}
    out = subprocess.run(["node", "-e", script], input=json.dumps(cfg), capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_returns_simplified_results_and_keeps_key_private():
    r = call_api()
    assert r["status"] == 200
    assert r["body"]["query"] == "moon landing staged"
    assert r["body"]["results"] == [
        {"claim": "The moon landing was staged", "claimant": "Social media posts", "rating": "Pants on Fire",
         "publisher": "PolitiFact", "title": "No", "url": "https://www.politifact.com/x", "date": "2020-01-02T00:00:00Z"},
        {"claim": "Water is wet", "claimant": None, "rating": "True", "publisher": "AFP", "title": None,
         "url": "http://afp.example/y", "date": None},
    ]
    assert "javascript:" not in json.dumps(r["body"])
    assert "secret-key" not in json.dumps(r["body"])
    assert "key=secret-key" in r["fetched"] and "query=moon+landing+staged" in r["fetched"]
    assert "s-maxage=3600" in r["headers"]["cache-control"]


def test_missing_api_key_is_reported_as_not_configured():
    r = call_api(key=None)
    assert (r["status"], r["body"]) == (503, {"error": "not_configured"})
    assert r["fetched"] is None


@pytest.mark.parametrize("query", [None, "ab", "x" * 301, "   "])
def test_rejects_missing_short_or_long_queries(query):
    r = call_api(query=query)
    assert (r["status"], r["body"]) == (400, {"error": "invalid_query"})
    assert r["fetched"] is None


def test_only_get_is_allowed():
    r = call_api(method="POST")
    assert r["status"] == 405 and r["headers"]["allow"] == "GET"


@pytest.mark.parametrize("upstream", [{"status": 403, "body": {}}, {"status": 200, "badJson": True}, {"throw": "TimeoutError"}])
def test_upstream_failures_become_502_without_details(upstream):
    r = call_api(upstream=upstream)
    assert r["status"] == 502
    assert r["body"]["error"] in {"upstream_error", "upstream_unavailable"}
    assert r["headers"]["cache-control"] == "no-store"


def test_no_matches_returns_empty_list():
    r = call_api(upstream={"status": 200, "body": {}})
    assert (r["status"], r["body"]["results"]) == (200, [])


def run_web(expr_list):
    script = f"""
        import {{ suggestQuery, ratingTone }} from {json.dumps(WEB.as_uri())};
        const cases = JSON.parse(await new Promise((r) => {{ let d = ""; process.stdin.on("data", (c) => (d += c)); process.stdin.on("end", () => r(d)); }}));
        console.log(JSON.stringify(cases.map(([fn, arg]) => (fn === "suggest" ? suggestQuery(arg) : ratingTone(arg)))));
    """
    out = subprocess.run(["node", "--input-type=module", "-e", script], input=json.dumps(expr_list), capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_suggest_query_uses_first_sentence_without_datelines():
    cases = [
        ("suggest", "WASHINGTON (Reuters) - The U.S. Senate approved the budget on Tuesday. More text follows here."),
        ("suggest", "BRUSSELS - EU ministers agreed to extend loans. Second sentence."),
        ("suggest", "BREAKING: You won't BELIEVE what they are hiding! Share now."),
        ("suggest", "\n\n  Headline without punctuation\nBody text."),
        ("suggest", " ".join(f"w{i}" for i in range(30)) + "."),
        ("suggest", "12345 !!!"),
    ]
    assert run_web(cases) == [
        "The U.S. Senate approved the budget on Tuesday",
        "EU ministers agreed to extend loans",
        "You won't BELIEVE what they are hiding",
        "Headline without punctuation",
        " ".join(f"w{i}" for i in range(14)),
        "",
    ]


def test_rating_tone():
    ratings = ["False", "Pants on Fire", "Mostly false", "Misleading", "True", "Mostly True", "Half true", "Needs context", None]
    assert run_web([["tone", r] for r in ratings]) == ["fake", "fake", "fake", "fake", "real", "real", "neutral", "neutral", "neutral"]
