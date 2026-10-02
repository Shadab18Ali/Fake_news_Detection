"""The browser classifier (web/classifier.js) must agree with the Python pipeline."""

import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from fake_news.export import export_pipeline, write_model
from fake_news.model import build_models
from fake_news.preprocessing import clean_text

CLASSIFIER_JS = Path(__file__).resolve().parent.parent / "web" / "classifier.js"

FAKE_WORDS = "shocking secret hoax exposed share elites hiding truth unbelievable scandal video café naïve".split()
REAL_WORDS = "ministry said officials parliament approved budget statement data showed lawmakers zürich straße".split()

EDGE_CASES = [
    "WASHINGTON (Reuters) - Officials said the budget was approved.",
    "SEOUL/TOKYO (Reuters) – Ministry data showed exports rose.",
    "Officials said [Photo: AP] see https://example.com/a?b=1 or www.x.org <b>now</b>!",
    "COVID19 case 2nd wave in 2017: the ministry said ¾ of them, ٣ times",
    "Café naïve Zürich straße ÉLITES Ελλάδα under_score It's SHOCKING!!!\n\nShare   now",
    "Officials [aside\rcut] said <b\u2028x> data [y\u2029z] showed",  # "." spans \r, \u2028, \u2029 in Python
    "a I be to of",  # only stop words and 1-letter tokens
    "",
]


def _corpus(seed=0, n=60):
    rng = random.Random(seed)
    texts, labels = [], []
    for label, words in ((0, FAKE_WORDS), (1, REAL_WORDS)):
        for _ in range(n):
            texts.append(" ".join(rng.choices(words + FAKE_WORDS[:3] + REAL_WORDS[:3], k=25)))
            labels.append(label)
    return texts, labels


def _node(script, *args, stdin=""):
    out = subprocess.run(
        ["node", "--input-type=module", "-e", script, *map(str, args)],
        input=stdin, capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def _run_node(model_path, texts):
    script = f"""
        import fs from "node:fs";
        import {{ cleanText, loadModel, predict }} from {json.dumps(CLASSIFIER_JS.as_uri())};
        const model = loadModel(JSON.parse(fs.readFileSync(process.argv[1], "utf8")));
        const texts = JSON.parse(fs.readFileSync(0, "utf8"));
        console.log(JSON.stringify(texts.map((t) => {{
            const r = predict(model, t);
            return {{ cleaned: cleanText(t), p: r ? r.probabilityReal : null }};
        }})));
    """
    return _node(script, model_path, stdin=json.dumps(texts))


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_browser_classifier_matches_sklearn(tmp_path):
    texts, labels = _corpus()
    pipeline = build_models(["lr"])["lr"]
    pipeline.fit(texts, labels)

    model_path = tmp_path / "model.json"
    write_model(export_pipeline(pipeline), model_path)

    samples = EDGE_CASES + _corpus(seed=1, n=10)[0]
    results = _run_node(model_path, samples)

    vectorizer = pipeline.named_steps["tfidf"]
    for text, result in zip(samples, results):
        assert result["cleaned"] == clean_text(text), text
        if vectorizer.transform([text]).nnz == 0:
            assert result["p"] is None, text
        else:
            assert result["p"] == pytest.approx(pipeline.predict_proba([text])[0][1], abs=1e-3), text


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_browser_rejects_malformed_models():
    texts, labels = _corpus()
    good = export_pipeline(build_models(["lr"])["lr"].fit(texts, labels))
    bad = [
        {**good, "idf": good["idf"][:-1]},
        {**good, "coef": good["coef"][:-1] + [None]},
        {**good, "intercept": "x"},
        {**good, "vocabulary": []},
        {k: v for k, v in good.items() if k != "stop_words"},
        None,
    ]
    script = f"""
        import fs from "node:fs";
        import {{ loadModel }} from {json.dumps(CLASSIFIER_JS.as_uri())};
        console.log(JSON.stringify(JSON.parse(fs.readFileSync(0, "utf8")).map((m) => {{
            try {{ loadModel(m); return "ok"; }} catch (e) {{ return "error"; }}
        }})));
    """
    assert _node(script, stdin=json.dumps([good] + bad)) == ["ok"] + ["error"] * len(bad)


def test_export_contains_what_the_browser_needs():
    texts, labels = _corpus()
    pipeline = build_models(["lr"])["lr"].fit(texts, labels)

    model = export_pipeline(pipeline, {"accuracy": 0.9})

    assert model["format_version"] == 1
    assert len(model["vocabulary"]) == len(model["idf"]) == len(model["coef"])
    assert "the" in model["stop_words"]
    assert model["metrics"] == {"accuracy": 0.9}
    json.dumps(model)


def test_export_rejects_settings_the_browser_cannot_reproduce():
    texts, labels = _corpus()
    pipeline = build_models(["lr"])["lr"].set_params(tfidf__sublinear_tf=True).fit(texts, labels)

    with pytest.raises(ValueError, match="sublinear_tf"):
        export_pipeline(pipeline)
