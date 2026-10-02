"""Train the Logistic Regression model and export it for the web app: ``python -m fake_news.export``.

The web app (``web/``) runs the model in the browser, so instead of a pickled
scikit-learn pipeline it needs plain numbers: the vocabulary, IDF weights,
coefficients and intercept. ``web/classifier.js`` re-implements ``clean_text``,
tokenization and TF-IDF scoring on top of them.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from fake_news.model import LABELS, build_models, load_dataset

FORMAT_VERSION = 1


def _check_supported(vectorizer):
    # classifier.js only implements the vectorizer settings build_models uses;
    # fail loudly rather than export a model the browser would score differently.
    expected = {
        "analyzer": "word", "ngram_range": (1, 1), "token_pattern": r"(?u)\b\w\w+\b",
        "tokenizer": None, "norm": "l2", "use_idf": True, "sublinear_tf": False, "binary": False,
    }
    params = vectorizer.get_params()
    unsupported = {k: params[k] for k, v in expected.items() if params[k] != v}
    if unsupported:
        raise ValueError(f"The web classifier does not support these TfidfVectorizer settings: {unsupported}")


def export_pipeline(pipeline, metrics=None):
    """Convert a fitted TF-IDF + LogisticRegression pipeline into a JSON-serializable dict."""
    vectorizer, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    _check_supported(vectorizer)
    if list(clf.classes_) != [0, 1]:
        raise ValueError(f"Expected binary classes [0, 1], got {list(clf.classes_)}")

    stop_words = vectorizer.get_stop_words() or frozenset()
    return {
        "format_version": FORMAT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels": {str(k): v for k, v in LABELS.items()},
        "vocabulary": vectorizer.get_feature_names_out().tolist(),
        "idf": [round(float(v), 5) for v in vectorizer.idf_],
        "coef": [round(float(v), 5) for v in clf.coef_[0]],
        "intercept": round(float(clf.intercept_[0]), 5),
        "stop_words": sorted(stop_words),
        "metrics": metrics or {},
    }


def write_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data", help="folder containing Fake.csv and True.csv (default: data)")
    parser.add_argument("--output", default="web/model.json")
    parser.add_argument(
        "--max-features", type=int, default=50000,
        help="vocabulary size; smaller means a faster-loading web page (default: 50000)",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args(argv)

    data = load_dataset(args.data_dir)
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"], data["class"],
        test_size=args.test_size, random_state=args.random_state, stratify=data["class"],
    )

    pipeline = build_models(["lr"], random_state=args.random_state)["lr"]
    pipeline.set_params(tfidf__max_features=args.max_features)
    pipeline.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(x_test))
    print(f"Logistic Regression test accuracy: {accuracy:.4f}")

    model = export_pipeline(pipeline, {
        "accuracy": round(accuracy, 4), "train_articles": len(x_train), "test_articles": len(x_test),
    })
    write_model(model, args.output)
    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"Exported {len(model['vocabulary'])} terms to {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
