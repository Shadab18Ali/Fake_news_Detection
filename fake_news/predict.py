"""Classify an article with the trained models: ``python -m fake_news.predict "text..."``.

With no text argument, the article is read from stdin.
"""

import argparse
import sys

from fake_news.model import LABELS, MODEL_NAMES, load_models, predict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="?", help="article text (reads stdin when omitted)")
    parser.add_argument("--model-path", default="models/fake_news_models.joblib")
    args = parser.parse_args(argv)

    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        parser.error("no article text given")

    results = predict(load_models(args.model_path), text)
    vote = results.pop("vote")
    for name, label in results.items():
        print(f"{MODEL_NAMES[name]:<20} {LABELS[label]}")
    print(f"\nMajority vote: {LABELS[vote]}")


if __name__ == "__main__":
    main()
