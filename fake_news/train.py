"""Train the classifiers and save them: ``python -m fake_news.train``."""

import argparse

from fake_news.model import MODEL_NAMES, load_dataset, save_models, train_and_evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="folder containing Fake.csv and True.csv (default: data)")
    parser.add_argument("--output", default="models/fake_news_models.joblib", help="where to save the trained models")
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_NAMES), default=list(MODEL_NAMES),
        help="which classifiers to train (default: all). 'gb' is by far the slowest.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args(argv)

    data = load_dataset(args.data_dir)
    print(f"Loaded {len(data)} unique articles ({(data['class'] == 0).sum()} fake, {(data['class'] == 1).sum()} real)")

    models, scores = train_and_evaluate(data, args.models, args.test_size, args.random_state)

    print("\nSummary:")
    for name, score in sorted(scores.items(), key=lambda item: -item[1]):
        print(f"  {MODEL_NAMES[name]:<20} {score:.4f}")

    save_models(models, args.output)
    print(f"\nSaved models to {args.output}")


if __name__ == "__main__":
    main()
