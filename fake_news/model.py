"""Dataset loading, model definitions, training and persistence."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from fake_news.preprocessing import clean_text

FAKE, REAL = 0, 1
LABELS = {FAKE: "Fake News", REAL: "True News"}

MODEL_NAMES = {
    "lr": "Logistic Regression",
    "dt": "Decision Tree",
    "gb": "Gradient Boosting",
    "rf": "Random Forest",
}


def load_dataset(data_dir="data", fake_file="Fake.csv", true_file="True.csv"):
    """Load the ISOT fake/true news CSVs into one shuffled ``text``/``class`` frame."""
    data_dir = Path(data_dir)
    paths = {FAKE: data_dir / fake_file, REAL: data_dir / true_file}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset not found: " + ", ".join(missing) + ". "
            "Download Fake.csv and True.csv from "
            "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset "
            f"and place them in {data_dir}/."
        )

    frames = []
    for label, path in paths.items():
        frame = pd.read_csv(path, usecols=["text"])
        frame["class"] = label
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["text"])
    data = data[data["text"].str.strip() != ""]
    # The dataset contains many copy-pasted articles; duplicates that land in
    # both train and test splits inflate the reported accuracy.
    data = data.drop_duplicates(subset=["text"])
    return data.sample(frac=1, random_state=0).reset_index(drop=True)


def build_models(names=None, random_state=0):
    """Return ``{name: Pipeline}`` with TF-IDF features feeding each classifier."""
    classifiers = {
        "lr": lambda: LogisticRegression(max_iter=1000),
        "dt": lambda: DecisionTreeClassifier(random_state=random_state),
        "gb": lambda: GradientBoostingClassifier(random_state=random_state),
        "rf": lambda: RandomForestClassifier(random_state=random_state, n_jobs=-1),
    }
    names = list(names or classifiers)
    unknown = sorted(set(names) - set(classifiers))
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}. Choose from {', '.join(classifiers)}.")

    return {
        name: Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=clean_text, stop_words="english", max_df=0.7)),
            ("clf", classifiers[name]()),
        ])
        for name in names
    }


def train_and_evaluate(data, names=None, test_size=0.25, random_state=0, verbose=True):
    """Fit each model on a stratified split and return ``(models, accuracies)``."""
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"], data["class"],
        test_size=test_size, random_state=random_state, stratify=data["class"],
    )

    models = build_models(names, random_state=random_state)
    scores = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        scores[name] = accuracy_score(y_test, pred)
        if verbose:
            print(f"\n=== {MODEL_NAMES[name]} ===")
            print(f"Accuracy: {scores[name]:.4f}")
            print(classification_report(y_test, pred, target_names=[LABELS[FAKE], LABELS[REAL]], zero_division=0))
    return models, scores


def save_models(models, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, path)


def load_models(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No trained models at {path}. Run `python -m fake_news.train` first.")
    return joblib.load(path)


def predict(models, text):
    """Return ``{name: label}`` for one article, plus a majority ``"vote"``."""
    results = {name: int(model.predict([text])[0]) for name, model in models.items()}
    real_votes = sum(results.values())
    # Ties go to "fake": flagging a real article for review is the cheaper mistake.
    results["vote"] = REAL if real_votes > len(models) / 2 else FAKE
    return results
