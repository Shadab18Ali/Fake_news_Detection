import pandas as pd
import pytest

from fake_news.model import FAKE, REAL, build_models, load_dataset, load_models, predict, save_models, train_and_evaluate
from fake_news.predict import main as predict_main

FAKE_TEXTS = [
    "SHOCKING you won't believe what this celebrity secretly did, share before it gets deleted",
    "BREAKING miracle cure doctors hate, the elites are hiding the truth from you",
    "WATCH this unbelievable video exposes the hoax they don't want you to see",
    "outrageous scandal exposed, insiders reveal shocking secret plot, share now",
]
REAL_TEXTS = [
    "WASHINGTON (Reuters) - The Senate voted on Tuesday to approve the budget bill, officials said.",
    "LONDON (Reuters) - The central bank held interest rates steady, according to a statement.",
    "TOKYO (Reuters) - The ministry said exports rose in the quarter, government data showed.",
    "BERLIN (Reuters) - The parliament approved the measure after debate, lawmakers said.",
]


def _write_dataset(tmp_path, repeat=5):
    # Vary each copy so de-duplication keeps them.
    for name, texts in (("Fake.csv", FAKE_TEXTS), ("True.csv", REAL_TEXTS)):
        rows = [f"{t} item {chr(97 + i)}" for i in range(repeat) for t in texts]
        pd.DataFrame({"title": "t", "text": rows, "subject": "s", "date": "d"}).to_csv(tmp_path / name, index=False)


def test_load_dataset_labels_dedupes_and_drops_empty(tmp_path):
    pd.DataFrame({"text": ["a fake story", "a fake story", "", None]}).to_csv(tmp_path / "Fake.csv", index=False)
    pd.DataFrame({"text": ["a real story"]}).to_csv(tmp_path / "True.csv", index=False)

    data = load_dataset(tmp_path)

    assert sorted(data.columns) == ["class", "text"]
    assert sorted(zip(data["text"], data["class"])) == [("a fake story", FAKE), ("a real story", REAL)]


def test_load_dataset_missing_files_explains_where_to_get_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle"):
        load_dataset(tmp_path)


def test_build_models_rejects_unknown_names():
    with pytest.raises(ValueError, match="svm"):
        build_models(["lr", "svm"])


def test_train_save_load_predict_roundtrip(tmp_path, capsys):
    _write_dataset(tmp_path)
    models, scores = train_and_evaluate(load_dataset(tmp_path), names=["lr", "dt", "rf"], verbose=False)
    assert set(scores) == {"lr", "dt", "rf"}

    path = tmp_path / "models" / "m.joblib"
    save_models(models, path)
    loaded = load_models(path)

    fake = predict(loaded, "SHOCKING secret hoax exposed, share before deleted")
    real = predict(loaded, "PARIS (Reuters) - The ministry said the parliament approved the budget, officials said.")
    assert fake["vote"] == FAKE
    assert real["vote"] == REAL

    predict_main(["--model-path", str(path), "The ministry said lawmakers approved the bill."])
    assert "Majority vote: True News" in capsys.readouterr().out


def test_load_models_missing_file_points_to_training(tmp_path):
    with pytest.raises(FileNotFoundError, match="fake_news.train"):
        load_models(tmp_path / "nope.joblib")
