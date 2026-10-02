# Fake News Detection

Classifies news articles as **fake** or **real** using TF-IDF text features and four scikit-learn models:
Logistic Regression, Decision Tree, Gradient Boosting and Random Forest. A majority vote combines them.

## Project layout

```
FAKE NEWS DETECTION.ipynb   step-by-step walkthrough: exploration, training, evaluation, manual testing
fake_news/
  preprocessing.py          clean_text(): the text normalization shared by training and prediction
  model.py                  dataset loading, model pipelines, training, saving/loading, prediction
  train.py                  CLI: train all models and save them
  predict.py                CLI: classify an article with the saved models
  export.py                 CLI: train the model and export it to web/model.json for the web app
web/                        static web app (deployed on Vercel), runs the model in the browser
vercel.json                 Vercel config: serves web/ with no build step
tests/                      pytest suite (runs without the dataset)
data/                       put Fake.csv and True.csv here (not committed)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `Fake.csv` and `True.csv` from the
[Fake and Real News dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
and put them in `data/`.

## Usage

**Notebook:** run `jupyter notebook "FAKE NEWS DETECTION.ipynb"` from the project root.

**Command line:**

```bash
# Train and save models to models/fake_news_models.joblib
python -m fake_news.train

# Gradient Boosting is slow on the full dataset; train a subset of models instead
python -m fake_news.train --models lr rf

# Classify an article (text as an argument, or piped through stdin)
python -m fake_news.predict "The ministry said on Tuesday that exports rose 3% in the quarter..."
cat article.txt | python -m fake_news.predict
```

**Tests:** `python -m pytest` (the web parity test also needs [Node.js](https://nodejs.org))

## Web app

`web/` is a dependency-free static site: paste an article and it shows a verdict, a fake ↔ real meter and the
words that pushed the score each way. The Logistic Regression model runs **in the browser**: `web/classifier.js`
re-implements the text cleaning and TF-IDF scoring, and `tests/test_web_parity.py` checks it gives the same
probabilities as scikit-learn. No server is needed, and pasted text never leaves the visitor's browser.

### Publish the model

The site needs `web/model.json`, which is generated from the dataset:

```bash
python -m fake_news.export            # ~1-2 MB with the default 50,000-word vocabulary
git add web/model.json && git commit -m "Update web model" && git push
```

Until the file exists, the site shows a notice instead of the analyze button.

### Deploy on Vercel

With the repository connected to Vercel, every push deploys automatically: `vercel.json` tells Vercel to serve
`web/` as-is (no install or build step), so no project settings need changing. Pushes to `main` go to
production and other branches get preview URLs.

To try it locally: `python -m http.server -d web 8000`, then open http://localhost:8000.

## How it works

1. **Load:** both CSVs are merged and labelled (`0` = fake, `1` = real). Empty and duplicate articles are dropped
   so the same text can't appear in both the training and test sets.
2. **Clean:** `clean_text` lowercases the text and removes URLs, HTML tags, bracketed text, tokens containing
   digits and punctuation.
3. **Vectorize:** `TfidfVectorizer` with English stop words removed.
4. **Classify:** each model is a scikit-learn `Pipeline` (cleaning + TF-IDF + classifier), saved with `joblib`.
   Raw text goes in at prediction time and gets exactly the same processing as during training.
5. **Evaluate:** stratified 75/25 train/test split with a fixed random seed, so results are reproducible.

### Note on accuracy

Nearly every real article in this dataset starts with a dateline like `WASHINGTON (Reuters) - `, which the fake
articles lack. A model trained on the raw text largely learns "mentions Reuters ⇒ real", and this is a big part of
why the original version of this project reported ~99% accuracy. The cleaning step strips that dateline and
duplicates are removed, so expect somewhat lower but more honest scores. Even then, a model trained on one
dataset's writing style won't reliably judge articles from other sources, so treat its predictions as a signal,
not a verdict.
