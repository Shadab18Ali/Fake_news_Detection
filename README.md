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
api/factcheck.js            Vercel function: searches published fact-checks via Google's Fact Check Tools API
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

`web/` is a dependency-free static site (plain HTML, CSS and ES modules, no build step). Paste an article and it
shows a prediction (likely real / likely fake / uncertain), the classifier's fake and real probabilities, the
confidence, and the words that pushed the score each way.

- **Private:** the Logistic Regression model runs in the browser (only the optional fact-check search sends anything,
  and then only the search phrase). `web/classifier.js` re-implements the text cleaning
  and TF-IDF scoring, and `tests/test_web_parity.py` checks it gives the same cleaned text and probabilities as
  scikit-learn. The page only downloads `model.json`; article text never leaves the browser, and the
  Content-Security-Policy in `vercel.json` only allows requests to the site itself.
- **Honest wording:** results describe writing patterns, never factual truth, and the page explains the model's
  limitations.
- **Robust:** input is validated (empty, under 10 words, over 100,000 characters), short or off-vocabulary texts get a
  reliability note, and a missing or malformed model shows a plain-language error instead of breaking the page.
- **Accessible:** semantic landmarks and headings, a skip link, labelled controls, visible focus, 44px touch targets,
  screen-reader announcements for status and results, light/dark themes and reduced-motion support.
- **SEO:** title, description, canonical URL, Open Graph/Twitter cards, `robots.txt` and `sitemap.xml`. The canonical
  URL is `https://fakenewsdetection-azure.vercel.app/`; update it in `web/index.html`, `web/robots.txt` and
  `web/sitemap.xml` if the site moves to another domain.

### Fact-check search (optional, free)

Below the analyzer, visitors can search fact-checks already published by organisations such as PolitiFact, Snopes
and AFP. The page sends only the short search phrase (pre-filled from the article's first sentence, editable) to
`api/factcheck.js`, which calls Google's free [Fact Check Tools API](https://developers.google.com/fact-check/tools/api)
with a key kept on the server. It only finds claims a fact-checker has already reviewed.

To turn it on:

1. In the [Google Cloud console](https://console.cloud.google.com/), create a project, enable the
   **Fact Check Tools API**, and create an API key under **APIs & Services → Credentials**. Restrict the key to the
   Fact Check Tools API.
2. In Vercel, open the project → **Settings → Environment Variables** and add `GOOGLE_FACT_CHECK_API_KEY` with the key
   as its value, for Production and Preview.
3. Redeploy. Until the key is set, the section says "Fact-check search isn't set up on this site yet."

### Publish the model

The site needs `web/model.json`, which is generated from the dataset:

```bash
python -m fake_news.export            # ~1-2 MB with the default 50,000-word vocabulary
git add web/model.json && git commit -m "Update web model" && git push
```

Until the file exists, the site shows "Unable to load the analysis model" and the browser console explains how to
generate it.

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
