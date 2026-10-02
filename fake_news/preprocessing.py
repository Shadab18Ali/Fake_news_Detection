"""Text cleaning used for both training and prediction."""

import re

# Nearly every article in True.csv opens with a dateline such as
# "WASHINGTON (Reuters) - ", while articles in Fake.csv almost never do.
# Left in place, the model learns "contains 'reuters' => real" instead of
# anything about the content, so we strip it before vectorizing.
_DATELINE = re.compile(r"^[^\n]{0,120}?\((?:Reuters|REUTERS)\)\s*[-–—]\s*")

_BRACKETED = re.compile(r"\[.*?\]")
_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG = re.compile(r"<.*?>+")
_WORD_WITH_DIGIT = re.compile(r"\w*\d\w*")
_NON_WORD = re.compile(r"[\W_]+")
_WHITESPACE = re.compile(r"\s+")


def strip_source_dateline(text):
    """Remove a leading news-agency dateline like ``"LONDON (Reuters) - "``."""
    return _DATELINE.sub("", text, count=1)


def clean_text(text):
    """Normalize raw article text for TF-IDF vectorization.

    Lowercases, drops the source dateline, bracketed text, URLs, HTML tags,
    tokens containing digits and punctuation, and collapses whitespace.
    Non-string input (e.g. ``NaN`` from pandas) becomes an empty string.
    """
    if not isinstance(text, str):
        return ""
    text = strip_source_dateline(text)
    text = text.lower()
    text = _BRACKETED.sub(" ", text)
    # URLs and tags must go before punctuation is stripped, otherwise
    # "https://example.com" is split into words that leak into the features.
    text = _URL.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _WORD_WITH_DIGIT.sub(" ", text)
    text = _NON_WORD.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()
