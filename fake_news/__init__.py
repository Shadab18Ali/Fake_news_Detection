"""Fake news detection with TF-IDF features and classic scikit-learn classifiers."""

from fake_news.preprocessing import clean_text, strip_source_dateline

__all__ = ["clean_text", "strip_source_dateline"]
__version__ = "1.0.0"
