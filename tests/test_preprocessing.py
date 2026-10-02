import math

from fake_news.preprocessing import clean_text, strip_source_dateline


def test_strips_reuters_dateline():
    assert strip_source_dateline("WASHINGTON (Reuters) - The head of...") == "The head of..."
    assert strip_source_dateline("SEOUL/WASHINGTON (Reuters) – Talks...") == "Talks..."


def test_keeps_reuters_mentioned_later_in_text():
    text = "Intro. " + "y" * 200 + " (Reuters) - more"
    assert strip_source_dateline(text) == text


def test_dateline_removed_before_features():
    assert "reuters" not in clean_text("LONDON (Reuters) - Prices rose.")


def test_removes_urls_tags_brackets_and_digits():
    text = "Read <b>this</b> at https://example.com/a?b=1 [Photo: AP] in 2017 or www.site.org now!"
    assert clean_text(text) == "read this at in or now"


def test_lowercases_and_collapses_whitespace_and_punctuation():
    assert clean_text("Hello,   WORLD!!\n\nIt's  under_score") == "hello world it s under score"


def test_non_string_input_is_empty():
    assert clean_text(None) == ""
    assert clean_text(math.nan) == ""
