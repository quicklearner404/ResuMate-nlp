"""Entity clean-up must trim whole connector words, never parts of real words."""

import pytest

pytest.importorskip("spacy")

from resumate.extraction.hybrid_parser import _trim_edges  # noqa: E402


@pytest.mark.parametrize("word", ["Berlin", "Kotlin", "India", "California", "Careem", "Data"])
def test_real_words_are_not_truncated(word):
    # The old clean-up matched raw suffixes: "Berlin" -> "Berl", "India" -> "Indi".
    assert _trim_edges(word) == word


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("based in Lahore", "Lahore"),
        ("Engineer at", "Engineer"),
        (". Data Analyst", "Data Analyst"),
        ("Salesforce in San Francisco,", "Salesforce in San Francisco"),
    ],
)
def test_connector_words_and_punctuation_are_trimmed(raw, clean):
    assert _trim_edges(raw) == clean
