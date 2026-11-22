"""Regression tests for the evaluation bug: wrong predictions must lower precision."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from evaluate import evaluate, matches, normalize, prf, score_field  # noqa: E402


def test_wrong_prediction_counts_as_false_positive():
    # The original script recorded only "label" or "MISSED", so this case scored
    # precision 1.0. Here the wrong prediction is a false positive.
    tp, fp, fn, wrong, missed = score_field({"washington"}, {"raytheon"}, "lenient")

    assert (tp, fp, fn) == (0, 1, 1)
    assert wrong == ["washington"] and missed == ["raytheon"]


def test_extra_prediction_lowers_precision_even_when_gold_is_found():
    tp, fp, fn, *_ = score_field({"google", "kubernet"}, {"google"}, "lenient")

    assert (tp, fp, fn) == (1, 1, 0)
    assert prf(tp, fp, fn)["precision"] == 0.5


def test_lenient_matching_follows_original_substring_rule():
    assert matches("london", "london uk", "lenient")
    assert not matches("london", "london uk", "exact")
    assert not matches("c", "c++", "lenient")  # too short to count


def test_normalize():
    assert normalize("  Senior   Engineer, ") == "senior engineer"


def test_unannotated_fields_are_not_penalised():
    data = [{"ground_truth": {"Name": ["Ada Lovelace"]}}]
    preds = [{"Name": ["Ada Lovelace"], "Location": ["London"]}]

    result = evaluate(preds, data, "exact")

    assert result["micro"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 1, "fp": 0, "fn": 0}
    assert result["resumes_fully_correct"] == "1/1"


def test_micro_average_over_fields():
    data = [{"ground_truth": {"Company": ["Careem"], "College": ["FAST NUCES"]}}]
    preds = [{"Company": ["care"], "College": ["FAST NUCES", "Careem"]}]

    result = evaluate(preds, data, "exact")

    assert result["micro"]["tp"] == 1
    assert result["micro"]["fp"] == 2
    assert result["micro"]["fn"] == 1
