"""Entities must cover whole words, never sub-word fragments like "care" for "Careem"."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "parsing"))
from hybridBert import BertExtractor  # noqa: E402

MODEL_ID = "quicklearner404/DistillbertNER"
LABELS = {0: "O", 3: "B-Company", 4: "I-Company"}


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.DistilBertTokenizerFast.from_pretrained(MODEL_ID)
    except Exception as exc:  # offline
        pytest.skip(f"tokenizer unavailable: {exc}")


class FirstSubtokenOnly:
    """Fake model: tags only the first sub-token of the target word as B-Company."""

    def __init__(self, tokenizer, text, word):
        enc = tokenizer(text, return_offsets_mapping=True)
        start = text.index(word)
        self.target = next(i for i, (s, _) in enumerate(enc["offset_mapping"]) if s == start)

    def eval(self):
        return self

    def __call__(self, input_ids, **_):
        logits = torch.zeros(1, input_ids.shape[1], 13)
        logits[0, :, 0] = 1.0
        logits[0, self.target, 3] = 5.0
        return SimpleNamespace(logits=logits)


def _extractor(tokenizer, model):
    ex = BertExtractor.__new__(BertExtractor)
    ex.tokenizer, ex.model, ex.device = tokenizer, model, torch.device("cpu")
    ex.id2label = LABELS
    return ex


def test_entity_spans_whole_word(tokenizer):
    text = "Worked as an engineer at Careem for three years."
    assert len(tokenizer.tokenize("Careem")) > 1  # the word really is split into pieces

    result = _extractor(tokenizer, FirstSubtokenOnly(tokenizer, text, "Careem")).predict(text)

    assert result["Company"] == ["Careem"]
