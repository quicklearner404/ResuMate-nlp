"""End-to-end smoke test of the app's pipeline: PDF -> text -> hybrid parser."""

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pdfplumber")

SAMPLE = Path(__file__).resolve().parents[1] / "app" / "samples" / "resume.pdf"


def test_sample_pdf_is_parsed_into_fields():
    from resumate.extraction.pdf_parser import process_resume

    try:
        with SAMPLE.open("rb") as f:
            text, fields = process_resume(f)
    except OSError as exc:  # model download unavailable offline
        pytest.skip(str(exc))

    assert text.strip()
    assert "Error" not in fields
    assert set(fields) >= {"Name", "Email", "Designation", "Company", "Skills"}
    assert fields["Skills"], "skill matcher found nothing in the sample resume"
