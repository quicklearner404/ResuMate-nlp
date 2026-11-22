"""
Generate the fictional sample resume used by the app and the README screenshot.

    python scripts/make_sample_resume.py   # -> app/samples/resume.pdf

The person, contact details and companies are made up (example.com is a
reserved domain), so the sample can be shared publicly.
"""

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parents[1] / "app" / "samples" / "resume.pdf"

NAME = "Hina Qureshi"
CONTACT = "hina.qureshi@example.com  |  +92 300 0000000  |  Lahore, Pakistan"

SECTIONS = [
    ("Summary", [
        "Machine learning engineer with 3 years of experience building NLP systems,",
        "from data pipelines to deployed models.",
    ]),
    ("Experience", [
        ("Machine Learning Engineer at Northwind Analytics", "2023 - Present"),
        "- Built a document classification service with Python, PyTorch and FastAPI.",
        "- Fine-tuned transformers models for named entity recognition on support tickets.",
        "- Deployed models with Docker and Kubernetes on AWS.",
        ("Data Analyst at Contoso Retail", "2021 - 2023"),
        "- Automated weekly sales reports with SQL, Pandas and Tableau.",
        "- Built churn prediction models with scikit-learn.",
    ]),
    ("Education", [
        ("BS Computer Science, FAST NUCES", "2017 - 2021"),
    ]),
    ("Skills", [
        "Python, SQL, PyTorch, Transformers, scikit-learn, Pandas, NumPy,",
        "NLP, FastAPI, Docker, Kubernetes, AWS, Git",
    ]),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=A4)
    width, height = A4
    left, y = 20 * mm, height - 22 * mm
    accent = HexColor("#2563eb")

    c.setFont("Helvetica-Bold", 20)
    c.drawString(left, y, NAME)
    y -= 7 * mm
    c.setFont("Helvetica", 10)
    c.setFillColor(HexColor("#475569"))
    c.drawString(left, y, CONTACT)
    c.setFillColor(HexColor("#000000"))
    y -= 10 * mm

    for title, lines in SECTIONS:
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(accent)
        c.drawString(left, y, title.upper())
        c.setStrokeColor(accent)
        c.line(left, y - 1.5 * mm, width - left, y - 1.5 * mm)
        c.setFillColor(HexColor("#000000"))
        y -= 7 * mm
        for line in lines:
            if isinstance(line, tuple):
                c.setFont("Helvetica-Bold", 10.5)
                c.drawString(left, y, line[0])
                c.setFont("Helvetica", 10)
                c.drawRightString(width - left, y, line[1])
            else:
                c.setFont("Helvetica", 10)
                c.drawString(left + (3 * mm if line.startswith("-") else 0), y, line)
            y -= 5.5 * mm
        y -= 4 * mm

    c.save()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
