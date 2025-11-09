import os
import pdfplumber
import pandas as pd

from entities import extract_entities
  # ✅ import entity extractor

INPUT_DIR = "data/resume/pdfs"
OUTPUT_FILE = "data/resume/parsed_resumes.csv"
def clean_text(text: str) -> str:
    """Basic text cleaning for resumes."""
    text = text.lower()
    text = text.replace("\n", " ")
    return " ".join(text.split())

def parse_pdf(file_path):
    """Extract raw text from PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text.strip()

def process_resume(pdf_file):
    """Parse + clean + extract entities from one resume."""
    raw_text = parse_pdf(pdf_file)
    cleaned = clean_text(raw_text)
    entities = extract_entities(cleaned)   # ✅ use entities.py
    return cleaned, entities

def main():
    results = []
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]

    for i, pdf_file in enumerate(pdf_files, 1):
        file_path = os.path.join(INPUT_DIR, pdf_file)
        cleaned, entities = process_resume(file_path)

        results.append({
            "filename": pdf_file,
            "cleaned_resume": cleaned,
            "entities": entities
        })

        if i % 10 == 0 or i == len(pdf_files):
            print(f"Processed {i}/{len(pdf_files)} PDFs")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{len(pdf_files)} PDFs processed and saved to {OUTPUT_FILE}\n")

    # Show first 3 samples
    for r in results[:3]:
        print(f"--- {r['filename']} ---")
        print("cleaned_resume:", r["cleaned_resume"][:400], "...")
        print("entities:", r["entities"])
        print()

if __name__ == "__main__":
    main()
