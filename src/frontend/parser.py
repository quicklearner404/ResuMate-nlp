import pdfplumber

def extract_text_from_pdf(pdf_path_or_file):
    text = ""
    with pdfplumber.open(pdf_path_or_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

extract_text_from_pdf("/home/bilal-asghar/Documents/ResuMate-nlp/src/frontend/resume.pdf")