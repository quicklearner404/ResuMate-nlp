import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import re
import textwrap

# Paths
csv_file = r"C:\Users\Dell\Documents\ResuMate-nlp\data\resume\gpt_dataset.csv"
output_folder = r"C:\Users\Dell\Documents\ResuMate-nlp\data\resume\pdfs"

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Read CSV
df = pd.read_csv(csv_file)

for i, row in df.iterrows():
    # Replace spaces with underscores
    category = row['Category'].replace(' ', '_')
    
    # Remove any character not allowed in filenames (Windows safe)
    category = re.sub(r'[<>:"/\\|?*]', '', category)
    
    filename = f"{category}_{i+1}.pdf"
    filepath = os.path.join(output_folder, filename)
    
    c = canvas.Canvas(filepath, pagesize=letter)
    text = c.beginText(40, 750)
    text.setFont("Helvetica", 12)
    
    text.textLine(f"Category: {row['Category']}")
    text.textLine("Resume:")
    text.textLine("")
    
    # Wrap text for readability
    for paragraph in row['Resume'].split("\n"):
        wrapped = textwrap.wrap(paragraph, width=90)  # width now applies
        for line in wrapped:
            text.textLine(line)
        text.textLine("")  # blank line between paragraphs

    c.drawText(text)
    c.save()

print(f"{len(df)} resumes converted to PDFs in {output_folder}")
