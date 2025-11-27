import sys
import os
import re
import pdfplumber
import unicodedata

# Ensure we can find your Champion Model
sys.path.append(os.getcwd())

try:
    from src.parsing.hybridBert import ResumeParser
    print("✅ Loaded Champion Logic from hybridBert")
except ImportError:
    try:
        from hybridBert import ResumeParser
    except:
        print("⚠️ Warning: Could not load hybridBert.")

def parse_pdf_robust(file_obj):
    text = ""
    try:
        file_obj.seek(0)
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text(x_tolerance=1, y_tolerance=1, layout=True)
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"❌ PDF Read Error: {e}")
        return ""
    return text

def advanced_clean_text(text: str) -> str:
    if not text: return ""
    text = unicodedata.normalize('NFKC', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    
    # Structure Fixes
    text = text.replace('|', '\n').replace('–', '\n').replace('—', '\n')
    text = re.sub(r'[\u2022\u2023\u25E6\u2043\u2219\*\>]', '\n', text)
    text = re.sub(r' {3,}', '\n', text)

    lines = text.split('\n')
    cleaned_lines = []
    headers = ["EXPERIENCE", "EDUCATION", "SKILLS", "PROJECTS", "SUMMARY", "WORK HISTORY"]

    for line in lines:
        line = line.strip()
        if not line: continue 
        upper_line = line.upper()
        
        # Header detection
        is_header = any(h == upper_line for h in headers) or (any(h in upper_line for h in headers) and len(line) < 30)
        
        if is_header:
            cleaned_lines.append("")     
            cleaned_lines.append(line)   
            cleaned_lines.append("")     
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def scrub_entities_aggressive(entities):
    """
    🧼 THE CLEANER: Removes junk words.
    """
    junk_words = [
        'is', 'was', 'are', 'were', 'at', 'in', 'of', 'to', 'from', 'with', 'for', 'on', 'by', 'as',
        'and', 'or', '&', 'current', 'present', 'a', 'an', 'the', 'he', 'she', 'his', 'her', 'my',
        'dr.', 'mr.', 'ms.', 'prof.', 'graduated', 'completed', 'earned', 'obtained',
        'works', 'worked', 'working', 'job', 'position', 'role'
    ]

    cleaned_entities = {}
    for label, items in entities.items():
        clean_items = []
        for txt in items:
            if not txt: continue
            txt = txt.strip()
            dirty = True
            while dirty:
                dirty = False
                txt_lower = txt.lower()
                for junk in junk_words:
                    if txt_lower.endswith(f" {junk}"):
                        txt = txt[:-len(junk)-1].strip()
                        dirty = True
                        txt_lower = txt.lower()
                    if txt_lower.startswith(f"{junk} "):
                        txt = txt[len(junk)+1:].strip()
                        dirty = True
                        txt_lower = txt.lower()

            if len(txt) < 2 and txt.lower() not in ['c', 'r']: continue
            if txt.lower() in junk_words: continue
            if txt not in clean_items: clean_items.append(txt)

        cleaned_entities[label] = clean_items
    return cleaned_entities

def smart_entity_fallback(text, entities):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    # 1. Name Fallback
    if not entities.get('Name'):
        for line in lines[:5]:
            if "@" in line:
                parts = line.split()
                potential_name = " ".join(parts[:2])
                if len(parts) > 1 and not any(c.isdigit() for c in potential_name):
                    entities['Name'] = [potential_name]
                    break
            words = line.split()
            if 2 <= len(words) <= 3 and not any(c.isdigit() for c in line) and not line.isupper():
                if "http" not in line and "phone" not in line.lower():
                    entities['Name'] = [line]
                    break

    # 2. Experience Fallback (Strict Description Filter)
    if not entities.get('Company') or not entities.get('Designation'):
        in_experience = False
        date_pattern = r'(20\d\d|Present|Current|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        # Words that start bullet points (Not Company Names!)
        verb_blocklist = ['LED', 'MANAGED', 'CREATED', 'BUILT', 'OPTIMIZED', 'DEVELOPED', 'ANALYZED', 'IMPLEMENTED', 'DESIGNED']

        for line in lines:
            if "EXPERIENCE" in line.upper():
                in_experience = True
                continue
            
            if in_experience:
                if line.isupper() and len(line) < 20: break # Hit next section
                
                # 🛑 FILTER 1: Skip Dates
                if re.search(date_pattern, line, re.IGNORECASE): continue
                
                # 🛑 FILTER 2: Skip Sentences (ending in period)
                if line.endswith('.'): continue
                
                # 🛑 FILTER 3: Skip Bullet Points (starts with Verb)
                first_word = line.split()[0].upper()
                if first_word in verb_blocklist: continue

                # 🛑 FILTER 4: Skip Long Lines (Companies usually < 6 words)
                if len(line.split()) > 7: continue

                # If it survived all filters, it's likely a Title or Company
                if len(line) > 3 and not "http" in line:
                    if not entities.get('Designation'):
                        entities['Designation'] = [line]
                    elif not entities.get('Company') and line != entities.get('Designation', [''])[0]:
                        entities['Company'] = [line]
                        # Don't break immediately, look for one more
    return entities

def process_resume(file_obj):
    print("🚀 Processing Resume...")
    
    # 1. Read
    raw_text = parse_pdf_robust(file_obj)
    if not raw_text:
        return "", {"Error": "Empty PDF"}
    
    # 2. Clean
    cleaned_text = advanced_clean_text(raw_text)
    
    # 3. Predict
    try:
        parser = ResumeParser()
        entities = parser.parse(cleaned_text)
    except Exception as e:
        print(f"❌ Model Error: {e}")
        return cleaned_text, {"Error": "Model Failed"}

    # 4. Scrub & Fallback
    entities = scrub_entities_aggressive(entities)
    entities = smart_entity_fallback(cleaned_text, entities)

    return cleaned_text, entities