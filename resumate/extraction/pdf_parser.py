import re
import unicodedata

import pdfplumber

from resumate.extraction.hybrid_parser import ResumeParser

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
        return ""
    return text

def advanced_clean_text(text: str) -> str:
    if not text: return ""
    text = unicodedata.normalize('NFKC', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    
    # Force split on pipes/dashes
    text = text.replace('|', '\n')
    text = text.replace('–', '-') 
    text = text.replace('—', '-') 
    
    # Fix Bullets
    text = re.sub(r'[\u2022\u2023\u25E6\u2043\u2219\*\>]', '\n', text)
    text = re.sub(r' {3,}', '\n', text) 

    lines = text.split('\n')
    cleaned_lines = []
    headers = ["EXPERIENCE", "EDUCATION", "SKILLS", "PROJECTS", "SUMMARY", "WORK HISTORY"]

    for line in lines:
        line = line.strip()
        if not line: continue 
        upper_line = line.upper()
        # Header Detection
        is_header = any(h == upper_line for h in headers) or (any(h in upper_line for h in headers) and len(line) < 30)
        
        if is_header:
            cleaned_lines.append("")     
            cleaned_lines.append(line)   
            cleaned_lines.append("")     
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def scrub_entities_aggressive(entities):
    junk_words = [
        'is', 'was', 'are', 'were', 'in', 'of', 'to', 'from', 'with', 'for', 'on', 'by', 'as',
        'and', 'or', '&', 'current', 'present', 'a', 'an', 'the', 'he', 'she', 'his', 'her', 'my',
        'dr.', 'mr.', 'ms.', 'prof.', 'graduated', 'completed', 'earned', 'obtained',
        'works', 'worked', 'working', 'job', 'position', 'role', 'senior', 'junior'
    ]
    cleaned_entities = {}
    for label, items in entities.items():
        clean_items = []
        for txt in items:
            if not txt: continue
            txt = txt.strip()
            words = txt.split()
            while words and words[0].lower() in junk_words: words.pop(0)
            while words and words[-1].lower() in junk_words: words.pop()
            
            clean_txt = " ".join(words)
            if len(clean_txt) > 2 and clean_txt.lower() not in junk_words:
                if clean_txt not in clean_items:
                    clean_items.append(clean_txt)
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

    # 2. Company/Role Hunter
    in_experience = False
    
    # Expanded Keyword Lists
    company_indicators = ['SOLUTIONS', 'SYSTEMS', 'TECHNOLOGIES', 'SERVICES', 'DIGITAL', 'LABS', 'INC', 'LLC', 'GROUP', 'CORP', 'LIMITED', 'PVT', 'AGENCY', 'STUDIO']
    role_indicators = ['ENGINEER', 'DEVELOPER', 'MANAGER', 'ANALYST', 'CONSULTANT', 'ADMIN', 'SPECIALIST', 'DESIGNER', 'DIRECTOR', 'HEAD', 'LEAD', 'SCIENTIST', 'ARCHITECT']
    
    # 🚀 FIX 1: Word Boundaries (\b) prevent "Jun" deleting "June" -> "e"
    date_pattern = r'\b(20\d\d|Present|Current|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\b'

    for i, line in enumerate(lines):
        upper_line = line.upper()
        
        if "EXPERIENCE" in upper_line:
            in_experience = True
            continue
        
        if in_experience:
            if line.isupper() and len(line) < 20: break 
            if line.endswith('.'): continue
            
            # Remove Dates cleanly
            clean_line = re.sub(date_pattern, '', line, flags=re.IGNORECASE).strip()
            clean_line = clean_line.strip('-,| ')
            
            if len(clean_line) < 3: continue
            upper_clean = clean_line.upper()

            # A. Company Check
            if any(ind in upper_clean for ind in company_indicators):
                if 'Company' not in entities: entities['Company'] = []
                if clean_line not in entities['Company']:
                    entities['Company'].append(clean_line)
                    
                    #  FIX 2: Look BACKWARDS for the Role
                    # If we found a Company, check the previous line for the Role
                    if i > 0:
                        prev_line = lines[i-1].strip()
                        prev_clean = re.sub(date_pattern, '', prev_line, flags=re.IGNORECASE).strip('-,| ')
                        
                        # If previous line has a Role Keyword or is Title Case
                        if any(ind in prev_clean.upper() for ind in role_indicators) or (prev_clean[0].isupper() and len(prev_clean.split()) < 5):
                            if 'Designation' not in entities: entities['Designation'] = []
                            if prev_clean not in entities['Designation'] and prev_clean != clean_line:
                                entities['Designation'].append(prev_clean)

            # B. Role Check
            elif any(ind in upper_clean for ind in role_indicators):
                if 'Designation' not in entities: entities['Designation'] = []
                if clean_line not in entities['Designation']:
                    entities['Designation'].append(clean_line)
                    
                    # C. Look FORWARD for the Company (Existing Logic)
                    if i + 1 < len(lines):
                        next_line = lines[i+1]
                        next_clean = re.sub(date_pattern, '', next_line, flags=re.IGNORECASE).strip('-,| ')
                        if next_clean and next_clean[0].isupper() and len(next_clean.split()) < 6:
                            # Only add if it's NOT a role
                            if not any(ind in next_clean.upper() for ind in role_indicators):
                                if 'Company' not in entities: entities['Company'] = []
                                if next_clean not in entities['Company']:
                                    entities['Company'].append(next_clean)

    return entities

_PARSER = None


def get_parser():
    """Load the DistilBERT parser once and reuse it across resumes."""
    global _PARSER
    if _PARSER is None:
        _PARSER = ResumeParser()
    return _PARSER


def process_resume(file_obj):
    raw_text = parse_pdf_robust(file_obj)
    if not raw_text: return "", {"Error": "Empty PDF"}

    cleaned_text = advanced_clean_text(raw_text)

    try:
        entities = get_parser().parse(cleaned_text)
    except Exception:
        entities = {}

    entities = scrub_entities_aggressive(entities)
    entities = smart_entity_fallback(cleaned_text, entities)

    return cleaned_text, entities