import os
import json
import torch
import re
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification

# 🔗 IMPORT YOUR RULE-BASED MODULE
try:
    from entities import extract_entities
except ImportError:
    try:
        from src.parsing.entities import extract_entities
    except ImportError:
        # Fallback for different folder structures
        from src.entities import extract_entities

# ==============================================================================
# 1. THE BERT MODEL ENGINE (Using Training Logic)
# ==============================================================================
class BertExtractor:
    def __init__(self, model_path="models/final_model"):
        print(f"🤖 Loading BERT Model from {model_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Running on: {self.device}")

        try:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            self.model = DistilBertForTokenClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.label2id = self.model.config.label2id
            self.id2label = self.model.config.id2label
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None

    def predict(self, text):
        if not self.model: return {}
        
        # 1. Clean Text (Crucial for model context)
        text = text.strip()

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
        offset_mapping = inputs.pop("offset_mapping")[0].cpu()
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu()
        predicted_labels = [self.id2label[p.item()] for p in predictions]
        
        entities = []
        current_ent = None
        
        for label, (start, end) in zip(predicted_labels, offset_mapping):
            if start == end == 0: continue
            
            tag = label[2:] if label != 'O' else 'O'
            prefix = label[0]
            
            # 🧠 SMART MERGE LOGIC 🧠
            should_merge = False
            if current_ent and current_ent['label'] == tag:
                # If it's an 'I' tag -> Merge
                if prefix == 'I':
                    should_merge = True
                # If it's a 'B' tag BUT it's right next to the previous one (e.g. "Abhishek" "Jha") -> Merge
                elif prefix == 'B' and (start - current_ent['end'] <= 1):
                    should_merge = True

            if should_merge:
                current_ent['text'] = text[current_ent['start']:end]
                current_ent['end'] = end
            
            # Start New Entity
            elif prefix == 'B' or (prefix == 'I' and not current_ent):
                if current_ent: entities.append(current_ent)
                current_ent = {
                    'label': tag, 
                    'text': text[start:end],
                    'start': start,
                    'end': end
                }
            else:
                if current_ent: entities.append(current_ent)
                current_ent = None
                
        if current_ent: entities.append(current_ent)
        
        # 3. Cleaning
        clean_results = {"Name": [], "Company": [], "Designation": [], "Location": [], "College": [], "Degree": []}
        
        # Extended Junk list from our optimization
        junk_edges = [
            'is', 'a', 'an', 'the', 'at', 'in', 'of', 'to', 'from', 'with', 'and', '&', 
            'based', 'works', 'served', 'serves', 'as', 'formerly', 'currently', 'previously',
            'holds', 'earned', 'completed', 'graduated', 'her', 'his', 'their', 'my', 'our',
            '.', ',', ':', '-', 'based in'
        ]
        bad_phrases = ["based in", "works as", "serves as", "acts as", "holds a", "graduated with", "completed her", "completed his"]

        for e in entities:
            txt = e['text'].strip()
            
            # Remove Titles
            if e['label'] == 'Name':
                for t in ["dr. ", "mr. ", "ms. ", "prof. "]: 
                    if txt.lower().startswith(t): txt = txt[len(t):]

            # Aggressive Scrubbing
            dirty = True
            while dirty:
                dirty = False
                txt_lower = txt.lower()
                
                for j in junk_edges:
                    if txt_lower.endswith(f" {j}") or txt_lower.endswith(j):
                        if txt_lower.endswith(j): txt = txt[:-len(j)].strip()
                        else: txt = txt[:-len(j)-1].strip()
                        dirty = True; txt_lower = txt.lower()
                    if txt_lower.startswith(f"{j} ") or txt_lower == j:
                        txt = txt[len(j)+1:].strip(); dirty = True; txt_lower = txt.lower()
                if ". " in txt: txt = txt.split(". ")[0]; dirty = True

            # Final check for bad phrases
            for phrase in bad_phrases:
                if phrase == txt.lower(): txt = ""

            if len(txt) > 1 and e['label'] in clean_results:
                if txt.lower() not in junk_edges:
                    clean_results[e['label']].append(txt)
                
        return {k: list(set(v)) for k,v in clean_results.items()}

# ==============================================================================
# 2. THE HYBRID PARSER
# ==============================================================================
class ResumeParser:
    def __init__(self):
        self.bert = BertExtractor()

    # ✅ RENAMED FROM 'predict' TO 'parse' (This fixes your error!)
    def parse(self, text):
        print("⚡ Parsing Resume...")
        
        # A. Run Rules
        rule_data = extract_entities(text)
        
        # B. Run BERT
        bert_ents = self.bert.predict(text)
        
        # C. Smart Merge
        raw_degrees = list(set(bert_ents.get("Degree", []) + rule_data.get("DEGREE", [])))
        final_degrees = []
        
        # Sort by length descending
        raw_degrees.sort(key=len, reverse=True)
        
        for d in raw_degrees:
            if not any(d in existing for existing in final_degrees):
                final_degrees.append(d)

        final_profile = {
            "Name": bert_ents.get("Name", []),
            "Email": rule_data.get("EMAIL", []),
            "Phone": rule_data.get("PHONE", []),
            "Designation": bert_ents.get("Designation", []),
            "Company": bert_ents.get("Company", []),
            "Location": bert_ents.get("Location", []),
            "College": bert_ents.get("College", []),
            "Degree": final_degrees,
            "Skills": []
        }
        
        skills_dict = rule_data.get("SKILLS", {})
        for category, skill_list in skills_dict.items():
            final_profile["Skills"].extend(skill_list)
            
        # Sanity Check for Name splits
        names = final_profile["Name"]
        locations = final_profile["Location"]
        for loc in locations[:]:
            for name in names:
                combined = f"{name} {loc}"
                if combined in text:
                    if combined not in final_profile["Name"]:
                        final_profile["Name"].append(combined)
                    if name in final_profile["Name"]: final_profile["Name"].remove(name)
                    if loc in final_profile["Location"]: final_profile["Location"].remove(loc)

        return final_profile

if __name__ == "__main__":
    parser = ResumeParser()
    sample_text = """
    Abhishek Jha
    Email: abhishek@gmail.com
    Software Engineer at Microsoft in Seattle.
    Skills: Python, Java, Docker, Generative AI.
    Education: B.S. in Computer Science from Stanford University.
    """
    print(json.dumps(parser.parse(sample_text), indent=2))