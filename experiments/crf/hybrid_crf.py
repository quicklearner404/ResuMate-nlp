import os
import re
import json
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from collections import Counter

# Always import CRF functions at top
from src.parsing.train_crf import get_crf_model, predict_complete_entities

# Always import rule-based extractor at top
from src.parsing.entities import extract_entities

print("🚀 INITIALIZING HYBRID RESUME PARSER...")

# Check if entities.py exists and import it
try:
    from src.parsing.entities import extract_entities
    print("✅ Successfully imported your rule-based system from entities.py")
except ImportError:
    print("❌ Could not import entities.py - creating mock function")
    # Mock function as fallback
    def extract_entities(text):
        return {
            "EMAIL": re.findall(r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+", text),
            "PHONE": re.findall(r"(\+?\d{1,3}[-\s]?)?\(?\d{2,4}\)?[-\s]?\d{3,4}[-\s]?\d{3,4}", text),
            "JOB_TITLE": [],
            "SKILLS": {
                "PROGRAMMING_LANGUAGE": ["python", "java", "javascript"],
                "FRAMEWORK": ["react", "django"],
                "DATABASE": ["mysql", "mongodb"],
                "CLOUD": ["aws", "docker"],
                "TOOL": ["git", "jenkins"]
            }
        }

def train_crf_for_structured_entities(file_path):
    """Train CRF only on entities it's good at"""
    print("📚 Loading training data...")
    training_data = []
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            data = json.loads(line)
            original_text = data['content']
            
            # Clean text
            text = re.sub(r'[•❖●➢〓…]', ' ', original_text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n+', ' ', text)
            
            entities = []
            for annotation in data['annotation']:
                label_list = annotation['label']
                if not label_list:
                    continue
                
                label = label_list[0]
                # Train on entities that CRF works well with
                if label in ['Name', 'Degree', 'College Name', 'Designation', 'Companies worked at', 'Location', 'Email Address']:
                    points = annotation['points'][0]
                    start = points['start']
                    end = points['end']
                    
                    original_entity = original_text[start:end]
                    cleaned_start = text.find(original_entity)
                    if cleaned_start != -1:
                        cleaned_end = cleaned_start + len(original_entity)
                        entities.append((cleaned_start, cleaned_end, label))
            
            if entities:
                training_data.append((text, {"entities": entities}))
    
    print(f"✅ Loaded {len(training_data)} training samples")
    return training_data

def word2features(sent, i):
    """Enhanced features for structured entities"""
    word = sent[i][0]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.length': len(word),
        
        # Enhanced education features
        'is_degree_word': word.lower() in {'bachelor', 'master', 'phd', 'b.tech', 'm.tech', 'be', 'b.e', 'mca', 'bsc', 'msc', 'ba', 'ma'},
        'is_education_keyword': word.lower() in {'university', 'college', 'institute', 'school', 'academy', 'faculty', 'department'},
        'is_degree_abbr': word.lower() in {'bs', 'ms', 'phd', 'btech', 'mtech', 'bca', 'mca'},
        'is_year': word.isdigit() and len(word) == 4 and 1980 < int(word) < 2030,
        
        # Company features
        'is_company_indicator': word.lower() in {'inc', 'ltd', 'corp', 'corporation', 'technologies', 'solutions', 'company', 'llc'},
        
        # Enhanced context
        'word.contains_edu': 'edu' in word.lower(),
        'word.contains_univ': 'univ' in word.lower(),
        'word.contains_college': 'college' in word.lower(),
        'word.contains_tech': 'tech' in word.lower(),
        
        # Context windows
        'prev_word': sent[i-1][0].lower() if i > 0 else '[START]',
        'next_word': sent[i+1][0].lower() if i < len(sent)-1 else '[END]',
    }
    
    # Extended context for better accuracy
    if i > 0:
        features.update({
            '-1:word.lower()': sent[i-1][0].lower(),
            '-1:word.istitle()': sent[i-1][0].istitle(),
        })
    if i < len(sent)-1:
        features.update({
            '+1:word.lower()': sent[i+1][0].lower(),
            '+1:word.istitle()': sent[i+1][0].istitle(),
        })
    
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def prepare_crf_data(training_data):
    """Convert to CRF format"""
    print("🔄 Preparing CRF training data...")
    labeled_sentences = []
    
    for text, annotations in training_data:
        tokens = text.split()
        labels = ['O'] * len(tokens)
        
        for start, end, label in annotations['entities']:
            entity_text = text[start:end]
            
            current_pos = 0
            for i, token in enumerate(tokens):
                token_start = text.find(token, current_pos)
                if token_start == -1:
                    continue
                token_end = token_start + len(token)
                current_pos = token_end
                
                if (token_start >= start and token_start < end) or (token_end > start and token_end <= end):
                    if labels[i] == 'O':
                        if i == 0 or labels[i-1] != f'B-{label}':
                            labels[i] = f'B-{label}'
                        else:
                            labels[i] = f'I-{label}'
        
        sentence = [(token, label) for token, label in zip(tokens, labels)]
        labeled_sentences.append(sentence)
    
    return labeled_sentences

def extract_entities_with_crf(text, crf_model):
    """Use CRF to extract structured entities"""
    tokens = text.split()
    if not tokens:
        return []
    
    features = [word2features([(t, 'O') for t in tokens], i) for i in range(len(tokens))]
    predictions = crf_model.predict([features])[0]
    
    entities = []
    current_entity = None
    
    for i, (token, label) in enumerate(zip(tokens, predictions)):
        if label.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            current_entity = {'text': token, 'label': label[2:], 'start': i, 'end': i}
        elif label.startswith('I-') and current_entity and current_entity['label'] == label[2:]:
            current_entity['text'] += ' ' + token
            current_entity['end'] = i
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

def analyze_resume(resume_text, crf_model):
    """Hybrid analysis - SIMPLIFIED VERSION"""
    if crf_model is None:
        crf_model = get_crf_model()  # only call once, no re-import
    
    cleaned_text = re.sub(r'\s+', ' ', resume_text)
    
    # Use rule-based system for skills
    rule_based_results = extract_entities(cleaned_text)
    
    # ✅ USE THE ENHANCED CRF PREDICTION FROM train_crf.py
    from src.parsing.train_crf import predict_complete_entities
    crf_entities = predict_complete_entities(cleaned_text, crf_model)
    
    
    # Combine results
    analysis = {
        'personal': {},
        'education': {},
        'experience': {},
        'skills': {}
    }
    
    # Process CRF entities with enhanced filtering
    for entity in crf_entities:
        label = entity['label']
        text = entity['text'].strip()
        
        # Enhanced filtering for false positives
        if label == 'Name':
            # Skip common false positives and invalid names
            false_positives = {
                'category', 'resume', 'email', 'phone', 'experience', 
                'education', 'skills', 'summary', 'objective', 'profile',
                'frontend', 'backend', 'developer', 'engineer', 'cloud'
            }
            
            if (text.lower() in false_positives or 
                text.endswith(':') or
                len(text) < 2 or
                text.isdigit() or
                any(char.isdigit() for char in text) or
                text.lower().startswith('category') or
                len(text.split()) > 3):  # Names usually 1-3 words
                continue
            analysis['personal']['name'] = text
            
        elif label == 'Location':
            analysis['personal']['location'] = text
        elif label == 'Degree':
            analysis['education']['degree'] = text
        elif label == 'College Name':
            analysis['education']['college'] = text
        elif label == 'Designation':
            analysis['experience']['designation'] = text
        elif label == 'Companies worked at':
            analysis['experience']['company'] = text
    
    # Add rule-based results
    analysis['personal']['email'] = rule_based_results.get('EMAIL', [])
    analysis['personal']['phone'] = rule_based_results.get('PHONE', [])
    analysis['experience']['job_titles'] = rule_based_results.get('JOB_TITLE', [])
    analysis['skills'] = rule_based_results.get('SKILLS', {})
    
    return analysis
    

def print_analysis(analysis):
    """Print formatted analysis"""
    print("\n" + "="*60)
    print(" RESUME ANALYSIS REPORT")
    print("="*60)
    
    print("\n PERSONAL INFORMATION:")
    if analysis['personal'].get('name'):
        print(f"  Name: {analysis['personal']['name']}")
    if analysis['personal'].get('email'):
        print(f"  Email: {', '.join(analysis['personal']['email'])}")
    if analysis['personal'].get('phone'):
        print(f"  Phone: {', '.join(analysis['personal']['phone'])}")
    if analysis['personal'].get('location'):
        print(f"  Location: {analysis['personal']['location']}")
    
    print("\n EDUCATION:")
    if analysis['education'].get('degree'):
        print(f"  Degree: {analysis['education']['degree']}")
    if analysis['education'].get('college'):
        print(f"  Institution: {analysis['education']['college']}")
    
    print("\n EXPERIENCE:")
    if analysis['experience'].get('designation'):
        print(f"  Designation: {analysis['experience']['designation']}")
    if analysis['experience'].get('company'):
        print(f"  Company: {analysis['experience']['company']}")
    if analysis['experience'].get('job_titles'):
        print(f"  Job Titles: {', '.join(analysis['experience']['job_titles'])}")
    
    print("\n TECHNICAL SKILLS:")
    skills_found = 0
    for category, skills in analysis['skills'].items():
        if skills:
            category_name = category.replace('_', ' ').title()
            print(f"  {category_name}: {', '.join(skills)}")
            skills_found += len(skills)
    
    print(f"\n📊 SKILLS SUMMARY: {skills_found} skills identified")
    print("="*60)


# Main execution
def main():
    print("🚀 HYBRID RESUME PARSER - LOADING MODEL")
    print("=" * 50)
    
    # 🎯 USE THE SAVED MODEL INSTEAD OF TRAINING
    from src.parsing.train_crf import get_crf_model
    crf_model = get_crf_model()  # This loads saved model or trains once
    
    print("✅ Model loaded successfully!")
    
    # Test with sample resumes (keep your existing test code)
    print("\n🔍 TESTING HYBRID SYSTEM")
    print("=" * 50)
    
    test_resumes = [
        """
        John Doe
        Email: john.doe@email.com | Phone: +1-555-123-4567
        Location: San Francisco, CA
        
        Senior Software Engineer at Google
        Previously: Backend Developer at Microsoft
        
        EDUCATION:
        Bachelor of Science in Computer Science
        Stanford University, 2020
        
        SKILLS:
        Programming: Python, Java, JavaScript, C++
        Frameworks: Django, React, Spring Boot, TensorFlow
        Databases: MySQL, PostgreSQL, MongoDB
        Cloud: AWS, Docker, Kubernetes
        Tools: Git, Jenkins, Terraform
        """,
        
        """
        Sarah Smith
        Software Developer
        Email: sarah.smith@tech.com
        Phone: 555-987-6543
        
        Currently working at Amazon as Full Stack Developer
        Education: Master of Computer Science from MIT
        
        Technical Skills:
        - Python, Java, React, Node.js
        - AWS, Azure, Docker
        - MySQL, MongoDB
        - Machine Learning, Data Science
        """
    ]
    
    for i, resume in enumerate(test_resumes, 1):
        print(f"\n📄 Resume {i} Analysis:")
        analysis = analyze_resume(resume, crf_model)
        print_analysis(analysis)

if __name__ == "__main__":
    main()