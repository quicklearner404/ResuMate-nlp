import re
import json
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter

# Expanded technical vocabulary
TECH_SKILLS = {
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'php', 'rust',
    'scala', 'swift', 'kotlin', 'r', 'matlab', 'html', 'css', 'sql', 'mysql', 'postgresql',
    'mongodb', 'redis', 'oracle', 'react', 'angular', 'vue', 'django', 'flask', 'spring',
    'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'jenkins', 'git', 'linux', 'machine learning',
    'data science', 'ai', 'nlp', 'computer vision', 'tensorflow', 'pytorch', 'pandas', 'numpy'
}

def enhanced_word2features(sent, i):
    """Enhanced features for all entities"""
    word = sent[i][0]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.length': len(word),
        
        # Technical features (for skills)
        'is_technical': word.lower() in TECH_SKILLS,
        
        # Education features
        'is_degree_word': word.lower() in {'bachelor', 'master', 'phd', 'b.tech', 'm.tech', 'be', 'b.e', 'mca', 'bsc', 'msc'},
        'is_education': word.lower() in {'university', 'college', 'institute', 'school', 'academy'},
        'is_year': word.isdigit() and len(word) == 4 and 1900 < int(word) < 2030,
        
        # Company/Designation features
        'is_company_indicator': word.lower() in {'inc', 'ltd', 'corp', 'corporation', 'technologies', 'solutions'},
        'is_designation': word.lower() in {'engineer', 'developer', 'manager', 'analyst', 'specialist', 'consultant'},
        
        # Context windows
        'prev_word': sent[i-1][0].lower() if i > 0 else '[START]',
        'next_word': sent[i+1][0].lower() if i < len(sent)-1 else '[END]',
    }
    
    # Extended context
    if i > 0:
        features.update({
            '-1:word.lower()': sent[i-1][0].lower(),
            '-1:word.istitle()': sent[i-1][0].istitle(),
            '-1:word.isdigit()': sent[i-1][0].isdigit(),
        })
    if i > 1:
        features.update({
            '-2:word.lower()': sent[i-2][0].lower(),
        })
    if i < len(sent)-1:
        features.update({
            '+1:word.lower()': sent[i+1][0].lower(),
            '+1:word.istitle()': sent[i+1][0].istitle(),
            '+1:word.isdigit()': sent[i+1][0].isdigit(),
        })
    if i < len(sent)-2:
        features.update({
            '+2:word.lower()': sent[i+2][0].lower(),
        })
        
    return features

def prepare_complete_training_data(file_path):
    """Prepare data that maintains all entity types"""
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
    
    return training_data

def sent2features(sent):
    return [enhanced_word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

def prepare_crf_data(training_data):
    """Convert to CRF format - keep all entities"""
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

def predict_complete_entities(text, crf_model):
    """Predict all entities using CRF"""
    tokens = text.split()
    if not tokens:
        return []
    
    features = [enhanced_word2features([(t, 'O') for t in tokens], i) for i in range(len(tokens))]
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

# Main execution - Focus on ALL entities
if __name__ == "__main__":
    print("🚀 TRAINING COMPLETE ENTITY MODEL...")
    
    # Load complete data
    training_data = prepare_complete_training_data("entities2.json")
    labeled_sentences = prepare_crf_data(training_data)
    
    print(f"Prepared {len(labeled_sentences)} labeled sentences")
    
    # Train-test split
    train_sentences, test_sentences = train_test_split(labeled_sentences, test_size=0.2, random_state=42)
    X_train = [sent2features(s) for s in train_sentences]
    y_train = [sent2labels(s) for s in train_sentences]
    X_test = [sent2features(s) for s in test_sentences]
    y_test = [sent2labels(s) for s in test_sentences]
    
    # Train CRF
    print("Training CRF model...")
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    
    crf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = crf.predict(X_test)
    y_test_flat = [label for sent in y_test for label in sent]
    y_pred_flat = [label for sent in y_pred for label in sent]
    
    print("\n📊 COMPLETE MODEL PERFORMANCE:")
    print(classification_report(y_test_flat, y_pred_flat, zero_division=0))
    
    # Test on comprehensive sample
    print("\n🔍 TESTING ON COMPREHENSIVE SAMPLE:")
    test_text = "John Doe is a Python Developer at Google. He has a Bachelor of Science from Stanford University in 2020. Skills include Java, React, and Docker."
    
    entities = predict_complete_entities(test_text, crf)
    
    print("Input:", test_text)
    print("Extracted Entities:")
    for entity in entities:
        print(f"  {entity['label']}: '{entity['text']}'")
    
    # Show entity distribution in predictions
    entity_counts = Counter()
    for label in y_pred_flat:
        if label != 'O':
            entity_counts[label] += 1
    
    print(f"\n🏷️ ENTITIES FOUND IN TEST SET:")
    for label, count in entity_counts.most_common():
        print(f"  {label}: {count}")