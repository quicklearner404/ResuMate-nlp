import json
import re
from collections import Counter

def advanced_clean_text(text):
    """Advanced text cleaning for CRF training"""
    # Replace bullet points and special chars
    bullet_chars = {'•', '❖', '●', '➢', '〓', '…'}
    for char in bullet_chars:
        text = text.replace(char, ' ')
    
    # Normalize whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    
    # Remove URLs and email patterns (keep the text, remove the pattern)
    text = re.sub(r'https?://\S+', '[URL]', text)
    text = re.sub(r'www\.\S+', '[URL]', text)
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    text = re.sub(r'indeed\.com/r/\S+', '[INDEED_PROFILE]', text)
    
    return text.strip()

def validate_entity_boundaries(text, entities):
    """Ensure entity boundaries make sense and aren't truncated"""
    valid_entities = []
    
    for start, end, label in entities:
        entity_text = text[start:end].strip()
        
        # Skip if entity is empty
        if not entity_text:
            continue
            
        # Skip if entity is too long (likely wrong boundary)
        if len(entity_text) > 200:
            continue
            
        # Skip if entity ends with incomplete word
        if entity_text and entity_text[-1].isalnum() and end < len(text) and text[end].isalnum():
            # Entity cuts off in middle of word - try to extend to word boundary
            extended_end = end
            while extended_end < len(text) and text[extended_end].isalnum():
                extended_end += 1
            entity_text = text[start:extended_end].strip()
            end = extended_end
        
        # Skip common problematic patterns
        if any(bad in entity_text for bad in ['\n', '...', '• •']):
            continue
            
        valid_entities.append((start, end, label))
    
    return valid_entities

def split_long_entities(text, entities):
    """Split very long entities (like skills lists) into smaller ones"""
    split_entities = []
    
    for start, end, label in entities:
        entity_text = text[start:end]
        
        # If it's a skills entity and too long, try to split
        if label == 'Skills' and len(entity_text) > 100:
            # Split by commas or common separators
            skills = re.split(r'[,•]|\band\b', entity_text)
            current_pos = start
            
            for skill in skills:
                skill = skill.strip()
                if skill and len(skill) > 2:  # Meaningful skill
                    skill_start = text.find(skill, current_pos)
                    if skill_start != -1:
                        skill_end = skill_start + len(skill)
                        split_entities.append((skill_start, skill_end, label))
                        current_pos = skill_end
        else:
            split_entities.append((start, end, label))
    
    return split_entities

def prepare_crf_training_data(file_path):
    """Prepare clean data ready for CRF training"""
    training_data = []
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line_num, line in enumerate(file, 1):
            try:
                data = json.loads(line)
                original_text = data['content']
                
                # Clean the text
                cleaned_text = advanced_clean_text(original_text)
                
                # Extract and validate entities
                entities = []
                for annotation in data['annotation']:
                    label_list = annotation['label']
                    if not label_list:
                        continue
                    
                    label = label_list[0]
                    points = annotation['points'][0]
                    start = points['start']
                    end = points['end']
                    
                    # Adjust positions for cleaned text
                    original_entity = original_text[start:end]
                    cleaned_start = cleaned_text.find(original_entity)
                    if cleaned_start != -1:
                        cleaned_end = cleaned_start + len(original_entity)
                        entities.append((cleaned_start, cleaned_end, label))
                
                # Validate and clean entity boundaries
                entities = validate_entity_boundaries(cleaned_text, entities)
                
                # Split long entities
                entities = split_long_entities(cleaned_text, entities)
                
                if entities:
                    training_data.append((cleaned_text, {"entities": entities}))
                    
            except Exception as e:
                print(f"⚠️ Skipped resume {line_num}: {e}")
                continue
    
    return training_data

def analyze_crf_readiness(training_data):
    """Analyze if data is ready for CRF training"""
    print("📊 CRF TRAINING READINESS ANALYSIS")
    print("=" * 50)
    
    entity_stats = Counter()
    entity_lengths = []
    
    for text, annotations in training_data:
        for start, end, label in annotations['entities']:
            entity_text = text[start:end]
            entity_stats[label] += 1
            entity_lengths.append(len(entity_text))
    
    print(f"Total training samples: {len(training_data)}")
    print(f"Total entities: {sum(entity_stats.values())}")
    print(f"Average entity length: {sum(entity_lengths)/len(entity_lengths):.1f} chars")
    
    print("\n🏷️ Entity distribution:")
    for label, count in entity_stats.most_common():
        print(f"  {label}: {count}")
    
    # CRF readiness check
    min_entities_per_class = 50
    ready_classes = [label for label, count in entity_stats.items() if count >= min_entities_per_class]
    
    print(f"\n🎯 CRF Readiness: {len(ready_classes)}/{len(entity_stats)} entity types have ≥{min_entities_per_class} examples")
    print("Ready classes:", ready_classes)

# Main execution
if __name__ == "__main__":
    print("🧹 PREPARING DATA FOR CRF TRAINING...")
    crf_data = prepare_crf_training_data("entities2.json")
    
    print(f"✅ Prepared {len(crf_data)} clean resumes for CRF")
    
    # Analyze readiness
    analyze_crf_readiness(crf_data)
    
    # Show sample of cleaned data
    if crf_data:
        print("\n🔍 SAMPLE CLEANED DATA:")
        text, entities = crf_data[0]
        print(f"Text length: {len(text)} chars")
        print(f"Entities: {len(entities['entities'])}")
        
        print("\nSample entities:")
        for start, end, label in entities['entities'][:8]:
            entity_text = text[start:end]
            print(f"  {label}: '{entity_text}'")