# clean_training_data.py
import json
import re
from collections import Counter

def clean_annotations(input_file, output_file):
    """Clean the messy annotations in your training data with detailed analysis"""
    
    print("🔄 Cleaning training data...")
    
    cleaned_data = []
    entity_stats = Counter()
    removed_annotations = 0
    total_annotations = 0
    input_resume_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            input_resume_count += 1  # Count all input resumes
            try:
                data = json.loads(line.strip())
                content = data['content']
                annotations = data['annotation']
                
                clean_annotations = []
                seen_entities = set()
                
                for ann in annotations:
                    label = ann['label'][0]
                    points = ann['points']
                    
                    for point in points:
                        total_annotations += 1
                        text = point['text'].strip()
                        
                        # Skip if empty or too long (not NER-friendly)
                        if not text:
                            removed_annotations += 1
                            continue
                            
                        if len(text) > 50:
                            removed_annotations += 1
                            continue
                        
                        # Skip skills annotations (they're not NER entities)
                        if label == "Skills":
                            removed_annotations += 1
                            continue
                        
                        # Fix wrong entity types
                        if label == "Designation" and any(degree_word in text.lower() for degree_word in ['b.','m.','phd','bachelor','master','degree','engineering']):
                            label = "Degree"
                        
                        # Skip non-email "email addresses"
                        if label == "Email Address" and '@' not in text:
                            removed_annotations += 1
                            continue
                        
                        # Skip year-only "graduation years" 
                        if label == "Graduation Year" and len(text.strip()) <= 4:
                            removed_annotations += 1
                            continue
                        
                        # Create clean annotation
                        entity_key = f"{label}:{text}"
                        if entity_key not in seen_entities:
                            clean_annotations.append({
                                "label": [label],
                                "points": [{"start": point['start'], "end": point['end'], "text": text}]
                            })
                            seen_entities.add(entity_key)
                            entity_stats[label] += 1
                        else:
                            removed_annotations += 1
            
                # Only keep if we have reasonable annotations
                if len(clean_annotations) >= 3:
                    cleaned_data.append({
                        "content": content,
                        "annotation": clean_annotations,
                        "extras": None
                    })
                    
            except Exception as e:
                print(f"❌ Error processing line {line_num}: {e}")
                continue
    
    # Save cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in cleaned_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Print detailed analysis - FIXED: using input_resume_count instead of line_num
    print(f"\n📊 DATA CLEANING REPORT:")
    print(f"   • Input resumes: {input_resume_count}")
    print(f"   • Cleaned resumes: {len(cleaned_data)}")
    print(f"   • Removed resumes: {input_resume_count - len(cleaned_data)}")
    print(f"   • Total annotations: {total_annotations}")
    print(f"   • Removed annotations: {removed_annotations}")
    print(f"   • Kept annotations: {total_annotations - removed_annotations}")
    
    print(f"\n📈 ENTITY DISTRIBUTION (cleaned data):")
    for entity_type, count in entity_stats.most_common():
        print(f"   • {entity_type}: {count}")
    
    # Calculate coverage
    if len(cleaned_data) > 0:
        avg_annotations = sum(len(item['annotation']) for item in cleaned_data) / len(cleaned_data)
        print(f"\n🎯 QUALITY METRICS:")
        print(f"   • Average annotations per resume: {avg_annotations:.1f}")
        print(f"   • Entity types covered: {len(entity_stats)}")
        
        # Check if we have enough data for training
        if len(cleaned_data) < 500:
            print(f"❌ INSUFFICIENT DATA: Only {len(cleaned_data)} resumes (need at least 500)")
        elif len(cleaned_data) < 1000:
            print(f"⚠️  LIMITED DATA: {len(cleaned_data)} resumes (recommended: 1000+)")
        else:
            print(f"✅ GOOD DATA: {len(cleaned_data)} resumes")
            
        # Check entity balance
        if entity_stats:
            min_entities = min(entity_stats.values())
            max_entities = max(entity_stats.values())
            if min_entities < 50:
                print(f"⚠️  IMBALANCED: Some entities have very few examples (<50)")
    
    return cleaned_data

def check_original_data_size(input_file):
    """Check how many resumes are in the original file"""
    print(f"\n🔍 CHECKING ORIGINAL DATA: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                line_count += 1
        
        print(f"   • Total resumes in original file: {line_count}")
        return line_count
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        return 0

# Run the cleaner with analysis
if __name__ == "__main__":
    input_file = "enhanced_tech_resumes.json"  # Your original data
    output_file = "cleaned_training_data.json"
    
    # First check original data size
    original_count = check_original_data_size(input_file)
    
    # Clean the data
    cleaned_data = clean_annotations(input_file, output_file)
    
    print(f"\n🎯 NEXT STEPS:")
    if len(cleaned_data) >= 500:
        print(f"✅ Ready to train! Use: python hybridBert_fixed.py")
        print(f"   Update training_file to: '{output_file}'")
    else:
        print(f"❌ Need more data! Options:")
        print(f"   1. Generate synthetic data first")
        print(f"   2. Find more labeled resumes")
        print(f"   3. Use data augmentation")