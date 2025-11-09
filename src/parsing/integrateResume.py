import os
import pandas as pd
import json
import re
from collections import Counter
import sklearn_crfsuite
from sklearn.model_selection import train_test_split

print("🚀 INTEGRATION WITH CORRECT PATHS")
print("=" * 50)

class FinalIntegrator:
    def __init__(self):
        self.crf_model = None
        
    def load_parsed_resumes(self):
        """Load the parsed resumes CSV file from the correct location"""
        # The exact path we found
        file_path = "../../data/resume/parsed_resumes.csv"  # From src/parsing to data/resume
        
        print(f"🔍 Loading from: {file_path}")
        print(f"📁 Absolute path: {os.path.abspath(file_path)}")
        
        if not os.path.exists(file_path):
            print("❌ File not found! Let me check what's available...")
            # Check what files exist in that folder
            folder_path = "../../data/resume"
            if os.path.exists(folder_path):
                print(f"📂 Files in {folder_path}:")
                for item in os.listdir(folder_path):
                    print(f"  - {item}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            print(f"✅ SUCCESS! Loaded {len(df)} resumes")
            print(f"📋 Columns: {list(df.columns)}")
            
            # Show detailed sample
            if len(df) > 0:
                print(f"\n📄 SAMPLE OF FIRST RESUME:")
                first_row = df.iloc[0]
                for col in df.columns:
                    value = str(first_row[col])
                    if len(value) > 150:
                        value = value[:150] + "..."
                    print(f"  {col}: {value}")
                    print()
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return pd.DataFrame()
    
    def analyze_dataset(self, df):
        """Comprehensive analysis of the dataset"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE DATASET ANALYSIS")
        print("="*60)
        
        print(f"📈 Dataset Overview:")
        print(f"  Total Resumes: {len(df)}")
        
        # Check the structure of the data
        if 'entities' in df.columns:
            print(f"\n🔍 ANALYZING ENTITIES STRUCTURE:")
            
            # Count resumes with different entity types
            has_skills = 0
            has_emails = 0
            has_job_titles = 0
            
            for idx, row in df.iterrows():
                try:
                    if pd.notna(row['entities']):
                        entities_dict = eval(row['entities'])
                        if 'SKILLS' in entities_dict and entities_dict['SKILLS']:
                            has_skills += 1
                        if 'EMAIL' in entities_dict and entities_dict['EMAIL']:
                            has_emails += 1
                        if 'JOB_TITLE' in entities_dict and entities_dict['JOB_TITLE']:
                            has_job_titles += 1
                except:
                    continue
            
            print(f"  Resumes with skills: {has_skills}/{len(df)} ({has_skills/len(df)*100:.1f}%)")
            print(f"  Resumes with emails: {has_emails}/{len(df)} ({has_emails/len(df)*100:.1f}%)")
            print(f"  Resumes with job titles: {has_job_titles}/{len(df)} ({has_job_titles/len(df)*100:.1f}%)")
        
        # Skills analysis
        if 'entities' in df.columns:
            print(f"\n🛠️ SKILLS BREAKDOWN:")
            
            skill_categories = Counter()
            total_skills = 0
            
            for idx, row in df.iterrows():
                try:
                    if pd.notna(row['entities']):
                        entities_dict = eval(row['entities'])
                        skills = entities_dict.get('SKILLS', {})
                        
                        for category, skill_list in skills.items():
                            skill_categories[category] += len(skill_list)
                            total_skills += len(skill_list)
                except:
                    continue
            
            print(f"  Total skills identified: {total_skills}")
            print(f"  Average skills per resume: {total_skills/len(df):.1f}")
            
            for category, count in skill_categories.most_common():
                category_name = category.replace('_', ' ').title()
                print(f"  {category_name}: {count} skills")
        
        # Category analysis
        if 'category' in df.columns:
            print(f"\n🎯 CATEGORY DISTRIBUTION:")
            categories = df['category'].value_counts()
            for category, count in categories.items():
                percentage = (count / len(df)) * 100
                print(f"  {category}: {count} resumes ({percentage:.1f}%)")
    
    def load_and_train_crf(self):
        """Load and train CRF model"""
        print("\n🤖 Training CRF model...")
        
        if not os.path.exists("entities2.json"):
            print("❌ entities2.json not found in current directory")
            return None
        
        training_data = []
        with open("entities2.json", 'r', encoding='utf-8', errors='replace') as file:
            for line in file:
                data = json.loads(line)
                original_text = data['content']
                
                text = re.sub(r'[•❖●➢〓…]', ' ', original_text)
                text = re.sub(r'\s+', ' ', text)
                
                entities = []
                for annotation in data['annotation']:
                    label_list = annotation['label']
                    if not label_list:
                        continue
                    
                    label = label_list[0]
                    if label in ['Name', 'Degree', 'College Name', 'Designation', 'Companies worked at', 'Location']:
                        points = annotation['points'][0]
                        start, end = points['start'], points['end']
                        
                        original_entity = original_text[start:end]
                        cleaned_start = text.find(original_entity)
                        if cleaned_start != -1:
                            entities.append((cleaned_start, cleaned_start + len(original_entity), label))
                
                if entities:
                    training_data.append((text, {"entities": entities}))
        
        # Train CRF model
        labeled_sentences = self.prepare_crf_data(training_data)
        train_sentences, _ = train_test_split(labeled_sentences, test_size=0.2, random_state=42)
        X_train = [self.sent2features(s) for s in train_sentences]
        y_train = [self.sent2labels(s) for s in train_sentences]
        
        self.crf_model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        
        self.crf_model.fit(X_train, y_train)
        print("✅ CRF model trained successfully!")
        return self.crf_model
    
    def word2features(self, sent, i):
        word = sent[i][0]
        features = {
            'bias': 1.0, 'word.lower()': word.lower(),
            'word.isupper()': word.isupper(), 'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(), 'word.length': len(word),
            'is_degree_word': word.lower() in {'bachelor', 'master', 'phd', 'b.tech', 'm.tech'},
            'is_education': word.lower() in {'university', 'college', 'institute', 'school'},
            'prev_word': sent[i-1][0].lower() if i > 0 else '[START]',
            'next_word': sent[i+1][0].lower() if i < len(sent)-1 else '[END]',
        }
        return features
    
    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]
    
    def sent2labels(self, sent):
        return [label for token, label in sent]
    
    def prepare_crf_data(self, training_data):
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
            
            labeled_sentences.append([(token, label) for token, label in zip(tokens, labels)])
        
        return labeled_sentences

def main():
    integrator = FinalIntegrator()
    
    print("📖 Loading parsed resumes...")
    df = integrator.load_parsed_resumes()
    
    if df.empty:
        print("\n❌ Could not load the dataset.")
        return
    
    # Analyze the dataset
    integrator.analyze_dataset(df)
    
    # Train CRF model
    print("\n🤖 Training CRF model for enhanced entity extraction...")
    crf_model = integrator.load_and_train_crf()
    
    if crf_model:
        print(f"\n🎉 SUCCESS! Your system is fully integrated.")
        print(f"📊 You have {len(df)} resumes loaded and analyzed.")
        print(f"🤖 CRF model is trained and ready.")
        print(f"\n🚀 Next: You can now run skill gap analysis on all {len(df)} resumes!")
    else:
        print(f"\n⚠️  Dataset loaded but CRF training failed.")
        print(f"📊 You still have {len(df)} resumes ready for analysis.")

if __name__ == "__main__":
    main()