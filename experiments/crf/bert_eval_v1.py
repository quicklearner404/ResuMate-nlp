"""
Proper F1 Evaluation Script for Resume NER
Calculates precision, recall, F1 for each entity type
"""

import json
import torch
from collections import defaultdict
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification

class ProperEvaluator:
    def __init__(self, model_path="models/improved_ner/final_model"):
        """Load trained model"""
        print(f"📥 Loading model from {model_path}...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForTokenClassification.from_pretrained(model_path)
        
        self.label2id = {
            'O': 0,
            'B-Name': 1, 'I-Name': 2,
            'B-Company': 3, 'I-Company': 4,
            'B-Degree': 5, 'I-Degree': 6,
            'B-College': 7, 'I-College': 8,
            'B-Location': 9, 'I-Location': 10,
            'B-Designation': 11, 'I-Designation': 12
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        print("✅ Model loaded!\n")
    
    def predict_entities(self, text):
        """
        Fixed prediction using character offsets
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop('offset_mapping')[0]
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=2)[0]
        predicted_labels = [self.id2label[pred.item()] for pred in predictions]
        
        entities = []
        current_entity = None
        
        for label, (start, end) in zip(predicted_labels, offset_mapping):
            if start == end == 0:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            token_text = text[start:end]
            
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'label': label[2:],
                    'text': token_text,
                    'start_char': start,
                    'end_char': end
                }
            elif label.startswith('I-') and current_entity:
                if current_entity['label'] == label[2:]:
                    current_entity['text'] = text[current_entity['start_char']:end]
                    current_entity['end_char'] = end
                else:
                    entities.append(current_entity)
                    current_entity = {
                        'label': label[2:],
                        'text': token_text,
                        'start_char': start,
                        'end_char': end
                    }
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        # Clean up
        final_entities = []
        for ent in entities:
            final_entities.append({
                'label': ent['label'],
                'text': ent['text'].strip().lower()  # Normalize for comparison
            })
        
        return final_entities
    
    def calculate_metrics(self, predicted, ground_truth, entity_type):
        """
        Calculate precision, recall, F1 for a specific entity type
        """
        # Extract entities of this type
        pred_entities = set([e['text'].lower() for e in predicted if e['label'] == entity_type])
        true_entities = set([e.lower() for e in ground_truth.get(entity_type, [])])
        
        # Calculate TP, FP, FN
        tp = len(pred_entities & true_entities)
        fp = len(pred_entities - true_entities)
        fn = len(true_entities - pred_entities)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def evaluate_on_test_set(self, test_file):
        """
        Evaluate on properly labeled test resumes
        
        Expected format:
        [
            {
                "text": "Ahmed Khan is a Software Engineer at Google...",
                "ground_truth": {
                    "Name": ["Ahmed Khan"],
                    "Designation": ["Software Engineer"],
                    "Company": ["Google"],
                    "Location": ["Lahore"],
                    "Degree": ["BS Computer Science"],
                    "College": ["LUMS"]
                }
            }
        ]
        """
        print(f"📂 Loading test set from {test_file}...")
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"📊 Evaluating on {len(test_data)} test resumes...\n")
        
        # Store metrics per entity type
        entity_types = ['Name', 'Company', 'Designation', 'Location', 'Degree', 'College']
        all_metrics = {et: [] for et in entity_types}
        
        # Evaluate each resume
        for i, test_case in enumerate(test_data, 1):
            text = test_case['text']
            ground_truth = test_case['ground_truth']
            
            # Predict
            predicted = self.predict_entities(text)
            
            # Calculate metrics for each entity type
            for entity_type in entity_types:
                metrics = self.calculate_metrics(predicted, ground_truth, entity_type)
                all_metrics[entity_type].append(metrics)
            
            # Show progress
            if i % 5 == 0:
                print(f"  Processed {i}/{len(test_data)} resumes...")
        
        # Calculate average metrics
        print("\n" + "="*70)
        print("📊 FINAL EVALUATION RESULTS")
        print("="*70)
        print(f"\n{'Entity Type':<15} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'TP':<6} {'FP':<6} {'FN':<6}")
        print("-"*70)
        
        overall_metrics = {}
        for entity_type in entity_types:
            metrics_list = all_metrics[entity_type]
            
            if metrics_list:
                avg_precision = sum(m['precision'] for m in metrics_list) / len(metrics_list)
                avg_recall = sum(m['recall'] for m in metrics_list) / len(metrics_list)
                avg_f1 = sum(m['f1'] for m in metrics_list) / len(metrics_list)
                total_tp = sum(m['tp'] for m in metrics_list)
                total_fp = sum(m['fp'] for m in metrics_list)
                total_fn = sum(m['fn'] for m in metrics_list)
                
                overall_metrics[entity_type] = {
                    'precision': avg_precision,
                    'recall': avg_recall,
                    'f1': avg_f1,
                    'tp': total_tp,
                    'fp': total_fp,
                    'fn': total_fn
                }
                
                print(f"{entity_type:<15} {avg_precision:<12.4f} {avg_recall:<12.4f} {avg_f1:<12.4f} {total_tp:<6} {total_fp:<6} {total_fn:<6}")
        
        # Calculate macro-averaged overall F1
        if overall_metrics:
            macro_precision = sum(m['precision'] for m in overall_metrics.values()) / len(overall_metrics)
            macro_recall = sum(m['recall'] for m in overall_metrics.values()) / len(overall_metrics)
            macro_f1 = sum(m['f1'] for m in overall_metrics.values()) / len(overall_metrics)
            
            print("-"*70)
            print(f"{'MACRO-AVERAGE':<15} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f}")
            print("="*70)
            
            return macro_f1, overall_metrics
        
        return 0.0, {}
    
    def show_detailed_examples(self, test_file, num_examples=5):
        """Show detailed predictions for first N examples"""
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print("\n" + "="*70)
        print("🔍 DETAILED EXAMPLES")
        print("="*70)
        
        for i, test_case in enumerate(test_data[:num_examples], 1):
            text = test_case['text']
            ground_truth = test_case['ground_truth']
            
            print(f"\n📄 Resume {i}:")
            print(f"Text: {text[:100]}...")
            
            predicted = self.predict_entities(text)
            
            print("\n  Ground Truth:")
            for entity_type, entities in ground_truth.items():
                if entities:
                    print(f"    {entity_type}: {entities}")
            
            print("\n  Predicted:")
            pred_by_type = defaultdict(list)
            for ent in predicted:
                pred_by_type[ent['label']].append(ent['text'])
            
            for entity_type, entities in pred_by_type.items():
                print(f"    {entity_type}: {entities}")
            
            # Show what was correct/wrong
            print("\n  Analysis:")
            for entity_type in ['Name', 'Company', 'Designation', 'Location', 'Degree', 'College']:
                pred_set = set([e['text'].lower() for e in predicted if e['label'] == entity_type])
                true_set = set([e.lower() for e in ground_truth.get(entity_type, [])])
                
                correct = pred_set & true_set
                missed = true_set - pred_set
                wrong = pred_set - true_set
                
                if correct:
                    print(f"    ✅ {entity_type} correct: {correct}")
                if missed:
                    print(f"    ❌ {entity_type} missed: {missed}")
                if wrong:
                    print(f"    ⚠️  {entity_type} wrong: {wrong}")


def create_test_template():
    """Create a template for manual test data annotation"""
    template = [
        {
            "text": "Ahmed Khan is a Software Engineer at Google in Lahore. He has a BS in Computer Science from LUMS.",
            "ground_truth": {
                "Name": ["Ahmed Khan"],
                "Designation": ["Software Engineer"],
                "Company": ["Google"],
                "Location": ["Lahore"],
                "Degree": ["BS in Computer Science"],
                "College": ["LUMS"]
            }
        },
        {
            "text": "Fatima Ali graduated from FAST NUCES with a Master's degree. She works as a Data Scientist at Systems Limited in Karachi.",
            "ground_truth": {
                "Name": ["Fatima Ali"],
                "College": ["FAST NUCES"],
                "Degree": ["Master's degree"],
                "Designation": ["Data Scientist"],
                "Company": ["Systems Limited"],
                "Location": ["Karachi"]
            }
        },
        # Add 18-28 more real examples here
    ]
    
    with open('test_resumes_template.json', 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    print("✅ Created test_resumes_template.json")
    print("📝 Fill in 20-30 real resume examples and save as test_resumes.json")


# ============================================
# MAIN
# ============================================

def main():
    print("🚀 PROPER F1 EVALUATION")
    print("="*70)
    
    # Check if test file exists
    import os
    if not os.path.exists('test_resumes.json'):
        print("\n⚠️  test_resumes.json not found!")
        print("Creating template...")
        create_test_template()
        print("\n📝 Next steps:")
        print("1. Open test_resumes_template.json")
        print("2. Add 20-30 REAL resume examples")
        print("3. Save as test_resumes.json")
        print("4. Run this script again")
        return
    
    # Load evaluator
    evaluator = ProperEvaluator(model_path="models/improved_ner/final_model")
    
    # Show detailed examples first
    print("\n🔍 Showing detailed predictions for first 5 resumes...")
    evaluator.show_detailed_examples('test_resumes.json', num_examples=5)
    
    # Run full evaluation
    print("\n📊 Running full evaluation...")
    macro_f1, entity_metrics = evaluator.evaluate_on_test_set('test_resumes.json')
    
    print(f"\n🎯 FINAL MACRO-AVERAGED F1 SCORE: {macro_f1:.4f}")
    
    # Decision point
    if macro_f1 >= 0.75:
        print("\n✅ GOOD RESULTS! (F1 >= 0.75)")
        print("💡 Recommendation: Train on FULL 2200 dataset for final model")
        print("   Expected improvement: +5-10% F1")
    elif macro_f1 >= 0.65:
        print("\n⚠️  MODERATE RESULTS (F1 = 0.65-0.75)")
        print("💡 Recommendations:")
        print("   1. Check if test data is too different from training data")
        print("   2. Consider data augmentation")
        print("   3. Try training on full 2200 dataset")
    else:
        print("\n❌ NEEDS IMPROVEMENT (F1 < 0.65)")
        print("💡 Recommendations:")
        print("   1. Check data quality (labels, formatting)")
        print("   2. Increase training epochs")
        print("   3. Adjust learning rate or class weights")
        print("   4. Review predicted vs actual examples above")


if __name__ == "__main__":
    main()