# test.py - PROPER EVALUATION WITH REAL GROUND TRUTH
import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from hybrid import analyze_resume
from train_crf import prepare_complete_training_data, prepare_crf_data, sent2features, sent2labels
import sklearn_crfsuite

class RealHybridEvaluator:
    def __init__(self):
        self.crf_model = None
    
    def train_hybrid_system(self):
        """Train hybrid system on 220 resumes"""
        print("🚀 TRAINING HYBRID SYSTEM")
        print("=" * 50)
        
        training_data = prepare_complete_training_data("entities2.json")
        labeled_sentences = prepare_crf_data(training_data)
        X_train = [sent2features(s) for s in labeled_sentences]
        y_train = [sent2labels(s) for s in labeled_sentences]
        
        self.crf_model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        self.crf_model.fit(X_train, y_train)
        print(f"✅ Trained on {len(training_data)} labeled resumes")
        return self.crf_model
    
    def load_real_ground_truth(self):
        """Load your real ground truth with 30 manually labeled resumes"""
        print("📖 LOADING REAL GROUND TRUTH")
        print("=" * 50)
        
        with open("../../ground_truth_labeled.json", 'r') as f:
            ground_truth = json.load(f)
        
        resumes = ground_truth['resumes']
        print(f"✅ Loaded {len(resumes)} manually labeled resumes")
        return resumes
    
    def evaluate_hybrid_system(self):
        """Real evaluation comparing hybrid system vs manual labels"""
        print("\n🎯 REAL HYBRID SYSTEM EVALUATION")
        print("=" * 50)
        
        # Load data
        ground_truth_resumes = self.load_real_ground_truth()
        
        print(f"🧪 Testing on {len(ground_truth_resumes)} MANUALLY LABELED resumes...")
        
        # Load production resumes to get the actual text
        df = pd.read_csv("../../data/resume/parsed_resumes.csv")
        resume_texts = {row['filename']: row['cleaned_resume'] for _, row in df.iterrows()}
        
        all_metrics = []
        detailed_results = []
        
        for resume_data in ground_truth_resumes:
            filename = resume_data['filename']
            resume_id = resume_data['resume_id']
            
            print(f"\n📄 {filename} ({resume_id})")
            
            # Get the actual resume text
            if filename not in resume_texts:
                print(f"   ❌ Could not find text for {filename}")
                continue
            
            text = resume_texts[filename]
            
            # Get hybrid system prediction
            result = analyze_resume(text, self.crf_model)
            
            # Extract predicted entities
            pred_entities = self._extract_predicted_entities(result)
            
            # Get ground truth entities
            true_entities = self._extract_true_entities(resume_data)
            
            # Calculate metrics
            metrics = self._calculate_metrics(true_entities, pred_entities)
            all_metrics.append(metrics)
            
            # Store detailed results
            detailed_results.append({
                'filename': filename,
                'resume_id': resume_id,
                'true_entities': true_entities,
                'pred_entities': pred_entities,
                'metrics': metrics
            })
            
            # Print sample comparison
            print(f"   True Skills: {len(true_entities.get('skills', []))}")
            print(f"   Pred Skills: {len(pred_entities.get('skills', []))}")
            print(f"   Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")
            
            # Show skill comparison
            true_skills = set(true_entities.get('skills', []))
            pred_skills = set(pred_entities.get('skills', []))
            
            missing_skills = true_skills - pred_skills
            extra_skills = pred_skills - true_skills
            
            if missing_skills:
                print(f"   ❌ Missing: {list(missing_skills)[:3]}")
            if extra_skills:
                print(f"   ⚠️  Extra: {list(extra_skills)[:3]}")
        
        # Calculate overall metrics
        if all_metrics:
            precision = sum(m['precision'] for m in all_metrics) / len(all_metrics)
            recall = sum(m['recall'] for m in all_metrics) / len(all_metrics)
            f1 = sum(m['f1'] for m in all_metrics) / len(all_metrics)
            
            print(f"\n📊 REAL EVALUATION RESULTS:")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall:    {recall:.3f}")
            print(f"   F1-Score:  {f1:.3f}")
            print(f"   Tested on: {len(all_metrics)} MANUALLY LABELED resumes")
            
            # Performance assessment
            if f1 >= 0.7:
                print("   🎉 EXCELLENT - Ready for production!")
            elif f1 >= 0.5:
                print("   ✅ GOOD - Solid performance")
            elif f1 >= 0.3:
                print("   ⚠️  FAIR - Needs improvement")
            else:
                print("   🔴 POOR - Major revisions needed")
            
            # Save detailed results
            self._save_detailed_results(detailed_results)
        
        return all_metrics
    
    def _extract_true_entities(self, resume_data):
        """Extract true entities from ground truth"""
        entities = {'skills': [], 'job_titles': []}
        
        # Extract skills from ground truth
        ground_truth = resume_data.get('ground_truth', {})
        skills_data = ground_truth.get('skills', {})
        
        # Flatten all skills from all categories
        for category, skills in skills_data.items():
            entities['skills'].extend(skills)
        
        # Extract job titles
        entities['job_titles'] = ground_truth.get('job_titles', [])
        
        return entities
    
    def _extract_predicted_entities(self, result):
        """Extract predicted entities from hybrid system"""
        entities = {'skills': [], 'job_titles': []}
        
        # Extract skills from hybrid result
        for category, skills in result['skills'].items():
            entities['skills'].extend(skills)
        
        # Extract job titles
        entities['job_titles'] = result['experience'].get('job_titles', [])
        
        return entities
    
    def _calculate_metrics(self, true_entities, pred_entities):
        """Calculate precision, recall, F1 for skills"""
        true_skills = set(skill.lower() for skill in true_entities['skills'])
        pred_skills = set(skill.lower() for skill in pred_entities['skills'])
        
        # Calculate metrics for skills
        tp = len(true_skills & pred_skills)
        fp = len(pred_skills - true_skills)
        fn = len(true_skills - pred_skills)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'true_skills_count': len(true_skills),
            'pred_skills_count': len(pred_skills)
        }
    
    def _save_detailed_results(self, detailed_results):
        """Save detailed evaluation results"""
        output = {
            'evaluation_summary': {
                'total_resumes': len(detailed_results),
                'average_precision': sum(r['metrics']['precision'] for r in detailed_results) / len(detailed_results),
                'average_recall': sum(r['metrics']['recall'] for r in detailed_results) / len(detailed_results),
                'average_f1': sum(r['metrics']['f1'] for r in detailed_results) / len(detailed_results)
            },
            'detailed_results': detailed_results
        }
        
        with open("hybrid_evaluation_results.json", "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"✅ Saved detailed results to hybrid_evaluation_results.json")
    
    def show_comparison_examples(self):
        """Show some detailed comparison examples"""
        print("\n🔍 DETAILED COMPARISON EXAMPLES")
        print("=" * 50)
        
        ground_truth_resumes = self.load_real_ground_truth()
        df = pd.read_csv("../../data/resume/parsed_resumes.csv")
        resume_texts = {row['filename']: row['cleaned_resume'] for _, row in df.iterrows()}
        
        # Show first 3 examples in detail
        for resume_data in ground_truth_resumes[:3]:
            filename = resume_data['filename']
            
            if filename not in resume_texts:
                continue
            
            text = resume_texts[filename]
            result = analyze_resume(text, self.crf_model)
            
            print(f"\n📋 {filename}")
            print("-" * 40)
            
            # Show ground truth vs predicted
            ground_truth = resume_data['ground_truth']
            extracted = resume_data['extracted_entities']
            
            print("🏷️ JOB TITLES:")
            print(f"   Manual: {ground_truth.get('job_titles', [])}")
            print(f"   Hybrid: {result['experience'].get('job_titles', [])}")
            
            print("🛠️ SKILLS COMPARISON:")
            true_skills = set()
            for category, skills in ground_truth['skills'].items():
                true_skills.update(skills)
            
            pred_skills = set()
            for category, skills in result['skills'].items():
                pred_skills.update(skills)
            
            print(f"   Manual found: {len(true_skills)} skills")
            print(f"   Hybrid found: {len(pred_skills)} skills")
            print(f"   Overlap: {len(true_skills & pred_skills)} skills")

def main():
    print("🧪 REAL HYBRID SYSTEM EVALUATION")
    print("=" * 60)
    print("Training: 220 labeled resumes")
    print("Testing: 30 MANUALLY LABELED resumes (REAL GROUND TRUTH)")
    print("=" * 60)
    
    evaluator = RealHybridEvaluator()
    
    # 1. Train hybrid system
    evaluator.train_hybrid_system()
    
    # 2. Run real evaluation
    evaluator.evaluate_hybrid_system()
    
    # 3. Show detailed examples
    evaluator.show_comparison_examples()
    
    print(f"\n🎉 REAL EVALUATION COMPLETED!")
    print("You now have ACTUAL F1 scores based on manual ground truth!")

if __name__ == "__main__":
    main()