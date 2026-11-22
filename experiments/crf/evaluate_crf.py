import json
import pandas as pd
from src.parsing.hybrid import analyze_resume,get_crf_model
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(base_dir, "data", "resume", "parsed_resumes.csv")
tests_dir = os.path.join(base_dir, "tests")  # Save results in tests folder
os.makedirs(tests_dir, exist_ok=True)

class RealHybridEvaluator:
    def __init__(self):
        pass
    
    def load_real_ground_truth(self):
        path = os.path.join(base_dir, "ground_truth_labeled.json")
        with open(path, 'r') as f:
            ground_truth = json.load(f)
        return ground_truth['resumes']

    def evaluate_hybrid_system(self):
        ground_truth_resumes = self.load_real_ground_truth()
        df = pd.read_csv(csv_path)
        resume_texts = {row['filename']: row['cleaned_resume'] for _, row in df.iterrows()}

        detailed_results = []

        for resume_data in ground_truth_resumes:
            filename = resume_data['filename']
            resume_id = resume_data['resume_id']

            if filename not in resume_texts:
                continue
            text = resume_texts[filename]
            # In your test evaluator
            crf_model = get_crf_model()  # only once
            result = analyze_resume(text, crf_model=crf_model)


            pred_entities = self._extract_predicted_entities(result)
            true_entities = self._extract_true_entities(resume_data)
            metrics = self._calculate_metrics(true_entities, pred_entities)

            detailed_results.append({
                'filename': filename,
                'resume_id': resume_id,
                'true_entities': true_entities,
                'pred_entities': pred_entities,
                'metrics': metrics
            })

        # Save both JSON and TXT summary
        self._save_detailed_json(detailed_results)
        self._save_summary_txt(detailed_results)

        return detailed_results

    def _extract_true_entities(self, resume_data):
        entities = {'skills': [], 'job_titles': []}
        gt = resume_data.get('ground_truth', {})
        for category, skills in gt.get('skills', {}).items():
            entities['skills'].extend(skills)
        entities['job_titles'] = gt.get('job_titles', [])
        return entities

    def _extract_predicted_entities(self, result):
        entities = {'skills': [], 'job_titles': []}
        for category, skills in result['skills'].items():
            entities['skills'].extend(skills)
        entities['job_titles'] = result['experience'].get('job_titles', [])
        return entities

    def _calculate_metrics(self, true_entities, pred_entities):
        metrics = {}
        for key in ['skills', 'job_titles']:
            true_set = set([x.lower() for x in true_entities.get(key, [])])
            pred_set = set([x.lower() for x in pred_entities.get(key, [])])

            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)

            precision = tp / (tp + fp) if (tp + fp) else 0
            recall = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2*precision*recall / (precision+recall) if (precision+recall) else 0

            metrics[key] = {
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'precision': round(precision,3),
                'recall': round(recall,3),
                'f1': round(f1,3),
                'true_count': len(true_set),
                'pred_count': len(pred_set)
            }
        return metrics

    def _save_detailed_json(self, detailed_results):
        path = os.path.join(tests_dir, "detailed_hybrid_full_report.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"✅ Detailed JSON report saved: {path}")

    def _save_summary_txt(self, detailed_results):
        # Calculate overall metrics
        summary = {}
        for key in ['skills', 'job_titles']:
            TP = sum(r['metrics'][key]['TP'] for r in detailed_results)
            FP = sum(r['metrics'][key]['FP'] for r in detailed_results)
            FN = sum(r['metrics'][key]['FN'] for r in detailed_results)
            precision = TP / (TP+FP) if (TP+FP) else 0
            recall = TP / (TP+FN) if (TP+FN) else 0
            f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0
            summary[key] = {'TP':TP,'FP':FP,'FN':FN,'precision':round(precision,3),'recall':round(recall,3),'f1':round(f1,3)}

        overall_tp = sum(summary[k]['TP'] for k in summary)
        overall_fp = sum(summary[k]['FP'] for k in summary)
        overall_fn = sum(summary[k]['FN'] for k in summary)
        overall_precision = overall_tp / (overall_tp+overall_fp) if (overall_tp+overall_fp) else 0
        overall_recall = overall_tp / (overall_tp+overall_fn) if (overall_tp+overall_fn) else 0
        overall_f1 = 2*overall_precision*overall_recall/(overall_precision+overall_recall) if (overall_precision+overall_recall) else 0

        txt_path = os.path.join(tests_dir, "hybrid_evaluation_summary.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("📊 HYBRID SYSTEM EVALUATION SUMMARY\n")
            f.write("="*50+"\n")
            f.write(f"{'Entity':<15}{'Precision':<10}{'Recall':<10}{'F1':<10}{'TP':<6}{'FP':<6}{'FN':<6}\n")
            for k,v in summary.items():
                f.write(f"{k:<15}{v['precision']:<10}{v['recall']:<10}{v['f1']:<10}{v['TP']:<6}{v['FP']:<6}{v['FN']:<6}\n")
            f.write("-"*50+"\n")
            f.write(f"{'Overall':<15}{round(overall_precision,3):<10}{round(overall_recall,3):<10}{round(overall_f1,3):<10}{overall_tp:<6}{overall_fp:<6}{overall_fn:<6}\n")
        print(f"✅ Summary TXT saved: {txt_path}")

def main():
    evaluator = RealHybridEvaluator()
    evaluator.evaluate_hybrid_system()

if __name__ == "__main__":
    main()
