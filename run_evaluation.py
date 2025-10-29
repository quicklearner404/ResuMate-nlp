from evaluate_ner import evaluate_ner, generate_report

print("🚀 FINAL NER EVALUATION - 30 RESUMES")
print("=" * 50)
results = evaluate_ner()
generate_report(results)