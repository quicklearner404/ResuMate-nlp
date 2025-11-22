import json
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict

# 1. Robust Import to find your Parser
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from hybridBert import ResumeParser
    print("✅ Loaded ResumeParser")
except ImportError:
    try:
        from src.parsing.hybridBert import ResumeParser
        print("✅ Loaded ResumeParser from src")
    except ImportError:
        from hybridBert import ResumeParser
        print("✅ Loaded ResumeParser local")

def generate_advanced_report(test_file):
    print("🚀 Initializing Advanced Analytics Pipeline...")
    parser = ResumeParser()
    
    # 2. Load Data
    print(f"📂 Loading Test Data: {test_file}")
    test_data = []
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            try:
                # Try loading as standard JSON array
                test_data = json.load(f)
            except json.JSONDecodeError:
                # Fallback for JSON Lines
                f.seek(0)
                test_data = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ File not found: {test_file}")
        return

    print(f"📊 Analyzing {len(test_data)} resumes...")

    y_true = []
    y_pred = []
    
    target_labels = ['Name', 'Email', 'Company', 'Designation', 'Degree', 'College', 'Location', 'Skills']
    
    # Store errors for detailed analysis
    error_log = []

    # 3. Run Inference & Comparison
    for i, item in enumerate(test_data):
        text = item.get('text') or item.get('content', '')
        if not text: continue

        # Ground Truth
        gt_data = item.get('ground_truth', {})
        
        # Prediction
        preds = parser.parse(text)

        # Compare
        for label in target_labels:
            # Get raw strings
            gt_list = gt_data.get(label, [])
            # Normalize to lowercase
            gt_set = set([g.strip().lower() for g in gt_list])
            
            pred_list = preds.get(label, [])
            pred_set = set([p.strip().lower() for p in pred_list])

            if len(gt_set) > 0:
                y_true.append(label)
                
                match_found = False
                
                # --- SMART EVALUATION LOGIC ---
                
                # 1. Check for Direct Overlap first (Fastest)
                if not gt_set.isdisjoint(pred_set):
                    match_found = True
                
                # 2. Check for Partial / Substring Match (The "Soft Evaluation")
                else:
                    for p in pred_set:
                        for g in gt_set:
                            # Logic: If Prediction is inside Truth OR Truth is inside Prediction
                            # e.g. "London" in "London UK" -> True
                            # e.g. "Software Engineer" in "Senior Software Engineer" -> True
                            if p in g or g in p:
                                # Safety Check: Only count if the word is long enough
                                # This prevents "C" matching "C++" or "a" matching "Java"
                                if len(p) > 2 and len(g) > 2:
                                    match_found = True
                                    break
                        if match_found: break
                
                # --- END LOGIC ---

                if match_found:
                    y_pred.append(label)
                else:
                    y_pred.append("MISSED")
                    # Log the specific error to see WHY it failed
                    error_log.append(f"Resume {i+1} [{label}]: Missed {gt_set} | Predicted {pred_set}")

    # ---------------------------------------------------------
    # 📉 1. GENERATE CSV REPORT
    # ---------------------------------------------------------
    print("\n💾 Saving Metrics to CSV...")
    clean_true = [t for t, p in zip(y_true, y_pred)]
    clean_pred = [p for t, p in zip(y_true, y_pred)]
    
    if not clean_true:
        print("❌ No ground truth data found to compare against.")
        return

    # Get dictionary report
    report_dict = classification_report(clean_true, clean_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report_dict).transpose()
    
    # Save to CSV
    df.to_csv("final_metrics.csv")
    print("✅ Saved 'final_metrics.csv'")
    print(df)

    # ---------------------------------------------------------
    # 📝 2. GENERATE ERROR LOG
    # ---------------------------------------------------------
    print("\n📝 Saving Error Log...")
    with open("error_analysis.txt", "w", encoding="utf-8") as f:
        f.write("=== DETAILED ERROR ANALYSIS ===\n")
        f.write(f"Total Errors: {len(error_log)}\n\n")
        for error in error_log:
            f.write(error + "\n")
    print("✅ Saved 'error_analysis.txt'")

    # ---------------------------------------------------------
    # 📊 3. VISUALIZATION: F1 SCORE BAR CHART
    # ---------------------------------------------------------
    print("\n🎨 Generating Visualizations...")
    
    plot_df = df.drop(['accuracy', 'macro avg', 'weighted avg'])
    if 'MISSED' in plot_df.index:
        plot_df = plot_df.drop('MISSED')

    plt.figure(figsize=(10, 6))
    # Fix for future warning: specify hue=x and legend=False
    sns.barplot(x=plot_df.index, y=plot_df['f1-score'], hue=plot_df.index, legend=False, palette='viridis')
    
    plt.title('Entity Recognition Performance (F1 Score)', fontsize=14)
    plt.ylabel('F1 Score')
    plt.xlabel('Entity Type')
    plt.ylim(0, 1.1)
    for i, v in enumerate(plot_df['f1-score']):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("entity_performance.png", dpi=300)
    print("✅ Saved 'entity_performance.png'")

    # ---------------------------------------------------------
    # 🎨 4. VISUALIZATION: CONFUSION MATRIX
    # ---------------------------------------------------------
    unique_labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.ylabel('Actual Entity')
    plt.xlabel('Predicted Entity')
    plt.title('Confusion Matrix: Hybrid System')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    print("✅ Saved 'confusion_matrix.png'")

if __name__ == "__main__":
    generate_advanced_report("test_resumes.json")