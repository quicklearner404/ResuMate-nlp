import json
import pandas as pd

INPUT_FILE = "extracted_entities.json"
OUTPUT_CSV = "ml_training_data.csv"

def extract_text_label(item):
    text = item["text"]
    titles = item["JOB_TITLE"]

    if len(titles) == 0:
        return None  # skip if no job title
    return text, titles[0].lower()

def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    rows = []
    for item in data:
        result = extract_text_label(item)
        if result:
            text, label = result
            rows.append({"text": text, "label": label})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print("Dataset saved as:", OUTPUT_CSV)
    print("Total samples:", len(df))
    print("Unique labels:", df["label"].unique())

if __name__ == "__main__":
    main()
