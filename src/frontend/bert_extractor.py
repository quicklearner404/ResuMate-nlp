import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification

class BertExtractor:
    def __init__(self, model_path="models/final_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            self.model = DistilBertForTokenClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.label2id = self.model.config.label2id
            self.id2label = self.model.config.id2label
        except:
            self.model = None

    def predict(self, text):
        if not self.model:
            return {}

        text = text.strip()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=512, return_offsets_mapping=True)

        offsets = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        preds = torch.argmax(outputs.logits, dim=2)[0]

        # Build extracted entities
        result = {"Name": [], "Company": [], "Designation": [], "Location": [], "College": [], "Degree": []}

        current = None
        for pred_id, (start, end) in zip(preds, offsets):
            if start == end == 0:
                continue

            label = self.id2label[pred_id.item()]
            if label == "O":
                if current:
                    result[current["label"]].append(current["text"])
                    current = None
                continue

            prefix = label[0]
            tag = label[2:]

            if prefix == "B":
                if current:
                    result[current["label"]].append(current["text"])
                current = {"label": tag, "text": text[start:end], "start": start, "end": end}

            elif prefix == "I" and current and current["label"] == tag:
                current["text"] += " " + text[start:end]
                current["end"] = end

        if current:
            result[current["label"]].append(current["text"])

        return {k: list(set(v)) for k, v in result.items()}
