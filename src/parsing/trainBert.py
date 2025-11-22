import os
import re
import json
import random
import shutil
import torch
import numpy as np
from collections import Counter
from datasets import Dataset, load_from_disk
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    TrainerCallback
)
from torch.nn import CrossEntropyLoss

# ==========================================
# 1. UTILITY & CALLBACK CLASSES
# ==========================================

class TestAfterEpochCallback(TrainerCallback):
    def __init__(self, model_wrapper, test_cases):
        self.model_wrapper = model_wrapper
        self.test_cases = test_cases

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        print(f"\n{'='*60}")
        print(f"🧪 TESTING AFTER EPOCH {epoch}")
        print(f"{'='*60}")
        correct = 0
        total = len(self.test_cases)
        for text, expected_types in self.test_cases:
            entities = self.model_wrapper.predict_entities(text)
            found_types = set([e['label'] for e in entities])

            # Check if we found the expected tags
            if expected_types.issubset(found_types):
                entity_strs = [f"{e['label']}:'{e['text']}'" for e in entities]
                print(f"✅ '{text}' \u2192 {entity_strs}")
                correct += 1
            else:
                # Don't print full failure spam, just a small note
                print(f"❌ '{text}'")

        accuracy = (correct / total) * 100
        print(f"\n📊 Epoch {epoch} Mini-Test Accuracy: {accuracy:.1f}%")
        print(f"{'='*60}\n")

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
             # Ensure weights are on the correct device
            if self.class_weights.device != logits.device:
                self.class_weights = self.class_weights.to(logits.device)

            loss_fct = CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        else:
            loss = outputs.get("loss")

        return (loss, outputs) if return_outputs else loss

# ==========================================
# 2. MAIN NER CLASS
# ==========================================

class ImprovedNER:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None

        # Define Labels exactly as they appear in your project
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

        # Initialize Model
        self.load_model()

    def load_model(self):
        model_dir = "models/improved_ner/final_model"
        if os.path.exists(model_dir):
            print(f"\U0001f5de Loading existing model from {model_dir}")
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
            self.model = DistilBertForTokenClassification.from_pretrained(
                model_dir, num_labels=len(self.label2id),
                id2label=self.id2label, label2id=self.label2id
            )
        else:
            print("\U0001f195 Initializing new DistilBERT model...")
            self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
            self.model = DistilBertForTokenClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=len(self.label2id),
                id2label=self.id2label, label2id=self.label2id
            )

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model.to('cuda')

    def compute_class_weights(self, dataset):
        """
        ⚖️ Calculating and BOOSTING Class Weights...
        """
        print("\u2696\ufe0f Calculating and BOOSTING Class Weights...")
        all_labels = []
        for batch in dataset:
            labels = [l for l in batch['labels'] if l != -100]
            all_labels.extend(labels)

        counter = Counter(all_labels)
        total = len(all_labels)

        weights = []
        for i in range(len(self.label2id)):
            label_name = self.id2label[i]
            count = counter.get(i, 0)

            # Standard Inverse Frequency
            if count == 0: w = 10.0
            else: w = total / (len(self.label2id) * count)

            # 🚀 MANUAL BOOST: Force model to learn these specific weak entities
            if "Location" in label_name:
                w *= 3.0  # Triple penalty for missing Location
            elif "Company" in label_name:
                w *= 2.0  # Double penalty for missing Company
            elif "Degree" in label_name:
                w *= 2.0  # Double penalty for missing Degree
            elif "Designation" in label_name:
                w *= 1.5

            weights.append(w)

        # Cap weights to prevent explosion
        weights_tensor = torch.tensor(weights, dtype=torch.float)
        weights_tensor = torch.clamp(weights_tensor, max=100.0)

        print(f"✅ Boosted Weights: {weights_tensor}")
        return weights_tensor

    def tokenize_and_align_labels(self, examples):
        """
        🔥 THE CRITICAL FIX:
        Uses return_offsets_mapping to map characters to tokens correctly.
        """
        tokenized_inputs = self.tokenizer(
            examples["content"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_offsets_mapping=True
        )

        labels = []

        for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
            annotations = examples["annotation"][i]

            # 1. Create Ground Truth Map [(start, end, label)]
            ground_truth = []
            mapping = {
                'Name': 'Name', 'Companies worked at': 'Company',
                'Designation': 'Designation', 'Degree': 'Degree',
                'College Name': 'College', 'Location': 'Location'
            }

            for ann in annotations:
                lab_list = ann.get("label")
                if not lab_list: continue
                label_type = mapping.get(lab_list[0])
                if not label_type: continue

                for point in ann.get("points", []):
                    start = point.get("start")
                    end = point.get("end")
                    if start is not None and end is not None:
                        ground_truth.append((start, end, label_type))

            # 2. Assign Labels to Tokens
            doc_labels = []
            for idx, (token_start, token_end) in enumerate(offsets):
                # Ignore special tokens (CLS, SEP, PAD)
                if token_start == token_end:
                    doc_labels.append(-100)
                    continue

                # Check overlap
                token_label = 'O'
                for (ent_start, ent_end, ent_type) in ground_truth:
                    if token_start >= ent_start and token_end <= ent_end:
                        if token_start == ent_start:
                            token_label = f"B-{ent_type}"
                        else:
                            token_label = f"I-{ent_type}"
                        break

                doc_labels.append(self.label2id.get(token_label, 0))

            labels.append(doc_labels)

        tokenized_inputs["labels"] = labels
        # Remove offsets before training so the model doesn't crash
        tokenized_inputs.pop("offset_mapping")
        return tokenized_inputs

    def prepare_data(self, training_file, subset_size=None):
        print("\U0001f4c2 Loading Raw Data...")
        raw_data = []
        with open(training_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if "content" in d and "annotation" in d:
                        raw_data.append(d)
                except: continue

        # Shuffle
        random.shuffle(raw_data)
        if subset_size: raw_data = raw_data[:subset_size]

        # Convert to HF Dataset
        return Dataset.from_list(raw_data)

    def train(self, training_file, epochs=8):
        print(f"\U0001f680 STARTING TRAINING (Epochs: {epochs})")

        # 1. Cleanup Old Data
        if os.path.exists("models/improved_ner"):
            try:
                shutil.rmtree("models/improved_ner/results")
                shutil.rmtree("models/improved_ner/logs")
            except: pass

        # 2. Prepare Data
        raw_dataset = self.prepare_data(training_file)

        print("\U0001f9e9 Tokenizing and Aligning Labels...")
        tokenized_dataset = raw_dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_dataset.column_names
        )

        # 3. Compute Weights
        class_weights = self.compute_class_weights(tokenized_dataset)

        # 4. Split
        dataset_split = tokenized_dataset.train_test_split(test_size=0.15)
        train_dataset = dataset_split["train"]
        eval_dataset = dataset_split["test"]

        training_args = TrainingArguments(
            output_dir="models/improved_ner/results",
            # 🚀 1. TRAIN LONGER
            num_train_epochs=15,             # Increased from 8 to 15.
                                             # Companies need more time to be memorized.

            # 🚀 2. GPU OPTIMIZATION
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,

            # 🚀 3. MEMORIZATION SETTING
            weight_decay=0.001,              # Decreased from 0.01.
                                             # Low decay = allows model to "memorize" rare degrees better.

            # 🚀 4. LABEL SMOOTHING (The Secret Weapon)
            label_smoothing_factor=0.1,      # New! Helps model distinguish ambiguous classes
                                             # (like Company vs Location).

            warmup_steps=200,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            learning_rate=4e-5               # Slightly higher learning rate to converge faster
        )

        test_cases = [
            ("Abhishek Jha is a developer", {'Name'}),
            ("Worked at Microsoft in Seattle", {'Company', 'Location'}),
        ]

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer=self.tokenizer),
            callbacks=[TestAfterEpochCallback(self, test_cases)]
        )

        trainer.train()

        print("\U0001f4c7 Saving Model...")
        self.model.save_pretrained("models/improved_ner/final_model")
        self.tokenizer.save_pretrained("models/improved_ner/final_model")
        print("\u2705 DONE!")

    def predict_entities(self, text):
        """
        🔥 FINAL VERSION: Aggressively cleans connecting words (is, at, from, graduated...)
        Returns clean, high-quality entities ready for F1 evaluation.
        """
        if not self.model: return []

        # 1. Run Model Inference
        # (Using offsets to map tokens back to original characters)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, return_offsets_mapping=True).to(self.model.device)
        offset_mapping = inputs.pop("offset_mapping")[0].cpu()

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu()
        predicted_labels = [self.id2label[p.item()] for p in predictions]

        # 2. Reconstruct Raw Entities
        entities = []
        current_ent = None

        for label, (start, end) in zip(predicted_labels, offset_mapping):
            if start == end == 0: continue # Skip [CLS], [SEP]

            tag = label[2:] if label != 'O' else 'O'
            prefix = label[0]

            if prefix == 'B':
                if current_ent: entities.append(current_ent)
                current_ent = {'label': tag, 'text': text[start:end], 'start': start, 'end': end}
            elif prefix == 'I' and current_ent and current_ent['label'] == tag:
                current_ent['text'] = text[current_ent['start']:end]
                current_ent['end'] = end
            else:
                if current_ent: entities.append(current_ent)
                current_ent = None
        if current_ent: entities.append(current_ent)

        # 3. 🧼 AGGRESSIVE CLEANING (The Fix)
        clean_entities = []

        # 🛑 The "Stop Word" List for Resumes
        # These are words that BERT often accidentally grabs at the start/end of an entity
        junk_words = [
            # Connectors
            'is', 'was', 'are', 'were',
            'at', 'in', 'of', 'to', 'from', 'with', 'for', 'on', 'by', 'as',
            'and', 'or', '&',
            # Articles/Pronouns
            'a', 'an', 'the', 'he', 'she', 'his', 'her', 'my',
            # Titles (Usually not part of the Name entity in ground truth)
            'dr.', 'mr.', 'ms.', 'prof.',
            # Resume Verbs (BERT loves to grab these next to Company/Degree)
            'graduated', 'completed', 'earned', 'obtained',
            'works', 'worked', 'working', 'job', 'position', 'role'
        ]

        for e in entities:
            txt = e['text'].strip()
            label = e['label']

            # 🔄 Iterative Scrubbing
            # We loop because sometimes you have "is a" (2 layers of junk)
            dirty = True
            while dirty:
                dirty = False
                txt_lower = txt.lower()

                for junk in junk_words:
                    # Remove suffix (e.g., "Google in")
                    if txt_lower.endswith(f" {junk}"):
                        txt = txt[:-len(junk)-1].strip()
                        dirty = True
                        txt_lower = txt.lower() # Update lower string for next check

                    # Remove prefix (e.g., "graduated from FAST")
                    if txt_lower.startswith(f"{junk} "):
                        txt = txt[len(junk)+1:].strip()
                        dirty = True
                        txt_lower = txt.lower()

            # 4. Final Sanity Checks
            # Drop empty strings or single punctuation
            if len(txt) < 2 and txt.lower() not in ['c', 'r']:
                continue

            # Drop it if the *entire* entity is just a junk word (e.g. model predicted "is")
            if txt.lower() in junk_words:
                continue

            clean_entities.append({'label': label, 'text': txt})

        return clean_entities

# ==========================================
# 3. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Update this path to point to your generated data
    TRAINING_FILE = "cleaned_training_data.json"

    # Disable Weights & Biases logging explicitly
    os.environ["WANDB_DISABLED"] = "true"

    ner = ImprovedNER()

    # Run training
    # subset_size=None means use ALL data
    #ner.train(TRAINING_FILE, epochs=15)
