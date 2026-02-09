#!/usr/bin/env python3
"""
Fine-tune RoBERTa for DCHA-UNGD Extended benchmark (1989-2024).

This script trains a RoBERTa-Large token classification model on the extended
training set (1989-2021) for cause/effect span extraction with BIO tagging.

Usage:
    python train_roberta.py

Output:
    - Model saved to ../models/roberta_extended_v1/

Requirements:
    - transformers>=4.35.0
    - torch>=2.0.0
    - datasets
    - accelerate

Note: This is a template. The training logic mirrors the original fine-tuning
approach used for the main paper's RoBERTa model.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

# === PATHS ===
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data" / "benchmark" / "dcha_ungd_extended_v1"
MODEL_OUTPUT_DIR = ROOT_DIR / "models" / "roberta_extended_v1"

# === CONFIGURATION ===
MODEL_NAME = "roberta-large"
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
SEED = 42

# BIO label scheme
LABEL2ID = {"O": 0, "B-CAUSE": 1, "I-CAUSE": 2, "B-EFFECT": 3, "I-EFFECT": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class DCHADataset(Dataset):
    """PyTorch Dataset for DCHA token classification."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sentence = row["sentence"]
        cause_span = row.get("cause_span_gold", "")
        effect_span = row.get("effect_span_gold", "")

        # Tokenize
        encoding = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            padding="max_length"
        )

        # Create labels
        labels = self._create_bio_labels(
            sentence, encoding, cause_span, effect_span
        )

        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(labels)
        }

    def _create_bio_labels(self, sentence, encoding, cause_span, effect_span):
        """Create BIO labels aligned with tokens."""
        offset_mapping = encoding["offset_mapping"]
        labels = [-100] * len(offset_mapping)  # -100 = ignore in loss

        # Find cause span character positions
        cause_start, cause_end = -1, -1
        if cause_span and cause_span in sentence:
            cause_start = sentence.find(cause_span)
            cause_end = cause_start + len(cause_span)

        # Find effect span character positions
        effect_start, effect_end = -1, -1
        if effect_span and effect_span in sentence:
            effect_start = sentence.find(effect_span)
            effect_end = effect_start + len(effect_span)

        # Assign labels to tokens
        in_cause = False
        in_effect = False

        for i, (start, end) in enumerate(offset_mapping):
            if start == end:  # Special token
                labels[i] = -100
                continue

            # Check if token is in cause span
            if cause_start <= start < cause_end:
                if not in_cause:
                    labels[i] = LABEL2ID["B-CAUSE"]
                    in_cause = True
                else:
                    labels[i] = LABEL2ID["I-CAUSE"]
            elif in_cause:
                in_cause = False
                labels[i] = LABEL2ID["O"]

            # Check if token is in effect span
            if effect_start <= start < effect_end:
                if not in_effect:
                    labels[i] = LABEL2ID["B-EFFECT"]
                    in_effect = True
                else:
                    labels[i] = LABEL2ID["I-EFFECT"]
            elif in_effect:
                in_effect = False
                if labels[i] == -100 or labels[i] == LABEL2ID["O"]:
                    labels[i] = LABEL2ID["O"]

            # Default to O if not set
            if labels[i] == -100 and start != end:
                labels[i] = LABEL2ID["O"]

        return labels


def load_data():
    """Load training and dev data."""
    gold_df = pd.read_csv(DATA_DIR / "candidates_gold.csv")

    with open(DATA_DIR / "splits.json") as f:
        splits = json.load(f)

    train_ids = set(splits["splits"]["train"]["candidate_ids"])
    dev_ids = set(splits["splits"]["dev"]["candidate_ids"])

    train_df = gold_df[gold_df["candidate_id"].isin(train_ids)]
    dev_df = gold_df[gold_df["candidate_id"].isin(dev_ids)]

    # Filter to attribution-positive for training spans
    train_pos = train_df[train_df["attrib_gold"] == True]
    dev_pos = dev_df[dev_df["attrib_gold"] == True]

    return train_pos, dev_pos


def main():
    print("=" * 80)
    print("RoBERTa Fine-tuning for DCHA-UNGD Extended")
    print("=" * 80)

    # Check if data exists
    if not (DATA_DIR / "candidates_gold.csv").exists():
        print(f"ERROR: Data not found at {DATA_DIR}")
        print("Run prepare_extended_data.py first.")
        return

    # Load data
    print("\nLoading data...")
    train_df, dev_df = load_data()
    print(f"  Train: {len(train_df)} attribution-positive samples")
    print(f"  Dev: {len(dev_df)} attribution-positive samples")

    # Initialize tokenizer
    print(f"\nLoading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = DCHADataset(train_df, tokenizer, MAX_LENGTH)
    dev_dataset = DCHADataset(dev_df, tokenizer, MAX_LENGTH)

    # Initialize model
    print(f"\nLoading model from {MODEL_NAME}...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=SEED,
        logging_dir=str(MODEL_OUTPUT_DIR / "logs"),
        logging_steps=50,
        report_to="none"
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train
    print("\nStarting training...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Output: {MODEL_OUTPUT_DIR}")

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(MODEL_OUTPUT_DIR / "final")
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR / "final")

    # Save training info
    training_info = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "train_samples": len(train_df),
        "dev_samples": len(dev_df),
        "timestamp": datetime.now().isoformat()
    }
    with open(MODEL_OUTPUT_DIR / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"\nModel saved to: {MODEL_OUTPUT_DIR / 'final'}")
    print("\nTo evaluate, run:")
    print(f"  python evaluate_finetuned_roberta.py --model-dir {MODEL_OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
