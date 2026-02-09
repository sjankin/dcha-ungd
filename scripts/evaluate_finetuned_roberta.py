#!/usr/bin/env python3
"""
Evaluate fine-tuned RoBERTa model on DCHA-UNGD Extended benchmark.

NOTE: For the extended dataset analysis, you should ideally:
1. Re-train RoBERTa on the extended training set (1989-2021)
2. Then evaluate on test set (2023-2024)

This script can use the existing model for initial comparison, but
re-training on the larger dataset is recommended for fair comparison.

Usage:
    python evaluate_finetuned_roberta.py [--split test]
    python evaluate_finetuned_roberta.py --split test --model-dir /path/to/retrained/model
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# === PATHS ===
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
# Default model directory (retrain with train_roberta.py or provide --model-dir)
DEFAULT_MODEL_DIR = ROOT_DIR / "models" / "roberta_finetuned"

DATA_DIR = ROOT_DIR / "data" / "benchmark" / "dcha_ungd_extended_v1"
RUNS_DIR = ROOT_DIR / "runs" / "roberta_finetuned"
OUTPUT_DIR = ROOT_DIR / "eval" / "outputs"

# === LEXICONS ===
DC_TERMS = [
    'climate change', 'changing climate', 'climate emergency', 'climate crisis', 'climate decay',
    'global warming', 'green house', 'temperature', 'extreme weather', 'global environmental change',
    'climate variability', 'greenhouse', 'greenhouse-gas', 'low carbon', 'ghge', 'ghges',
    'renewable energy', 'carbon emission', 'carbon emissions', 'carbon dioxide', 'carbon-dioxide',
    'co2 emission', 'co2 emissions', 'climate pollutant', 'climate pollutants', 'decarbonization',
    'decarbonisation', 'carbon neutral', 'carbon-neutral', 'carbon neutrality', 'climate neutrality',
    'climate action', 'net-zero', 'net zero'
]

DH_TERMS = [
    'malaria', 'diarrhoea', 'infection', 'disease', 'diseases', 'sars', 'measles', 'pneumonia',
    'epidemic', 'epidemics', 'pandemic', 'pandemics', 'epidemiology', 'healthcare', 'health',
    'mortality', 'morbidity', 'nutrition', 'illness', 'illnesses', 'ncd', 'ncds', 'air pollution',
    'malnutrition', 'malnourishment', 'mental disorder', 'mental disorders', 'stunting'
]

# BIO label mapping
ID2LABEL = {0: "O", 1: "B-CAUSE", 2: "I-CAUSE", 3: "B-EFFECT", 4: "I-EFFECT"}
LABEL2ID = {"O": 0, "B-CAUSE": 1, "I-CAUSE": 2, "B-EFFECT": 3, "I-EFFECT": 4}


def contains_term(text: str, terms: list) -> bool:
    """Check if text contains any term from the list (case-insensitive)."""
    if not text or pd.isna(text):
        return False
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in terms)


def derive_link_type(attrib: bool, cause_span: str, effect_span: str) -> str:
    """Derive link type from spans using lexicon matching."""
    if not attrib:
        return "NO_CAUSAL_EXTRACTION"

    cause_has_dc = contains_term(cause_span, DC_TERMS)
    cause_has_dh = contains_term(cause_span, DH_TERMS)
    effect_has_dc = contains_term(effect_span, DC_TERMS)
    effect_has_dh = contains_term(effect_span, DH_TERMS)

    if cause_has_dc and effect_has_dh:
        return "C2H_HARM"
    if cause_has_dh and effect_has_dc:
        return "H2C_JUST"

    return "OTHER_UNCLEAR"


def bio_tags_to_spans(tokens: list, labels: list) -> tuple:
    """Convert BIO tags to cause and effect span strings."""
    cause_tokens = []
    effect_tokens = []
    current_type = None

    for token, label in zip(tokens, labels):
        if label == "B-CAUSE":
            current_type = "CAUSE"
            cause_tokens.append(token)
        elif label == "I-CAUSE":
            if current_type != "CAUSE":
                current_type = "CAUSE"
            cause_tokens.append(token)
        elif label == "B-EFFECT":
            current_type = "EFFECT"
            effect_tokens.append(token)
        elif label == "I-EFFECT":
            if current_type != "EFFECT":
                current_type = "EFFECT"
            effect_tokens.append(token)
        else:
            current_type = None

    def join_tokens(token_list):
        if not token_list:
            return ""
        result = token_list[0]
        for t in token_list[1:]:
            if t.startswith("##"):
                result += t[2:]
            elif t.startswith("Ġ"):
                result += t[1:]
            else:
                result += " " + t
        return result.strip()

    return join_tokens(cause_tokens), join_tokens(effect_tokens)


def reconstruct_spans_from_offsets(sentence: str, input_ids: list, labels: list,
                                    tokenizer, offset_mapping: list = None) -> tuple:
    """Reconstruct cause and effect spans using character offsets."""
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    cause_start, cause_end = None, None
    effect_start, effect_end = None, None

    for i, label in enumerate(labels):
        if label == "B-CAUSE":
            cause_start = i
            cause_end = i
        elif label == "I-CAUSE":
            if cause_start is None:
                cause_start = i
            cause_end = i
        elif label == "B-EFFECT":
            effect_start = i
            effect_end = i
        elif label == "I-EFFECT":
            if effect_start is None:
                effect_start = i
            effect_end = i

    if offset_mapping is not None and len(offset_mapping) == len(labels):
        cause_span = ""
        effect_span = ""

        if cause_start is not None and cause_end is not None:
            start_char = offset_mapping[cause_start][0]
            end_char = offset_mapping[cause_end][1]
            if start_char < len(sentence) and end_char <= len(sentence):
                cause_span = sentence[start_char:end_char].strip()

        if effect_start is not None and effect_end is not None:
            start_char = offset_mapping[effect_start][0]
            end_char = offset_mapping[effect_end][1]
            if start_char < len(sentence) and end_char <= len(sentence):
                effect_span = sentence[start_char:end_char].strip()

        return cause_span, effect_span

    return bio_tags_to_spans(tokens, labels)


class RoBERTaPredictor:
    """Wrapper for fine-tuned RoBERTa model."""

    def __init__(self, model_dir: Path):
        print(f"Loading model from {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")

    def predict(self, sentence: str) -> dict:
        """Run inference on a single sentence."""
        encoding = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            padding=True
        )

        offset_mapping = encoding.pop("offset_mapping")[0].tolist()
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()

        labels = [ID2LABEL[p] for p in predictions]

        cause_span, effect_span = reconstruct_spans_from_offsets(
            sentence,
            encoding["input_ids"][0].tolist(),
            labels,
            self.tokenizer,
            offset_mapping
        )

        has_cause = any(l in ["B-CAUSE", "I-CAUSE"] for l in labels)
        has_effect = any(l in ["B-EFFECT", "I-EFFECT"] for l in labels)
        attrib = has_cause or has_effect

        return {
            "attrib": attrib,
            "cause_span": cause_span if attrib else "",
            "effect_span": effect_span if attrib else "",
            "labels": labels
        }


def load_gold_data(split: str = "test") -> pd.DataFrame:
    """Load gold data for specified split."""
    gold_df = pd.read_csv(DATA_DIR / "candidates_gold.csv")

    with open(DATA_DIR / "splits.json") as f:
        splits = json.load(f)

    split_ids = set(splits["splits"][split]["candidate_ids"])
    return gold_df[gold_df["candidate_id"].isin(split_ids)]


def run_evaluation(predictions_df: pd.DataFrame, gold_df: pd.DataFrame, split: str) -> dict:
    """Compute all evaluation metrics."""
    merged = gold_df.merge(
        predictions_df[["candidate_id", "attrib", "cause_span", "effect_span", "link_type"]],
        on="candidate_id",
        suffixes=("_gold", "_pred")
    )

    # Task A: Attribution detection
    y_true = merged["attrib_gold"].astype(bool)
    y_pred = merged["attrib"].astype(bool)

    tp = ((y_true == True) & (y_pred == True)).sum()
    fp = ((y_true == False) & (y_pred == True)).sum()
    fn = ((y_true == True) & (y_pred == False)).sum()
    tn = ((y_true == False) & (y_pred == False)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(merged)

    task_a = {
        "n": len(merged),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }

    # Task E: DCHA detection
    def is_dcha(row, prefix):
        if not row[f"attrib{prefix}"]:
            return False
        cause = row[f"cause_span{prefix}"] if prefix == "_gold" else row["cause_span"]
        effect = row[f"effect_span{prefix}"] if prefix == "_gold" else row["effect_span"]
        return contains_term(cause, DC_TERMS) and contains_term(effect, DH_TERMS)

    merged["dcha_gold"] = merged.apply(lambda r: is_dcha(r, "_gold"), axis=1)
    merged["dcha_pred"] = merged.apply(lambda r: is_dcha(r, ""), axis=1)

    dcha_tp = ((merged["dcha_gold"] == True) & (merged["dcha_pred"] == True)).sum()
    dcha_fp = ((merged["dcha_gold"] == False) & (merged["dcha_pred"] == True)).sum()
    dcha_fn = ((merged["dcha_gold"] == True) & (merged["dcha_pred"] == False)).sum()

    dcha_precision = dcha_tp / (dcha_tp + dcha_fp) if (dcha_tp + dcha_fp) > 0 else 0
    dcha_recall = dcha_tp / (dcha_tp + dcha_fn) if (dcha_tp + dcha_fn) > 0 else 0
    dcha_f1 = 2 * dcha_precision * dcha_recall / (dcha_precision + dcha_recall) if (dcha_precision + dcha_recall) > 0 else 0

    task_e = {
        "n": len(merged),
        "support_positive": int(merged["dcha_gold"].sum()),
        "precision": dcha_precision,
        "recall": dcha_recall,
        "f1": dcha_f1
    }

    # Task B/C: Span extraction
    attrib_pos = merged[merged["attrib_gold"] == True]

    def token_f1(gold_span, pred_span):
        if pd.isna(gold_span) or gold_span == "":
            return 0.0 if pred_span else 1.0
        if pd.isna(pred_span) or pred_span == "":
            return 0.0

        gold_tokens = set(str(gold_span).lower().split())
        pred_tokens = set(str(pred_span).lower().split())

        if len(pred_tokens) == 0:
            return 0.0

        overlap = len(gold_tokens & pred_tokens)
        precision = overlap / len(pred_tokens) if pred_tokens else 0
        recall = overlap / len(gold_tokens) if gold_tokens else 0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    if len(attrib_pos) > 0:
        cause_f1s = attrib_pos.apply(
            lambda r: token_f1(r["cause_span_gold"], r["cause_span"]), axis=1
        )
        effect_f1s = attrib_pos.apply(
            lambda r: token_f1(r["effect_span_gold"], r["effect_span"]), axis=1
        )

        task_bc = {
            "n": len(attrib_pos),
            "cause_span_f1": cause_f1s.mean(),
            "effect_span_f1": effect_f1s.mean(),
            "combined_f1": (cause_f1s.mean() + effect_f1s.mean()) / 2
        }
    else:
        task_bc = {"n": 0, "cause_span_f1": 0, "effect_span_f1": 0, "combined_f1": 0}

    # Task D: Link type classification
    link_types = ["NO_CAUSAL_EXTRACTION", "OTHER_UNCLEAR", "C2H_HARM", "C2H_COBEN", "H2C_JUST"]

    per_class_f1 = {}
    for lt in link_types:
        gold_is_lt = merged["link_type_gold"] == lt
        pred_is_lt = merged["link_type"] == lt

        lt_tp = (gold_is_lt & pred_is_lt).sum()
        lt_fp = (~gold_is_lt & pred_is_lt).sum()
        lt_fn = (gold_is_lt & ~pred_is_lt).sum()

        lt_p = lt_tp / (lt_tp + lt_fp) if (lt_tp + lt_fp) > 0 else 0
        lt_r = lt_tp / (lt_tp + lt_fn) if (lt_tp + lt_fn) > 0 else 0
        lt_f1 = 2 * lt_p * lt_r / (lt_p + lt_r) if (lt_p + lt_r) > 0 else 0

        per_class_f1[lt] = {
            "support": int(gold_is_lt.sum()),
            "precision": lt_p,
            "recall": lt_r,
            "f1": lt_f1
        }

    macro_f1 = np.mean([v["f1"] for v in per_class_f1.values()])

    task_d = {
        "n": len(merged),
        "macro_f1": macro_f1,
        "per_class": per_class_f1
    }

    return {
        "split": split,
        "task_a_attribution": task_a,
        "task_e_dcha": task_e,
        "task_bc_spans": task_bc,
        "task_d_link": task_d
    }


def save_metrics_csvs(metrics: dict, run_name: str, split: str):
    """Save metrics in the same CSV format as LLM evaluations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Task A
    task_a = metrics["task_a_attribution"]
    df_a = pd.DataFrame([{
        "run": run_name,
        "split": split,
        "n": task_a["n"],
        "accuracy": task_a["accuracy"],
        "precision": task_a["precision"],
        "recall": task_a["recall"],
        "f1": task_a["f1"]
    }])
    df_a.to_csv(OUTPUT_DIR / f"metrics_taskA_{run_name}_{split}.csv", index=False)

    # Task E
    task_e = metrics["task_e_dcha"]
    df_e = pd.DataFrame([{
        "run": run_name,
        "split": split,
        "n": task_e["n"],
        "support_positive": task_e["support_positive"],
        "precision": task_e["precision"],
        "recall": task_e["recall"],
        "f1": task_e["f1"]
    }])
    df_e.to_csv(OUTPUT_DIR / f"metrics_taskE_{run_name}_{split}.csv", index=False)

    # Task B/C
    task_bc = metrics["task_bc_spans"]
    rows = [
        {"run": run_name, "split": split, "span_type": "cause_span", "n": task_bc["n"],
         "token_f1": task_bc["cause_span_f1"]},
        {"run": run_name, "split": split, "span_type": "effect_span", "n": task_bc["n"],
         "token_f1": task_bc["effect_span_f1"]},
        {"run": run_name, "split": split, "span_type": "combined", "n": task_bc["n"],
         "token_f1": task_bc["combined_f1"]}
    ]
    df_bc = pd.DataFrame(rows)
    df_bc.to_csv(OUTPUT_DIR / f"metrics_taskB_C_{run_name}_{split}.csv", index=False)

    # Task D
    task_d = metrics["task_d_link"]
    rows = []
    for lt, vals in task_d["per_class"].items():
        rows.append({
            "run": run_name,
            "split": split,
            "link_type": lt,
            "support": vals["support"],
            "precision": vals["precision"],
            "recall": vals["recall"],
            "f1": vals["f1"]
        })
    rows.append({
        "run": run_name,
        "split": split,
        "link_type": "MACRO_AVG",
        "support": task_d["n"],
        "f1": task_d["macro_f1"]
    })
    df_d = pd.DataFrame(rows)
    df_d.to_csv(OUTPUT_DIR / f"metrics_taskD_{run_name}_{split}.csv", index=False)

    print(f"Saved metrics to {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned RoBERTa on DCHA-UNGD Extended")
    parser.add_argument("--split", default="test", help="Split to evaluate (train/dev/test)")
    parser.add_argument("--run-name", default=None, help="Name for this run")
    parser.add_argument("--model-dir", default=None, help="Path to model directory")
    args = parser.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else DEFAULT_MODEL_DIR
    date_str = datetime.now().strftime("%Y-%m-%d")
    run_name = args.run_name or f"{date_str}_roberta_finetuned_extended"

    print("=" * 80)
    print("Fine-tuned RoBERTa Evaluation (Extended Dataset)")
    print("=" * 80)
    print(f"Model: {model_dir}")
    print(f"Data:  {DATA_DIR}")

    # Load model
    predictor = RoBERTaPredictor(model_dir)

    # Load gold data
    print(f"\nLoading {args.split} split...")
    gold_df = load_gold_data(args.split)
    print(f"  {len(gold_df)} candidates")

    # Run inference
    print("\nRunning inference...")
    predictions = []
    for idx, row in gold_df.iterrows():
        result = predictor.predict(row["sentence"])
        link_type = derive_link_type(
            result["attrib"],
            result["cause_span"],
            result["effect_span"]
        )

        predictions.append({
            "candidate_id": row["candidate_id"],
            "attrib": result["attrib"],
            "cause_span": result["cause_span"],
            "effect_span": result["effect_span"],
            "link_type": link_type
        })

        if (len(predictions) % 50) == 0:
            print(f"  Processed {len(predictions)}/{len(gold_df)}")

    predictions_df = pd.DataFrame(predictions)

    # Save predictions
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(run_dir / "predictions.csv", index=False)
    print(f"\nSaved predictions to {run_dir / 'predictions.csv'}")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = run_evaluation(predictions_df, gold_df, args.split)

    # Save metrics
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    save_metrics_csvs(metrics, run_name, args.split)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\nTask A (Attribution Detection):")
    ta = metrics["task_a_attribution"]
    print(f"  Precision: {ta['precision']:.3f}")
    print(f"  Recall:    {ta['recall']:.3f}")
    print(f"  F1:        {ta['f1']:.3f}")

    print(f"\nTask E (DCHA Detection):")
    te = metrics["task_e_dcha"]
    print(f"  Precision: {te['precision']:.3f}")
    print(f"  Recall:    {te['recall']:.3f}")
    print(f"  F1:        {te['f1']:.3f}")

    print(f"\nTask B/C (Span Extraction, n={metrics['task_bc_spans']['n']}):")
    tbc = metrics["task_bc_spans"]
    print(f"  Cause span F1:  {tbc['cause_span_f1']:.3f}")
    print(f"  Effect span F1: {tbc['effect_span_f1']:.3f}")
    print(f"  Combined F1:    {tbc['combined_f1']:.3f}")

    print(f"\nTask D (Link Type Classification):")
    td = metrics["task_d_link"]
    print(f"  Macro F1: {td['macro_f1']:.3f}")

    print("\n" + "=" * 80)
    print("TABLE ROW")
    print("=" * 80)
    print(f"| RoBERTa (fine-tuned) | {ta['f1']:.2f} | {te['f1']:.2f} | {tbc['combined_f1']:.2f} | {td['macro_f1']:.2f} |")


if __name__ == "__main__":
    main()
