#!/usr/bin/env python3
"""
Compute majority baseline metrics for DCHA-UNGD Extended benchmark.

For each task, the majority baseline predicts the most common class for all instances.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# === PATHS ===
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data" / "benchmark" / "dcha_ungd_extended_v1"
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


def contains_term(text: str, terms: list) -> bool:
    """Check if text contains any term from the list (case-insensitive)."""
    if not text or pd.isna(text):
        return False
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in terms)


def load_gold_data(split: str = "test") -> pd.DataFrame:
    """Load gold data for specified split."""
    gold_df = pd.read_csv(DATA_DIR / "candidates_gold.csv")

    with open(DATA_DIR / "splits.json") as f:
        splits = json.load(f)

    split_ids = set(splits["splits"][split]["candidate_ids"])
    return gold_df[gold_df["candidate_id"].isin(split_ids)]


def compute_majority_baseline(gold_df: pd.DataFrame) -> dict:
    """Compute majority baseline metrics for all tasks."""

    n = len(gold_df)

    # === Task A: Attribution Detection ===
    attrib_counts = gold_df["attrib_gold"].value_counts()
    majority_attrib = attrib_counts.idxmax()
    n_positive = (gold_df["attrib_gold"] == True).sum()
    n_negative = (gold_df["attrib_gold"] == False).sum()

    if not majority_attrib:
        task_a = {
            "majority_class": "negative (ATTRIB=0)",
            "n": n,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": n_negative / n,
            "confusion_matrix": {"TN": n_negative, "FP": 0, "FN": n_positive, "TP": 0}
        }
    else:
        task_a = {
            "majority_class": "positive (ATTRIB=1)",
            "n": n,
            "precision": n_positive / n,
            "recall": 1.0,
            "f1": 2 * (n_positive / n) / (1 + n_positive / n),
            "accuracy": n_positive / n,
            "confusion_matrix": {"TN": 0, "FP": n_negative, "FN": 0, "TP": n_positive}
        }

    # === Task E: DCHA Detection ===
    def is_dcha(row):
        if row["attrib_gold"] != True:
            return False
        return (contains_term(row["cause_span_gold"], DC_TERMS) and
                contains_term(row["effect_span_gold"], DH_TERMS))

    gold_df = gold_df.copy()
    gold_df["dcha_gold"] = gold_df.apply(is_dcha, axis=1)
    n_dcha_pos = gold_df["dcha_gold"].sum()
    n_dcha_neg = n - n_dcha_pos

    task_e = {
        "majority_class": "negative (DCHA=0)",
        "n": n,
        "support_positive": n_dcha_pos,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "accuracy": n_dcha_neg / n
    }

    # === Task D: Link Type Classification ===
    link_types = ["NO_CAUSAL_EXTRACTION", "OTHER_UNCLEAR", "C2H_HARM", "C2H_COBEN", "H2C_JUST"]
    link_counts = gold_df["link_type_gold"].value_counts()
    majority_link = link_counts.idxmax()

    per_class_f1 = {}
    for lt in link_types:
        support = link_counts.get(lt, 0)
        if lt == majority_link:
            tp = support
            fp = n - support
            fn = 0
            precision = support / n
            recall = 1.0 if support > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            tp = 0
            fp = 0
            fn = support
            precision = 0.0
            recall = 0.0
            f1 = 0.0

        per_class_f1[lt] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    macro_f1 = np.mean([v["f1"] for v in per_class_f1.values()])

    task_d = {
        "majority_class": majority_link,
        "n": n,
        "macro_f1": macro_f1,
        "per_class": per_class_f1
    }

    # === Task B/C: Span Extraction ===
    attrib_pos = gold_df[gold_df["attrib_gold"] == True]
    task_bc = {
        "majority_class": "empty spans",
        "n": len(attrib_pos),
        "cause_span_f1": 0.0,
        "effect_span_f1": 0.0,
        "combined_f1": 0.0
    }

    return {
        "task_a_attribution": task_a,
        "task_e_dcha": task_e,
        "task_d_link": task_d,
        "task_bc_spans": task_bc
    }


def main():
    print("=" * 80)
    print("Majority Baseline Metrics (Extended Dataset)")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for split in ["test", "dev", "train"]:
        print(f"\n{'=' * 40}")
        print(f"Split: {split}")
        print("=" * 40)

        gold_df = load_gold_data(split)
        print(f"Loaded {len(gold_df)} candidates")

        metrics = compute_majority_baseline(gold_df)
        all_results[split] = metrics

        print(f"\nTask A (Attribution Detection):")
        ta = metrics["task_a_attribution"]
        print(f"  Majority class: {ta['majority_class']}")
        print(f"  Accuracy: {ta['accuracy']:.3f}")
        print(f"  F1: {ta['f1']:.3f}")

        print(f"\nTask E (DCHA Detection):")
        te = metrics["task_e_dcha"]
        print(f"  Majority class: {te['majority_class']}")
        print(f"  Support positive: {te['support_positive']}")
        print(f"  F1: {te['f1']:.3f}")

        print(f"\nTask D (Link Type Classification):")
        td = metrics["task_d_link"]
        print(f"  Majority class: {td['majority_class']}")
        print(f"  Macro F1: {td['macro_f1']:.3f}")

        print(f"\nTask B/C (Span Extraction):")
        tbc = metrics["task_bc_spans"]
        print(f"  Combined F1: {tbc['combined_f1']:.3f}")

        if split == "test":
            print("\n" + "-" * 40)
            print("TABLE ROW (for paper):")
            print("-" * 40)
            print(f"| Majority baseline | — | {ta['f1']:.2f} | {te['f1']:.2f} | {tbc['combined_f1']:.2f} | {td['macro_f1']:.2f} |")

    # Save results
    output_file = OUTPUT_DIR / "majority_baseline_extended.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()
