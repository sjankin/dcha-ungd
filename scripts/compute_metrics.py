#!/usr/bin/env python3
"""
DCHA-UNGD Extended Evaluation Metrics Computation
==================================================

Computes evaluation metrics for the extended dataset (1989-2024).

Usage:
    python compute_metrics.py

Output:
    - Prints all metrics to stdout
    - Saves metrics JSON to ../data/computed_metrics_extended.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# === PATHS ===
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / 'data'
BENCHMARK_DIR = DATA_DIR / 'benchmark' / 'dcha_ungd_extended_v1'


def load_data():
    """Load gold data and splits."""
    gold_df = pd.read_csv(BENCHMARK_DIR / 'candidates_gold.csv')

    with open(BENCHMARK_DIR / 'splits.json') as f:
        splits_data = json.load(f)

    # Add split column
    train_ids = set(splits_data['splits']['train']['candidate_ids'])
    dev_ids = set(splits_data['splits']['dev']['candidate_ids'])
    test_ids = set(splits_data['splits']['test']['candidate_ids'])

    def get_split(cid):
        if cid in train_ids:
            return 'train'
        elif cid in dev_ids:
            return 'dev'
        elif cid in test_ids:
            return 'test'
        return 'unknown'

    gold_df['split'] = gold_df['candidate_id'].apply(get_split)

    return gold_df, splits_data


def compute_dataset_statistics(df, splits_data):
    """Compute dataset-level statistics."""
    stats = {
        'time_span': {
            'start': int(df['year'].min()),
            'end': int(df['year'].max())
        },
        'candidate_level': {
            'total_candidates': len(df),
            'attrib_positive': int(df['attrib_gold'].sum()),
            'attrib_negative': int((~df['attrib_gold']).sum())
        },
        'link_type_distribution': df['link_type_gold'].value_counts().to_dict(),
        'by_split': {
            'train': int((df['split'] == 'train').sum()),
            'dev': int((df['split'] == 'dev').sum()),
            'test': int((df['split'] == 'test').sum())
        },
        'by_year': df.groupby('year').size().to_dict()
    }
    return stats


def compute_label_distribution(df):
    """Compute label distribution."""
    dist = df['link_type_gold'].value_counts()
    total = len(df)

    distribution = {
        label: {
            'count': int(count),
            'share': round(count / total, 3)
        }
        for label, count in dist.items()
    }

    return distribution


def normalize_span(span):
    """Normalize span for comparison."""
    if span is None or (isinstance(span, float) and pd.isna(span)):
        return ''
    return str(span).lower().strip()


def compute_span_f1(predicted_span, human_span):
    """Compute token-level F1 between predicted and gold spans."""
    pred = normalize_span(predicted_span)
    gold = normalize_span(human_span)

    if pred == '' and gold == '':
        return 1.0
    if pred == '' or gold == '':
        return 0.0

    pred_tokens = set(pred.split())
    gold_tokens = set(gold.split())

    intersection = len(pred_tokens & gold_tokens)

    precision = intersection / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = intersection / len(gold_tokens) if len(gold_tokens) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main():
    print("=" * 70)
    print("DCHA-UNGD Extended Metrics Computation")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df, splits_data = load_data()
    print(f"  Loaded {len(df)} candidates")
    print(f"  Year range: {df['year'].min()}-{df['year'].max()}")

    # Compute all metrics
    metrics = {}

    # 1. Dataset statistics
    print("\n1. Computing dataset statistics...")
    metrics['dataset_statistics'] = compute_dataset_statistics(df, splits_data)

    stats = metrics['dataset_statistics']
    print(f"   Time span: {stats['time_span']['start']}-{stats['time_span']['end']}")
    print(f"   Candidates: {stats['candidate_level']['total_candidates']}")
    print(f"   ATTRIB=1: {stats['candidate_level']['attrib_positive']}")
    print(f"   Train/Dev/Test: {stats['by_split']['train']}/{stats['by_split']['dev']}/{stats['by_split']['test']}")

    # 2. Label distribution
    print("\n2. Computing label distribution...")
    metrics['label_distribution'] = compute_label_distribution(df)

    for label, data in metrics['label_distribution'].items():
        print(f"   {label}: {data['count']} ({data['share']*100:.1f}%)")

    # 3. Split statistics
    print("\n3. Split statistics:")
    for split in ['train', 'dev', 'test']:
        split_df = df[df['split'] == split]
        n_attrib = split_df['attrib_gold'].sum()
        n_directed = split_df['link_type_gold'].isin(['C2H_HARM', 'C2H_COBEN', 'H2C_JUST']).sum()
        print(f"   {split}: {len(split_df)} total, {n_attrib} ATTRIB=1, {n_directed} directed")

    # Save metrics
    output_path = DATA_DIR / 'computed_metrics_extended.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {output_path}")

    print("\n" + "=" * 70)
    print("Metrics computation complete!")
    print("=" * 70)

    return metrics


if __name__ == '__main__':
    main()
