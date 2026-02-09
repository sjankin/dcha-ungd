#!/usr/bin/env python3
"""
Prepare negative control dataset for LLM false positive testing.

Samples random sentences from 1946-1988 UN speeches (before climate-health
discourse existed). Any DCHA claims found by LLMs in this period would be
false positives.

Usage:
    python prepare_negative_control.py [--n-samples 200]
"""

import argparse
import ast
import hashlib
import json
import os
import random
from pathlib import Path

import pandas as pd

# === PATHS ===
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
# Source: UN General Debate Corpus (not included in this repo).
# Download from: https://doi.org/10.1177/00223433241275335
# Override with: UNGD_SOURCE_DIR environment variable
UNGD_DIR = Path(os.environ.get('UNGD_SOURCE_DIR', ROOT_DIR.parent / 'UNGD'))
COMPLETE_PATH = UNGD_DIR / "COMPLETE.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "benchmark" / "negative_control"

# === CONFIGURATION ===
PRE_CLIMATE_CUTOFF = 1989  # Before this year, climate-health discourse was minimal
SEED = 42
MIN_SENTENCE_WORDS = 10  # Filter out very short sentences
MAX_SENTENCE_WORDS = 200  # Filter out very long sentences


def create_candidate_id(iso3: str, year: int, sentence: str) -> str:
    """Create stable candidate_id: ISO3_YEAR_hash8."""
    sent_hash = hashlib.sha1(sentence.encode('utf-8')).hexdigest()[:8]
    return f"{iso3}_{year}_{sent_hash}"


def extract_sentences(speech_sentences_str: str) -> list:
    """Parse the sentence list from COMPLETE.csv format."""
    try:
        # The column contains a Python list as string
        sentences = ast.literal_eval(speech_sentences_str)
        if isinstance(sentences, list):
            return sentences
    except (ValueError, SyntaxError):
        pass
    return []


def filter_sentence(sentence: str) -> bool:
    """Check if sentence is suitable for evaluation."""
    words = sentence.split()
    if len(words) < MIN_SENTENCE_WORDS:
        return False
    if len(words) > MAX_SENTENCE_WORDS:
        return False
    # Filter out sentences that are too short or clearly not statements
    if sentence.count('.') > 3:  # Likely multiple sentences concatenated
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare negative control dataset")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of sentences to sample (default: 200)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 70)
    print("Negative Control Dataset Preparation")
    print("=" * 70)

    # Load COMPLETE.csv
    print(f"\nLoading {COMPLETE_PATH}...")
    if not COMPLETE_PATH.exists():
        print(f"ERROR: {COMPLETE_PATH} not found")
        return

    df = pd.read_csv(COMPLETE_PATH)
    print(f"  Loaded {len(df)} speeches")

    # Filter to pre-climate era
    print(f"\nFiltering to pre-{PRE_CLIMATE_CUTOFF} speeches...")
    pre_climate = df[df['Year'] < PRE_CLIMATE_CUTOFF].copy()
    print(f"  {len(pre_climate)} speeches from 1946-{PRE_CLIMATE_CUTOFF-1}")

    # Extract all sentences
    print("\nExtracting sentences...")
    all_sentences = []
    for _, row in pre_climate.iterrows():
        sentences = extract_sentences(row['Speech Sentences'])
        for sent in sentences:
            if filter_sentence(sent):
                all_sentences.append({
                    'year': row['Year'],
                    'country': row['Country'],
                    'sentence': sent.strip()
                })

    print(f"  Extracted {len(all_sentences)} suitable sentences")

    # Sample
    print(f"\nSampling {args.n_samples} sentences...")
    if len(all_sentences) < args.n_samples:
        print(f"  WARNING: Only {len(all_sentences)} sentences available")
        sampled = all_sentences
    else:
        sampled = random.sample(all_sentences, args.n_samples)

    # Create candidate IDs
    for item in sampled:
        item['candidate_id'] = create_candidate_id(
            item['country'], item['year'], item['sentence']
        )

    # Create DataFrame in benchmark format
    control_df = pd.DataFrame(sampled)
    control_df = control_df.rename(columns={'country': 'iso3'})

    # Add gold labels (all should be negative)
    control_df['attrib_gold'] = False
    control_df['cause_span_gold'] = ''
    control_df['effect_span_gold'] = ''
    control_df['link_type_gold'] = 'NO_CAUSAL_EXTRACTION'

    # Reorder columns
    control_df = control_df[[
        'candidate_id', 'iso3', 'year', 'sentence',
        'attrib_gold', 'cause_span_gold', 'effect_span_gold', 'link_type_gold'
    ]]

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    control_df.to_csv(OUTPUT_DIR / 'candidates_gold.csv', index=False)
    print(f"\nSaved {len(control_df)} candidates to {OUTPUT_DIR / 'candidates_gold.csv'}")

    # Create splits.json (all in "test" for evaluation)
    splits = {
        'version': 'v1.0-negative-control',
        'description': f'Negative control: random sentences from 1946-{PRE_CLIMATE_CUTOFF-1}',
        'expected_positives': 0,
        'purpose': 'False positive testing - any DCHA claims found are false positives',
        'splits': {
            'test': {
                'years': f'1946-{PRE_CLIMATE_CUTOFF-1}',
                'candidate_ids': control_df['candidate_id'].tolist()
            }
        }
    }

    with open(OUTPUT_DIR / 'splits.json', 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits.json")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Negative Control Dataset:
  Sentences: {len(control_df)}
  Year range: {control_df['year'].min()}-{control_df['year'].max()}
  Countries: {control_df['iso3'].nunique()}
  Expected DCHA: 0 (any found = false positive)

Year distribution:
""")
    for decade in range(1940, 1990, 10):
        count = ((control_df['year'] >= decade) & (control_df['year'] < decade + 10)).sum()
        if count > 0:
            print(f"  {decade}s: {count} sentences")

    print(f"""
To run LLM evaluation:
  python run_llm_baseline.py --data-dir ../data/benchmark/negative_control --split test

False positive rate = (predicted ATTRIB=1) / {len(control_df)}
""")


if __name__ == '__main__':
    main()
