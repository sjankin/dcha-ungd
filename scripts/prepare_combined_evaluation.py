#!/usr/bin/env python3
"""
Prepare combined evaluation dataset for efficient LLM runs.

Combines:
- 5,000 pre-1989 sentences (negative control - false positive testing)
- 907 post-1989 candidates (extended benchmark - full evaluation)

Total: 5,907 samples per LLM run.

Usage:
    python prepare_combined_evaluation.py [--neg-samples 5000]
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
EXTENDED_DATA = ROOT_DIR / "data" / "benchmark" / "dcha_ungd_extended_v1" / "candidates_gold.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "benchmark" / "combined_evaluation"

# === CONFIGURATION ===
PRE_CLIMATE_CUTOFF = 1989
SEED = 42
MIN_SENTENCE_WORDS = 10
MAX_SENTENCE_WORDS = 200


def create_candidate_id(iso3: str, year: int, sentence: str) -> str:
    """Create stable candidate_id: ISO3_YEAR_hash8."""
    sent_hash = hashlib.sha1(sentence.encode('utf-8')).hexdigest()[:8]
    return f"{iso3}_{year}_{sent_hash}"


def extract_sentences(speech_sentences_str: str) -> list:
    """Parse the sentence list from COMPLETE.csv format."""
    try:
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
    if sentence.count('.') > 3:
        return False
    return True


def prepare_negative_control(n_samples: int) -> pd.DataFrame:
    """Sample n_samples sentences from pre-1989 speeches."""
    print(f"\n{'='*70}")
    print("PREPARING NEGATIVE CONTROL (Pre-1989)")
    print(f"{'='*70}")

    print(f"\nLoading {COMPLETE_PATH}...")
    df = pd.read_csv(COMPLETE_PATH)
    print(f"  Loaded {len(df)} speeches")

    # Filter to pre-climate era
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
                    'iso3': row['Country'],
                    'sentence': sent.strip()
                })

    print(f"  Extracted {len(all_sentences)} suitable sentences")

    # Sample
    print(f"\nSampling {n_samples} sentences...")
    if len(all_sentences) < n_samples:
        print(f"  WARNING: Only {len(all_sentences)} sentences available, using all")
        sampled = all_sentences
    else:
        sampled = random.sample(all_sentences, n_samples)

    # Create DataFrame
    neg_df = pd.DataFrame(sampled)

    # Create candidate IDs
    neg_df['candidate_id'] = neg_df.apply(
        lambda r: create_candidate_id(r['iso3'], r['year'], r['sentence']),
        axis=1
    )

    # Add gold labels (all negative for pre-1989)
    neg_df['attrib_gold'] = False
    neg_df['cause_span_gold'] = ''
    neg_df['effect_span_gold'] = ''
    neg_df['link_type_gold'] = 'NO_CAUSAL_EXTRACTION'
    neg_df['era'] = 'pre_1989'  # Tag for analysis

    print(f"  Created {len(neg_df)} negative control samples")
    return neg_df


def load_extended_data() -> pd.DataFrame:
    """Load all 907 extended benchmark candidates."""
    print(f"\n{'='*70}")
    print("LOADING EXTENDED BENCHMARK (1989-2024)")
    print(f"{'='*70}")

    ext_df = pd.read_csv(EXTENDED_DATA)
    print(f"  Loaded {len(ext_df)} candidates")

    # Add era tag
    ext_df['era'] = 'post_1989'

    # Show breakdown
    attrib_count = ext_df['attrib_gold'].sum()
    dcha_types = ['C2H_HARM', 'C2H_COBEN', 'H2C_JUST']
    dcha_count = ext_df['link_type_gold'].isin(dcha_types).sum()

    print(f"  ATTRIB=1: {attrib_count}")
    print(f"  DCHA (directed): {dcha_count}")
    print(f"  Year range: {ext_df['year'].min()}-{ext_df['year'].max()}")

    return ext_df


def main():
    parser = argparse.ArgumentParser(description="Prepare combined evaluation dataset")
    parser.add_argument("--neg-samples", type=int, default=5000,
                        help="Number of pre-1989 sentences (default: 5000)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 70)
    print("COMBINED EVALUATION DATASET PREPARATION")
    print("=" * 70)
    print(f"\nGoal: {args.neg_samples} pre-1989 + all post-1989 candidates")

    # Prepare both datasets
    neg_df = prepare_negative_control(args.neg_samples)
    ext_df = load_extended_data()

    # Ensure same columns
    columns = ['candidate_id', 'iso3', 'year', 'sentence',
               'attrib_gold', 'cause_span_gold', 'effect_span_gold',
               'link_type_gold', 'era']

    neg_df = neg_df[columns]
    ext_df = ext_df[columns]

    # Combine
    print(f"\n{'='*70}")
    print("COMBINING DATASETS")
    print(f"{'='*70}")

    combined_df = pd.concat([neg_df, ext_df], ignore_index=True)
    print(f"\nTotal combined samples: {len(combined_df)}")
    print(f"  Pre-1989 (negative control): {len(neg_df)}")
    print(f"  Post-1989 (extended benchmark): {len(ext_df)}")

    # Shuffle for randomized evaluation order
    combined_df = combined_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / 'candidates_gold.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")

    # Create splits.json (all in "test" since LLMs are zero-shot)
    splits = {
        'version': 'v1.0-combined',
        'description': f'Combined evaluation: {len(neg_df)} pre-1989 + {len(ext_df)} post-1989',
        'splits': {
            'test': {
                'description': 'Full combined evaluation set',
                'n_total': len(combined_df),
                'n_pre_1989': len(neg_df),
                'n_post_1989': len(ext_df),
                'candidate_ids': combined_df['candidate_id'].tolist()
            }
        },
        'analysis_notes': {
            'pre_1989': 'Negative control - any ATTRIB=1 predictions are FALSE POSITIVES',
            'post_1989': 'Extended benchmark - compute standard metrics (F1, etc.)'
        }
    }

    with open(OUTPUT_DIR / 'splits.json', 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits.json")

    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"""
Combined Evaluation Dataset:
  Total samples: {len(combined_df)}

  Pre-1989 (Negative Control):
    Samples: {len(neg_df)}
    Year range: {neg_df['year'].min()}-{neg_df['year'].max()}
    Expected ATTRIB=1: 0 (any = false positive)
    Expected DCHA: 0 (any = false positive)

  Post-1989 (Extended Benchmark):
    Samples: {len(ext_df)}
    Year range: {ext_df['year'].min()}-{ext_df['year'].max()}
    Gold ATTRIB=1: {ext_df['attrib_gold'].sum()}
    Gold DCHA: {ext_df['link_type_gold'].isin(['C2H_HARM', 'C2H_COBEN', 'H2C_JUST']).sum()}

To run LLM evaluation:
  python run_llm_baseline.py --data-dir ../data/benchmark/combined_evaluation --split test

Analysis will separate results by 'era' column (pre_1989 vs post_1989).
""")


if __name__ == '__main__':
    main()
