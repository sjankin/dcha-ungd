#!/usr/bin/env python3
"""
Prepare extended DCHA-UNGD benchmark data (1989-2024).

Transforms GOLD_1946-2024.csv into benchmark format and creates splits.

Creates:
- data/benchmark/dcha_ungd_extended_v1/candidates_gold.csv
- data/benchmark/dcha_ungd_extended_v1/splits.json
"""

import os
import pandas as pd
import json
import hashlib
from pathlib import Path

# === PATHS ===
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
# Source: UN General Debate annotated corpus (not included in this repo).
# Download from: https://doi.org/10.1177/00223433241275335
# Override with: UNGD_SOURCE_DIR environment variable
UNGD_DIR = Path(os.environ.get('UNGD_SOURCE_DIR', ROOT_DIR.parent / 'UNGD'))
SOURCE_FILE = UNGD_DIR / 'GOLD_1946-2024.csv'
OUTPUT_DIR = ROOT_DIR / 'data' / 'benchmark' / 'dcha_ungd_extended_v1'

# === CONFIGURATION ===
# Filter to years where climate-health discourse appears in UN speeches
# Data before 1989 contains no climate-health co-mentions
MIN_YEAR = 1989

# Splits: same dev/test as original paper for comparability
# Extended training includes 1989-2021
SPLIT_CONFIG = {
    'train': {'min_year': MIN_YEAR, 'max_year': 2021},
    'dev': {'min_year': 2022, 'max_year': 2022},
    'test': {'min_year': 2023, 'max_year': 2024}
}


def create_candidate_id(iso3: str, year: int, sentence: str) -> str:
    """Create stable candidate_id: ISO3_YEAR_hash8."""
    sent_hash = hashlib.sha1(sentence.encode('utf-8')).hexdigest()[:8]
    return f"{iso3}_{year}_{sent_hash}"


def determine_link_type(row: pd.Series) -> str:
    """
    Determine link_type from boolean columns.

    Priority: C2H_HARM > H2C_JUST > C2H_COBEN > OTHER_UNCLEAR > NO_CAUSAL_EXTRACTION
    """
    if not row['ATTRIB']:
        return 'NO_CAUSAL_EXTRACTION'

    # Check directed link types
    if row.get('C→H_HARM', False) == True:
        return 'C2H_HARM'
    if row.get('H→C_JUST', False) == True:
        return 'H2C_JUST'
    if row.get('C→H_COBEN', False) == True:
        return 'C2H_COBEN'

    # If ATTRIB but no directed type, it's OTHER_UNCLEAR
    return 'OTHER_UNCLEAR'


def main():
    print("=" * 70)
    print("DCHA-UNGD Extended Benchmark Data Preparation")
    print("=" * 70)

    # Load source data
    print(f"\nLoading source data from {SOURCE_FILE}...")
    if not SOURCE_FILE.exists():
        print(f"ERROR: Source file not found: {SOURCE_FILE}")
        print("Please ensure GOLD_1946-2024.csv exists in the UNGD folder.")
        return

    df = pd.read_csv(SOURCE_FILE)
    print(f"  Loaded {len(df)} total candidates")

    # Show year distribution
    print(f"\n  Year range in source: {df['Year'].min()} - {df['Year'].max()}")
    print(f"  Candidates by decade:")
    for decade in range(1940, 2030, 10):
        count = ((df['Year'] >= decade) & (df['Year'] < decade + 10)).sum()
        if count > 0:
            print(f"    {decade}s: {count}")

    # Filter to MIN_YEAR onwards
    print(f"\n  Filtering to {MIN_YEAR}+ (excluding pre-climate-discourse era)...")
    df = df[df['Year'] >= MIN_YEAR].copy()
    print(f"  Retained {len(df)} candidates")

    # Create candidate_id
    print("\nCreating candidate IDs...")
    df['candidate_id'] = df.apply(
        lambda row: create_candidate_id(row['Country'], row['Year'], row['Sentence']),
        axis=1
    )

    # Check for duplicates
    n_dups = df['candidate_id'].duplicated().sum()
    if n_dups > 0:
        print(f"  WARNING: {n_dups} duplicate candidate_ids found!")
        # Remove duplicates, keeping first occurrence
        df = df.drop_duplicates(subset='candidate_id', keep='first')
        print(f"  After deduplication: {len(df)} candidates")
    else:
        print(f"  All {len(df)} candidate_ids are unique")

    # Determine link types
    print("\nDetermining link types...")
    df['link_type_gold'] = df.apply(determine_link_type, axis=1)

    print("\n  Link type distribution:")
    for lt, count in df['link_type_gold'].value_counts().items():
        pct = 100 * count / len(df)
        print(f"    {lt}: {count} ({pct:.1f}%)")

    # Prepare output dataframe
    print("\nCreating benchmark format...")
    gold_df = pd.DataFrame({
        'candidate_id': df['candidate_id'],
        'iso3': df['Country'],
        'year': df['Year'],
        'sentence': df['Sentence'],
        'attrib_gold': df['ATTRIB'].astype(bool),
        'cause_span_gold': df['CAUSE'].fillna(''),
        'effect_span_gold': df['EFFECT'].fillna(''),
        'link_type_gold': df['link_type_gold']
    })

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save candidates_gold.csv
    gold_df.to_csv(OUTPUT_DIR / 'candidates_gold.csv', index=False)
    print(f"  Saved {len(gold_df)} rows to candidates_gold.csv")

    # Create splits
    print("\nCreating splits...")
    splits = {
        'version': 'v1.0-extended',
        'description': f'Chronological splits for DCHA-UNGD extended benchmark ({MIN_YEAR}-2024)',
        'splits': {}
    }

    for split_name, config in SPLIT_CONFIG.items():
        mask = (gold_df['year'] >= config['min_year']) & (gold_df['year'] <= config['max_year'])
        split_ids = gold_df[mask]['candidate_id'].tolist()
        splits['splits'][split_name] = {
            'years': f"{config['min_year']}-{config['max_year']}" if config['min_year'] != config['max_year'] else str(config['min_year']),
            'candidate_ids': split_ids
        }
        print(f"  {split_name}: {len(split_ids)} candidates ({config['min_year']}-{config['max_year']})")

    # Save splits.json
    with open(OUTPUT_DIR / 'splits.json', 'w') as f:
        json.dump(splits, f, indent=2)
    print("  Saved splits.json")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    n_attrib = gold_df['attrib_gold'].sum()
    n_dcha = (gold_df['link_type_gold'].isin(['C2H_HARM', 'C2H_COBEN', 'H2C_JUST'])).sum()

    print(f"\n  Total candidates: {len(gold_df)}")
    print(f"  ATTRIB=1: {n_attrib} ({100*n_attrib/len(gold_df):.1f}%)")
    print(f"  DCHA (directed): {n_dcha} ({100*n_dcha/len(gold_df):.1f}%)")

    # Compare to original dataset
    print("\n  Comparison to original (2014-2024):")
    orig_count = len(gold_df[gold_df['year'] >= 2014])
    extended_count = len(gold_df[gold_df['year'] < 2014])
    print(f"    Original period (2014-2024): {orig_count} candidates")
    print(f"    Extended period (1989-2013): {extended_count} candidates")
    print(f"    Additional data: +{extended_count} ({100*extended_count/orig_count:.1f}% increase)")

    # Check pre-2014 DCHA
    pre_2014_dcha = (gold_df['year'] < 2014) & gold_df['link_type_gold'].isin(['C2H_HARM', 'C2H_COBEN', 'H2C_JUST'])
    print(f"\n  Pre-2014 DCHA claims: {pre_2014_dcha.sum()}")
    if pre_2014_dcha.sum() > 0:
        print("    Years with DCHA:")
        dcha_years = gold_df[pre_2014_dcha]['year'].value_counts().sort_index()
        for year, count in dcha_years.items():
            print(f"      {year}: {count}")

    print("\n" + "=" * 70)
    print("Extended benchmark data created successfully!")
    print("=" * 70)
    print(f"\nOutput files in: {OUTPUT_DIR}")
    print("  - candidates_gold.csv")
    print("  - splits.json")


if __name__ == '__main__':
    main()
