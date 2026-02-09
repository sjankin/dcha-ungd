#!/usr/bin/env python3
"""
build_paper_data.py - Statistics pipeline for DCHA-UNGD Extended benchmark (1989-2024)

Generates statistics for the extended dataset.

Usage:
    python build_paper_data.py

Outputs:
    - data/extended_statistics.json: All statistics for extended dataset
    - data/speech_level_indicators.csv: Speech-level COM/ATTRIB/DCHA
    - data/country_dcha_counts.csv: Country-level DCHA counts
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# === CONFIGURATION ===
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
GOLD_PATH = ROOT_DIR / "data/benchmark/dcha_ungd_extended_v1/candidates_gold.csv"
SPLITS_PATH = ROOT_DIR / "data/benchmark/dcha_ungd_extended_v1/splits.json"
OUTPUT_DIR = ROOT_DIR / "data"

# For speech-level stats (if external covariates available).
# Source: assembled from ND-GAIN, WDI, and OWID (see paper Appendix).
# Override with: UNGD_SOURCE_DIR environment variable
UNGD_DIR = Path(os.environ.get('UNGD_SOURCE_DIR', ROOT_DIR.parent / 'UNGD'))
EXTERNAL_COV_PATH = UNGD_DIR / "covariates.csv"

# === LEXICONS (from paper Appendix) ===
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


def contains_term(text, terms):
    """Check if text contains any of the terms (case-insensitive)."""
    if pd.isna(text):
        return False
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in terms)


def derive_dcha(row):
    """
    Derive DCHA flag per paper definition:
    DCHA=1 if ATTRIB=1 AND CAUSE span contains DC term AND EFFECT span contains DH term
    """
    if row['attrib_gold'] != True:
        return False
    cause_has_dc = contains_term(row['cause_span_gold'], DC_TERMS)
    effect_has_dh = contains_term(row['effect_span_gold'], DH_TERMS)
    return cause_has_dc and effect_has_dh


def main():
    print("=" * 80)
    print("DCHA-UNGD Extended Statistics Pipeline")
    print("=" * 80)

    # === LOAD DATA ===
    print(f"\nLoading gold data from: {GOLD_PATH}")
    if not GOLD_PATH.exists():
        print(f"ERROR: {GOLD_PATH} not found. Run prepare_extended_data.py first.")
        return None

    gold = pd.read_csv(GOLD_PATH)
    print(f"  Loaded {len(gold)} candidates")

    # Load splits
    with open(SPLITS_PATH) as f:
        splits_data = json.load(f)

    # === DERIVE DCHA ===
    print("\nDeriving DCHA flags...")
    gold['dcha'] = gold.apply(derive_dcha, axis=1)
    gold['attrib'] = gold['attrib_gold'].astype(bool)
    gold['com'] = True

    # === SENTENCE-LEVEL STATISTICS ===
    stats = {
        '_description': 'Extended dataset statistics (1989-2024)',
        '_source': str(GOLD_PATH)
    }

    # Basic counts
    stats['n_candidates'] = len(gold)
    stats['n_attrib'] = int(gold['attrib'].sum())
    stats['n_dcha_sentences'] = int(gold['dcha'].sum())

    # Link type distribution
    link_dist = gold['link_type_gold'].value_counts().to_dict()
    stats['link_type_distribution'] = link_dist
    stats['n_no_causal'] = link_dist.get('NO_CAUSAL_EXTRACTION', 0)
    stats['n_other_unclear'] = link_dist.get('OTHER_UNCLEAR', 0)
    stats['n_c2h_harm'] = link_dist.get('C2H_HARM', 0)
    stats['n_c2h_coben'] = link_dist.get('C2H_COBEN', 0)
    stats['n_h2c_just'] = link_dist.get('H2C_JUST', 0)

    # Percentages
    total = len(gold)
    stats['pct_no_causal'] = round(stats['n_no_causal'] / total * 100, 1)
    stats['pct_other_unclear'] = round(stats['n_other_unclear'] / total * 100, 1)
    stats['pct_c2h_harm'] = round(stats['n_c2h_harm'] / total * 100, 1)
    stats['pct_directed'] = round((stats['n_c2h_harm'] + stats['n_c2h_coben'] + stats['n_h2c_just']) / total * 100, 1)

    # === SPLIT STATISTICS ===
    for split_name, split_info in splits_data['splits'].items():
        split_ids = set(split_info['candidate_ids'])
        split_mask = gold['candidate_id'].isin(split_ids)
        stats[f'n_{split_name}'] = int(split_mask.sum())

        # Per-split details
        split_df = gold[split_mask]
        stats[f'{split_name}_n_attrib'] = int(split_df['attrib'].sum())
        stats[f'{split_name}_n_dcha'] = int(split_df['dcha'].sum())

    # === YEAR DISTRIBUTION ===
    year_dist = gold.groupby('year').agg({
        'attrib': 'sum',
        'dcha': 'sum',
        'candidate_id': 'count'
    }).rename(columns={'candidate_id': 'total'}).to_dict('index')
    stats['year_distribution'] = {int(k): v for k, v in year_dist.items()}

    # === LINGUISTIC STATISTICS ===
    gold['sent_tokens'] = gold['sentence'].apply(lambda x: len(str(x).split()))
    gold['cause_tokens'] = gold['cause_span_gold'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    gold['effect_tokens'] = gold['effect_span_gold'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

    stats['sent_len_min'] = int(gold['sent_tokens'].min())
    stats['sent_len_mean'] = round(gold['sent_tokens'].mean(), 1)
    stats['sent_len_max'] = int(gold['sent_tokens'].max())

    attrib_pos = gold[gold['attrib'] == True]
    stats['cause_len_min'] = int(attrib_pos['cause_tokens'].min())
    stats['cause_len_mean'] = round(attrib_pos['cause_tokens'].mean(), 1)
    stats['cause_len_max'] = int(attrib_pos['cause_tokens'].max())

    stats['effect_len_min'] = int(attrib_pos['effect_tokens'].min())
    stats['effect_len_mean'] = round(attrib_pos['effect_tokens'].mean(), 1)
    stats['effect_len_max'] = int(attrib_pos['effect_tokens'].max())

    # === CAUSAL MARKERS ===
    import re
    markers = {
        'cause': r'\bcause[sd]?\b',
        'lead_to': r'\blead(?:s|ing)?\s+to\b',
        'contribute_to': r'\bcontribute[sd]?\s+to\b',
        'result_in': r'\bresult[sd]?\s+in\b'
    }

    for name, pattern in markers.items():
        count = attrib_pos['sentence'].str.contains(pattern, case=False, regex=True).sum()
        stats[f'marker_{name}'] = int(count)

    any_marker_pattern = r'\b(cause[sd]?|lead(?:s|ing)?\s+to|contribute[sd]?\s+to|result[sd]?\s+in)\b'
    with_markers = attrib_pos['sentence'].str.contains(any_marker_pattern, case=False, regex=True).sum()
    stats['n_explicit_markers'] = int(with_markers)
    stats['n_implicit'] = int(len(attrib_pos) - with_markers)
    stats['pct_implicit'] = round(stats['n_implicit'] / len(attrib_pos) * 100, 0)

    # === SPEECH-LEVEL AGGREGATION ===
    print("\nAggregating to speech level...")
    speech = gold.groupby(['iso3', 'year']).agg({
        'com': 'max',
        'attrib': 'max',
        'dcha': 'max',
        'candidate_id': 'count'
    }).rename(columns={'candidate_id': 'n_candidates'}).reset_index()

    speech['com'] = speech['com'].astype(int)
    speech['attrib'] = speech['attrib'].astype(int)
    speech['dcha'] = speech['dcha'].astype(int)

    stats['n_speeches_with_candidates'] = len(speech)
    stats['n_speeches_with_attrib'] = int(speech['attrib'].sum())
    stats['n_speeches_with_dcha'] = int(speech['dcha'].sum())

    # === COUNTRY-LEVEL DCHA COUNTS ===
    print("\nCalculating country-level DCHA counts...")
    country_stats = gold.groupby('iso3').agg({
        'dcha': 'sum',
        'attrib': 'sum',
        'candidate_id': 'count'
    }).rename(columns={'candidate_id': 'total', 'attrib': 'attrib_count', 'dcha': 'dcha_count'}).reset_index()
    country_stats = country_stats.sort_values('dcha_count', ascending=False)

    stats['top_countries'] = country_stats.head(15).to_dict('records')
    stats['n_countries'] = gold['iso3'].nunique()

    # === COMPARISON WITH ORIGINAL ===
    orig_period = gold[gold['year'] >= 2014]
    extended_period = gold[gold['year'] < 2014]

    stats['comparison'] = {
        'original_period': {
            'years': '2014-2024',
            'n_candidates': len(orig_period),
            'n_attrib': int(orig_period['attrib'].sum()),
            'n_dcha': int(orig_period['dcha'].sum())
        },
        'extended_period': {
            'years': '1989-2013',
            'n_candidates': len(extended_period),
            'n_attrib': int(extended_period['attrib'].sum()),
            'n_dcha': int(extended_period['dcha'].sum())
        },
        'increase_pct': round(len(extended_period) / len(orig_period) * 100, 1) if len(orig_period) > 0 else 0
    }

    # === SAVE OUTPUTS ===
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stats_path = OUTPUT_DIR / "extended_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics to: {stats_path}")

    speech_path = OUTPUT_DIR / "speech_level_indicators.csv"
    speech.to_csv(speech_path, index=False)
    print(f"Saved speech-level indicators to: {speech_path}")

    country_path = OUTPUT_DIR / "country_dcha_counts.csv"
    country_stats.to_csv(country_path, index=False)
    print(f"Saved country DCHA counts to: {country_path}")

    # === PRINT SUMMARY ===
    print("\n" + "=" * 80)
    print("EXTENDED DATASET STATISTICS SUMMARY")
    print("=" * 80)
    print(f"""
SENTENCE-LEVEL:
  Total candidates: {stats['n_candidates']}
  ATTRIB=1: {stats['n_attrib']}
  DCHA=1 sentences: {stats['n_dcha_sentences']}

LINK TYPE DISTRIBUTION:
  NO_CAUSAL_EXTRACTION: {stats['n_no_causal']} ({stats['pct_no_causal']}%)
  OTHER_UNCLEAR: {stats['n_other_unclear']} ({stats['pct_other_unclear']}%)
  C2H_HARM: {stats['n_c2h_harm']} ({stats['pct_c2h_harm']}%)
  C2H_COBEN: {stats['n_c2h_coben']}
  H2C_JUST: {stats['n_h2c_just']}
  Directed total: {stats['pct_directed']}%

SPLITS:
  Train ({splits_data['splits']['train']['years']}): {stats['n_train']}
  Dev ({splits_data['splits']['dev']['years']}): {stats['n_dev']}
  Test ({splits_data['splits']['test']['years']}): {stats['n_test']}

COMPARISON WITH ORIGINAL:
  Original period (2014-2024): {stats['comparison']['original_period']['n_candidates']} candidates
  Extended period (1989-2013): {stats['comparison']['extended_period']['n_candidates']} candidates
  Increase: +{stats['comparison']['increase_pct']}%

  Pre-2014 DCHA: {stats['comparison']['extended_period']['n_dcha']} claims
""")

    print("TOP COUNTRIES BY DCHA:")
    for i, c in enumerate(stats['top_countries'][:10], 1):
        print(f"  {i}. {c['iso3']}: DCHA={c['dcha_count']}, Attrib={c['attrib_count']}, Total={c['total']}")

    print("\n" + "=" * 80)
    print("Pipeline complete!")
    print("=" * 80)

    return stats


if __name__ == "__main__":
    main()
