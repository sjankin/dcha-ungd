#!/usr/bin/env python3
"""
Analyze negative control results to compute false positive rates.

Usage:
    python analyze_negative_control.py

Reads predictions from runs/neg_control_*/ directories.
"""

import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
RUNS_DIR = ROOT_DIR / "runs"
CONTROL_DATA = ROOT_DIR / "data" / "benchmark" / "negative_control" / "candidates_gold.csv"


def analyze_run(run_dir: Path) -> dict:
    """Analyze a single negative control run."""
    pred_file = run_dir / "predictions.csv"
    if not pred_file.exists():
        return None

    pred_df = pd.read_csv(pred_file)
    n_total = len(pred_df)

    # Count false positives (any ATTRIB=True)
    if 'attrib' in pred_df.columns:
        n_fp_attrib = pred_df['attrib'].sum()
    else:
        n_fp_attrib = 0

    # Count DCHA false positives (link_type is directed)
    directed_types = ['C2H_HARM', 'C2H_COBEN', 'H2C_JUST']
    if 'link_type' in pred_df.columns:
        n_fp_dcha = pred_df['link_type'].isin(directed_types).sum()
    else:
        n_fp_dcha = 0

    return {
        'run_name': run_dir.name,
        'n_total': n_total,
        'n_fp_attrib': int(n_fp_attrib),
        'fp_rate_attrib': n_fp_attrib / n_total if n_total > 0 else 0,
        'n_fp_dcha': int(n_fp_dcha),
        'fp_rate_dcha': n_fp_dcha / n_total if n_total > 0 else 0
    }


def main():
    print("=" * 70)
    print("Negative Control Analysis: LLM False Positive Rates")
    print("=" * 70)

    # Find negative control runs
    neg_runs = list(RUNS_DIR.glob("neg_control_*"))

    if not neg_runs:
        print("\nNo negative control runs found.")
        print("Run ./run_negative_control.sh first.")
        return

    print(f"\nFound {len(neg_runs)} negative control runs.\n")

    results = []
    for run_dir in sorted(neg_runs):
        result = analyze_run(run_dir)
        if result:
            results.append(result)
            print(f"{result['run_name']}:")
            print(f"  ATTRIB FP: {result['n_fp_attrib']}/{result['n_total']} ({result['fp_rate_attrib']*100:.1f}%)")
            print(f"  DCHA FP:   {result['n_fp_dcha']}/{result['n_total']} ({result['fp_rate_dcha']*100:.1f}%)")
            print()

    if results:
        # Summary table
        print("=" * 70)
        print("SUMMARY: False Positive Rates on Pre-1989 Data")
        print("=" * 70)
        print(f"\n{'Model':<30} {'ATTRIB FP Rate':<20} {'DCHA FP Rate':<20}")
        print("-" * 70)
        for r in results:
            model = r['run_name'].replace('neg_control_', '')
            print(f"{model:<30} {r['fp_rate_attrib']*100:>6.1f}%{'':<13} {r['fp_rate_dcha']*100:>6.1f}%")

        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)
        print("""
- ATTRIB FP Rate: LLM incorrectly identifies causal attribution in non-causal text
- DCHA FP Rate: LLM incorrectly identifies climate→health attribution

Lower is better. High rates indicate the model hallucinates causal claims.
Pre-1989 UN speeches should have essentially zero climate-health content.
""")

        # Save results
        results_df = pd.DataFrame(results)
        output_file = ROOT_DIR / "eval" / "outputs" / "negative_control_results.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
