#!/usr/bin/env python3
"""
Analyze combined evaluation results.

Splits LLM predictions by era:
- Pre-1989: Computes FALSE POSITIVE rates (any ATTRIB/DCHA = hallucination)
- Post-1989: Computes standard metrics (Precision, Recall, F1)

Usage:
    python analyze_combined_results.py [--run-dir runs/run_name]
    python analyze_combined_results.py --all  # Analyze all runs
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
RUNS_DIR = ROOT_DIR / "runs"
GOLD_DATA = ROOT_DIR / "data" / "benchmark" / "combined_evaluation" / "candidates_gold.csv"
OUTPUT_DIR = ROOT_DIR / "eval" / "outputs"


def load_gold_data() -> pd.DataFrame:
    """Load gold standard with era labels."""
    return pd.read_csv(GOLD_DATA)


def compute_metrics(y_true, y_pred, task_name: str) -> dict:
    """Compute P/R/F1 for binary classification."""
    # Handle edge cases
    if sum(y_true) == 0 and sum(y_pred) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 0}
    if sum(y_true) == 0:
        # No positives in gold, any prediction is false positive
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {
        'precision': round(p, 3),
        'recall': round(r, 3),
        'f1': round(f1, 3),
        'support': int(sum(y_true))
    }


def analyze_run(run_dir: Path, gold_df: pd.DataFrame) -> dict:
    """Analyze a single run, splitting by era."""
    pred_file = run_dir / "predictions.csv"
    if not pred_file.exists():
        return None

    pred_df = pd.read_csv(pred_file)

    # Merge with gold to get era labels
    merged = gold_df.merge(pred_df, on='candidate_id', suffixes=('_gold', '_pred'))

    # Split by era
    pre_1989 = merged[merged['era'] == 'pre_1989']
    post_1989 = merged[merged['era'] == 'post_1989']

    results = {
        'run_name': run_dir.name,
        'n_total': len(merged),
        'n_pre_1989': len(pre_1989),
        'n_post_1989': len(post_1989)
    }

    # === PRE-1989: FALSE POSITIVE ANALYSIS ===
    if len(pre_1989) > 0:
        # ATTRIB false positives
        attrib_col = 'attrib' if 'attrib' in pre_1989.columns else 'attrib_pred'
        if attrib_col in pre_1989.columns:
            n_fp_attrib = pre_1989[attrib_col].sum()
            results['pre_1989_attrib_fp'] = int(n_fp_attrib)
            results['pre_1989_attrib_fp_rate'] = round(n_fp_attrib / len(pre_1989), 4)
        else:
            results['pre_1989_attrib_fp'] = 0
            results['pre_1989_attrib_fp_rate'] = 0.0

        # DCHA false positives (directed link types)
        dcha_types = ['C2H_HARM', 'C2H_COBEN', 'H2C_JUST']
        link_col = 'link_type' if 'link_type' in pre_1989.columns else 'link_type_pred'
        if link_col in pre_1989.columns:
            n_fp_dcha = pre_1989[link_col].isin(dcha_types).sum()
            results['pre_1989_dcha_fp'] = int(n_fp_dcha)
            results['pre_1989_dcha_fp_rate'] = round(n_fp_dcha / len(pre_1989), 4)
        else:
            results['pre_1989_dcha_fp'] = 0
            results['pre_1989_dcha_fp_rate'] = 0.0

    # === POST-1989: STANDARD METRICS ===
    if len(post_1989) > 0:
        # Task A: Attribution
        attrib_col = 'attrib' if 'attrib' in post_1989.columns else 'attrib_pred'
        if attrib_col in post_1989.columns:
            y_true_attrib = post_1989['attrib_gold'].astype(bool).astype(int)
            y_pred_attrib = post_1989[attrib_col].astype(bool).astype(int)
            results['post_1989_task_a'] = compute_metrics(y_true_attrib, y_pred_attrib, 'ATTRIB')

        # Task E: DCHA (derived from link_type)
        dcha_types = ['C2H_HARM', 'C2H_COBEN', 'H2C_JUST']
        link_col = 'link_type' if 'link_type' in post_1989.columns else 'link_type_pred'
        if link_col in post_1989.columns:
            y_true_dcha = post_1989['link_type_gold'].isin(dcha_types).astype(int)
            y_pred_dcha = post_1989[link_col].isin(dcha_types).astype(int)
            results['post_1989_task_e'] = compute_metrics(y_true_dcha, y_pred_dcha, 'DCHA')

        # Task D: Link type (macro F1)
        if link_col in post_1989.columns:
            link_types = ['NO_CAUSAL_EXTRACTION', 'OTHER_UNCLEAR', 'C2H_HARM', 'C2H_COBEN', 'H2C_JUST']
            y_true_link = post_1989['link_type_gold']
            y_pred_link = post_1989[link_col]

            # Compute per-class F1
            per_class_f1 = {}
            for lt in link_types:
                true_binary = (y_true_link == lt).astype(int)
                pred_binary = (y_pred_link == lt).astype(int)
                if true_binary.sum() > 0 or pred_binary.sum() > 0:
                    _, _, f1, _ = precision_recall_fscore_support(
                        true_binary, pred_binary, average='binary', zero_division=0
                    )
                    per_class_f1[lt] = round(f1, 3)

            macro_f1 = sum(per_class_f1.values()) / len(per_class_f1) if per_class_f1 else 0
            results['post_1989_task_d'] = {
                'macro_f1': round(macro_f1, 3),
                'per_class_f1': per_class_f1
            }

    return results


def print_results(results: dict):
    """Pretty print results for a single run."""
    print(f"\n{'='*70}")
    print(f"RUN: {results['run_name']}")
    print(f"{'='*70}")

    print(f"\nSamples: {results['n_total']} total")
    print(f"  Pre-1989:  {results['n_pre_1989']}")
    print(f"  Post-1989: {results['n_post_1989']}")

    # Pre-1989 false positives
    print(f"\n--- PRE-1989 FALSE POSITIVE ANALYSIS ---")
    print(f"(Any ATTRIB/DCHA prediction = hallucination)")
    print(f"  ATTRIB FP: {results.get('pre_1989_attrib_fp', 'N/A')}/{results['n_pre_1989']} "
          f"({results.get('pre_1989_attrib_fp_rate', 0)*100:.2f}%)")
    print(f"  DCHA FP:   {results.get('pre_1989_dcha_fp', 'N/A')}/{results['n_pre_1989']} "
          f"({results.get('pre_1989_dcha_fp_rate', 0)*100:.2f}%)")

    # Post-1989 metrics
    print(f"\n--- POST-1989 BENCHMARK METRICS ---")
    if 'post_1989_task_a' in results:
        m = results['post_1989_task_a']
        print(f"  Task A (Attribution): P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    if 'post_1989_task_e' in results:
        m = results['post_1989_task_e']
        print(f"  Task E (DCHA):        P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    if 'post_1989_task_d' in results:
        m = results['post_1989_task_d']
        print(f"  Task D (Link Type):   Macro-F1={m['macro_f1']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze combined evaluation results")
    parser.add_argument("--run-dir", type=str, help="Specific run directory to analyze")
    parser.add_argument("--all", action="store_true", help="Analyze all runs in runs/")
    args = parser.parse_args()

    # Load gold data
    if not GOLD_DATA.exists():
        print(f"ERROR: Gold data not found at {GOLD_DATA}")
        print("Run prepare_combined_evaluation.py first.")
        return

    gold_df = load_gold_data()
    print(f"Loaded gold data: {len(gold_df)} samples")

    # Find runs to analyze
    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    elif args.all:
        run_dirs = sorted(RUNS_DIR.glob("**/combined_*"))
        run_dirs = [d for d in run_dirs if d.is_dir() and (d / "predictions.csv").exists()]
    else:
        # Look for combined_* runs by default
        run_dirs = sorted(RUNS_DIR.glob("**/combined_*"))
        run_dirs = [d for d in run_dirs if d.is_dir() and (d / "predictions.csv").exists()]

    if not run_dirs:
        print("\nNo runs found to analyze.")
        print("Run LLM baselines first, or specify --run-dir or --all")
        return

    print(f"\nAnalyzing {len(run_dirs)} run(s)...")

    all_results = []
    for run_dir in run_dirs:
        result = analyze_run(run_dir, gold_df)
        if result:
            all_results.append(result)
            print_results(result)

    # Summary comparison table
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY COMPARISON")
        print(f"{'='*70}")

        print(f"\n{'Model':<25} {'ATTRIB FP%':<12} {'DCHA FP%':<12} {'Task A F1':<12} {'Task E F1':<12}")
        print("-" * 73)
        for r in all_results:
            model = r['run_name'].replace('combined_', '')
            attrib_fp = r.get('pre_1989_attrib_fp_rate', 0) * 100
            dcha_fp = r.get('pre_1989_dcha_fp_rate', 0) * 100
            task_a_f1 = r.get('post_1989_task_a', {}).get('f1', 0)
            task_e_f1 = r.get('post_1989_task_e', {}).get('f1', 0)
            print(f"{model:<25} {attrib_fp:>10.2f}% {dcha_fp:>10.2f}% {task_a_f1:>10.3f} {task_e_f1:>10.3f}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "combined_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
