#!/bin/bash
# Run all baselines for DCHA-UNGD Extended benchmark
#
# Usage:
#   ./run_all_baselines.sh                # Run all baselines
#   ./run_all_baselines.sh --llm-only     # Run only LLM baselines
#   ./run_all_baselines.sh --no-llm       # Run everything except LLMs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "DCHA-UNGD Extended Benchmark Evaluation"
echo "========================================"
echo ""

# Parse arguments
RUN_LLM=true
RUN_OTHERS=true

for arg in "$@"; do
    case $arg in
        --llm-only)
            RUN_OTHERS=false
            ;;
        --no-llm)
            RUN_LLM=false
            ;;
    esac
done

# Step 1: Prepare data (if not already done)
echo "Step 1: Checking data preparation..."
if [ ! -f "../data/benchmark/dcha_ungd_extended_v1/candidates_gold.csv" ]; then
    echo "  Running prepare_extended_data.py..."
    python3 prepare_extended_data.py
else
    echo "  Data already prepared."
fi

# Step 2: Generate statistics
if [ "$RUN_OTHERS" = true ]; then
    echo ""
    echo "Step 2: Generating statistics..."
    python3 build_paper_data.py
fi

# Step 3: Majority baseline
if [ "$RUN_OTHERS" = true ]; then
    echo ""
    echo "Step 3: Computing majority baseline..."
    python3 compute_majority_baseline.py
fi

# Step 4: Fine-tuned RoBERTa
if [ "$RUN_OTHERS" = true ]; then
    echo ""
    echo "Step 4: Evaluating fine-tuned RoBERTa..."
    echo "  NOTE: Using model trained on 2014-2021 data."
    echo "  For fair comparison, retrain on 1989-2021 data."
    python3 evaluate_finetuned_roberta.py --split test
fi

# Step 5: LLM baselines
if [ "$RUN_LLM" = true ]; then
    echo ""
    echo "Step 5: Running LLM baselines..."

    # DeepSeek R1
    echo "  Running DeepSeek R1..."
    python3 run_llm_baseline.py --provider deepseek --model deepseek-reasoner --split test

    # GPT-5.2
    echo "  Running GPT-5.2..."
    python3 run_llm_baseline.py --provider openai --model gpt-5.2 --split test

    # Claude Opus 4
    echo "  Running Claude Opus 4..."
    python3 run_llm_baseline.py --provider anthropic --model claude-opus-4-20250514 --split test

    # Gemini 3 Pro
    echo "  Running Gemini 3 Pro..."
    python3 run_llm_baseline.py --provider google --model gemini-3-pro --split test
fi

echo ""
echo "========================================"
echo "All baselines complete!"
echo "========================================"
echo ""
echo "Results saved in:"
echo "  - ../data/extended_statistics.json"
echo "  - ../eval/outputs/*.csv"
echo "  - ../runs/*/"
