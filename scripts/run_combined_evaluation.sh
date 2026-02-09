#!/bin/bash
# Run all LLMs on combined evaluation dataset (pre-1989 + post-1989)
#
# Models (matching original paper + Opus 4.5 upgrade):
#   - DeepSeek R1 (deepseek-reasoner) - fewshot
#   - GPT-5.2 (gpt-5.2) - fewshot
#   - Claude Opus 4.5 (claude-opus-4-5-20251101) - zeroshot
#   - Gemini 3 Pro (gemini-3-pro-preview) - zeroshot
#
# Usage:
#   ./run_combined_evaluation.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Combined Evaluation: Pre + Post 1989"
echo "========================================"
echo ""
echo "Dataset: 5,000 pre-1989 + 907 post-1989 = 5,907 samples"
echo ""

# Check if combined data exists
if [ ! -f "../data/benchmark/combined_evaluation/candidates_gold.csv" ]; then
    echo "Preparing combined dataset..."
    python3 prepare_combined_evaluation.py --neg-samples 5000
fi

DATA_DIR="data/benchmark/combined_evaluation"

# Create runs directory
mkdir -p ../runs

echo ""
echo "Starting LLM evaluations..."
echo "Logs will be written to ../runs/combined_*.log"
echo ""

# Function to run model with unbuffered output
run_model() {
    provider=$1
    model=$2
    variant=$3
    run_name=$4

    echo "Starting $run_name ($model, $variant)..."

    PYTHONUNBUFFERED=1 nohup python3 run_llm_baseline.py \
        --provider "$provider" \
        --model "$model" \
        --variant "$variant" \
        --split test \
        --data-dir "$DATA_DIR" \
        --run-name "$run_name" \
        > "../runs/${run_name}.log" 2>&1 &

    echo "  PID: $!"
    echo "  Log: ../runs/${run_name}.log"
    echo ""
}

# Run all models in parallel (they use different APIs)
run_model "deepseek" "deepseek-reasoner" "fewshot" "combined_deepseek"
run_model "openai" "gpt-5.2" "fewshot" "combined_gpt5"
run_model "anthropic" "claude-opus-4-5-20251101" "zeroshot" "combined_claude"
run_model "google" "gemini-3-pro-preview" "zeroshot" "combined_gemini"

echo "========================================"
echo "All models started!"
echo "========================================"
echo ""
echo "Monitor progress:"
echo "  tail -f ../runs/combined_*.log"
echo ""
echo "Check status:"
echo "  ps aux | grep run_llm_baseline"
echo ""
echo "After completion, analyze results:"
echo "  python3 analyze_combined_results.py"
echo ""

# Save PIDs for later reference
ps aux | grep "run_llm_baseline" | grep -v grep > ../runs/running_pids.txt 2>/dev/null || true
echo "Running processes saved to ../runs/running_pids.txt"
