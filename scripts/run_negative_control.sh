#!/bin/bash
# Run LLMs on negative control dataset (1946-1988 sentences)
# to measure false positive rate
#
# Usage:
#   ./run_negative_control.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Negative Control: LLM False Positive Test"
echo "========================================"
echo ""
echo "Testing LLMs on 1946-1988 sentences where no DCHA should exist."
echo "Any DCHA claims found = false positives."
echo ""

# Check if negative control data exists
if [ ! -f "../data/benchmark/negative_control/candidates_gold.csv" ]; then
    echo "Preparing negative control dataset..."
    python3 prepare_negative_control.py --n-samples 200
fi

NC_DIR="data/benchmark/negative_control"

echo ""
echo "Running LLM baselines on negative control..."
echo ""

# DeepSeek R1
echo "1. DeepSeek R1..."
python3 run_llm_baseline.py --provider deepseek --model deepseek-reasoner \
    --split test --data-dir "$NC_DIR" --run-name neg_control_deepseek

# GPT-5.2
echo ""
echo "2. GPT-5.2..."
python3 run_llm_baseline.py --provider openai --model gpt-5.2 \
    --split test --data-dir "$NC_DIR" --run-name neg_control_gpt5

# Claude Opus 4
echo ""
echo "3. Claude Opus 4..."
python3 run_llm_baseline.py --provider anthropic --model claude-opus-4-20250514 \
    --split test --data-dir "$NC_DIR" --run-name neg_control_claude

# Gemini 3 Pro
echo ""
echo "4. Gemini 3 Pro..."
python3 run_llm_baseline.py --provider google --model gemini-3-pro \
    --split test --data-dir "$NC_DIR" --run-name neg_control_gemini

echo ""
echo "========================================"
echo "Negative Control Complete"
echo "========================================"
echo ""
echo "Check results in ../runs/neg_control_*/"
echo ""
echo "To compute false positive rates:"
echo "  grep -l 'attrib.*true' ../runs/neg_control_*/predictions.csv"
