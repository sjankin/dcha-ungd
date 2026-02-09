# DCHA-UNGD v1.0: Directed Climate-Health Attribution in UN General Debate Speeches

Dataset and replication materials for:

> **DCHA-UNGD: An Extended Dataset and Benchmarks for Directed Climate-Health Attribution in UN General Debate Speeches, 1989-2024**
> Hannah Bechara, Krishnamoorthy Manohara, Niheer Dasandi, Slava Jankin.
> KDD 2026 (Datasets & Benchmarks Track).

## Overview

DCHA-UNGD v1.0 is an extended dataset and benchmark suite for **Directed Climate-Health Attribution (DCHA)** in elite political speech. It provides:

- **907 annotated candidate sentences** from UN General Debate speeches (1989-2024) with hierarchical labels for attribution presence, cause/effect spans, and directed linkage type (5-way taxonomy)
- **5,000 negative control sentences** from pre-1989 speeches (1946-1988), where climate-health attribution is anachronistic
- **Evaluation toolkit** for attribution detection, span extraction, and directed linkage classification
- **LLM baselines** with predictions from DeepSeek-R1, GPT-5.2, and Claude Opus 4.5
- **Construct validation** demonstrating that DCHA captures distinct information from co-mention

### Key Findings

1. **DCHA specificity**: The two-stage DCHA construct (attribution + directed span matching) achieves near-zero false positive rates (<0.1%) even when attribution alone has 20-29% false positives on the negative control corpus
2. **Fine-tuning matches LLMs**: Fine-tuned RoBERTa matches frontier LLM performance with only 632 training examples
3. **Co-mention overestimates**: Speech-level co-mention overestimates directed climate-health attribution by 91%

## Dataset

### Label Schema

| Level | Label | Description |
|-------|-------|-------------|
| 1 | ATTRIB | Binary: does the sentence assert a causal claim? |
| 1 | CAUSE/EFFECT spans | Contiguous text spans for cause and effect |
| 2 | C2H_HARM | Climate impacts cause health harms |
| 2 | C2H_COBEN | Climate action produces health co-benefits |
| 2 | H2C_JUST | Health outcomes justify climate action |
| 2 | OTHER_UNCLEAR | Causal but ambiguous direction |
| 2 | NO_CAUSAL_EXTRACTION | No causal claim despite lexical co-mention |

### Statistics

| | Count |
|---|---|
| Annotated candidates (1989-2024) | 907 |
| Attribution present (ATTRIB=1) | 304 |
| Derived DCHA=1 | 72 |
| Negative control (1946-1988) | 5,000 |
| Train / Dev / Test split | 632 / 125 / 150 |

### Files

```
data/
├── benchmark/
│   ├── dcha_ungd_extended_v1/
│   │   ├── candidates_gold.csv      # 907 annotated candidates
│   │   ├── splits.json              # Chronological train/dev/test
│   │   └── lexicons/                # DC (34 terms) and DH (28 terms) lexicons
│   └── negative_control/
│       ├── candidates_gold.csv      # 5,000 pre-1989 sentences
│       └── splits.json
├── speech_level_indicators.csv      # Speech-level binary indicators
├── country_dcha_counts.csv          # Country-level DCHA counts
└── extended_statistics.json         # Dataset-level statistics
```

## Quick Start

### Evaluate a model

```python
import pandas as pd
from scripts.compute_metrics import compute_attribution_metrics

gold = pd.read_csv("data/benchmark/dcha_ungd_extended_v1/candidates_gold.csv")
predictions = pd.read_csv("your_predictions.csv")

metrics = compute_attribution_metrics(gold, predictions)
print(f"Attribution F1: {metrics['f1']:.3f}")
```

### Run LLM baselines

```bash
# 1. Set up environment
pip install -r requirements.txt
cp .env.example .env  # then fill in your API keys

# 2. Run a single model on the test set
python scripts/run_llm_baseline.py \
    --provider openai \
    --model gpt-5.2 \
    --data-dir data/benchmark/dcha_ungd_extended_v1 \
    --split test

# 3. Analyse results
python scripts/analyze_combined_results.py --all
```

### Fine-tune RoBERTa

```bash
python scripts/train_roberta.py \
    --data-dir data/benchmark/dcha_ungd_extended_v1 \
    --output-dir models/roberta_finetuned \
    --epochs 10

python scripts/evaluate_finetuned_roberta.py \
    --model-dir models/roberta_finetuned \
    --data-dir data/benchmark/dcha_ungd_extended_v1 \
    --split test
```

## Repository Structure

```
dcha-ungd/
├── data/                  # Released dataset (CC BY 4.0)
├── scripts/               # All Python and shell scripts (MIT)
├── prompts/v1/            # LLM prompt templates and schema
├── runs/                  # LLM prediction outputs for replication
├── eval/outputs/          # Aggregated evaluation results
├── analysis/outputs/      # Construct validation results
├── paper/                 # Paper source (LaTeX + figures)
├── docs/                  # Annotation guidelines
├── .env.example           # API key template
├── requirements.txt       # Python dependencies
└── LICENSE                # CC BY 4.0 (data) + MIT (code)
```

## Benchmark Tasks

| Task | Description | Metric |
|------|-------------|--------|
| A | Attribution detection (ATTRIB) | F1 |
| B | Cause span extraction | Token-level F1 |
| C | Effect span extraction | Token-level F1 |
| D | Directed linkage classification (5-way) | Macro F1 |
| E | DCHA detection (derived flag) | F1 |

## Replication

To reproduce all results from the paper:

```bash
# 1. Run LLM baselines on full candidate set + negative control
bash scripts/run_all_baselines.sh

# 2. Analyse results
python scripts/analyze_combined_results.py --all
python scripts/analyze_negative_control.py --all

# 3. Generate paper statistics
python scripts/build_paper_data.py
```

Pre-computed LLM predictions are provided in `runs/` for verification without re-running the models.

## Citation

```bibtex
@inproceedings{bechara2026dcha,
  title     = {{DCHA-UNGD}: An Extended Dataset and Benchmarks for Directed
               Climate--Health Attribution in {UN} General Debate Speeches,
               1989--2024},
  author    = {Bechara, Hannah and Manohara, Krishnamoorthy and
               Dasandi, Niheer and Jankin, Slava},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge
               Discovery and Data Mining (KDD '26)},
  year      = {2026},
  address   = {Jeju, Korea}
}
```

## License

- **Data** (`data/` directory): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Code** (all other files): [MIT License](LICENSE)

## Acknowledgements

This project has received funding from the European Union's Horizon Europe research and innovation program under Grant Agreement No 101057131, Climate Action To Advance HeaLthY Societies in Europe (CATALYSE).

Source corpus: [UN General Debate Corpus](https://doi.org/10.1177/00223433241275335) (Jankin, Baturo, and Dasandi, 2025).
