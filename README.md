# Survey Translation Agent

An agent-based pipeline for translating English social-science survey items into culturally localized Standard German using open-source large language models.

The project compares **Mistral-7B-Instruct-v0.3** and **Qwen2.5-7B-Instruct** in a three-stage workflow:

```text
English survey item
        ↓
1. Initial translation
        ↓
2. Cross-model review
        ↓
3. Reviewer-guided self-correction
        ↓
Automatic and trap-focused evaluation
```

Mistral translations are reviewed by Qwen, and Qwen translations are reviewed by Mistral. Each original model then corrects its own output using the other model's feedback.

## Research objectives

The project investigates:

1. How well open-source instruction-tuned LLMs translate survey questions and response scales from English into German.
2. Whether cross-model review can identify translation problems without seeing the human German reference.
3. Whether reviewer-guided self-correction improves translation quality.
4. Whether automatic translation metrics agree with performance on explicitly designed translation traps.
5. Whether correction improves when the model receives a precise description of the known error.

## Datasets

### European Social Survey

The ESS dataset contains **109 aligned English--German items** derived from the European Social Survey Round 11 questionnaires:

| Item type | Count |
|---|---:|
| Attitudinal | 71 |
| Sociodemographic | 26 |
| Behavioral | 12 |
| **Total** | **109** |

The experimental dataset, including translated response scales, is located at:

```text
data/processed/ess11_items_with_scale.json
```

### Custom diagnostic dataset

The custom experiment uses **41 survey items** containing known translation difficulties:

```text
data/custom/custom_test_items_de_CH.json
```

The items cover 12 categories:

- Lexical accuracy
- Sense disambiguation
- Conceptual equivalence
- Cultural and institutional referents
- Vague quantifier calibration
- Register and politeness
- Grammatical gender
- Idioms and metaphors
- Negation and polarity
- Scale-label equivalence
- Numeric and unit conversion
- Reading load and length

Each custom item includes an English source, English response scale, German reference, German reference scale, item type, diagnostic category, and description of the intended translation trap.

## Models

| Key | Model | Role |
|---|---|---|
| `mistral` | Mistral-7B-Instruct-v0.3 | Translation, cross-review, and self-correction |
| `qwen` | Qwen2.5-7B-Instruct | Translation, cross-review, and self-correction |
| Oracle/judge | Qwen3-32B | Trap evaluation and expert-feedback correction experiment |

The model locations and generation limits are configured in `config.yaml`. The current configuration expects locally available model weights.

## Shared prompting strategy

The translator, reviewer, and corrector use a common survey-translation framework. The prompts emphasize:

- Formal German `Sie` register
- Gender-inclusive language
- Preservation of measurement intent
- Semantic, polarity, frequency, and intensity equivalence
- Correct and complete response scales
- Functional localization of placeholders and institutions
- Avoidance of false friends and literal idiom translation
- Metric-unit and currency adaptation where appropriate
- Natural and concise Standard German
- Strict machine-readable JSON output

The correction prompt also instructs the model to evaluate reviewer feedback critically and avoid changing translations when the review is incorrect.

## Pipeline

### Step 1: Translation

Translate one model and dataset:

```bash
python agent/run_translation.py --model mistral --dataset custom
python agent/run_translation.py --model qwen --dataset ess
```

Run all model/dataset combinations:

```bash
python agent/run_translation.py --model all --dataset all
```

Outputs follow this naming pattern:

```text
data/outputs/step1_<model>_<dataset>_translated.json
```

The runner checkpoints results after every completed item.

### Step 2: Cross-review

```bash
python agent/reviewer.py --translated all --dataset all
```

Cross-review mapping:

```text
Mistral translation → Qwen review
Qwen translation    → Mistral review
```

Outputs:

```text
data/outputs/step2_<model>_<dataset>_reviewed.json
```

The reviewer checks register, gender language, semantic drift, response scales, naturalness, idioms, institutional references, units, numbers, and completeness.

### Step 3: Self-correction

```bash
python agent/corrected.py --translated all --dataset all
```

Outputs:

```text
data/outputs/step3_<model>_<dataset>_corrected.json
```

Items without identified problems pass through unchanged. Flagged or structurally broken items are regenerated and validated.

## Evaluation

### Reference-based metrics

The main evaluation calculates:

- chrF
- Multilingual BERTScore F1 with `xlm-roberta-large`
- Reference-based COMET with `Unbabel/wmt22-comet-da`

Run:

```bash
python evaluation/evaluate.py \
  --data-dir data \
  --steps-dir data/outputs \
  --out-dir results \
  --metrics chrf bertscore comet
```

Generated tables include:

```text
results/scores_per_item.csv
results/summary.csv
results/summary_with_delta.csv
```

For each metric, the evaluator now reports:

- `question_*`: question-only quality
- `scale_*`: response-scale-only quality
- `combined_*`: question and response scale evaluated together

The headline `chrf`, `bertscore_f1`, and `comet` columns use the **combined** question-and-scale score. Empty translated scales are retained and penalized rather than skipped.

### Reference-free COMET-QE

```bash
python agent/2.py
```

This evaluates question translations using `Unbabel/wmt20-comet-qe-da` without a German reference and writes:

```text
results/cometkiwi_results.csv
```

### Trap-focused evaluation

The custom dataset includes an explicit problem description for every item. `agent/check_traps.py` uses Qwen3-32B to judge whether the model successfully avoided that problem.

An additional oracle experiment in `agent/oracle_corrector.py` supplies the known problem directly to Qwen3-32B as expert feedback before reevaluating the translation.

## Main results

### Corrected-minus-raw question-only results from the completed run

| Dataset | Model | chrF Δ | BERTScore Δ | COMET Δ |
|---|---|---:|---:|---:|
| Custom | Mistral | -1.392 | -0.00409 | -0.01406 |
| Custom | Qwen | +0.229 | +0.00150 | +0.01148 |
| ESS | Mistral | +0.257 | -0.00112 | -0.00328 |
| ESS | Qwen | -0.781 | -0.00039 | -0.00250 |

These values were produced by the earlier question-only evaluation run. Rerunning `evaluation/evaluate.py` produces separate question, scale, and combined results; the combined columns become the new headline metrics. In the completed question-only run, Qwen on the custom dataset is the only configuration that improves across chrF, BERTScore, reference-based COMET, and reference-free COMET-QE.

### Trap avoidance

| Output | Traps avoided |
|---|---:|
| Raw Mistral | 16/41 (39.0%) |
| Corrected Mistral | 15/41 (36.6%) |
| Raw Qwen | 17/41 (41.5%) |
| Corrected Qwen | 16/41 (39.0%) |
| Oracle-corrected output | 21/41 (51.2%) |

The results show that ordinary self-correction does not consistently improve survey translation. The stronger oracle result suggests that the quality and specificity of error diagnosis are central to successful correction.

## Project structure

```text
survey_translation_agent/
├── agent/                 Translation, review, correction, and analysis code
├── data/
│   ├── custom/            Custom diagnostic datasets
│   ├── raw/               ESS source PDFs
│   ├── processed/         Aligned ESS datasets
│   └── outputs/           Step 1, Step 2, and Step 3 model outputs
├── evaluation/            Metric calculation and plotting
├── prompts/               Translation, review, and correction prompts
├── results/               Metric tables, audits, and figures
├── scripts/               SLURM job scripts
├── report/                LaTeX seminar report
├── config.yaml            Model and path configuration
└── requirements.txt       Python dependencies
```

## Installation

The experiments were developed with Python 3.10 and GPU inference. Create a virtual environment and install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Additional packages used by the complete project include `PyYAML`, `pdfplumber`, `matplotlib`, and `seaborn`.

The COMET and BERTScore evaluations may download model checkpoints if they are not already cached.

## Seminar report

The full project report is available at:

```text
report/report.tex
```

Compile it from the repository root with a LaTeX distribution:

```bash
latexmk -pdf -interaction=nonstopmode -output-directory=report report/report.tex
```

Alternatively, upload `report/report.tex` and the figures in `results/` to Overleaf.

## Summary

This project demonstrates that an LLM review-and-correction pipeline can produce useful improvements in some settings, but additional agent stages do not guarantee higher translation quality. The experiments highlight the importance of precise error diagnosis, survey-specific constraints, response-scale evaluation, and targeted human validation.
