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
| Oracle/judge | Qwen3.5-27B | Trap evaluation and expert-feedback correction experiment |

The model locations and generation limits are configured in `config.yaml`. The current configuration expects locally available model weights.

## Setup

The experiments were developed with Python 3.10. Run all commands from the repository root. A CUDA-capable GPU is strongly recommended for model inference and COMET/BERTScore evaluation; the Qwen3.5-27B judge and oracle experiments require substantially more GPU memory than the 7B pipeline models.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
COMET and BERTScore may download model checkpoints when they are not already cached.

Before running inference, update the local model paths in `config.yaml`. The trap judge and oracle corrector have a separate default Qwen3.5-27B path, which can be overridden with `--model-path`.

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
results/paired_deltas.csv
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

The custom dataset includes an explicit problem description for every item. `agent/check_traps.py` uses Qwen3.5-27B to judge whether the model successfully avoided that problem.

```bash
python agent/check_traps.py --model-path /path/to/Qwen3.5-27B
```

By default, this checks the Step 1 and Step 3 custom outputs for both models and writes per-item judgments plus a summary to `results/trap_checks/`.

An additional oracle experiment in `agent/oracle_corrector.py` supplies the known problem directly to Qwen3.5-27B as expert feedback before reevaluating the translation.

```bash
python agent/oracle_corrector.py \
  --models mistral qwen \
  --model-path /path/to/Qwen3.5-27B

python agent/check_traps.py \
  --inputs \
    data/outputs/step4_oracle_mistral_custom_corrected.json \
    data/outputs/step4_oracle_qwen_custom_corrected.json \
  --output-dir results/oracle_trap_checks \
  --model-path /path/to/Qwen3.5-27B
```

The oracle correction is applied only to Step 3 items that failed the initial trap check. Its outputs are written to `data/outputs/step4_oracle_*`, with run summaries in `results/oracle_correction_summary.json` and `results/oracle_trap_checks/`.

### Plots

After the metric, trap, and oracle evaluations are available, generate report-ready PNG and PDF figures with:

```bash
python evaluation/plot.py
```

This creates combined-score, correction-delta, question-versus-scale, and trap-avoidance figures in `results/plots/` in both PNG and PDF formats. Use `--formats png` or `--formats pdf` to generate only one format.

## Main results

### Automatic metric improvements (Step 1 → Step 3)

The pipeline evaluated 300 aligned item pairs (150 items × 2 models × 2 stages) using reference-based metrics on both questions and response scales separately.

| Dataset | Model | Combined chrF Δ | Combined BERTScore Δ | Combined COMET Δ | Interpretation |
|---|---|---:|---:|---:|---|
| Custom | Mistral | -0.002 | -0.00130 | -0.00100 | Essentially unchanged |
| Custom | Qwen | +1.232 | +0.01256 | +0.01250 | Improved (2 items corrected) |
| ESS | Mistral | -0.511 | +0.00320 | -0.00355 | Mixed, mostly unchanged/slightly worse |
| ESS | Qwen | +2.671 | +0.04188 | +0.01364 | Combined improvement via scale recovery |

**Key finding**: Only Qwen improves across all three metrics on both datasets. Mistral correction is inconsistent or slightly detrimental. Qwen's ESS improvement is driven primarily by filling ten missing scales; its question-only quality actually declines (−0.657 chrF, −0.00142 BERTScore, −0.00362 COMET).

### Structural completeness

| Measure | Step 1 | Step 3 | Improvement |
|---|---:|---:|---|
| Empty questions | 0 | 0 | — |
| Empty scales | 12 | 0 | **All recovered** |

Step 3's strongest measurable success is **recovering all 12 missing response scales**: one Mistral/ESS, one Qwen/custom, and ten Qwen/ESS. No questions are empty at any stage.

### Reviewer performance (Step 2)

The cross-reviewer correctly flags **42/300 items (14.0%)** with 80 total issues recorded. However, manual audit shows:

- **Useful precision (among flagged items)**: ~60–81% depending on model/dataset pair
- **Severe recall failure**: Reviewer misses many obvious translation problems (grammar errors, semantic shifts, malformed questions)
- **False positives common**: Even correctly flagged items often contain false or contradictory explanations
- **Scale detection strong**: 51.3% of findings concern response scales; naturalness and idiom issues are almost absent

**Conclusion**: Useful as a partial error detector but inadequate as an autonomous quality gate without additional rule-based checks.

### Trap avoidance on custom dataset

| Output | Complete traps avoided | Interpretation |
|---|---:|---|
| Mistral Step 1 | 16/41 (39.0%) | Baseline performance |
| Mistral Step 3 | 16/41 (39.0%) | No net improvement from ordinary review/correction |
| Qwen Step 1 | 21/41 (51.2%) | Stronger baseline |
| Qwen Step 3 | 21/41 (51.2%) | No net improvement |
| **Mistral Step 4 oracle** | **34/41 (82.9%)** | With exact error diagnosis |
| **Qwen Step 4 oracle** | **36/41 (87.8%)** | With exact error diagnosis |

**Ordinary self-correction produces partial component repairs but zero net increase in complete trap resolution.** Example: Qwen/item_11 changes from Chinese to valid German (question component passes) but scale-component equivalence still fails.

**Oracle feedback is transformative**: Providing the exact annotated problem to Qwen3.5-27B enables it to resolve 18/25 failed Mistral traps and 15/20 failed Qwen traps. This demonstrates that **error diagnosis is the bottleneck**: accurate problem identification is substantially more effective than generic reviewer feedback. (Note: This is an upper-bound experimental result; the same Qwen3.5-27B performs both correction and judging, and the judge sees the human reference.)

## Detailed analysis and human audit

Comprehensive manual reviews of each pipeline stage are available in the `reviewed/` directory:

- **[translation_agent_review.md](reviewed/translation_agent_review.md)**: Step 1 output audit. Both models produce understandable German for most items, but neither output is production-ready without human validation. Mistral omits only one scale; Qwen omits eleven scales and contains severe errors including Chinese text.

- **[step2_reviewer_performance.md](reviewed/step2_reviewer_performance.md)**: Step 2 cross-reviewer analysis. Reviewer useful precision is 60–81% but recall is severely limited. It excels at detecting missing scales but misses grammar errors, semantic shifts, and malformed questions. The reviewer is useful as a partial error detector but inadequate as an autonomous quality gate.

- **[step3_correction_results.md](reviewed/step3_correction_results.md)**: Step 3 correction quality. All 12 missing Step 2 scales are recovered. However, manual audit shows only a minority of the 44 changed items are unambiguously better end-to-end; many corrections are partial, cosmetic, or harmful when the corrector follows inaccurate feedback.

- **[final_results_summary.md](reviewed/final_results_summary.md)**: Complete pipeline summary with component-level trap analysis, oracle experiment details, and final conclusions.

## Project structure

```text
survey_translation_agent/
├── agent/                 Translation, review, correction, and analysis code
│   ├── run_translation.py     Step 1: translate items
│   ├── reviewer.py            Step 2: cross-review translations
│   ├── corrected.py           Step 3: self-correct based on reviews
│   ├── check_traps.py          Trap evaluation with judge model
│   ├── oracle_corrector.py     Step 4: correction with oracle feedback
│   └── extract_ess_data.py     ESS data alignment and preprocessing
├── data/
│   ├── custom/            Custom diagnostic datasets (41 items with annotated traps)
│   ├── raw/               ESS source PDFs
│   ├── processed/         Aligned ESS datasets
│   └── outputs/           Step 1, 2, 3, and 4 model outputs
├── evaluation/            Metric calculation and plotting
│   ├── evaluate.py        Reference-based metrics (chrF, BERTScore, COMET)
│   ├── metrics.py         Metric computation helpers
│   └── plot.py            Visualization and report figures
├── reviewed/              Human audit reports for each pipeline stage
├── prompts/               Translation, review, and correction prompts
├── results/               Metric tables, trap checks, and oracle correction summaries
├── scripts/               SLURM batch job templates
├── config.yaml            Model paths, generation limits, and pipeline configuration
└── requirements.txt       Python dependencies
```

## Summary

This project demonstrates that an LLM review-and-correction pipeline can produce useful improvements in some settings. The experiments highlight the importance of precise error diagnosis, survey-specific constraints, response-scale evaluation and human validation.
