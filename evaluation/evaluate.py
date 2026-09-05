import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sacrebleu.metrics.chrf import CHRF

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# CONFIG
MODELS = ["mistral", "qwen"]
DATASETS = ["ess", "custom"]

SOURCE_FIELD = "source_en"
ID_FIELD = "id"

GOLD_FILES = {
    "ess": "processed/ess11_items_with_scale.json",
    "custom": "custom/custom_test_items_de_CH.json",
}

# gold field names per dataset
GOLD_FIELDS = {
    "ess": ("trans_de", "scale_de"),
    "custom": ("trans_de", "scale_de"),
}

STEP1_TEMPLATE = "step1_{model}_{dataset}_translated.json"
STEP3_TEMPLATE = "step3_{model}_{dataset}_corrected.json"

STEP1_FIELDS = ("model_trans_de", "model_trans_scale")
STEP3_FIELDS = ("corrected_de", "corrected_scale")

BERTSCORE_MODEL = "xlm-roberta-large"
COMET_MODEL = "Unbabel/wmt22-comet-da"  # standard reference-based COMET model

# Data loading helpers
def load_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected a list of objects in {path}")
    return data


def index_by_id(items: list[dict], path: Path) -> dict:
    """Map IDs to items and fail on missing or duplicate IDs."""
    out = {}
    for i, it in enumerate(items):
        key = it.get(ID_FIELD)
        if not key:
            raise ValueError(f"Missing '{ID_FIELD}' in {path} at item {i}")
        if key in out:
            raise ValueError(f"Duplicate ID '{key}' in {path}")
        out[key] = it
    return out


@dataclass
class AlignedItem:
    item_id: str
    dataset: str
    model: str
    stage: str
    source_en: str
    source_scale: str
    hyp_item: str
    hyp_scale: str
    ref_item: str
    ref_scale: str


def build_aligned_items(data_dir: Path, steps_dir: Path) -> list:
    """Load gold plus Step 1/Step 3 and require exact ID alignment."""
    aligned = []

    for dataset in DATASETS:
        gold_path = data_dir / GOLD_FILES[dataset]
        if not gold_path.exists():
            raise FileNotFoundError(f"Gold file not found for '{dataset}': {gold_path}")
        gold_items = index_by_id(load_json(gold_path), gold_path)
        item_gold_field, scale_gold_field = GOLD_FIELDS[dataset]

        for model in MODELS:
            for stage, template, fields in (
                ("raw", STEP1_TEMPLATE, STEP1_FIELDS),
                ("corrected", STEP3_TEMPLATE, STEP3_FIELDS),
            ):
                fpath = steps_dir / template.format(model=model, dataset=dataset)
                if not fpath.exists():
                    raise FileNotFoundError(f"Required pipeline output not found: {fpath}")

                item_field, scale_field = fields
                pred_items = load_json(fpath)
                pred_by_id = index_by_id(pred_items, fpath)
                missing_ids = sorted(set(gold_items) - set(pred_by_id))
                extra_ids = sorted(set(pred_by_id) - set(gold_items))
                if missing_ids or extra_ids:
                    raise ValueError(
                        f"ID mismatch in {fpath}: missing={missing_ids}, extra={extra_ids}"
                    )

                for key, gold in gold_items.items():
                    pred = pred_by_id[key]
                    hyp_item = pred.get(item_field) or ""
                    ref_item = gold.get(item_gold_field) or ""
                    if not ref_item.strip():
                        raise ValueError(f"Empty German question reference for '{key}' in {gold_path}")

                    source_en = gold.get(SOURCE_FIELD, "") or ""
                    pred_source = pred.get(SOURCE_FIELD, source_en) or ""
                    if pred_source.strip() != source_en.strip():
                        raise ValueError(f"English source mismatch for '{key}' in {fpath}")

                    aligned.append(
                        AlignedItem(
                            item_id=str(key),
                            dataset=dataset,
                            model=model,
                            stage=stage,
                            source_en=source_en,
                            source_scale=gold.get("scale", "") or "",
                            hyp_item=hyp_item,
                            hyp_scale=pred.get(scale_field) or "",
                            ref_item=ref_item,
                            ref_scale=gold.get(scale_gold_field) or "",
                        )
                    )
    return aligned

# Metric computation
def compute_chrf(hyps: list, refs: list) -> list:
    chrf = CHRF()
    return [chrf.sentence_score(h, [r]).score for h, r in zip(hyps, refs)]


def compute_bertscore(hyps: list, refs: list, model_type: str = BERTSCORE_MODEL) -> list:
    from bert_score import BERTScorer

    scorer = BERTScorer(model_type=model_type, lang="de", rescale_with_baseline=False)
    _, _, F1 = scorer.score(hyps, refs)
    return F1.tolist()


def compute_comet(sources: list, hyps: list, refs: list, model_name: str = COMET_MODEL) -> list:
    from comet import download_model, load_from_checkpoint

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hyps, refs)]
    output = model.predict(data, batch_size=16, gpus=1 if _gpu_available() else 0)
    scores = getattr(output, "scores", None)
    if scores is None:
        scores = output["scores"]
    return list(scores)


def _gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# Main evaluation
def run_metrics(aligned: list, metrics: list) -> pd.DataFrame:
    df = pd.DataFrame([a.__dict__ for a in aligned])
    if df.empty:
        log.error("No aligned items found — check paths and field names in CONFIG.")
        return df

    # Score questions and scales separately. The combined score is their
    # arithmetic mean so a short response scale is not hidden by a long question.
    question_hyps = df["hyp_item"].fillna("").tolist()
    question_refs = df["ref_item"].fillna("").tolist()
    scale_hyps = df["hyp_scale"].fillna("").tolist()
    scale_refs = df["ref_scale"].fillna("").tolist()
    question_sources = df["source_en"].fillna("").tolist()
    scale_sources = df["source_scale"].fillna("").tolist()

    if "chrf" in metrics:
        log.info("Computing question, scale, and combined chrF for %d items...", len(df))
        df["question_chrf"] = compute_chrf(question_hyps, question_refs)
        df["scale_chrf"] = compute_chrf(scale_hyps, scale_refs)
        df["combined_chrf"] = (df["question_chrf"] + df["scale_chrf"]) / 2

    if "bertscore" in metrics:
        log.info("Computing question, scale, and combined BERTScore (%s) for %d items...", BERTSCORE_MODEL, len(df))
        # Score both views in one batch so the model is loaded only once.
        all_hyps = question_hyps + scale_hyps
        all_refs = question_refs + scale_refs
        all_scores = compute_bertscore(all_hyps, all_refs)
        n = len(df)
        df["question_bertscore_f1"] = all_scores[:n]
        df["scale_bertscore_f1"] = all_scores[n:2*n]
        df["combined_bertscore_f1"] = (
            df["question_bertscore_f1"] + df["scale_bertscore_f1"]
        ) / 2

    if "comet" in metrics:
        log.info("Computing question, scale, and combined COMET (%s) for %d items...", COMET_MODEL, len(df))
        # Score both views in one model call. Empty model scales are kept
        # deliberately so missing response-scale translations are penalized.
        all_sources = question_sources + scale_sources
        all_hyps = question_hyps + scale_hyps
        all_refs = question_refs + scale_refs
        all_scores = compute_comet(all_sources, all_hyps, all_refs)
        n = len(df)
        df["question_comet"] = all_scores[:n]
        df["scale_comet"] = all_scores[n:2*n]
        df["combined_comet"] = (df["question_comet"] + df["scale_comet"]) / 2

    return df


def summarize(df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    score_cols = [
        c for c in [
            "question_chrf", "scale_chrf", "combined_chrf",
            "question_bertscore_f1", "scale_bertscore_f1",
            "combined_bertscore_f1",
            "question_comet", "scale_comet", "combined_comet",
        ]
        if c in df.columns
    ]
    summary = (
        df.groupby(["dataset", "model", "stage"])[score_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.to_flat_index()]
    return summary

def add_improvement_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """Create one flat row per dataset/model with raw, corrected, and delta means."""
    mean_columns = [
        c for c in summary.columns
        if c.endswith("_mean") and c not in {"dataset_mean", "model_mean", "stage_mean"}
    ]
    raw = summary[summary["stage"] == "raw"].set_index(["dataset", "model"])
    corrected = summary[summary["stage"] == "corrected"].set_index(["dataset", "model"])
    if not raw.index.equals(corrected.index):
        raise ValueError("Raw and corrected summary groups do not align")

    rows = []
    for dataset, model in raw.index:
        row = {"dataset": dataset, "model": model}
        for mean_column in mean_columns:
            metric = mean_column.removesuffix("_mean")
            raw_value = raw.loc[(dataset, model), mean_column]
            corrected_value = corrected.loc[(dataset, model), mean_column]
            row[f"{metric}_raw"] = raw_value
            row[f"{metric}_corrected"] = corrected_value
            row[f"{metric}_delta"] = corrected_value - raw_value
        rows.append(row)
    return pd.DataFrame(rows)


def build_paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-item corrected-minus-raw changes for every computed score."""
    score_cols = [
        column for column in df.columns
        if column.startswith(("question_", "scale_", "combined_"))
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    keys = ["item_id", "dataset", "model"]
    raw = df[df["stage"] == "raw"][keys + score_cols].copy()
    corrected = df[df["stage"] == "corrected"][keys + score_cols].copy()
    raw = raw.rename(columns={score: f"{score}_raw" for score in score_cols})
    corrected = corrected.rename(
        columns={score: f"{score}_corrected" for score in score_cols}
    )
    paired = raw.merge(corrected, on=keys, how="inner", validate="one_to_one")
    if len(paired) * 2 != len(df):
        raise ValueError("Raw and corrected per-item rows do not align")
    for score in score_cols:
        paired[f"{score}_delta"] = (
            paired[f"{score}_corrected"] - paired[f"{score}_raw"]
        )
    return paired

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                         help="Directory containing gold-standard JSON files")
    parser.add_argument("--steps-dir", type=Path, default=Path("data/outputs"),
                         help="Directory containing step1/step3 output JSON files")
    parser.add_argument("--out-dir", type=Path, default=Path("results"),
                         help="Where to write per-item, summary, and delta CSV files")
    parser.add_argument("--metrics", nargs="+", default=["chrf", "bertscore", "comet"],
                         choices=["chrf", "bertscore", "comet"],
                         help="Which metrics to compute (default: all)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Aligning items from gold + step1/step3 outputs...")
    aligned = build_aligned_items(args.data_dir, args.steps_dir)
    log.info("Aligned %d (item, model, stage) rows.", len(aligned))
    expected_rows = sum(len(load_json(args.data_dir / GOLD_FILES[d])) for d in DATASETS) * len(MODELS) * 2
    if len(aligned) != expected_rows:
        raise ValueError(f"Expected {expected_rows} aligned rows, found {len(aligned)}")

    df = run_metrics(aligned, args.metrics)
    if df.empty:
        return

    per_item_path = args.out_dir / "scores_per_item.csv"
    df.to_csv(per_item_path, index=False, encoding="utf-8")
    log.info("Wrote per-item scores to %s", per_item_path)

    summary = summarize(df, args.metrics)
    summary_csv = args.out_dir / "summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8")
    pivot_summary = add_improvement_metrics(summary)
    pivot_path = args.out_dir / "summary_with_delta.csv"
    pivot_summary.to_csv(pivot_path, index=False, encoding="utf-8")
    log.info("Wrote delta summary to %s", pivot_path)

    paired_path = args.out_dir / "paired_deltas.csv"
    build_paired_deltas(df).to_csv(paired_path, index=False, encoding="utf-8")
    log.info("Wrote paired item-level deltas to %s", paired_path)

    print("\nSummary (mean scores by dataset x model x stage)")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
