import json
import yaml
import argparse
import copy
from pathlib import Path
from datetime import datetime
from translator import Translator

# Config 
MODELS = {
    "mistral": {
        "name": "Mistral-7B-Instruct-v0.3",
        "path": "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models/Mistral-7B-Instruct-v0.3"
    },
    "qwen": {
        "name": "Qwen2.5-7B-Instruct",
        "path": "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models/Qwen2.5-7B-Instruct"
    }
}

DATASETS = {
    "ess":    "data/processed/ess11_items_with_scale.json",
    "custom": "data/custom/custom_test_items_de_CH.json"
}

OUTPUT_DIR = Path("data/outputs")


# Helpers 
def load_dataset(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} items from {path}")
    return data


def save_results(results: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(results)} results to {path}")


def print_sample(results: list, n: int = 3):
    """Print sample translations for quick inspection."""
    print(f"\n--- Sample Translations (first {n}) ---")
    for item in results[:n]:
        print(f"\n  ID:       {item.get('id', '')}")
        print(f"  Type:     {item.get('item_type', '')}")
        print(f"  EN:       {item['source_en'][:100]}")
        print(f"  DE:       {item.get('translation_de', 'N/A')[:100]}")
        if item.get('gold_de'):
            print(f"  Gold DE:  {item['gold_de'][:100]}")
        print(f"  Model:    {item.get('model', '')}")


def print_stats(results: list):
    """Print quick stats about translation results."""
    from collections import Counter
    type_counts = Counter(item.get('item_type', '') for item in results)

    print(f"\n--- Stats ---")
    print(f"  Total items translated: {len(results)}")
    print(f"  By item type:")
    for t, c in type_counts.most_common():
        print(f"    {t:<20} {c}")

    # check for empty translations
    empty = [r for r in results if not r.get('translation_de', '').strip()]
    if empty:
        print(f"\n  WARNING: {len(empty)} empty translations!")
        for item in empty:
            print(f"    {item.get('id', '')}")
    else:
        print(f"\n  All translations non-empty ✓")


# Main Translation Run

def run_translation(model_key: str, dataset_key: str, config: dict):
    """
    Run translation for one model on one dataset.
    Saves results immediately after translation.
    """
    print(f"\n{'='*60}")
    print(f"STEP 1 — TRANSLATION")
    print(f"Model:   {MODELS[model_key]['name']}")
    print(f"Dataset: {dataset_key.upper()}")
    print(f"Time:    {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    # update config with current model
    run_config = copy.deepcopy(config)
    run_config["model"]["name"] = MODELS[model_key]["name"]
    run_config["model"]["path"] = MODELS[model_key]["path"]

    # load dataset
    print(f"\nLoading dataset...")
    items = load_dataset(DATASETS[dataset_key])

    # init translator — loads model
    print(f"\nLoading model...")
    translator = Translator(run_config)

    # translate all items
    print(f"\nTranslating {len(items)} items...")
    results = translator.translate_batch(items)

    # save results immediately
    output_path = OUTPUT_DIR / f"step1_{model_key}_{dataset_key}_translated.json"
    print(f"\nSaving results...")
    save_results(results, str(output_path))

    # show sample and stats
    print_stats(results)
    print_sample(results)

    print(f"\nDone: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Results: {output_path}")

    return results


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run translation step for survey items"
    )
    parser.add_argument(
        "--model",
        choices=["mistral", "qwen", "all"],
        default="all",
        help="Which model to run (default: all)"
    )
    parser.add_argument(
        "--dataset",
        choices=["ess", "custom", "all"],
        default="all",
        help="Which dataset to use (default: all)"
    )
    args = parser.parse_args()

    # load base config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # determine which models and datasets to run
    models   = ["mistral", "qwen"] if args.model   == "all" else [args.model]
    datasets = ["ess", "custom"]   if args.dataset == "all" else [args.dataset]

    print("\n" + "="*60)
    print("Survey Translation Agent — Step 1: Translation")
    print("="*60)
    print(f"Models:   {models}")
    print(f"Datasets: {datasets}")
    print(f"Total runs: {len(models) * len(datasets)}")
    print(f"\nOutput files:")
    for m in models:
        for d in datasets:
            print(f"  data/outputs/step1_{m}_{d}_translated.json")

    # run all combinations
    # NOTE: each model is loaded fresh per run
    # this avoids GPU memory conflicts between models
    for model_key in models:
        for dataset_key in datasets:
            run_translation(model_key, dataset_key, config)

    print("\n" + "="*60)
    print("Translation step complete.")
    print("Check data/outputs/ for results.")
    print("Next step: run_review.py")
    print("="*60)


if __name__ == "__main__":
    main()