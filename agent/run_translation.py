import json
import yaml
import argparse
import copy
from pathlib import Path
from datetime import datetime
from translator import Translator

# Config
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "ess": "data/processed/ess11_items_with_scale.json",
    "custom": "data/custom/custom_test_items_de_CH.json"
}

OUTPUT_DIR = PROJECT_ROOT / "data/outputs"


# Helpers 
def load_dataset(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from {path}")
    return data


def save_results(results: list, path: str, quiet: bool = False):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    temporary_path.replace(output_path)
    if not quiet:
        print(f"Saved {len(results)} results to {path}")


def print_sample(results: list, n: int = 3):
    """Print sample translations for quick inspection."""
    print(f"\nSample Translations (first {n}) ---")
    for item in results[:n]:
        print(f"\nID: {item.get('id', '')}")
        print(f"Type: {item.get('item_type', '')}")
        print(f"EN: {item['source_en'][:100]}")
        print(f"DE: {item.get('model_trans_de', 'N/A')[:100]}")
        print(f"Model: {item.get('model', '')}")


def print_stats(results: list):
    """Print quick stats about translation results."""
    from collections import Counter
    type_counts = Counter(item.get('item_type', '') for item in results)

    print(f"\nStats ")
    print(f"Total items translated: {len(results)}")
    print(f"By item type:")
    for t, c in type_counts.most_common():
        print(f"{t:<20} {c}")

    # check for empty translations
    empty = [r for r in results if not r.get('model_trans_de', '').strip()]
    empty_scales = [
        r for r in results
        if r.get('scale', '').strip() and not r.get('model_trans_scale', '').strip()
    ]
    if empty:
        print(f"\nWARNING: {len(empty)} empty translations!")
        for item in empty:
            print(f"{item.get('id', '')}")
    else:
        print(f"\nAll translations non-empty :)")

    if empty_scales:
        print(f"WARNING: {len(empty_scales)} empty translated scales!")


# Main Translation Run
def run_translation(model_key: str, dataset_key: str, config: dict):
    """
    Run translation for one model on one dataset.
    Saves results immediately after translation.
    """
    print(f"\n{'='*60}")
    print(f"STEP 1 — TRANSLATION")
    if model_key not in config.get("models", {}):
        raise KeyError(f"Model '{model_key}' is not defined in config.yaml")

    print(f"Model: {config['models'][model_key]['name']}")
    print(f"Dataset: {dataset_key.upper()}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    # update config with current model
    run_config = copy.deepcopy(config)
    run_config["model"] = copy.deepcopy(config["models"][model_key])
    prompt_path = Path(run_config["paths"]["translate_prompt"])
    if not prompt_path.is_absolute():
        run_config["paths"]["translate_prompt"] = str(PROJECT_ROOT / prompt_path)

    # load dataset
    print(f"\nLoading dataset...")
    dataset_path = Path(DATASETS[dataset_key])
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    items = load_dataset(str(dataset_path))

    # init translator — loads model
    print(f"\nLoading model...")
    translator = Translator(run_config)

    output_path = OUTPUT_DIR / f"step1_{model_key}_{dataset_key}_translated.json"

    # translate all items and checkpoint after each completed item
    print(f"\nTranslating {len(items)} items...")
    results = translator.translate_batch(
        items,
        on_result=lambda partial: save_results(partial, str(output_path), quiet=True)
    )

    print(f"\nSaving final results...")
    save_results(results, str(output_path))

    # show sample and stats
    print_stats(results)
    print_sample(results)

    print(f"\nDone: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Results: {output_path}")

    return results


# Entry Point
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
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # determine which models and datasets to run
    models = ["mistral", "qwen"] if args.model   == "all" else [args.model]
    datasets = ["ess", "custom"]   if args.dataset == "all" else [args.dataset]

    print("\n" + "="*60)
    print("Survey Translation Agent — Step 1: Translation")
    print("="*60)
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Total runs: {len(models) * len(datasets)}")
    print(f"\nOutput files:")
    for m in models:
        for d in datasets:
            print(f"  data/outputs/step1_{m}_{d}_translated.json")

    # Run all combinations
    # Each model is loaded fresh per run
    # This avoids GPU memory conflicts between models
    for model_key in models:
        for dataset_key in datasets:
            run_translation(model_key, dataset_key, config)

    print("\n" + "="*60)
    print("Translation step complete.")
    print("Check data/outputs/ for results.")
    print("Next step: REVIEW TIME !")
    print("="*60)

if __name__ == "__main__":
    main()
