import re
import json
import yaml
import argparse
import gc
import torch
from pathlib import Path
from datetime import datetime
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM


OUTPUT_DIR = Path("data/outputs")

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

CROSS_REVIEW = {
    "mistral": "qwen",      # Qwen reviews Mistral translations
    "qwen": "mistral"    # Mistral reviews Qwen translations
}

DATASETS = ["ess", "custom"]


# Reviewer
class Reviewer:
    """
    Reviews German translation against original English source only.
    """

    def __init__(self, config: dict, reviewer_model: str):
        self.config = config
        self.reviewer_model = reviewer_model
        self.max_new_tokens = config["models"][reviewer_model]["max_new_tokens"]

        if reviewer_model not in MODELS:
            raise ValueError(f"reviewer_model must be 'mistral' or 'qwen', got: {reviewer_model}")

        self.model_name = MODELS[reviewer_model]["name"]
        self.model_path = MODELS[reviewer_model]["path"]

        prompt_path = config["paths"]["review_prompt"]
        print(f"Loading review prompt from: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        print("Review prompt loaded.")

        self._load_model()


    def _load_model(self):
        print(f"Loading reviewer model: {self.model_name}")
        print(f"From path: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype = torch.float16, device_map = "auto", trust_remote_code = True)
        self.model.eval()
        print(f"Reviewer loaded: {self.model_name}")


    def review(self, source_en:str, scale:str, model_trans_de:str, model_trans_scale:str, item_type:str) -> dict:
        """Review a single translation against the original English source."""
        prompt = self.prompt_template
        prompt = prompt.replace("{item_type}", item_type)
        prompt = prompt.replace("{source_en}", source_en)
        prompt = prompt.replace("{scale}", scale if scale else "No scale")
        prompt = prompt.replace("{model_trans_de}", model_trans_de if model_trans_de else "Not provided")
        prompt = prompt.replace("{model_trans_scale}", model_trans_scale if model_trans_scale.strip() else "[EMPTY - NO SCALE PROVIDED]")

        messages  = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

        inputs = self.tokenizer(formatted, return_tensors = "pt", truncation = True, max_length = 3072).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens = self.max_new_tokens, do_sample = False, temperature = 1.0, pad_token_id = self.tokenizer.eos_token_id) # type: ignore

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens = True).strip()

        result = self._parse_output(raw_output)
        result["reviewer_model"] = self.model_name

        return result
    
    def is_broken_translation(self, model_trans_de: str, model_trans_scale: str) -> bool:
        """Checks both translation and scale for structural integrity."""
        combined_text = f"{model_trans_de} {model_trans_scale}"

        # Foreign Language Detection
        if re.search(r'[\u4e00-\u9fff\u0600-\u06ff\u0400-\u04ff]', combined_text):
            return True
        
        # Formatting: Markdown/JSON/Code block artifacts
        if "```" in combined_text or '{"model_trans_de"' in combined_text:
           return True
        
        # Newline Traps: Check for vertical lists or excessive \n
        if "\n" in model_trans_de or "\n" in model_trans_scale:
            return True
        
        # Minimal Content Check
        if len(model_trans_de.strip()) < 3:
            return True

        return False


    def _parse_output(self, raw_output: str) -> dict:
        cleaned = re.sub(r"```json|```", "", raw_output).strip()
        try:
            parsed = json.loads(cleaned)
            if "issues" not in parsed:
                parsed["issues"] = []
            if "has_issues" not in parsed:
                parsed["has_issues"] = len(parsed["issues"]) > 0
            return parsed
        except json.JSONDecodeError:
            print(f"WARNING: Failed to parse reviewer JSON. Raw: {raw_output[:200]}")
            return {
                "has_issues": True,
                "issues": [{
                    "category": "PARSE_ERROR",
                    "problem": "Reviewer output could not be parsed as JSON",
                    "suggested_fix": raw_output[:1000]
                }]
            }

    def review_batch(self, items: list) -> list:
        """Review a batch of translated items."""
        results = []
        total = len(items)
        issues_count = 0

        for i, item in enumerate(items):
            print(f"  [{i+1}/{total}] Reviewing: {item.get('id', '')} — {item.get('item_type', '')}")

            review = self.review(
                source_en = item.get("source_en", ""),
                scale = item.get("scale", ""),
                model_trans_de = item.get("model_trans_de", ""),
                model_trans_scale = item.get("model_trans_scale", ""),
                item_type = item.get("item_type", "attitudinal")
            )
            if not review.get("has_issues", False) and self.is_broken_translation(item.get("model_trans_de", ""), item.get("model_trans_scale", "")):
                print(f"  [{i+1}/{total}] LLM missed error. Fallback Bouncer triggered for {item.get('id')}")
                review = {
                    "has_issues": True,
                    "reviewer_model": "FALLBACK_BOUNCER",
                    "issues": [{
                        "category": "FATAL_FORMAT_ERROR",
                        "problem": "Formatting error missed by LLM: Code blocks or illegal characters detected.",
                        "suggested_fix": "Regenerate as clean, plain-text German."
                    }]
                }

            has_issues = review.get("has_issues", False)
            if has_issues:
                issues_count += 1

            result = item.copy()
            result["review"] = review
            results.append(result)

        print(f"Done — {issues_count}/{total} items flagged with issues")
        return results

def load_results(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from {path}")
    return data


def save_results(results: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} results to {path}")


def print_review_summary(results: list):
    total = len(results)
    has_issues = sum(1 for r in results if r.get("review", {}).get("has_issues", False))
    no_issues  = total - has_issues

    category_counts = Counter()
    for item in results:
        for issue in item.get("review", {}).get("issues", []):
            category_counts[issue.get("category", "UNKNOWN")] += 1

    print(f"\nReview Summary:")
    print(f"Total items: {total}")
    print(f"No issues: {no_issues} ({no_issues/total*100:.1f}%)")
    print(f"Issues found: {has_issues} ({has_issues/total*100:.1f}%)")

    if category_counts:
        print(f"\nIssues by category:")
        for cat, count in category_counts.most_common():
            print(f" {cat:<25} {count}")


def print_sample_issues(results: list, n: int = 3):
    items_with_issues = [r for r in results if r.get("review", {}).get("has_issues", False)]
    print(f"\nSample Items With Issues (first {n})")
    for item in items_with_issues[:n]:
        print(f"\nID: {item.get('id', '')}")
        print(f"EN: {item['source_en'][:80]}")
        print(f"Trans DE: {item.get('model_trans_de', '')[:100]}")
        for issue in item.get("review", {}).get("issues", []):
            print(f"[{issue.get('category', '')}] {issue.get('problem', '')[:500]}")


def run_review(translated_model: str, dataset: str, config: dict):
    """Run cross-review for one translated model on one dataset."""
    reviewer_model = CROSS_REVIEW[translated_model]

    print(f"\n{'='*60}")
    print(f"STEP 2 — CROSS-REVIEW (source_en vs model_trans_de only)")
    print(f"Translated by: {translated_model.upper()}")
    print(f"Reviewed by: {reviewer_model.upper()}")
    print(f"Dataset: {dataset.upper()}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    input_path = OUTPUT_DIR / f"step1_{translated_model}_{dataset}_translated.json"
    print(f"\nLoading translations from: {input_path}")
    items = load_results(str(input_path))

    print(f"\nLoading reviewer model ({reviewer_model})...")
    reviewer = Reviewer(config, reviewer_model=reviewer_model)

    print(f"\nReviewing {len(items)} items...")
    results = reviewer.review_batch(items)

    output_path = OUTPUT_DIR / f"step2_{translated_model}_{dataset}_reviewed.json"
    print(f"\nSaving results...")
    save_results(results, str(output_path))

    print_review_summary(results)
    print_sample_issues(results)

    print(f"\nDone: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Results: {output_path}")

    # Clear GPU memory before the next model loads
    del reviewer.model
    del reviewer.tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results

# Execution
def main():
    parser = argparse.ArgumentParser(
        description="Cross-review translated survey items"
    )
    parser.add_argument("--translated", choices=["mistral", "qwen", "all"], default="all")
    parser.add_argument("--dataset", choices=["ess", "custom", "all"], default="all")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    translated_models = ["mistral", "qwen"] if args.translated == "all" else [args.translated]
    datasets = ["ess", "custom"] if args.dataset == "all" else [args.dataset]

    print("\n" + "="*60)
    print("Survey Translation Agent: Cross-Review")
    print("="*60)
    print("Reviewer sees ONLY source_en. No gold standard.")
    print(f"\nCross-review mapping:")
    print(f"Mistral translations → reviewed by Qwen")
    print(f"Qwen translations → reviewed by Mistral")
    print(f"\nDatasets: {datasets}")
    print(f"Total runs: {len(translated_models) * len(datasets)}")

    for translated_model in translated_models:
        for dataset in datasets:
            run_review(translated_model, dataset, config)

    print("\n" + "="*60)
    print("Cross-review complete.")
    print("Next step: python corrector.py")
    print("="*60)


if __name__ == "__main__":
    main()
