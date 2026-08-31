import re
import json
import yaml
import argparse
import gc
import torch
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Any, cast
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

# Each model corrects its OWN output
SELF_CORRECT = {
    "mistral": "mistral",
    "qwen": "qwen"
}

# Corrector Class 
class Corrector:
    """
    Corrects German translation based on reviewer feedback.
    Each model corrects its own output — self-correction based on
    cross-reviewer feedback.
    """

    def __init__(self, config: dict, translated_model: str):
        """
        Args:
            config : config dict from config.yaml
            translated_model : "mistral" or "qwen" — who originally translated
                               (same model will correct its own output)
        """
        self.config = config
        self.translated_model = translated_model
        self.max_new_tokens  = config["models"][translated_model]["max_new_tokens"]

        self.model_name = MODELS[translated_model]["name"]
        self.model_path = MODELS[translated_model]["path"]

        # load correction prompt
        prompt_path = config["paths"]["self_correct_prompt"]
        print(f"Loading correction prompt from: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        print("Correction prompt loaded.")

        self._load_model()


    def _load_model(self):
        print(f"Loading corrector model: {self.model_name}")
        print(f"From path: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype = torch.float16,
            device_map = "auto",
            trust_remote_code = True
        )
        self.model.eval()
        print(f"Corrector loaded: {self.model_name}")


    def _is_broken(self, model_trans_de: str, model_trans_scale: str, scale: str) -> bool:
        """
        Detect broken translator output that needs cleanup before correction.
        These are cases the reviewer should have flagged but we also
        catch here as a safety net.
        """
        if not model_trans_de or not model_trans_de.strip():
            return True

        # contains non-German characters (Chinese, Arabic, Cyrillic etc.)
        if re.search(r'[\u4e00-\u9fff\u0600-\u06ff\u0400-\u04ff]', model_trans_de):
            return True

        # contains raw JSON or code blocks
        if "```" in model_trans_de or '{"model_trans_de"' in model_trans_de:
            return True

        # scale missing when source has one
        if scale and scale.strip() and not model_trans_scale.strip():
            return True

        return False


    def _format_issues(self, review: dict) -> str:
        """Format reviewer issues into a readable string for the prompt."""
        issues = review.get("issues", [])
        if not issues:
            return "No specific issues listed — general quality improvement needed."

        lines = []
        for i, issue in enumerate(issues, 1):
            category = issue.get("category", "")
            problem  = issue.get("problem", "")
            fix = issue.get("suggested_fix", "")
            lines.append(f"Issue {i} [{category}]: {problem}")
            if fix:
                lines.append(f"  Suggested fix: {fix}")
        return "\n".join(lines)


    def correct(self, source_en: str, scale: str, model_trans_de: str, model_trans_scale: str, 
                item_type: str, review: dict, max_attempts: int = 3) -> dict:
        last_result: dict[str, Any] = {
            "corrected_de": model_trans_de,
            "corrected_scale": model_trans_scale,
        }

        attempt = 0
        while attempt < max_attempts:
            # Prepare prompt
            issues_str = self._format_issues(review)
            if attempt > 0:
                issues_str = f"PREVIOUS ATTEMPT FAILED FORMAT CHECK. " \
                             f"Do not use code blocks, JSON artifacts, or non-German characters.\n" + issues_str
            
            prompt = self.prompt_template.format(
                item_type=item_type, source_en=source_en, scale=scale or "not specified",
                model_trans_de=model_trans_de, model_trans_scale=model_trans_scale or "EMPTY",
                issues=issues_str
            )

            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=3072,
            ).to(self.model.device)

            # Generate the corrected output deterministically.
            with torch.no_grad():
                outputs = cast(Any, self.model).generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            result = self._parse_output(raw_output)
            last_result = result

            # POST-CORRECTION VALIDATOR (The Check)
            if not self._is_broken(result.get("corrected_de", ""), result.get("corrected_scale", ""), scale):
                # Success: Output is clean
                return result
            
            # If broken, loop again
            attempt += 1
            print(f"Attempt {attempt} failed validation for source item. Retrying...")

        # Final fallback if all attempts fail
        last_result["has_issues"] = True
        last_result["issues"] = [{
            "category": "PERSISTENT_FAILURE",
            "problem": "Failed to correct format after 3 attempts.",
            "suggested_fix": "Regenerate the correction as clean JSON with corrected_de and corrected_scale.",
        }]

        if not last_result.get("corrected_scale") or not last_result["corrected_scale"].strip():
            print(f"Fallback: Restoring original scale because 3rd attempt was empty.")
            last_result["corrected_scale"] = model_trans_scale
            
        return last_result


    def _parse_output(self, raw_output: str) -> dict:
        """Parse JSON output from corrector model."""
        cleaned = re.sub(r"```json|```", "", raw_output).strip()

        try:
            parsed = json.loads(cleaned)
            return {
                "corrected_de": parsed.get("corrected_de", "").strip(),
                "corrected_scale": parsed.get("corrected_scale", "").strip()
            }
        except json.JSONDecodeError:
            # try regex extraction as fallback
            de_match    = re.search(r'"corrected_de"\s*:\s*"([^"]*)"', cleaned)
            scale_match = re.search(r'"corrected_scale"\s*:\s*"([^"]*)"', cleaned)
            return {
                "corrected_de": de_match.group(1).strip() if de_match else "",
                "corrected_scale": scale_match.group(1).strip() if scale_match else ""
            }


    def correct_batch(self, items: list) -> list:
        """
        Correct a batch of reviewed items.
        Items with has_issues: false are passed through unchanged.
        Items with has_issues: true are corrected.
        """
        results = []
        total = len(items)
        corrected = 0
        passed_through = 0

        for i, item in enumerate(items):
            review_obj = item.get("review", {})
            has_issues = review_obj.get("has_issues", False)
            broken = self._is_broken(
                item.get("model_trans_de", ""),
                item.get("model_trans_scale", ""),
                item.get("scale", "")
            )

            result = item.copy()

            if has_issues or broken:
                print(f"  [{i+1}/{total}] Correcting: {item.get('id', '')} — {item.get('item_type', '')}")

                correction = self.correct(
                    source_en = item.get("source_en", ""),
                    scale = item.get("scale", ""),
                    model_trans_de  = item.get("model_trans_de", ""),
                    model_trans_scale  = item.get("model_trans_scale", ""),
                    item_type = item.get("item_type", "attitudinal"),
                    review = item.get("review", {})
                )

                result["corrected_de"] = correction["corrected_de"]
                result["corrected_scale"] = correction["corrected_scale"]
                result["was_corrected"]   = True
                corrected += 1

            else:
                print(f"  [{i+1}/{total}] Passing through: {item.get('id', '')} — no issues")
                # pass through original translation unchanged
                result["corrected_de"] = item.get("model_trans_de", "")
                result["corrected_scale"] = item.get("model_trans_scale", "")
                result["was_corrected"] = False
                passed_through += 1

            result["corrector_model"] = self.model_name
            results.append(result)

        print(f"\nDone — {corrected} corrected, {passed_through} passed through")
        return results


# Helpers 

def load_results(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} items from {path}")
    return data


def save_results(results: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} results to {path}")


def print_correction_summary(results: list):
    total = len(results)
    corrected = sum(1 for r in results if r.get("was_corrected", False))
    passed_through = total - corrected

    print(f"\n--- Correction Summary ---")
    print(f"Total items: {total}")
    print(f"Corrected: {corrected} ({corrected/total*100:.1f}%)")
    print(f"Passed through: {passed_through} ({passed_through/total*100:.1f}%)")


def print_sample_corrections(results: list, n: int = 3):
    corrected_items = [r for r in results if r.get("was_corrected", False)]
    print(f"\n--- Sample Corrections (first {n}) ---")
    for item in corrected_items[:n]:
        print(f"\nID: {item.get('id', '')}")
        print(f"EN: {item['source_en'][:500]}")
        print(f"Original DE: {item.get('model_trans_de', '')[:500]}")
        print(f"Corrected DE: {item.get('corrected_de', '')[:500]}")
        if item.get("model_trans_scale") != item.get("corrected_scale"):
            print(f"Original scale: {item.get('model_trans_scale', '')[:300]}")
            print(f"Corrected scale: {item.get('corrected_scale', '')[:300]}")


def run_correction(translated_model: str, dataset: str, config: dict):
    """Run correction for one translated model on one dataset."""
    print(f"\n{'='*60}")
    print(f"STEP 3 — CORRECTION")
    print(f"Translated by: {translated_model.upper()} (self-corrects)")
    print(f"Dataset: {dataset.upper()}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    input_path = OUTPUT_DIR / f"step2_{translated_model}_{dataset}_reviewed.json"
    print(f"\nLoading reviewed results from: {input_path}")
    items = load_results(str(input_path))

    print(f"\nLoading corrector model ({translated_model})...")
    corrector = Corrector(config, translated_model=translated_model)

    print(f"\nCorrecting {len(items)} items...")
    results = corrector.correct_batch(items)

    output_path = OUTPUT_DIR / f"step3_{translated_model}_{dataset}_corrected.json"
    print(f"\nSaving results...")
    save_results(results, str(output_path))

    print_correction_summary(results)
    print_sample_corrections(results)

    print(f"\nDone: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Results: {output_path}")

    # clear GPU memory before next model loads
    del corrector.model
    del corrector.tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


# Main function

def main():
    parser = argparse.ArgumentParser(
        description="Correct translated survey items based on reviewer feedback"
    )
    parser.add_argument("--translated", choices=["mistral", "qwen", "all"], default="all")
    parser.add_argument("--dataset", choices=["ess", "custom", "all"],   default="all")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    translated_models = ["mistral", "qwen"] if args.translated == "all" else [args.translated]
    datasets = ["ess", "custom"] if args.dataset == "all" else [args.dataset]

    print("\n" + "="*60)
    print("Survey Translation Agent — Step 3: Correction")
    print("="*60)
    print("Each model self-corrects based on cross-reviewer feedback.")
    print(f"\nDatasets: {datasets}")
    print(f"Total runs: {len(translated_models) * len(datasets)}")
    print(f"\nOutput files:")
    for m in translated_models:
        for d in datasets:
            print(f"  data/outputs/step3_{m}_{d}_corrected.json")

    for translated_model in translated_models:
        for dataset in datasets:
            run_correction(translated_model, dataset, config)

    print("\n" + "="*60)
    print("Correction step complete.")
    print("Next step: python evaluate.py")
    print("="*60)


if __name__ == "__main__":
    main()