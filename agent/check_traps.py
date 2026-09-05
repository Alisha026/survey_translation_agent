import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_INPUTS = [
    "data/outputs/step1_mistral_custom_translated.json",
    "data/outputs/step1_qwen_custom_translated.json",
    "data/outputs/step3_mistral_custom_corrected.json",
    "data/outputs/step3_qwen_custom_corrected.json",
]
DEFAULT_GOLD_FILE = "data/custom/custom_test_items_de_CH.json"
DEFAULT_OUTPUT_DIR = "results/trap_checks"
DEFAULT_MODEL_PATH = "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models/Qwen3.5-27B"

def load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected a list of objects in {path}")
    return data


def index_by_id(items: list[dict], path: Path) -> dict[str, dict]:
    indexed = {}
    for position, item in enumerate(items):
        item_id = item.get("id")
        if not item_id:
            raise ValueError(f"Missing ID in {path} at position {position}")
        if item_id in indexed:
            raise ValueError(f"Duplicate ID '{item_id}' in {path}")
        indexed[item_id] = item
    return indexed


def load_model(model_path: str):
    print(f"Loading judge model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def select_translation(item: dict) -> tuple[str, str, str]:
    """Select oracle fields, then Step 3 fields, then Step 1 fields."""
    if "oracle_corrected_de" in item or "oracle_corrected_scale" in item:
        return (
            item.get("oracle_corrected_de") or "",
            item.get("oracle_corrected_scale") or "",
            "oracle",
        )
    if "corrected_de" in item or "corrected_scale" in item:
        return (
            item.get("corrected_de") or "",
            item.get("corrected_scale") or "",
            "corrected",
        )
    return (
        item.get("model_trans_de") or "",
        item.get("model_trans_scale") or "",
        "raw",
    )


def build_prompt(gold: dict, model_trans_de: str, model_trans_scale: str) -> str:
    return f"""You are an expert bilingual judge evaluating an English-to-German survey translation.

This item has one annotated translation trap. Judge ONLY whether the model avoided that
specific annotated trap. Do not fail the item for unrelated style, grammar, wording, or
formatting differences. A valid paraphrase does not need to copy the human reference.

If the annotated trap concerns only the question, set scale_trap_avoided to null.
If it concerns only the response scale, set question_trap_avoided to null.
If it concerns both, judge both. The overall trap_avoided value is true only when every
relevant component avoids the annotated trap.

[SOURCE]
Category: {gold['category']}
English question: {gold['source_en']}
English scale: {gold['scale']}

[ANNOTATED TRAP]
{gold['problem']}

[HUMAN REFERENCE - EVIDENCE, NOT REQUIRED WORDING]
German question: {gold['trans_de']}
German scale: {gold['scale_de']}

[MODEL TRANSLATION]
German question: {model_trans_de or '[EMPTY]'}
German scale: {model_trans_scale or '[EMPTY]'}

Return only this JSON structure:
{{
  "trap_avoided": true,
  "question_trap_avoided": true,
  "scale_trap_avoided": null,
  "reason": "Brief evidence tied directly to the annotated trap"
}}
"""


def parse_judgment(raw: str) -> dict:
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found")
    parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed.get("trap_avoided"), bool):
        raise ValueError("'trap_avoided' must be a boolean")
    for field in ("question_trap_avoided", "scale_trap_avoided"):
        if parsed.get(field) not in (True, False, None):
            raise ValueError(f"'{field}' must be true, false, or null")
    if not isinstance(parsed.get("reason"), str) or not parsed["reason"].strip():
        raise ValueError("'reason' must be a non-empty string")
    return parsed


def check_trap(tokenizer, model, gold: dict, translation: dict) -> dict:
    model_trans_de, model_trans_scale, stage = select_translation(translation)
    prompt = build_prompt(gold, model_trans_de, model_trans_scale)
    messages = [{"role": "user", "content": prompt}]

    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    # Qwen3 supports disabling hidden reasoning. Fall back for other tokenizers.
    try:
        formatted = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **template_kwargs
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(messages, **template_kwargs)

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    base = {
        "id": gold["id"],
        "category": gold["category"],
        "problem": gold["problem"],
        "stage": stage,
        "model_trans_de": model_trans_de,
        "model_trans_scale": model_trans_scale,
    }
    try:
        return {**base, "judge_status": "ok", **parse_judgment(raw)}
    except (json.JSONDecodeError, ValueError) as error:
        return {
            **base,
            "judge_status": "parse_error",
            "trap_avoided": None,
            "question_trap_avoided": None,
            "scale_trap_avoided": None,
            "reason": str(error),
            "raw_judge_output": raw,
        }


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def evaluate_file(tokenizer, model, input_path: Path, gold_by_id: dict, output_dir: Path):
    translations = load_json_list(input_path)
    translations_by_id = index_by_id(translations, input_path)
    if set(translations_by_id) != set(gold_by_id):
        missing = sorted(set(gold_by_id) - set(translations_by_id))
        extra = sorted(set(translations_by_id) - set(gold_by_id))
        raise ValueError(f"ID mismatch in {input_path}: missing={missing}, extra={extra}")

    output_path = output_dir / f"{input_path.stem}_trap_results.json"
    results = []
    total = len(gold_by_id)
    print(f"\nEvaluating {input_path.name} ({total} items)")

    for position, (item_id, gold) in enumerate(gold_by_id.items(), start=1):
        result = check_trap(tokenizer, model, gold, translations_by_id[item_id])
        results.append(result)
        save_json(results, output_path)  # checkpoint after every expensive judgment

        if result["judge_status"] != "ok":
            status = "JUDGE ERROR"
        elif result["trap_avoided"]:
            status = "AVOIDED"
        else:
            status = "FAILED"
        print(f"[{position:02d}/{total:02d}] {item_id:<10} | {status}")

    valid = [result for result in results if result["judge_status"] == "ok"]
    passed = sum(result["trap_avoided"] for result in valid)
    errors = total - len(valid)
    summary = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "total_items": total,
        "valid_judgments": len(valid),
        "judge_errors": errors,
        "traps_avoided": passed,
        "traps_failed": len(valid) - passed,
        "pass_rate_valid_percent": round(100 * passed / len(valid), 1) if valid else None,
    }
    print(
        f"Summary: {passed}/{len(valid)} valid judgments avoided the trap "
        f"({summary['pass_rate_valid_percent']}%); judge errors={errors}"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--gold-file", type=Path, default=Path(DEFAULT_GOLD_FILE))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    gold_items = load_json_list(args.gold_file)
    gold_by_id = index_by_id(gold_items, args.gold_file)
    tokenizer, model = load_model(args.model_path)

    summaries = []
    for file_name in args.inputs:
        summaries.append(
            evaluate_file(tokenizer, model, Path(file_name), gold_by_id, args.output_dir)
        )
    save_json(summaries, args.output_dir / "summary.json")
    print(f"\nSaved trap-evaluation results to {args.output_dir}")

if __name__ == "__main__":
    main()
