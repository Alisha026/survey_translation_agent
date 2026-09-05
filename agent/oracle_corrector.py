import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models/Qwen3.5-27B"
GOLD_FILE = Path("data/custom/custom_test_items_de_CH.json")
MODEL_FILES = {
    "mistral": {
        "input": Path("data/outputs/step3_mistral_custom_corrected.json"),
        "traps": Path("results/trap_checks/step3_mistral_custom_corrected_trap_results.json"),
        "output": Path("data/outputs/step4_oracle_mistral_custom_corrected.json"),
    },
    "qwen": {
        "input": Path("data/outputs/step3_qwen_custom_corrected.json"),
        "traps": Path("results/trap_checks/step3_qwen_custom_corrected_trap_results.json"),
        "output": Path("data/outputs/step4_oracle_qwen_custom_corrected.json"),
    },
}


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


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def load_model(model_path: str):
    print(f"Loading oracle corrector from: {model_path}")
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


def build_prompt(gold: dict, question_de: str, scale_de: str) -> str:
    return f"""You are an expert English-to-German survey translator.

The translation below failed one specifically annotated translation trap. Correct that
trap while preserving every part that is already correct. Use formal German "Sie",
natural respondent-facing language, and preserve meaning, polarity, time period,
target group, response-option count, order, and response type. Do not invent response
categories. Return a German scale whenever the English scale exists.

[ENGLISH SOURCE]
Question: {gold['source_en']}
Scale: {gold['scale']}

[CURRENT GERMAN TRANSLATION]
Question: {question_de}
Scale: {scale_de}

[EXACT PROBLEM TO FIX]
{gold['problem']}

Return only valid JSON:
{{
  "oracle_corrected_de": "corrected German question",
  "oracle_corrected_scale": "corrected German response scale"
}}
"""

def parse_output(raw: str) -> tuple[str, str]:
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found")
    parsed = json.loads(cleaned[start : end + 1])
    question = parsed.get("oracle_corrected_de")
    scale = parsed.get("oracle_corrected_scale")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Missing non-empty 'oracle_corrected_de'")
    if not isinstance(scale, str):
        raise ValueError("Missing string 'oracle_corrected_scale'")
    return question.strip(), scale.strip()


def validate_output(question: str, scale: str, english_scale: str):
    combined = f"{question} {scale}"
    if re.search(r"[\u4e00-\u9fff\u0600-\u06ff\u0400-\u04ff]", combined):
        raise ValueError("Non-German writing system detected")
    if "```" in combined or "{" in combined or "}" in combined:
        raise ValueError("Code or JSON artifact detected")
    if "\n" in question or "\n" in scale:
        raise ValueError("Embedded newline detected")
    if english_scale.strip() and not scale:
        raise ValueError("German scale is empty although the English scale exists")


def generate_once(tokenizer, model, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        formatted = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **template_kwargs
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(messages, **template_kwargs)
    inputs = tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=4096
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def oracle_correct(tokenizer, model, item: dict, gold: dict) -> dict:
    result = deepcopy(item)
    current_question = item.get("corrected_de") or item.get("model_trans_de") or ""
    current_scale = item.get("corrected_scale") or item.get("model_trans_scale") or ""
    prompt = build_prompt(gold, current_question, current_scale)
    last_error = "Unknown generation failure"

    for attempt in range(1, 3):
        raw = generate_once(tokenizer, model, prompt)
        try:
            question, scale = parse_output(raw)
            validate_output(question, scale, gold.get("scale", ""))
            result["pre_oracle_de"] = current_question
            result["pre_oracle_scale"] = current_scale
            result["oracle_corrected_de"] = question
            result["oracle_corrected_scale"] = scale
            changed = question != current_question or scale != current_scale
            result["was_oracle_corrected"] = changed
            result["oracle_status"] = "corrected" if changed else "unchanged"
            result["oracle_attempts"] = attempt
            result["oracle_corrector_model"] = "Qwen3.5-27B"
            return result
        except (json.JSONDecodeError, ValueError) as error:
            last_error = str(error)
            prompt += "\nYour previous response was invalid. Return only the required JSON object."

    result["pre_oracle_de"] = current_question
    result["pre_oracle_scale"] = current_scale
    result["oracle_corrected_de"] = current_question
    result["oracle_corrected_scale"] = current_scale
    result["was_oracle_corrected"] = False
    result["oracle_status"] = "generation_failed"
    result["oracle_attempts"] = 2
    result["oracle_error"] = last_error
    result["oracle_corrector_model"] = "Qwen3.5-27B"
    return result


def mark_skipped(item: dict) -> dict:
    result = deepcopy(item)
    question = item.get("corrected_de") or item.get("model_trans_de") or ""
    scale = item.get("corrected_scale") or item.get("model_trans_scale") or ""
    result["pre_oracle_de"] = question
    result["pre_oracle_scale"] = scale
    result["oracle_corrected_de"] = question
    result["oracle_corrected_scale"] = scale
    result["was_oracle_corrected"] = False
    result["oracle_status"] = "skipped_trap_already_avoided"
    result["oracle_attempts"] = 0
    result["oracle_corrector_model"] = "Qwen3.5-27B"
    return result


def process_model(tokenizer, model, model_name: str, gold_by_id: dict):
    paths = MODEL_FILES[model_name]
    input_by_id = index_by_id(load_json_list(paths["input"]), paths["input"])
    trap_by_id = index_by_id(load_json_list(paths["traps"]), paths["traps"])
    expected_ids = set(gold_by_id)
    if set(input_by_id) != expected_ids or set(trap_by_id) != expected_ids:
        raise ValueError(f"{model_name} input/trap IDs do not match the gold dataset")

    results = []
    failed_count = sum(
        trap_by_id[item_id].get("trap_avoided") is False for item_id in gold_by_id
    )
    print(f"\n{model_name.upper()}: oracle-correcting {failed_count}/41 failed traps")
    corrected_position = 0

    for item_id, gold in gold_by_id.items():
        trap = trap_by_id[item_id]
        if trap.get("judge_status") != "ok":
            raise ValueError(f"Invalid trap judgment for {model_name}/{item_id}")
        if trap.get("trap_avoided") is True:
            result = mark_skipped(input_by_id[item_id])
        else:
            corrected_position += 1
            print(f"[{corrected_position:02d}/{failed_count:02d}] Correcting {item_id}")
            result = oracle_correct(tokenizer, model, input_by_id[item_id], gold)
        results.append(result)
        save_json(results, paths["output"])

    status_counts = {}
    for item in results:
        status = item["oracle_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    print(f"Saved {paths['output']}: {status_counts}")
    return {
        "model": model_name,
        "input_file": str(paths["input"]),
        "trap_results_file": str(paths["traps"]),
        "output_file": str(paths["output"]),
        "failed_traps_selected": failed_count,
        "status_counts": status_counts,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=sorted(MODEL_FILES), default=sorted(MODEL_FILES)
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    gold_by_id = index_by_id(load_json_list(GOLD_FILE), GOLD_FILE)
    tokenizer, model = load_model(args.model_path)
    summaries = [
        process_model(tokenizer, model, model_name, gold_by_id)
        for model_name in args.models
    ]
    save_json(summaries, Path("results/oracle_correction_summary.json"))
    print("\nOracle correction complete.")


if __name__ == "__main__":
    main()
