from unittest import result

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re

class Translator:
    """
    Translates English survey items into German.
    Loads model from shared cluster storage.
    Uses translate_prompt.txt as instruction template.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model"]["name"]
        self.model_path = config["model"]["path"]
        self.max_new_tokens = config["model"]["max_new_tokens"]

        # load prompt template
        prompt_path = config["paths"]["translate_prompt"]
        print(f"Loading prompt from: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        print("Prompt loaded.")

        # load model
        self._load_model()


    def device(self) -> str:
        if torch.cuda.is_available():
            print("Device: CUDA GPU")
            return "cuda"
        elif torch.backends.mps.is_available():
            print("Device: Apple MPS")
            return "mps"
        else:
            print("Device: CPU")
            return "cpu"

    def _load_model(self):
        # Model Loading 
        # Load model and tokenizer from shared cluster storage
        print(f"Loading model: {self.model_name}")
        print(f"From path: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained( self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype = torch.float16, device_map = "auto", trust_remote_code = True)
        self.model.eval()

        print(f"Model loaded: {self.model_name}")


    # Translation 
    def translate(self, source_en: str, item_type: str, scale: str = "") -> dict:
        """
        Translate a single English survey item into German.
        Args:
            source_en : English survey item text
            item_type : attitudinal / behavioral / sociodemographic / likert
            scale     : response scale or empty string

        Returns:
            Dictionary containing the German translation and scale
        """
        # fill prompt template
        prompt = self.prompt_template.format(
            item_type = item_type,
            source_en = source_en,
            scale = scale if scale else "not specified"
        )

        # apply chat template
        messages  = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True
        )

        # tokenize
        inputs = self.tokenizer(
            formatted,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 2048
        ).to(self.model.device)

        # generate — deterministic for reproducibility
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens = self.max_new_tokens,
                do_sample      = False,
                temperature    = 1.0,
                pad_token_id   = self.tokenizer.eos_token_id
            )

        # decode new tokens only — skip the prompt
        new_tokens  = outputs[0][inputs["input_ids"].shape[1]:]
        translation = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens = True
        ).strip()

        cleaned_output = re.sub(r"```json|```", "", translation).strip()
        
        # 2. Parse the JSON
        try:
            parsed_translation = json.loads(cleaned_output)
            return parsed_translation
        except json.JSONDecodeError:
            print(f"WARNING: Failed to parse JSON. Raw output: {translation}")
            # Fallback in case the model hallucinates
            return {"trans_de": translation, "trans_scale": ""}


    # Batch Translation
    def translate_batch(self, items: list) -> list:
        """
        Translate a list of survey items.

        Args:
            items: list of dicts with keys:
                   id, source_en, item_type, scale

        Returns:
            list of dicts with added translation_de field
        """
        results = []
        total   = len(items)

        for i, item in enumerate(items):
            print(f"  [{i+1}/{total}] {item.get('id', '')} — {item.get('item_type', '')}")

            translation = self.translate(
                source_en = item["source_en"],
                item_type = item.get("item_type", "attitudinal"),
                scale     = item.get("scale", "")
            )

            result = item.copy()
            result["model_trans_de"] = translation.get("trans_de", "")
            result["model_trans_scale"] = translation.get("trans_scale", "")
            result["model"] = self.model_name
            results.append(result)

        print(f"  Done — {len(results)} items translated")
        return results