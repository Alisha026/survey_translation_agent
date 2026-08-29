import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from typing import Callable, Optional

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
        self.max_new_tokens = config["model"].get("max_new_tokens", 1024)

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
    def _generate(self, prompt: str) -> str:
        """Generate and decode one model response."""
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        ).strip()

    @staticmethod
    def _parse_translation(raw_output: str) -> dict:
        """Extract the model translation fields from a JSON response."""
        cleaned = re.sub(r"```(?:json)?\s*|```", "", raw_output, flags=re.IGNORECASE).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Some models add a short explanation around an otherwise valid object.
            start = cleaned.find("{")
            if start == -1:
                raise ValueError("response does not contain a JSON object")
            try:
                parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
            except json.JSONDecodeError as exc:
                raise ValueError("response contains malformed JSON") from exc

        if not isinstance(parsed, dict):
            raise ValueError("JSON response is not an object")

        model_trans_de = parsed.get("model_trans_de")
        model_trans_scale = parsed.get("model_trans_scale")
        if not isinstance(model_trans_de, str):
            raise ValueError("model_trans_de is missing or is not text")
        if not isinstance(model_trans_scale, str):
            raise ValueError("model_trans_scale is missing or is not text")

        return {
            "model_trans_de": model_trans_de.strip(),
            "model_trans_scale": model_trans_scale.strip()
        }

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

        last_error = None
        for attempt in range(2):
            attempt_prompt = prompt
            if attempt == 1:
                attempt_prompt += (
                    "\n\nYour previous response did not match the required format. "
                    "Return only one valid JSON object with non-empty string values "
                    "for model_trans_de and model_trans_scale."
                )
            raw_output = self._generate(attempt_prompt)
            try:
                return self._parse_translation(raw_output)
            except ValueError as exc:
                last_error = exc
                print(
                    f"WARNING: Invalid model output (attempt {attempt + 1}/2): {exc}. "
                    f"Raw output: {raw_output[:500]}"
                )

        raise ValueError(f"Model returned invalid output twice: {last_error}")


    # Batch Translation
    def translate_batch(
        self,
        items: list,
        on_result: Optional[Callable[[list], None]] = None
    ) -> list:
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

            result = item.copy()
            try:
                translation = self.translate(
                    source_en=item["source_en"],
                    item_type=item.get("item_type", "attitudinal"),
                    scale=item.get("scale", "")
                )
                result["model_trans_de"] = translation["model_trans_de"]
                result["model_trans_scale"] = translation["model_trans_scale"]
            except (KeyError, ValueError) as exc:
                print(f"ERROR: Translation failed for {item.get('id', '')}: {exc}")
                result["model_trans_de"] = ""
                result["model_trans_scale"] = ""

            result["model"] = self.model_name
            results.append(result)
            if on_result is not None:
                on_result(results)

        print(f"Done — {len(results)} items translated")
        return results
