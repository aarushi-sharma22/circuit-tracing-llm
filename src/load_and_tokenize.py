from typing import List, Dict, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_qwen_instruct_4bit(
    model_id: str,
    device_map: str = "auto",
    torch_dtype=torch.bfloat16,
):
    """
    Load a CausalLM (e.g., Qwen/Qwen2.5-7B-Instruct).
    Tries 4-bit (bitsandbytes) and falls back to fp16/bf16 if bnb is unavailable.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    quantization_config = None
    use_4bit = True
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    except Exception:
        print("[WARN] bitsandbytes not available or misconfigured; will fall back to standard weights.")
        use_4bit = False

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config if use_4bit else None,
        )
    except Exception as e:
        print("[INFO] Falling back to standard model (fp16/bf16). Error:", str(e))
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )

    return model, tokenizer


def apply_qwen_chat_template(tokenizer, questions: List[str], add_generation_prompt: bool = True) -> List[str]:
    """
    Render each plain question into a single-turn chat string using the model's chat template.
    """
    rendered: List[str] = []
    for q in questions:
        messages = [{"role": "user", "content": q}]
        s = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            truncation=False,
        )
        rendered.append(s)
    return rendered


def batch_tokenize(tokenizer, prompts: List[str], max_length: int = 4096) -> Dict[str, torch.Tensor]:
    """
    Tokenize a list of already-rendered prompts.
    """
    batch = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    return batch


def yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    """
    Get token ids for 'Yes' and 'No' (single-token best effort).
    """
    candidates = ["Yes", " Yes", "No", " No"]
    ids = {c: tokenizer.encode(c, add_special_tokens=False) for c in candidates}

    def pick(base: str) -> int:
        for variant in [base, " " + base]:
            pieces = ids[variant]
            if len(pieces) == 1:
                return pieces[0]
        # Fall back to the first piece of the shortest tokenization
        v = min([base, " " + base], key=lambda x: len(ids[x]))
        return ids[v][0]

    return pick("Yes"), pick("No")
