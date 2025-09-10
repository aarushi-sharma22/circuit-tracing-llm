from typing import List, Dict, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_qwen_instruct_4bit(
    model_id: str,
    device_map: str = "auto",
    torch_dtype=torch.bfloat16,
):
    """
    Load Qwen2.5-7B-Instruct (or any Hugging Face CausalLM) in 4-bit using bitsandbytes.
    Works in Google Colab (Linux + GPU).
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    return model, tokenizer


def apply_qwen_chat_template(tokenizer, questions: List[str]) -> List[str]:
    """
    Apply Qwen-style chat template to a list of user questions.
    Returns a list of rendered strings.
    """
    rendered = []
    for q in questions:
        messages = [{"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        rendered.append(text)
    return rendered


def batch_tokenize(
    tokenizer, rendered_prompts: List[str], max_length: int = 256
) -> Dict[str, torch.Tensor]:
    """
    Tokenize multiple rendered prompts into a batch.
    """
    enc = tokenizer(
        rendered_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return enc


def yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    """
    Get token ids for 'Yes' and 'No' (single-token best effort).
    """
    cands = ["Yes", " Yes", "No", " No"]
    ids = {c: tokenizer.encode(c, add_special_tokens=False) for c in cands}

    def pick(base: str) -> int:
        for variant in [base, " " + base]:
            pieces = ids[variant]
            if len(pieces) == 1:
                return pieces[0]
        v = min([base, " " + base], key=lambda x: len(ids[x]))
        return ids[v][0]

    return pick("Yes"), pick("No")
