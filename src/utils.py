
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-Math-1.5B"

def load_model(
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    dtype: str = "auto",
    attn_impl: str = "eager",  # <- key: request eager attention
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dtype == "auto":
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,                # new param name in recent transformers
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,    # <- force eager so attentions are returned
    )
    model.eval()
    return tok, model, device

@torch.no_grad()
def get_attentions(tok, model, prompt: str):
    # safety: ensure eager at runtime too (older HF versions may ignore load flag)
    try:
        model.config.attn_implementation = "eager"
    except Exception:
        pass

    inputs = tok(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    out = model(
        input_ids=input_ids,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
    )
    return {
        "attentions": out.attentions,        # List[num_layers] or tuple
        "hidden_states": out.hidden_states,  # tuple(len = num_layers+1)
        "input_ids": input_ids,
        "tokens": tok.convert_ids_to_tokens(input_ids[0].tolist()),
    }

@torch.no_grad()
def logit_lens(tok, model, hidden_states, position: int = -1):
    lm_head = model.lm_head
    tops = []
    for layer, h in enumerate(hidden_states[1:]):  # skip embeddings
        vec = h[0, position, :]
        logits = (vec @ lm_head.weight.T).float()
        topk = torch.topk(logits, k=5)
        toks = tok.convert_ids_to_tokens(topk.indices.tolist())
        vals = topk.values.tolist()
        tops.append({"layer": layer, "top_tokens": toks, "logits": vals})
    return tops
