import torch
from typing import Dict
from transformers import PreTrainedModel


def run_with_patched_activation(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    cache_source: Dict[str, torch.Tensor],
    layer_name: str,
    source_idx: int,
    target_idx: int,
) -> torch.Tensor:
    handles = []

    def patch_hook(module, inp, out):
        act_src = cache_source[layer_name][source_idx].to(out.device)
        out_new = out.clone()
        out_new[target_idx] = act_src
        return out_new

    # Decode layer number and activation kind
    layer_part, kind = layer_name.split(".")  # "layer_0", "mlp_out"
    layer_num = int(layer_part.split("_")[1])  # → 0
    block = model.model.layers[layer_num]

    if kind == "mlp_out":
        handles.append(block.mlp.register_forward_hook(patch_hook))
    elif kind == "resid_out":
        handles.append(block.register_forward_hook(patch_hook))
    else:
        raise ValueError(f"Unsupported layer kind: {kind}")

    with torch.no_grad():
        outputs = model(**{k: v.to(model.device) for k, v in batch.items()})
        logits = outputs.logits

    for h in handles:
        h.remove()

    return logits
