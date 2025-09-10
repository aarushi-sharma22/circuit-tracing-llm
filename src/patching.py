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
    """
    Run the model on a batch, but patch one activation for one prompt
    with the cached activation from another prompt.

    Args:
        model: Qwen model (Colab GPU, bitsandbytes 4-bit)
        batch: tokenized batch
        cache_source: cache dict from a previous run (Step 2)
        layer_name: which activation to patch, e.g. "layer_12.mlp_out"
        source_idx: index of the prompt to copy from
        target_idx: index of the prompt to patch into

    Returns:
        logits: [batch, seq, vocab] after patching
    """
    handles = []

    def patch_hook(module, inp, out):
        # Grab source activation from cache
        act_src = cache_source[layer_name][source_idx].to(out.device)

        # Clone and overwrite target row
        out_new = out.clone()
        out_new[target_idx] = act_src
        return out_new

    # Decode layer number and activation kind
    layer_num = int(layer_name.split("_")[1])
    kind = layer_name.split(".")[-1]  # "mlp_out" or "resid_out"
    block = model.model.layers[layer_num]

    # Register patch hook
    if kind == "mlp_out":
        handles.append(block.mlp.register_forward_hook(patch_hook))
    elif kind == "resid_out":
        handles.append(block.register_forward_hook(patch_hook))
    else:
        raise ValueError(f"Unsupported layer kind: {kind}")

    # Forward pass with patch applied
    with torch.no_grad():
        outputs = model(**{k: v.to(model.device) for k, v in batch.items()})
        logits = outputs.logits  # [batch, seq, vocab]

    # Clean up hooks
    for h in handles:
        h.remove()

    return logits
