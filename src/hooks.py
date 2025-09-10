import torch
from typing import Dict
from transformers import PreTrainedModel


def get_activation_cache(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    zero_attn: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Run model forward and cache residual/MLP activations.
    All activations are stored on CPU in float16 to save GPU memory.
    """
    cache = {}
    handles = []

    def save_hook(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                cache[name] = out.detach().cpu().to(torch.float16)
            else:  # sometimes (out,) is a tuple
                cache[name] = out[0].detach().cpu().to(torch.float16)
            return out
        return hook

    def save_mlp_in(name):
        def hook(module, inp, out):
            hidden_in = inp[0]  # input to the MLP
            cache[name] = hidden_in.detach().cpu().to(torch.float16)
            return out
        return hook

    # Register hooks on each transformer block
    for i in range(model.config.num_hidden_layers):
        block = model.model.layers[i]

        # Residual stream after block
        handles.append(block.register_forward_hook(
            save_hook(f"layer_{i}.resid_out")
        ))

        # MLP input (before feed-forward)
        handles.append(block.mlp.register_forward_hook(
            save_mlp_in(f"layer_{i}.mlp_in")
        ))

        # MLP output (after feed-forward)
        handles.append(block.mlp.register_forward_hook(
            save_hook(f"layer_{i}.mlp_out")
        ))

    # Forward pass (fills cache)
    with torch.no_grad():
        _ = model(**{k: v.to(model.device) for k, v in batch.items()})

    # Remove hooks
    for h in handles:
        h.remove()

    # Memory usage estimate
    total_elems = sum(v.numel() for v in cache.values())
    total_bytes = total_elems * 2  # float16 = 2 bytes
    print(f"[Cache] Stored {len(cache)} tensors, ~{total_bytes/1e6:.2f} MB in float16 on CPU")

    return cache
