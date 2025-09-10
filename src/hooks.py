import torch
from typing import Dict, List
from transformers import PreTrainedModel


def _get_blocks(model: PreTrainedModel):
    """
    Try common transformer layouts to return the list/ModuleList of blocks.
    Covers Qwen2.5, LLaMA, GPT-NeoX, GPT-2.
    """
    # Qwen/LLaMA style
    for path in ["model.layers", "model.model.layers", "model.transformer.layers"]:
        cur = model
        ok = True
        for name in path.split("."):
            if hasattr(cur, name):
                cur = getattr(cur, name)
            else:
                ok = False
                break
        if ok and isinstance(cur, (list, torch.nn.ModuleList)) and len(cur) > 0:
            return cur

    # GPT-2 style
    try:
        h = model.transformer.h
        if isinstance(h, (list, torch.nn.ModuleList)) and len(h) > 0:
            return h
    except AttributeError:
        pass

    # GPT-NeoX style
    try:
        layers = model.gpt_neox.layers
        if isinstance(layers, (list, torch.nn.ModuleList)) and len(layers) > 0:
            return layers
    except AttributeError:
        pass

    raise RuntimeError("Could not locate transformer blocks on this model.")


def get_activation_cache(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    zero_attn: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Run model forward and cache residual/MLP activations for every block.
    Keys are 'mlp_out.{i}' and 'resid_out.{i}'.
    Stored on CPU in float16 to save GPU memory.
    """
    cache: Dict[str, torch.Tensor] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    blocks = _get_blocks(model)

    def _to_cpu_f16(x: torch.Tensor) -> torch.Tensor:
        return x.detach().to("cpu", dtype=torch.float16)

    def _ensure_tensor(out):
        # Some modules return tuples; we want the main hidden states tensor
        if isinstance(out, tuple):
            return out[0]
        return out

    # Register hooks
    for i, block in enumerate(blocks):
        # MLP output
        if hasattr(block, "mlp"):
            def mlp_hook(module, inp, out, i=i):
                x = _ensure_tensor(out)
                cache[f"mlp_out.{i}"] = _to_cpu_f16(x)
            handles.append(block.mlp.register_forward_hook(mlp_hook))

        # Block (residual stream) output
        def block_hook(module, inp, out, i=i):
            x = _ensure_tensor(out)
            cache[f"resid_out.{i}"] = _to_cpu_f16(x)
        handles.append(block.register_forward_hook(block_hook))

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
