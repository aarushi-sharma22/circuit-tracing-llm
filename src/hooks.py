import torch
from typing import Dict, List
from transformers import PreTrainedModel


def _get_blocks(model: PreTrainedModel):
    """
    Try common transformer layouts to return the list/ModuleList of blocks.
    Covers Qwen2.5, LLaMA, GPT-NeoX, GPT-2.
    """
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

    try:
        h = model.transformer.h
        if isinstance(h, (list, torch.nn.ModuleList)) and len(h) > 0:
            return h
    except AttributeError:
        pass

    try:
        layers = model.gpt_neox.layers
        if isinstance(layers, (list, torch.nn.ModuleList)) and len(layers) > 0:
            return layers
    except AttributeError:
        pass

    raise RuntimeError("Could not locate transformer blocks on this model.")


def _ensure_tensor(out):
    # Some modules return tuples; we want the main hidden states tensor
    return out[0] if isinstance(out, tuple) else out


def _to_cpu_f16(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu", dtype=torch.float16)


def _find_mlp_out_module(block) -> torch.nn.Module:
    """
    Return a stable submodule that produces the post-MLP tensor added back to the residual.
    Prefer the last Linear in the MLP stack (e.g., down_proj). Fall back to block.mlp.
    """
    # common names
    for attr in ["mlp", "ffn", "feed_forward"]:
        if hasattr(block, attr):
            mlp = getattr(block, attr)
            # Prefer a specific final projection if present
            for name in ["down_proj", "proj_out", "o_proj", "out_proj"]:
                if hasattr(mlp, name) and isinstance(getattr(mlp, name), torch.nn.Module):
                    return getattr(mlp, name)
            # Otherwise, find the last Linear within the MLP
            last_linear = None
            for _, mod in mlp.named_modules():
                if isinstance(mod, torch.nn.Linear):
                    last_linear = mod
            if last_linear is not None:
                return last_linear
            return mlp  # fallback
    return None


def get_activation_cache(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    zero_attn: bool = False,  # kept for API compatibility
) -> Dict[str, torch.Tensor]:
    """
    Run model forward and cache residual/MLP activations for every block.
    Keys are 'mlp_out.{i}' and 'resid_out.{i}'.
    Stored on CPU in float16 to save GPU memory.
    """
    cache: Dict[str, torch.Tensor] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    blocks = _get_blocks(model)

    # Register hooks
    for i, block in enumerate(blocks):
        # MLP output (robust)
        mlp_target = _find_mlp_out_module(block)
        if mlp_target is not None:
            def mlp_hook(module, inp, out, i=i):
                x = _ensure_tensor(out)
                cache[f"mlp_out.{i}"] = _to_cpu_f16(x)
            handles.append(mlp_target.register_forward_hook(mlp_hook))

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
