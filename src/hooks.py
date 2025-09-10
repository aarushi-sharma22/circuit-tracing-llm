# hooks.py — capture resid_pre at the first per-layer norm; resid_post; mlp_out

import torch
from typing import Dict, List
from transformers import PreTrainedModel


def _get_blocks(model: PreTrainedModel):
    for path in ["model.layers", "model.model.layers", "model.transformer.layers"]:
        cur = model; ok = True
        for name in path.split("."):
            if hasattr(cur, name): cur = getattr(cur, name)
            else: ok = False; break
        if ok and isinstance(cur, (list, torch.nn.ModuleList)) and len(cur) > 0:
            return cur
    try:
        h = model.transformer.h
        if isinstance(h, (list, torch.nn.ModuleList)) and len(h) > 0: return h
    except AttributeError: pass
    try:
        layers = model.gpt_neox.layers
        if isinstance(layers, (list, torch.nn.ModuleList)) and len(layers) > 0: return layers
    except AttributeError: pass
    raise RuntimeError("Could not locate transformer blocks on this model.")


def _ensure_tensor(out): return out[0] if isinstance(out, tuple) else out
def _to_cpu_f16(x: torch.Tensor) -> torch.Tensor: return x.detach().to("cpu", dtype=torch.float16)

def _find_mlp_out_module(block) -> torch.nn.Module:
    for attr in ["mlp","ffn","feed_forward"]:
        if hasattr(block, attr):
            mlp = getattr(block, attr)
            for name in ["down_proj","proj_out","o_proj","out_proj"]:
                if hasattr(mlp, name) and isinstance(getattr(mlp, name), torch.nn.Module):
                    return getattr(mlp, name)
            last_linear = None
            for _, mod in mlp.named_modules():
                if isinstance(mod, torch.nn.Linear): last_linear = mod
            return last_linear if last_linear is not None else mlp
    return None

def _find_first_norm(block) -> torch.nn.Module:
    for name in ["input_layernorm", "ln_1", "pre_attention_layernorm", "norm1"]:
        if hasattr(block, name) and isinstance(getattr(block, name), torch.nn.Module):
            return getattr(block, name)
    for _, mod in block.named_modules():
        cls = mod.__class__.__name__.lower()
        if isinstance(mod, torch.nn.LayerNorm) or "rmsnorm" in cls or "layernorm" in cls:
            return mod
    return None


def get_activation_cache(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    zero_attn: bool = False,  # kept for API compatibility
) -> Dict[str, torch.Tensor]:
    """
    Cache per-layer activations:
      - resid_pre.{i}: input to the first norm inside block i (canonical block input)
      - resid_post.{i}: output of block i
      - resid_out.{i}: alias to resid_post.{i} for compatibility
      - mlp_out.{i}: final MLP projection output inside block i
    """
    cache: Dict[str, torch.Tensor] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    blocks = _get_blocks(model)

    for i, block in enumerate(blocks):
        # --- resid_pre at the first per-layer norm input ---
        norm = _find_first_norm(block)
        if norm is not None:
            def resid_pre_hook(module, inp, i=i):
                x = inp[0] if isinstance(inp, tuple) else inp
                cache[f"resid_pre.{i}"] = _to_cpu_f16(x)
            handles.append(norm.register_forward_pre_hook(resid_pre_hook))
        else:
            # Fallback: block pre-hook (less robust but better than nothing)
            def resid_pre_block(module, inp, i=i):
                x = inp[0] if isinstance(inp, tuple) else inp
                cache[f"resid_pre.{i}"] = _to_cpu_f16(x)
            handles.append(block.register_forward_pre_hook(resid_pre_block))

        # --- resid_post / resid_out at block output ---
        def resid_post_hook(module, inp, out, i=i):
            x = _ensure_tensor(out)
            x_cpu = _to_cpu_f16(x)
            cache[f"resid_post.{i}"] = x_cpu
            cache[f"resid_out.{i}"]  = x_cpu
        handles.append(block.register_forward_hook(resid_post_hook))

        # --- mlp_out (final projection preferred) ---
        mlp_target = _find_mlp_out_module(block)
        if mlp_target is not None:
            def mlp_hook(module, inp, out, i=i):
                cache[f"mlp_out.{i}"] = _to_cpu_f16(_ensure_tensor(out))
            handles.append(mlp_target.register_forward_hook(mlp_hook))

    with torch.no_grad():
        _ = model(**{k: v.to(model.device) for k, v in batch.items()})

    for h in handles: h.remove()

    total_bytes = sum(v.numel() for v in cache.values()) * 2
    n_pre  = sum(1 for k in cache if k.startswith("resid_pre."))
    n_post = sum(1 for k in cache if k.startswith("resid_post."))
    n_mlp  = sum(1 for k in cache if k.startswith("mlp_out."))
    print(f"[Cache] Stored {len(cache)} tensors "
          f"(pre={n_pre}, post={n_post}, mlp={n_mlp}), ~{total_bytes/1e6:.2f} MB in float16 on CPU")
    return cache
