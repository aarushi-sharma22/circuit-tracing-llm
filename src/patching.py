# patching.py — robust residual/MLP patching with try/finally hook cleanup

import torch
from typing import Dict, List
from transformers import PreTrainedModel


def _get_blocks(model: PreTrainedModel):
    """
    Return the list/ModuleList of transformer blocks for common decoder stacks.
    """
    for path in ["model.layers", "model.model.layers", "model.transformer.layers"]:
        cur = model; ok = True
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
    return out[0] if isinstance(out, tuple) else out


def _repack_like(original_out, new_tensor):
    if isinstance(original_out, tuple):
        lst = list(original_out)
        lst[0] = new_tensor
        return tuple(lst)
    return new_tensor


def _find_mlp_out_module(block) -> torch.nn.Module:
    """
    Prefer the final MLP projection (down_proj / o_proj / out_proj).
    Fallback to last Linear inside the MLP.
    """
    for attr in ["mlp", "ffn", "feed_forward"]:
        if hasattr(block, attr):
            mlp = getattr(block, attr)
            for name in ["down_proj", "proj_out", "o_proj", "out_proj"]:
                if hasattr(mlp, name) and isinstance(getattr(mlp, name), torch.nn.Module):
                    return getattr(mlp, name)
            last_linear = None
            for _, mod in mlp.named_modules():
                if isinstance(mod, torch.nn.Linear):
                    last_linear = mod
            return last_linear if last_linear is not None else mlp
    return None


def _find_first_norm(block) -> torch.nn.Module:
    """
    Find the first per-layer normalization module inside a block
    (works for LLaMA/Qwen-style RMSNorm or LayerNorm).
    """
    for name in ["input_layernorm", "ln_1", "pre_attention_layernorm", "norm1"]:
        if hasattr(block, name) and isinstance(getattr(block, name), torch.nn.Module):
            return getattr(block, name)
    # Fallback: scan submodules for a norm-like module
    for _, mod in block.named_modules():
        cls = mod.__class__.__name__.lower()
        if isinstance(mod, torch.nn.LayerNorm) or "rmsnorm" in cls or "layernorm" in cls:
            return mod
    return None


def run_with_patched_activation(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    cache_source: Dict[str, torch.Tensor],
    kind: str,                 # "mlp_out" or "resid_out"
    layer_idx: int,
    source_idx: int,
    target_idx: int,
    inputs_on_device: Dict[str, torch.Tensor] = None,
) -> torch.Tensor:
    """
    - 'mlp_out': patch the final MLP projection output inside block[layer_idx] via forward hook.
    - 'resid_out': patch the residual stream *entering* block[layer_idx] by pre-hooking
      its first LayerNorm/RMSNorm (i.e., the canonical block input). Uses cache 'resid_pre.{layer_idx}'
      if present; else falls back to 'resid_out.{layer_idx}' and patches the next block's input.

    Hooks are always removed via try/finally so crashes don’t leave dangling hooks.
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []
    blocks = _get_blocks(model)
    n_layers = len(blocks)

    # Choose cache key & placement
    if kind == "resid_out":
        if f"resid_pre.{layer_idx}" in cache_source:
            cache_key = f"resid_pre.{layer_idx}"
            where = "this_block_pre_norm"
        elif f"resid_out.{layer_idx}" in cache_source:
            cache_key = f"resid_out.{layer_idx}"
            where = "next_block_pre"
        else:
            raise KeyError(f"Cache missing 'resid_pre.{layer_idx}' and 'resid_out.{layer_idx}'. Rebuild cache.")
    else:
        cache_key = f"{kind}.{layer_idx}"
        where = None

    if cache_key not in cache_source:
        raise KeyError(f"Cache missing key '{cache_key}'.")

    src_act = cache_source[cache_key]  # CPU fp16

    def patch_forward_hook(module, inp, out):
        current = _ensure_tensor(out)
        patched = current.clone()
        to_insert = src_act.to(device=current.device, dtype=current.dtype)
        if patched.shape != to_insert.shape:
            raise RuntimeError(f"Shape mismatch at {cache_key}: {tuple(patched.shape)} vs {tuple(to_insert.shape)}")
        patched[target_idx, :, :] = to_insert[source_idx, :, :]
        return _repack_like(out, patched)

    def patch_pre_hook(module, inp):
        x = inp[0] if isinstance(inp, tuple) else inp
        x = x.clone()
        to_insert = src_act.to(device=x.device, dtype=x.dtype)
        if x.shape != to_insert.shape:
            raise RuntimeError(f"Shape mismatch at {cache_key} (pre): {tuple(x.shape)} vs {tuple(to_insert.shape)}")
        x[target_idx, :, :] = to_insert[source_idx, :, :]
        if isinstance(inp, tuple):
            lst = list(inp); lst[0] = x; return tuple(lst)
        return x

    try:
        if kind == "mlp_out":
            block = blocks[layer_idx]
            mlp_target = _find_mlp_out_module(block)
            if mlp_target is None:
                raise ValueError(f"Block {layer_idx} has no MLP-like submodule; cannot patch mlp_out.")
            handles.append(mlp_target.register_forward_hook(patch_forward_hook))

        elif kind == "resid_out":
            if where == "this_block_pre_norm":
                block = blocks[layer_idx]
                norm = _find_first_norm(block)
                if norm is not None:
                    handles.append(norm.register_forward_pre_hook(patch_pre_hook))
                else:
                    # Fallback: pre-hook the block itself
                    handles.append(block.register_forward_pre_hook(patch_pre_hook))
            else:
                # Fallback: use resid_post to patch next block's input
                if layer_idx < n_layers - 1:
                    handles.append(blocks[layer_idx + 1].register_forward_pre_hook(patch_pre_hook))
                else:
                    # Last layer fallback: patch its output
                    handles.append(blocks[layer_idx].register_forward_hook(patch_forward_hook))
        else:
            raise ValueError(f"Unsupported layer kind: {kind}")

        with torch.inference_mode():
            io = inputs_on_device if inputs_on_device is not None else {k: v.to(model.device) for k, v in batch.items()}
            logits = model(**io).logits
        return logits
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
