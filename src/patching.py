import torch
from typing import Dict, List
from transformers import PreTrainedModel


def _get_blocks(model: PreTrainedModel):
    for path in ["model.layers","model.model.layers","model.transformer.layers"]:
        cur = model; ok = True
        for name in path.split("."):
            if hasattr(cur, name): cur = getattr(cur, name)
            else: ok = False; break
        if ok and isinstance(cur,(list,torch.nn.ModuleList)) and len(cur)>0:
            return cur
    try:
        h = model.transformer.h
        if isinstance(h,(list,torch.nn.ModuleList)) and len(h)>0: return h
    except AttributeError: pass
    try:
        layers = model.gpt_neox.layers
        if isinstance(layers,(list,torch.nn.ModuleList)) and len(layers)>0: return layers
    except AttributeError: pass
    raise RuntimeError("Could not locate transformer blocks on this model.")


def _ensure_tensor(out):
    return out[0] if isinstance(out, tuple) else out


def _repack_like(original_out, new_tensor):
    if isinstance(original_out, tuple):
        lst = list(original_out); lst[0] = new_tensor; return tuple(lst)
    return new_tensor


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


def run_with_patched_activation(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    cache_source: Dict[str, torch.Tensor],
    kind: str,                 # "mlp_out" or "resid_out" (prefers resid_pre)
    layer_idx: int,
    source_idx: int,
    target_idx: int,
    inputs_on_device: Dict[str, torch.Tensor] = None,
) -> torch.Tensor:
    """
    - For 'mlp_out': patch the MLP final projection output inside block[layer_idx].
    - For 'resid_out': prefer patching the residual stream *before* block[layer_idx]
      using cache key 'resid_pre.{layer_idx}'. If not available, fall back to
      patching the *input* of the next block using 'resid_out.{layer_idx}'.
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []
    blocks = _get_blocks(model)
    n_layers = len(blocks)

    # Choose cache key & hook strategy for residuals
    use_pre = False
    if kind == "resid_out":
        key_pre  = f"resid_pre.{layer_idx}"
        key_post = f"resid_out.{layer_idx}"
        if key_pre in cache_source:
            cache_key = key_pre
            use_pre = True
        elif key_post in cache_source:
            cache_key = key_post
            use_pre = False
        else:
            raise KeyError(f"Cache missing both '{key_pre}' and '{key_post}'. Rebuild cache with updated hooks.")
    else:
        cache_key = f"{kind}.{layer_idx}"

    if cache_key not in cache_source:
        raise KeyError(f"Cache missing key '{cache_key}'. Did you call get_activation_cache on the same model/batch?")

    src_act = cache_source[cache_key]  # CPU fp16

    # --- hook bodies ---
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

    # --- attach the right hook ---
    if kind == "mlp_out":
        block = blocks[layer_idx]
        mlp_target = _find_mlp_out_module(block)
        if mlp_target is None:
            raise ValueError(f"Block {layer_idx} has no MLP-like submodule; cannot patch mlp_out.")
        handles.append(mlp_target.register_forward_hook(patch_forward_hook))

    elif kind == "resid_out":
        if use_pre:
            # Replace *input* to this very block (resid_pre)
            handles.append(blocks[layer_idx].register_forward_pre_hook(patch_pre_hook))
        else:
            # Fallback: replace input to next block with resid_post from this one
            if layer_idx < n_layers - 1:
                handles.append(blocks[layer_idx + 1].register_forward_pre_hook(patch_pre_hook))
            else:
                # Last layer fallback: patch the block's output
                handles.append(blocks[layer_idx].register_forward_hook(patch_forward_hook))
    else:
        raise ValueError(f"Unsupported layer kind: {kind}")

    with torch.inference_mode():
        io = inputs_on_device if inputs_on_device is not None else {k: v.to(model.device) for k, v in batch.items()}
        logits = model(**io).logits

    for h in handles:
        h.remove()

    return logits
