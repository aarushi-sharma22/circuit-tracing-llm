import torch
from typing import Dict, List
from transformers import PreTrainedModel


def _get_blocks(model: PreTrainedModel):
    # Must mirror hooks._get_blocks so indices align
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


def run_with_patched_activation(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    cache_source: Dict[str, torch.Tensor],
    kind: str,                 # "mlp_out" or "resid_out"
    layer_idx: int,
    source_idx: int,
    target_idx: int,
) -> torch.Tensor:
    """
    Replace the activation at (kind, layer_idx) for the target batch item
    with the cached activation from source_idx, then run the forward pass.
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _ensure_tensor(out):
        return out[0] if isinstance(out, tuple) else out

    def _repack_like(original_out, new_tensor):
        if isinstance(original_out, tuple):
            # Replace the first element; keep any auxiliary outputs
            as_list = list(original_out)
            as_list[0] = new_tensor
            return tuple(as_list)
        return new_tensor

    # Fetch the cached tensor and remember device/dtype conversion
    cache_key = f"{kind}.{layer_idx}"
    if cache_key not in cache_source:
        raise KeyError(f"Cache missing key '{cache_key}'. Did you call get_activation_cache on the same model/batch?")

    blocks = _get_blocks(model)
    block = blocks[layer_idx]

    src_act = cache_source[cache_key]  # on CPU float16
    # We'll move this to the hook's device/dtype when applying

    def patch_hook(module, inp, out):
        current = _ensure_tensor(out)
        # Clone to avoid in-place on autograd graph (we're in no_grad anyway, but keep clean)
        patched = current.clone()

        # ensure dtype/device match
        to_insert = src_act.to(device=current.device, dtype=current.dtype)

        # Replace the whole sequence for the target sample with source sample
        patched[target_idx, :, :] = to_insert[source_idx, :, :]

        return _repack_like(out, patched)

    if kind == "mlp_out":
        if not hasattr(block, "mlp"):
            raise ValueError(f"Block {layer_idx} has no .mlp; cannot patch mlp_out.")
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
