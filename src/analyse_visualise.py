import time
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Iterable
from transformers import PreTrainedModel
from patching import run_with_patched_activation
from load_and_tokenize import yes_no_token_ids


def _num_blocks(model: PreTrainedModel) -> int:
    for path in ["model.layers", "model.model.layers", "model.transformer.layers"]:
        cur = model
        ok = True
        for name in path.split("."):
            if hasattr(cur, name):
                cur = getattr(cur, name)
            else:
                ok = False
                break
        if ok and isinstance(cur, (list, torch.nn.ModuleList)):
            return len(cur)
    try:
        return len(model.transformer.h)
    except Exception:
        pass
    try:
        return len(model.gpt_neox.layers)
    except Exception:
        pass
    raise RuntimeError("Could not determine number of transformer blocks.")


def sweep_patch_layers(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    cache_source: Dict[str, torch.Tensor],
    source_idx: int,
    target_idx: int,
    kind: str = "mlp_out",                  # or "resid_out"
    layers: Optional[Iterable[int]] = None, # e.g., range(5) to test first 5 layers
    progress: bool = True,
) -> List[float]:
    """
    For each layer i, patch (kind, i) on target_idx with source_idx's cached activation.
    Measure ΔP(Yes) at the final token for the target item.
    """
    # Get Yes/No token IDs
    yes_id, _ = yes_no_token_ids(model.tokenizer)

    # Move inputs once
    inputs_on_device = {k: v.to(model.device) for k, v in batch.items()}

    # Baseline (no patch) probability of "Yes" for target
    with torch.inference_mode():
        logits_base = model(**inputs_on_device).logits
    probs_base = F.softmax(logits_base[target_idx, -1], dim=-1)
    p_yes_base = probs_base[yes_id].item()

    # Which layers to sweep
    if layers is None:
        n_layers = _num_blocks(model)
        layer_list = list(range(n_layers))
    else:
        layer_list = list(layers)

    effects: List[float] = []
    t0 = time.time()
    for j, layer_idx in enumerate(layer_list):
        t_layer = time.time()
        with torch.inference_mode():
            patched_logits = run_with_patched_activation(
                model,
                batch,
                cache_source,
                kind=kind,
                layer_idx=layer_idx,
                source_idx=source_idx,
                target_idx=target_idx,
                inputs_on_device=inputs_on_device,
            )
        probs_patched = F.softmax(patched_logits[target_idx, -1], dim=-1)
        delta = probs_patched[yes_id].item() - p_yes_base
        effects.append(delta)

        if progress:
            print(f"[patch] {kind} L{layer_idx} done "
                  f"(ΔYes={delta:+.4f}, {time.time()-t_layer:.2f}s, step {j+1}/{len(layer_list)})")

    if progress:
        print(f"[patch] Completed {len(layer_list)} layers in {time.time()-t0:.2f}s")

    return effects
