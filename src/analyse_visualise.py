import torch
import torch.nn.functional as F
from typing import Dict, List
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
    kind: str = "mlp_out",  # or "resid_out"
) -> List[float]:
    """
    For each layer i, patch (kind, i) on target_idx with source_idx's cached activation.
    Measure ΔP(Yes) at the final token for the target item.
    """
    # Get Yes/No token IDs
    yes_id, _ = yes_no_token_ids(model.tokenizer)

    # Baseline (no patch) probability of "Yes" for target
    with torch.no_grad():
        logits_base = model(**{k: v.to(model.device) for k, v in batch.items()}).logits
    probs_base = F.softmax(logits_base[target_idx, -1], dim=-1)
    p_yes_base = probs_base[yes_id].item()

    n_layers = _num_blocks(model)
    effects: List[float] = []

    for layer_idx in range(n_layers):
        patched_logits = run_with_patched_activation(
            model,
            batch,
            cache_source,
            kind=kind,
            layer_idx=layer_idx,
            source_idx=source_idx,
            target_idx=target_idx,
        )

        # Compute new Yes probability
        probs_patched = F.softmax(patched_logits[target_idx, -1], dim=-1)
        p_yes_patched = probs_patched[yes_id].item()

        # Record effect size
        effects.append(p_yes_patched - p_yes_base)

    return effects
