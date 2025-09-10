import torch
import torch.nn.functional as F
from typing import Dict, List
from transformers import PreTrainedModel
from patching import run_with_patched_activation
from load_and_tokenize import yes_no_token_ids


def sweep_patch_layers(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    cache_source: Dict[str, torch.Tensor],
    source_idx: int,
    target_idx: int,
    kind: str = "mlp_out",  # or "resid_out"
) -> List[float]:
    """
    Patch source -> target at each layer, measure shift in Yes prob.

    Args:
        model: Qwen model (on GPU in Colab)
        batch: tokenized batch
        cache_source: cached activations from previous run
        source_idx: index of prompt to copy from
        target_idx: index of prompt to patch into
        kind: which activation to patch ("mlp_out" or "resid_out")

    Returns:
        effects: list of change in Yes prob for each layer
    """
    # Get Yes/No token IDs
    yes_id, no_id = yes_no_token_ids(model.tokenizer)

    # Baseline (no patch) probability of "Yes" for target
    with torch.no_grad():
        logits_base = model(**{k: v.to(model.device) for k, v in batch.items()}).logits
    probs_base = F.softmax(logits_base[target_idx, -1], dim=-1)
    p_yes_base = probs_base[yes_id].item()

    effects = []
    for layer_num in range(model.config.num_hidden_layers):
        layer_name = f"layer_{layer_num}.{kind}"

        # Run forward with patched activation
        patched_logits = run_with_patched_activation(
            model,
            batch,
            cache_source,
            layer_name=layer_name,
            source_idx=source_idx,
            target_idx=target_idx,
        )

        # Compute new Yes probability
        probs_patched = F.softmax(patched_logits[target_idx, -1], dim=-1)
        p_yes_patched = probs_patched[yes_id].item()

        # Record effect size
        effects.append(p_yes_patched - p_yes_base)

    return effects
