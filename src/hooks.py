# hooks.py — capture embed_out, resid_post/out, mlp_out; derive resid_pre offline

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


def _ensure_tensor(out):
    return out[0] if isinstance(out, tuple) else out


def _to_cpu_f16(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu", dtype=torch.float16)


def _find_mlp_out_module(block) -> torch.nn.Module:
    for attr in ["mlp", "ffn", "feed_forward"]:
        if hasattr(block, attr):
            mlp = getattr(block, attr)
            for name in ["down_proj", "proj_out", "o_proj", "out_proj"]:
                if hasattr(mlp, name) and isinstance(getattr(mlp, name), torch.nn.Module):
                    return getattr(mlp, name)
            last_linear = None
            for _, mod in mlp.named_modules():
                if isinstance(mod, torch.nn.Linear): last_linear = mod
            return last_linear if last_linear is not None else mlp
    return None


def _find_embed_module(model: PreTrainedModel):
    # Common names across Qwen/LLaMA/GPT2/NeoX families
    for path in [
        "model.embed_tokens",
        "model.model.embed_tokens",
        "model.transformer.embed_tokens",
        "model.transformer.wte",
        "transformer.wte",
        "tok_embeddings",
        "gpt_neox.embed_in",
    ]:
        cur = model; ok = True
        for name in path.split("."):
            if hasattr(cur, name): cur = getattr(cur, name)
            else: ok = False; break
        if ok and isinstance(cur, torch.nn.Module):
            return cur
    # Fallback: first nn.Embedding found
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Embedding):
            return m
    return None


def get_activation_cache(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    zero_attn: bool = False,  # API compatibility
) -> Dict[str, torch.Tensor]:
    """
    Cache:
      - embed_out
      - resid_post.{i} (alias resid_out.{i})
      - mlp_out.{i}
      - resid_pre.{0} = embed_out
      - resid_pre.{i} = resid_post.{i-1} for i>=1
    """
    cache: Dict[str, torch.Tensor] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    blocks = _get_blocks(model)

    # --- embeddings ---
    embed = _find_embed_module(model)
    if embed is not None:
        def embed_hook(module, inp, out):
            cache["embed_out"] = _to_cpu_f16(_ensure_tensor(out))
        handles.append(embed.register_forward_hook(embed_hook))

    # --- per-block outputs + MLP ---
    for i, block in enumerate(blocks):
        # residual after the block
        def resid_post_hook(module, inp, out, i=i):
            x = _ensure_tensor(out)
            x_cpu = _to_cpu_f16(x)
            cache[f"resid_post.{i}"] = x_cpu
            cache[f"resid_out.{i}"]  = x_cpu  # alias for compat
        handles.append(block.register_forward_hook(resid_post_hook))

        # mlp_out at final projection (or last Linear fallback)
        mlp_target = _find_mlp_out_module(block)
        if mlp_target is not None:
            def mlp_hook(module, inp, out, i=i):
                cache[f"mlp_out.{i}"] = _to_cpu_f16(_ensure_tensor(out))
            handles.append(mlp_target.register_forward_hook(mlp_hook))

    # --- run once to populate ---
    with torch.no_grad():
        _ = model(**{k: v.to(model.device) for k, v in batch.items()})

    # cleanup hooks
    for h in handles: h.remove()

    # --- derive resid_pre offline ---
    n_layers = len(blocks)
    if "embed_out" in cache:
        cache["resid_pre.0"] = cache["embed_out"]
    else:
        # fallback if embed_out missing; you'll still have resid_pre.{i>=1}
        pass

    for i in range(1, n_layers):
        prev_post = f"resid_post.{i-1}"
        if prev_post in cache and f"resid_pre.{i}" not in cache:
            cache[f"resid_pre.{i}"] = cache[prev_post]

    total_bytes = sum(v.numel() for v in cache.values()) * 2
    n_pre  = sum(1 for k in cache if k.startswith("resid_pre."))
    n_post = sum(1 for k in cache if k.startswith("resid_post."))
    n_mlp  = sum(1 for k in cache if k.startswith("mlp_out."))
    print(f"[Cache] Stored {len(cache)} tensors (pre={n_pre}, post={n_post}, mlp={n_mlp}), "
          f"~{total_bytes/1e6:.2f} MB in float16 on CPU")
    return cache
