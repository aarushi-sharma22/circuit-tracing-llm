import os
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Iterable, Any

import torch
import torch.nn.functional as F
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


def _hash_list_of_strings(xs: List[str]) -> str:
    m = hashlib.sha256()
    for s in xs:
        m.update(s.encode("utf-8"))
        m.update(b"\n")
    return m.hexdigest()[:12]


def _mkdir_p(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _append_csv_row(path: str, header: List[str], row: List[Any]) -> None:
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if not exists:
            f.write(",".join(header) + "\n")
        # basic CSV escaping for commas in strings
        def esc(x):
            if isinstance(x, str):
                if ("," in x) or ("\n" in x) or ('"' in x):
                    return '"' + x.replace('"', '""') + '"'
                return x
            return str(x)
        f.write(",".join(esc(x) for x in row) + "\n")


def _model_meta(model: PreTrainedModel) -> Dict[str, Any]:
    meta = {
        "dtype": str(getattr(model, "dtype", None)),
        "device": str(getattr(model, "device", None)),
        "class": model.__class__.__name__,
    }
    # try to capture model id if available
    for attr in ["name_or_path", "model_name", "config"]:
        try:
            val = getattr(model, attr)
            if isinstance(val, str):
                meta["name_or_path"] = val
                break
            if hasattr(val, "name_or_path"):
                meta["name_or_path"] = val.name_or_path
                break
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            meta["cuda_device"] = torch.cuda.get_device_name()
            meta["cuda_capability"] = torch.cuda.get_device_capability()
    except Exception:
        pass
    return meta


def sweep_patch_layers(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    cache_source: Dict[str, torch.Tensor],
    source_idx: int,
    target_idx: int,
    kind: str = "mlp_out",                  # or "resid_out"
    layers: Optional[Iterable[int]] = None, # e.g., range(5) to test first 5 layers
    progress: bool = True,
    # --- new persistence knobs ---
    save_dir: Optional[str] = None,         # folder to persist outputs (CSV + JSON)
    run_name: Optional[str] = None,         # optional label; used in filenames
    extra_meta: Optional[Dict[str, Any]] = None,  # user-provided metadata
) -> List[float]:
    """
    For each layer i, patch (kind, i) on target_idx with source_idx's cached activation.
    Measure ΔP(Yes) at the final token for the target item.
    Optionally persist per-layer CSV rows + summary JSON in save_dir.
    """
    t_start = time.time()
    # Get Yes/No token IDs
    yes_id, _ = yes_no_token_ids(model.tokenizer)

    # Move inputs once
    inputs_on_device = {k: v.to(model.device) for k, v in batch.items()}

    # Baseline (no patch) probability of "Yes" for target (final token)
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

    # Prepare persistence
    csv_path = None
    summary_path = None
    meta_written = False
    run_id = None

    if save_dir is not None:
        _mkdir_p(save_dir)

        # Try to extract prompts to hash (for identification) if available
        # We can't reconstruct rendered strings here; store input_ids hash instead
        try:
            ids_hash = hashlib.sha256(batch["input_ids"].cpu().numpy().tobytes()).hexdigest()[:12]
        except Exception:
            ids_hash = "unknown"

        # Basic run id and filenames
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{run_name or 'run'}_{kind}_s{source_idx}_t{target_idx}_{timestamp}_{ids_hash}"

        csv_path = os.path.join(save_dir, f"{run_id}.csv")
        summary_path = os.path.join(save_dir, f"{run_id}.summary.json")

        # Write a minimal meta header file early (so you can identify partial runs)
        if extra_meta is None:
            extra_meta = {}
        initial_meta = {
            "run_id": run_id,
            "kind": kind,
            "source_idx": source_idx,
            "target_idx": target_idx,
            "timestamp_utc": timestamp,
            "model": _model_meta(model),
            "batch_shapes": {k: tuple(v.shape) for k, v in batch.items()},
            "p_yes_base": p_yes_base,
            "num_layers": len(layer_list),
            "layers": layer_list,
            "extra_meta": extra_meta,
        }
        _write_json(summary_path, initial_meta)
        meta_written = True

    # Sweep
    effects: List[float] = []
    header = ["layer", "delta_yes", "elapsed_layer_s", "kind", "source_idx", "target_idx", "t_utc"]

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

        # Persist per-layer row
        if csv_path is not None:
            _append_csv_row(
                csv_path,
                header,
                [
                    layer_idx,
                    float(delta),
                    round(time.time() - t_layer, 5),
                    kind,
                    source_idx,
                    target_idx,
                    datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                ],
            )

    if progress:
        print(f"[patch] Completed {len(layer_list)} layers in {time.time()-t_start:.2f}s")

    # Finalize summary with effects + timing
    if summary_path is not None and meta_written:
        final_meta = {
            "run_id": run_id,
            "kind": kind,
            "source_idx": source_idx,
            "target_idx": target_idx,
            "p_yes_base": p_yes_base,
            "effects": effects,
            "num_layers": len(layer_list),
            "elapsed_total_s": round(time.time() - t_start, 3),
        }
        # Merge into existing summary (keep earlier fields)
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
        existing.update(final_meta)
        _write_json(summary_path, existing)

    return effects
