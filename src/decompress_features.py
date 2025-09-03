# src/decompress_features.py
# Decompress top SAE features: decode to tokens and (optionally) show top-activating examples.

import os, json, math, heapq, argparse
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- import your SAE + streaming helpers ----
import sys
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)
from sae_layer23 import SAE, decode_features, stream_batches, set_seed, dev

def load_top_features(report_path: str, k: int) -> List[int]:
    with open(report_path, "r") as f:
        rep = json.load(f)
    feats = rep["top_feature_indices"][:k]
    return [int(x) for x in feats]

@torch.no_grad()
def decode_tokens_for_features(model, tok, sae, feature_ids: List[int], topk_tokens: int = 20) -> Dict[int, List[Dict]]:
    return decode_features(model, tok, sae, feature_ids, topk_tokens=topk_tokens)

def build_sae(out_dir: str) -> Tuple[SAE, int]:
    # load meta + state
    with open(os.path.join(out_dir, "sae_meta.json"), "r") as f:
        meta = json.load(f)
    d_code = int(meta["d_code"])
    state = torch.load(os.path.join(out_dir, "sae.pt"), map_location="cpu")
    d_in = state["encoder.weight"].shape[1]
    sae = SAE(d_in, d_code)
    sae.load_state_dict(state)
    sae.eval()
    return sae, d_code

def gen_addition_batch(bs: int, two_digit: bool, include_equals: bool) -> List[str]:
    import random
    xs = []
    for _ in range(bs):
        a = random.randint(10, 99) if two_digit else random.randint(1, 99)
        b = random.randint(10, 99) if two_digit else random.randint(1, 99)
        s = f"{a}+{b}"
        if include_equals: s += "="
        xs.append(s)
    return xs

@torch.no_grad()
def top_activations_examples(model, tok, sae: SAE, layer: int, feats: List[int],
                             n_prompts: int = 8000, batch_prompts: int = 32,
                             max_len: int = 32, two_digit: bool = True,
                             include_equals: bool = True, keep_per_feat: int = 5) -> Dict[int, List[Dict]]:
    """
    Stream a modest number of synthetic prompts and keep the top-k positions per feature by activation.
    Returns: {feat_id: [ {act, text, pos, token}, ... ]} sorted by descending act.
    """
    sae = sae.to(dev()).eval()

    # min-heaps (store negative activation to use heapq as max-heap)
    heaps: Dict[int, List[Tuple[float, Tuple[str, int, int]]]] = {f: [] for f in feats}

    # We reuse stream_batches from sae_layer23; it yields (x_flat, tok_flat) per forward
    # Wrap the text generator to hit exactly n_prompts with batch_prompts each call
    seen = 0
    while seen < n_prompts:
        bs = min(batch_prompts, n_prompts - seen)
        # make the texts
        texts = gen_addition_batch(bs, two_digit, include_equals)

        # Tokenize and run model so the hook fires (we’ll inline the needed part here)
        # Quick local tokenize (same options as in sae_layer23)
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len, add_special_tokens=False)
        input_ids = enc["input_ids"].to(dev())
        attention_mask = enc["attention_mask"].to(dev())

        # We need the MLP input at given layer. Rather than duplicating hooks, we’ll
        # reroute through a tiny local capture that mirrors stream_batches.
        capture = {}
        blocks, n_layers, kind = None, None, None
        # lightweight accessors:
        def try_get_module(model, path: str):
            obj = model
            for p in path.split('.'):
                if not hasattr(obj, p): return None
                obj = getattr(obj, p)
            return obj
        def get_block_list(model):
            h = try_get_module(model, "transformer.h")
            if h is not None: return h, len(h), "gpt2"
            h = try_get_module(model, "model.layers")
            if h is not None: return h, len(h), "llama"
            h = try_get_module(model, "gpt_neox.layers")
            if h is not None: return h, len(h), "neox"
            h = try_get_module(model, "model.decoder.layers")
            if h is not None: return h, len(h), "mistral"
            raise ValueError("Unsupported architecture")
        def get_mlp_module(block, kind: str):
            if hasattr(block, "mlp"): return block.mlp
            if hasattr(block, "feed_forward"): return block.feed_forward
            for name, m in block.named_modules():
                if any(k in name.lower() for k in ["mlp", "feed_forward", "ff"]):
                    return m
            raise ValueError("No MLP submodule")
        blocks, n_layers, kind = get_block_list(model)
        mlp_mod = get_mlp_module(blocks[layer], kind)
        def pre_hook(_m, args):
            capture["x"] = args[0]
            return None
        h = mlp_mod.register_forward_pre_hook(pre_hook)

        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        h.remove()

        x = capture["x"]                  # [B,T,d]
        valid = attention_mask.bool()
        x_flat = x[valid]                 # [Npos,d]
        toks = input_ids[valid]           # [Npos]

        # Encode with SAE in chunks
        B = 65536
        start = 0
        while start < x_flat.size(0):
            xb = x_flat[start:start+B].to(dev())
            _, z = sae(xb)                # [n,d_code]
            z = z.detach().cpu()
            tb = toks[start:start+B].cpu()
            # For each feature, update heap with each position’s activation
            for fi in feats:
                col = z[:, fi]            # [n]
                for j in range(col.size(0)):
                    act = float(col[j])
                    # keep only strong activations
                    if len(heaps[fi]) < keep_per_feat or -heaps[fi][0][0] < act:
                        # identify text and token position
                        # We do not track per-token absolute position back into text here;
                        # instead, record the actual token string for context.
                        token_str = tok.convert_ids_to_tokens([int(tb[j])])[0]
                        heapq.heappush(heaps[fi], (-act, (token_str, start + j, fi)))
                        if len(heaps[fi]) > keep_per_feat:
                            heapq.heappop(heaps[fi])
            start += B

        seen += bs

    # Convert heaps to sorted lists
    out = {}
    for fi in feats:
        items = [heapq.heappop(heaps[fi]) for _ in range(len(heaps[fi]))]
        items.sort()  # since we stored negative activations, this sorts ascending
        items = items[::-1]  # descending by activation
        out[fi] = [{"activation": -score, "token": token, "position_index": pos}
                   for score, (token, pos, _feat) in items]
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="sae_out_stream",
                   help="Directory with sae.pt, sae_meta.json, layer23_plus_report.json")
    p.add_argument("--report", default="layer23_plus_report.json")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--topn_features", type=int, default=10)
    p.add_argument("--topk_tokens", type=int, default=20)
    p.add_argument("--show_examples", action="store_true",
                   help="Also stream a small corpus and show top-activating tokens per feature")
    p.add_argument("--n_prompts", type=int, default=8000)
    p.add_argument("--batch_prompts", type=int, default=32)
    p.add_argument("--keep_per_feat", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    set_seed(args.seed)

    out_dir = args.out_dir
    report_path = os.path.join(out_dir, args.report)

    # 1) top-N features from report
    feats = load_top_features(report_path, args.topn_features)
    print(f"Top {len(feats)} features from report: {feats}")

    # 2) load SAE
    sae, d_code = build_sae(out_dir)

    # 3) load model + tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32, device_map=None).to(dev()).eval()

    # 4) token decodes
    dec = decode_tokens_for_features(model, tok, sae, feats, topk_tokens=args.topk_tokens)

    print("\n=== Token decodes per feature (top aligned tokens) ===")
    for f in feats:
        toksims = dec.get(int(f), [])[:args.topk_tokens]
        summary = ", ".join([f"{e['token']}:{e['sim']:.3f}" for e in toksims[:10]])
        print(f"feature {f:4d}: {summary}")

    # 5) optional: show top-activating examples per feature
    if args.show_examples:
        print("\nStreaming a small corpus to collect top-activating tokens per feature...")
        ex = top_activations_examples(
            model, tok, sae, layer=23, feats=feats,
            n_prompts=args.n_prompts, batch_prompts=args.batch_prompts,
            keep_per_feat=args.keep_per_feat
        )
        save_path = os.path.join(out_dir, "top_activation_examples.json")
        with open(save_path, "w") as f:
            json.dump(ex, f, indent=2)
        print(f"\nSaved top-activation examples to {save_path}")
        for f in feats:
            rows = ex[f]
            pretty = "; ".join([f"{r['token']} (act={r['activation']:.3f})" for r in rows])
            print(f"feature {f:4d}: {pretty}")

if __name__ == "__main__":
    main()
