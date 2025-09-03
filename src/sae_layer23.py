# sae_layer23_stream.py
# Streamed (memory-light) SAE training + "+" analysis at Layer 23 MLP input.
# No activation tensor is saved; only SAE weights and compact JSON reports.

import argparse, os, math, random, json
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# -------------- Utilities --------------

def set_seed(seed: int = 123):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def try_get_module(model, path: str):
    obj = model
    for p in path.split('.'):
        if not hasattr(obj, p): return None
        obj = getattr(obj, p)
    return obj

def get_block_list(model):
    # GPT-2
    h = try_get_module(model, "transformer.h")
    if h is not None: return h, len(h), "gpt2"
    # LLaMA/Qwen
    h = try_get_module(model, "model.layers")
    if h is not None: return h, len(h), "llama"
    # NeoX
    h = try_get_module(model, "gpt_neox.layers")
    if h is not None: return h, len(h), "neox"
    # Mistral-like
    h = try_get_module(model, "model.decoder.layers")
    if h is not None: return h, len(h), "mistral"
    raise ValueError("Unsupported architecture: cannot locate blocks")

def get_mlp_module(block, kind: str):
    if hasattr(block, "mlp"): return block.mlp
    if hasattr(block, "feed_forward"): return block.feed_forward
    # fallback scan
    for name, m in block.named_modules():
        if any(k in name.lower() for k in ["mlp", "feed_forward", "ff"]):
            return m
    raise ValueError("Could not find MLP submodule")

@torch.no_grad()
def tokenize(tokenizer, texts: List[str], max_len: int):
    out = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_len, add_special_tokens=False
    )
    return out["input_ids"], out["attention_mask"]

def gen_addition_batch(bs: int, two_digit: bool, include_equals: bool) -> List[str]:
    xs = []
    for _ in range(bs):
        a = random.randint(10, 99) if two_digit else random.randint(1, 99)
        b = random.randint(10, 99) if two_digit else random.randint(1, 99)
        s = f"{a}+{b}"
        if include_equals: s += "="
        xs.append(s)
    return xs

# -------------- SAE --------------

class SAE(nn.Module):
    # z = ReLU(x W_e + b_e); x_hat = z W_d + b_d
    def __init__(self, d_in: int, d_code: int):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_code)
        self.decoder = nn.Linear(d_code, d_in)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())
    def forward(self, x):
        z = torch.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

# -------------- Hooked passes --------------

def make_mlp_input_hook(capture: dict):
    def pre_hook(_module, args):
        capture["x"] = args[0]
        return None
    return pre_hook

def stream_batches(model, tok, layer_idx: int, total_texts: int, collect_bs: int,
                   max_len: int, two_digit: bool, include_equals: bool):
    blocks, n_layers, kind = get_block_list(model)
    if layer_idx >= n_layers: raise ValueError(f"layer_idx {layer_idx} >= {n_layers}")
    mlp_mod = get_mlp_module(blocks[layer_idx], kind)

    capture = {}
    h = mlp_mod.register_forward_pre_hook(make_mlp_input_hook(capture))
    model.eval()

    done = 0
    with torch.no_grad():
        pbar = tqdm(total=total_texts, desc="Streaming")
        while done < total_texts:
            bs = min(collect_bs, total_texts - done)
            texts = gen_addition_batch(bs, two_digit, include_equals)
            input_ids, attn_mask = tokenize(tok, texts, max_len)
            input_ids = input_ids.to(dev()); attn_mask = attn_mask.to(dev())
            _ = model(input_ids=input_ids, attention_mask=attn_mask)

            x = capture["x"]
            valid = attn_mask.bool()
            x_flat = x[valid].detach()
            tok_flat = input_ids[valid].detach()
            yield x_flat, tok_flat

            done += bs; pbar.update(bs)
        pbar.close()
    h.remove()

# -------------- Training (online) --------------

def train_sae_stream(model, tok, layer_idx: int, d_code: int,
                     total_texts: int, collect_bs: int, max_len: int,
                     two_digit: bool, include_equals: bool,
                     opt_lr: float, l1_weight: float, epochs: int,
                     microbatch: int):
    first_stream = stream_batches(model, tok, layer_idx, total_texts=collect_bs,
                                  collect_bs=collect_bs, max_len=max_len,
                                  two_digit=two_digit, include_equals=include_equals)
    x0, _ = next(first_stream)
    d_in = x0.shape[1]

    sae = SAE(d_in, d_code).to(dev())
    opt = torch.optim.AdamW(sae.parameters(), lr=opt_lr)

    def step_on_chunk(xb):
        xb = xb.to(dev()).to(dtype=sae.encoder.weight.dtype)
        for p in sae.parameters():
            p.requires_grad_(True)
        with torch.enable_grad():
            xhat, z = sae(xb)
            mse = ((xhat - xb) ** 2).mean()
            l1 = z.abs().mean()
            loss = mse + l1_weight * l1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return float(loss.item()), float(mse.item()), float(l1.item())

    print(f"[SAE] d_in={d_in}, d_code={d_code}")
    for ep in range(1, epochs+1):
        print(f"\nEpoch {ep}/{epochs}")
        seen = 0; running = 0.0
        s = stream_batches(model, tok, layer_idx, total_texts=total_texts,
                           collect_bs=collect_bs, max_len=max_len,
                           two_digit=two_digit, include_equals=include_equals)
        for x_flat, _ in s:
            for i in range(0, x_flat.size(0), microbatch):
                xb = x_flat[i:i+microbatch]
                loss, mse, l1 = step_on_chunk(xb)
                running += loss * xb.size(0)
                seen += xb.size(0)
        print(f"[ep {ep}] mean_loss={running/max(seen,1):.6f} over {seen} positions")
    return sae

# -------------- Analysis pass --------------

@torch.no_grad()
def analyze_plus_stream(model, tok, sae: SAE, layer_idx: int,
                        total_texts: int, collect_bs: int, max_len: int,
                        two_digit: bool, include_equals: bool, batch_positions: int = 65536,
                        topk_features: int = 25) -> Dict:
    sae = sae.to(dev()).eval()

    plus_id = tok.convert_tokens_to_ids("+")
    if plus_id is None or plus_id == tok.unk_token_id:
        plus_id = tok.encode("+", add_special_tokens=False)[0]

    d_code = sae.encoder.out_features
    sum_all = torch.zeros(d_code)
    sum_plus = torch.zeros(d_code)
    count_all = 0
    count_plus = 0

    s = stream_batches(model, tok, layer_idx, total_texts, collect_bs, max_len, two_digit, include_equals)
    for x_flat, tok_flat in s:
        for i in range(0, x_flat.size(0), batch_positions):
            xb = x_flat[i:i+batch_positions].to(dev())
            _, z = sae(xb)
            z = z.detach().cpu()
            tb = tok_flat[i:i+batch_positions].cpu()
            sum_all += z.sum(0)
            count_all += z.size(0)
            mask = (tb == plus_id)
            if mask.any():
                sum_plus += z[mask].sum(0)
                count_plus += int(mask.sum().item())

    mean_all = (sum_all / max(1, count_all))
    mean_plus = (sum_plus / max(1, count_plus)) if count_plus > 0 else torch.zeros_like(mean_all)
    lift = mean_plus - mean_all
    top_vals, top_idx = torch.topk(lift, k=min(topk_features, d_code))
    report = {
        "positions_seen": int(count_all),
        "plus_positions": int(count_plus),
        "top_feature_indices": top_idx.tolist(),
        "lift_values": top_vals.tolist(),
        "mean_plus_activation": mean_plus[top_idx].tolist(),
        "mean_all_activation": mean_all[top_idx].tolist(),
    }
    return report

# -------------- decode_features (top 20 tokens) --------------

@torch.no_grad()
def decode_features(model, tok, sae: SAE, feature_indices: List[int], topk_tokens: int = 20) -> Dict:
    device = next(sae.parameters()).device
    E = model.get_input_embeddings().weight.detach().to(device)   # [V, d_in]
    E = nn.functional.normalize(E, dim=1)
    W = sae.decoder.weight.detach().to(device)                    # [d_in, d_code]

    out: Dict[int, List[Dict]] = {}
    for f in feature_indices:
        f = int(f)
        feat_dir = W[:, f]
        feat_dir = nn.functional.normalize(feat_dir, dim=0)
        sims = E @ feat_dir                                       # [V]
        vals, idxs = torch.topk(sims, k=topk_tokens)
        toks = tok.convert_ids_to_tokens(idxs.tolist())
        out[f] = [{"token": t, "sim": float(v)} for t, v in zip(toks, vals.tolist())]
    return out

# -------------- Main --------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--total_texts", type=int, default=40000)
    ap.add_argument("--collect_bs", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--two_digit", action="store_true")
    ap.add_argument("--include_equals", action="store_true")
    # SAE hyperparams
    ap.add_argument("--d_code", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l1", type=float, default=1e-3)
    ap.add_argument("--microbatch", type=int, default=8192)
    # Analysis
    ap.add_argument("--topk_features", type=int, default=25)
    ap.add_argument("--topk_tokens", type=int, default=20)   # default: top 20 tokens
    ap.add_argument("--analysis_chunk", type=int, default=65536)
    # IO
    ap.add_argument("--out_dir", default="sae_out_stream")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading tokenizer/model (no gradient)...")
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.convert_ids_to_tokens(0)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=None
    ).to(dev()).eval()

    sae = train_sae_stream(
        model, tok, layer_idx=args.layer, d_code=args.d_code,
        total_texts=args.total_texts, collect_bs=args.collect_bs, max_len=args.max_len,
        two_digit=args.two_digit, include_equals=args.include_equals,
        opt_lr=args.lr, l1_weight=args.l1, epochs=args.epochs, microbatch=args.microbatch
    )

    torch.save(sae.state_dict(), os.path.join(args.out_dir, "sae.pt"))
    with open(os.path.join(args.out_dir, "sae_meta.json"), "w") as f:
        json.dump({"d_code": args.d_code, "layer": args.layer}, f, indent=2)

    plus_report = analyze_plus_stream(
        model, tok, sae, layer_idx=args.layer,
        total_texts=args.total_texts, collect_bs=args.collect_bs, max_len=args.max_len,
        two_digit=args.two_digit, include_equals=args.include_equals,
        batch_positions=args.analysis_chunk, topk_features=args.topk_features
    )
    with open(os.path.join(args.out_dir, "layer23_plus_report.json"), "w") as f:
        json.dump(plus_report, f, indent=2)

    if plus_report.get("top_feature_indices"):
        dec = decode_features(model, tok, sae, plus_report["top_feature_indices"], args.topk_tokens)
        with open(os.path.join(args.out_dir, "layer23_plus_feature_decodes.json"), "w") as f:
            json.dump(dec, f, indent=2)

    print("\nSummary")
    print(f"- SAE saved to: {os.path.join(args.out_dir,'sae.pt')}")
    print(f"- '+' positions seen: {plus_report.get('plus_positions', 0)} / {plus_report.get('positions_seen', 0)}")
    if plus_report.get("top_feature_indices"):
        for i,(fid,lift) in enumerate(zip(plus_report["top_feature_indices"], plus_report["lift_values"])):
            print(f"  {i+1:2d}. feature {fid}  lift {lift:.4f}")
    print("Reports: layer23_plus_report.json, layer23_plus_feature_decodes.json")

if __name__ == "__main__":
    main()
