import argparse
import contextlib
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from src.utils import load_model, get_attentions, logit_lens, DEFAULT_MODEL

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

TARGET_LAYERS = [23, 24, 25, 26]

# ---------- debug helpers ----------
def now(): return time.strftime("%H:%M:%S")
def log(msg): print(f"[{now()}] {msg}", flush=True)
def stage(msg): print(f"\n[{now()}] === {msg} ===", flush=True)
def done(): print(f"[{now()}] STAGE COMPLETED\n", flush=True)

# ---------- arithmetic helpers ----------
def true_sum_str(prompt: str) -> str:
    s = prompt.strip()
    if not s.endswith("="):
        s += "="
    a, b = s[:-1].split("+")
    return str(int(a) + int(b))

@torch.no_grad()
def p_correct_next_token(tok, model, prompt: str) -> float:
    """Probability of the correct next token at '=' position (no generate())."""
    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=True)
    logits = out.logits[0, -1, :]
    target_id = tok.convert_tokens_to_ids(true_sum_str(prompt)[0])
    return torch.softmax(logits.float(), dim=-1)[target_id].item()

# ---------- ablation helpers ----------
@dataclass
class HeadAblationSpec:
    layer: int
    heads: List[int]

class HeadAblator:
    """Qwen/LLaMA-safe head ablator: zero the slice per head in the tensor fed to attention.o_proj."""
    def __init__(self, model, specs: List[HeadAblationSpec]):
        self.model = model
        self.specs = specs
        self.handles = []
        self._prepare()

    def _prepare(self):
        # group requested heads by layer
        by_layer: Dict[int, List[int]] = {}
        for s in self.specs:
            by_layer.setdefault(s.layer, []).extend(s.heads)

        cfg_heads = getattr(self.model.config, "num_attention_heads", None)
        if cfg_heads is None:
            raise RuntimeError("Cannot read num_attention_heads from model.config")

        for layer_idx, heads in by_layer.items():
            block = self.model.model.layers[layer_idx]
            attn = block.self_attn
            if not hasattr(attn, "o_proj"):
                raise RuntimeError("Attention module missing o_proj; cannot ablate per head.")
            o_proj = attn.o_proj
            hidden_size = o_proj.in_features
            if hidden_size % cfg_heads != 0:
                raise RuntimeError(f"hidden_size {hidden_size} not divisible by num_heads {cfg_heads}")
            head_dim = hidden_size // cfg_heads
            slices = [(h * head_dim, (h + 1) * head_dim) for h in heads]

            def make_hook(slices_):
                def pre_hook(module, inputs):
                    (x,) = inputs  # (B, T, hidden_size)
                    x = x.clone()
                    for s0, s1 in slices_:
                        x[..., s0:s1] = 0
                    return (x,)
                return pre_hook

            log(f"Installing head-ablation hook: L{layer_idx}, heads {heads} "
                f"(num_heads={cfg_heads}, head_dim={head_dim})")
            h = o_proj.register_forward_pre_hook(make_hook(slices))
            self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

class ComponentKnockout:
    """Zero either attention.o_proj input or mlp.down_proj input at a given layer."""
    def __init__(self, model, layer: int, kind: str):
        assert kind in ("attn", "mlp")
        self.model, self.layer, self.kind = model, layer, kind
        self.handle = None
        self._prepare()

    def _prepare(self):
        block = self.model.model.layers[self.layer]
        if self.kind == "attn":
            o_proj = block.self_attn.o_proj
            def hook(module, inputs): (x,) = inputs; return (torch.zeros_like(x),)
            self.handle = o_proj.register_forward_pre_hook(hook)
        else:
            if not hasattr(block.mlp, "down_proj"):
                raise RuntimeError("MLP missing down_proj for knockout.")
            down_proj = block.mlp.down_proj
            def hook(module, inputs): (x,) = inputs; return (torch.zeros_like(x),)
            self.handle = down_proj.register_forward_pre_hook(hook)

    def remove(self):
        if self.handle: self.handle.remove()

@contextlib.contextmanager
def apply_head_ablation(model, specs): ab = HeadAblator(model, specs);  yield; ab.remove()
@contextlib.contextmanager
def apply_component_knockout(model, layer, kind): ko = ComponentKnockout(model, layer, kind); yield; ko.remove()

# ---------- experiments (single prompt) ----------
def head_importance_single(tok, model, prompt: str, layers: List[int]) -> Tuple[np.ndarray, float]:
    base = p_correct_next_token(tok, model, prompt)
    n_heads = getattr(model.config, "num_attention_heads", None)
    if n_heads is None:
        raise RuntimeError("Cannot determine number of heads from model.config")

    drops = np.zeros((len(layers), n_heads), dtype=np.float32)
    iterator = range(len(layers))
    if tqdm: iterator = tqdm(iterator, desc="Head ablation (single)")

    for li in iterator:
        L = layers[li]
        heads_iter = range(n_heads)
        if tqdm: heads_iter = tqdm(heads_iter, leave=False, desc=f"L{L} heads")
        for h in heads_iter:
            log(f"Ablating L{L} H{h} ...")
            with apply_head_ablation(model, [HeadAblationSpec(layer=L, heads=[h])]):
                prob = p_correct_next_token(tok, model, prompt)
            drops[li, h] = max(0.0, base - prob)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    return drops, base

def component_importance_single(tok, model, prompt: str, layers: List[int]) -> Dict[int, Dict[str, float]]:
    base = p_correct_next_token(tok, model, prompt)
    out = {}
    for L in layers:
        log(f"Knockout at L{L}: attention")
        with apply_component_knockout(model, L, "attn"):
            pa = p_correct_next_token(tok, model, prompt)
        log(f"Knockout at L{L}: mlp")
        with apply_component_knockout(model, L, "mlp"):
            pm = p_correct_next_token(tok, model, prompt)
        out[L] = {"attn": max(0.0, base - pa), "mlp": max(0.0, base - pm)}
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out

def info_flow_from_equals(tok, model, prompt: str, layer: int):
    pack = get_attentions(tok, model, prompt)
    A = pack["attentions"][layer][0]  # (n_heads, seq, seq)
    tokens = pack["tokens"]
    meanA = A.mean(dim=0).detach().cpu().numpy()
    eq_pos = len(tokens) - 1
    return tokens, meanA[eq_pos]

# ---------- plotting helpers ----------
def save_plot(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"[saved] {path}")

# ---------- master + separate ----------
def run_all(tok, model, prompt="55+56=", focus_layers=TARGET_LAYERS, flow_layer=23,
            out_dir="assets/focus_23_26", also_master=True):

    os.makedirs(out_dir, exist_ok=True)

    # 1) Answer emergence restricted to 23–26
    stage("1) Answer emergence (L23–L26)")
    pack = get_attentions(tok, model, prompt)
    hs = pack["hidden_states"]; pos = len(pack["tokens"]) - 1
    correct_id = tok.convert_tokens_to_ids(true_sum_str(prompt)[0])
    probs = []
    for row in logit_lens(tok, model, hs, position=pos):
        h = hs[1 + row["layer"]][0, pos, :]
        logits = (h @ model.lm_head.weight.T).float()
        probs.append(torch.softmax(logits, dim=-1)[correct_id].item())
    # Slice to 23–26
    xs = focus_layers
    ys = [probs[L] for L in xs]
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, marker="o")
    ax.set_title("Answer emergence by layer (L23–L26)")
    ax.set_xlabel("Layer"); ax.set_ylabel("P(correct)")
    save_plot(fig, os.path.join(out_dir, "emergence_L23_26.png"))
    done()

    # 2) Head importance heatmap
    stage("2) Head importance heatmap (L23–L26)")
    drops, base = head_importance_single(tok, model, prompt, focus_layers)
    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    im = ax.imshow(drops, aspect="auto")
    ax.set_title("Critical heads (probability drop)")
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_yticks(range(len(focus_layers))); ax.set_yticklabels([f"L{L}" for L in focus_layers])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_plot(fig, os.path.join(out_dir, "critical_heads_L23_26.png"))
    done()

    # 3) Emergence layer bar (single prompt)
    stage("3) Emergence threshold (single prompt)")
    earliest = max([i for i,L in enumerate(range(len(probs))) if probs[L] == max(probs)])  # index of max prob
    fig = plt.figure(figsize=(3.8, 3.6))
    ax = fig.add_subplot(111)
    ax.bar(["55+56"], [earliest if earliest >= 0 else 0])
    ax.set_title("Earliest emergence layer (argmax prob)")
    save_plot(fig, os.path.join(out_dir, "emergence_single_bar.png"))
    done()

    # 4) Component importance
    stage("4) Component importance (L23–L26)")
    comp = component_importance_single(tok, model, prompt, focus_layers)
    attn_drop = [comp[L]["attn"] for L in focus_layers]
    mlp_drop  = [comp[L]["mlp"]  for L in focus_layers]
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    x = np.arange(len(focus_layers))
    ax.bar(x, attn_drop, label="Attention")
    ax.bar(x, mlp_drop, bottom=attn_drop, label="MLP")
    ax.set_xticks(x); ax.set_xticklabels([f"L{L}" for L in focus_layers])
    ax.set_title("MLP vs Attention importance"); ax.legend()
    save_plot(fig, os.path.join(out_dir, "comp_importance_L23_26.png"))
    done()

    # 5) Info flow at chosen layer
    stage(f"5) Information flow at L{flow_layer}")
    tokens, weights = info_flow_from_equals(tok, model, prompt, flow_layer)
    fig = plt.figure(figsize=(5.5, 3.8))
    ax = fig.add_subplot(111)
    xs_t = np.arange(len(tokens))
    ax.scatter(xs_t, [0]*len(tokens))
    for i, t in enumerate(tokens): ax.text(xs_t[i], 0.02, t, ha="center", fontsize=9, rotation=90)
    eq_pos = len(tokens)-1
    for j, w in enumerate(weights):
        if j != eq_pos and w > weights.max()*0.15:
            ax.annotate("", xy=(eq_pos,0.4), xytext=(j,0.05), arrowprops=dict(arrowstyle="->", lw=1+5*w))
    ax.set_ylim(-0.05,0.5); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Info flow at L{flow_layer} (from '=')")
    save_plot(fig, os.path.join(out_dir, "info_flow_L23.png"))
    done()

    # 6) Top heads list
    stage("6) Top heads by probability drop")
    head_drops = [(f"L{L}-H{h}", float(d)) for i,L in enumerate(focus_layers) for h,d in enumerate(drops[i])]
    head_drops.sort(key=lambda x: x[1], reverse=True)
    topN = head_drops[:12]
    fig = plt.figure(figsize=(6.5, 4))
    ax = fig.add_subplot(111)
    ax.bar([k for k,_ in topN], [v for _,v in topN])
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Top heads (probability drop)")
    save_plot(fig, os.path.join(out_dir, "top_heads_L23_26.png"))
    done()

    # Optional: also compose the master figure you had before
    if also_master:
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(15, 10)); gs = GridSpec(2, 3, figure=fig)

        # Re-embed the saved content quickly
        # Emergence mini
        ax = fig.add_subplot(gs[0,0]); ax.plot(xs, ys, marker="o")
        ax.set_title("Answer emergence by layer (L23–L26)"); ax.set_xlabel("Layer"); ax.set_ylabel("P(correct)")

        # Heatmap
        ax = fig.add_subplot(gs[0,1]); im = ax.imshow(drops, aspect="auto")
        ax.set_title("Critical heads heatmap"); ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_yticks(range(len(focus_layers))); ax.set_yticklabels([f"L{L}" for L in focus_layers])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Emergence bar
        ax = fig.add_subplot(gs[0,2]); ax.bar(["55+56"], [earliest if earliest>=0 else 0])
        ax.set_title("Earliest emergence layer")

        # Component importance
        ax = fig.add_subplot(gs[1,0]); x = np.arange(len(focus_layers))
        ax.bar(x, attn_drop, label="Attention"); ax.bar(x, mlp_drop, bottom=attn_drop, label="MLP")
        ax.set_xticks(x); ax.set_xticklabels([f"L{L}" for L in focus_layers])
        ax.set_title("MLP vs Attention importance"); ax.legend()

        # Info flow
        ax = fig.add_subplot(gs[1,1]); xs_t = np.arange(len(tokens))
        ax.scatter(xs_t, [0]*len(tokens))
        for i, t in enumerate(tokens): ax.text(xs_t[i], 0.02, t, ha="center", fontsize=9, rotation=90)
        eq_pos = len(tokens)-1
        for j, w in enumerate(weights):
            if j != eq_pos and w > weights.max()*0.15:
                ax.annotate("", xy=(eq_pos,0.4), xytext=(j,0.05),
                            arrowprops=dict(arrowstyle="->", lw=1+5*w))
        ax.set_ylim(-0.05,0.5); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Info flow at L{flow_layer}")

        # Top heads
        ax = fig.add_subplot(gs[1,2]); ax.bar([k for k,_ in topN], [v for _,v in topN])
        ax.tick_params(axis="x", rotation=45); ax.set_title("Top heads")

        fig.suptitle("Arithmetic Circuit Focus (55+56; L23–L26)")
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, "master_23_26.png"), dpi=200, bbox_inches="tight")
        plt.close(fig); log(f"[saved] {os.path.join(out_dir, 'master_23_26.png')}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--attn-impl", default="eager", choices=["eager","sdpa","flash_attention_2"])
    ap.add_argument("--prompt", default="55+56=")
    ap.add_argument("--out-dir", default="assets/focus_23_26")
    ap.add_argument("--no-master", action="store_true", help="Skip creating the combined master figure.")
    args = ap.parse_args()

    stage("Boot: load model")
    tok, model, device = load_model(args.model, attn_impl=args.attn_impl)
    model.eval(); done()
    log(f"Device: {device}")

    with torch.inference_mode():
        run_all(tok, model, prompt=args.prompt, out_dir=args.out_dir, also_master=not args.no_master)

    log("All done.")

if __name__ == "__main__":
    main()
