import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_model, get_attentions, logit_lens, DEFAULT_MODEL

def _aggregate_layer_attention(layer_tensor, aggregate="mean", head_idx=None):
    """
    layer_tensor: (batch=1, n_heads, seq, seq) -> return (seq, seq) numpy
    """
    A = layer_tensor[0]  # (n_heads, seq, seq)
    if aggregate == "mean":
        return A.mean(dim=0).detach().cpu().numpy()
    elif aggregate == "max":
        return A.max(dim=0).values.detach().cpu().numpy()
    elif aggregate == "head":
        if head_idx is None:
            raise ValueError("--aggregate head requires --head <idx>")
        return A[head_idx].detach().cpu().numpy()
    else:
        raise ValueError(f"Unknown aggregate: {aggregate}")

def _plot_matrix(M, tokens, title, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    im = ax.imshow(M)
    ax.set_xticks(range(len(tokens))); ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def save_per_layer_maps(attentions, tokens, layers=None, aggregate="mean", head_idx=None, all_heads=False):
    """
    Save one PNG per layer under assets/attn_layers/.
    If all_heads=True, also save each head's map for each layer.
    """
    out_dir = "assets/attn_layers"
    os.makedirs(out_dir, exist_ok=True)
    num_layers = len(attentions)
    if layers is None:
        layers = list(range(num_layers))

    # Shared color scale across layers (for fairness)
    vmin, vmax = np.inf, -np.inf
    mats = []
    for L in layers:
        if aggregate == "head":
            M = _aggregate_layer_attention(attentions[L], aggregate="head", head_idx=head_idx)
            mats.append([("head", head_idx, M)])
            vmin, vmax = min(vmin, M.min()), max(vmax, M.max())
        else:
            M = _aggregate_layer_attention(attentions[L], aggregate=aggregate)
            mats.append([(aggregate, None, M)])
            vmin, vmax = min(vmin, M.min()), max(vmax, M.max())

        if all_heads:
            # compute vmin/vmax including per-head maps
            A = attentions[L][0].detach().cpu().numpy()  # (n_heads, seq, seq)
            vmin = min(vmin, A.min())
            vmax = max(vmax, A.max())

    # Now actually plot with fixed vmin/vmax
    for idx, L in enumerate(layers):
        # primary map (aggregate or head)
        for tag, h, M in mats[idx]:
            title = f"Layer {L} ({'head '+str(h) if tag=='head' else tag})"
            out_path = os.path.join(out_dir, f"L{L:02d}_{tag if tag!='head' else f'H{h}'} .png").replace(" ", "")
            fig, ax = plt.subplots(figsize=(4.5, 4.2))
            im = ax.imshow(M, vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(tokens))); ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90)
            ax.set_yticklabels(tokens)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            plt.close(fig)

        # optional: all heads for this layer
        if all_heads:
            A = attentions[L][0].detach().cpu().numpy()  # (n_heads, seq, seq)
            for h in range(A.shape[0]):
                title = f"Layer {L} Head {h}"
                out_path = os.path.join(out_dir, f"L{L:02d}_H{h}.png")
                fig, ax = plt.subplots(figsize=(4.5, 4.2))
                im = ax.imshow(A[h], vmin=vmin, vmax=vmax)
                ax.set_xticks(range(len(tokens))); ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=90)
                ax.set_yticklabels(tokens)
                ax.set_title(title)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                fig.savefig(out_path, dpi=200)
                plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default="55+56=")
    ap.add_argument("--attn-impl", default="eager", choices=["eager","sdpa","flash_attention_2"])

    # per-layer dump options
    ap.add_argument("--dump-layers", action="store_true",
                    help="Save one attention PNG per layer to assets/attn_layers/")
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=27)
    ap.add_argument("--aggregate", default="mean", choices=["mean","max","head"],
                    help="How to combine heads for each per-layer map.")
    ap.add_argument("--head", type=int, default=0,
                    help="Used when --aggregate head.")
    ap.add_argument("--all-heads", action="store_true",
                    help="Also dump each head per layer.")

    args = ap.parse_args()

    tok, model, device = load_model(args.model, attn_impl=args.attn_impl)
    pack = get_attentions(tok, model, args.prompt)
    attentions = pack["attentions"]
    tokens = pack["tokens"]
    hidden_states = pack["hidden_states"]

    if attentions is None:
        raise RuntimeError(
            "Model did not return attentions. Re-run with --attn-impl eager "
            "or ensure your Transformers version supports returning attentions."
        )

    if args.dump_layers:
        layers = list(range(args.layer_start, args.layer_end + 1))
        save_per_layer_maps(
            attentions, tokens, layers=layers,
            aggregate=args.aggregate, head_idx=args.head, all_heads=args.all_heads
        )
        print(f"[saved] per-layer attention maps to assets/attn_layers/")

    # Optional: still emit a quick logit-lens plot and summary
    pos = len(tokens) - 1
    tops = logit_lens(tok, model, hidden_states, position=pos)
    print("Layer\tTop-5 tokens at last position")
    for row in tops:
        toks = ", ".join(row["top_tokens"])
        print(f"{row['layer']:>3}\t{toks}")

    max_logits = [max(row["logits"]) for row in tops]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(len(max_logits)), max_logits, marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Max logit (arb.)")
    ax.set_title("Logit-lens: max vocab logit per layer")
    fig.tight_layout()
    os.makedirs("assets", exist_ok=True)
    out_path = "assets/logit_lens_max.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()
