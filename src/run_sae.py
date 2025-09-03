# src/inspect_sae.py
# Load and inspect a trained SAE checkpoint (sae.pt)

import torch, json, os
from sae_layer23 import SAE  # import your SAE class

def main():
    out_dir = "sae_out_stream"   # adjust if your sae.pt lives elsewhere
    state_path = os.path.join(out_dir, "sae.pt")
    meta_path  = os.path.join(out_dir, "sae_meta.json")

    # Load meta (contains d_code and layer info)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    d_code = int(meta["d_code"])
    layer  = int(meta["layer"])
    print(f"Metadata: layer={layer}, d_code={d_code}")

    # Load state dict
    state = torch.load(state_path, map_location="cpu")

    # Infer d_in from encoder weight
    d_in = state["encoder.weight"].shape[1]

    # Recreate SAE
    sae = SAE(d_in, d_code)
    sae.load_state_dict(state)

    print("\nLoaded SAE model:")
    print(sae)

    # Print some shapes
    print("\nWeight shapes:")
    print(" encoder.weight:", tuple(sae.encoder.weight.shape))
    print(" decoder.weight:", tuple(sae.decoder.weight.shape))

if __name__ == "__main__":
    main()
