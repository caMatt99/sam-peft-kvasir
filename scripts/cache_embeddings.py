"""Calcola e salva gli embedding dell'image encoder originale (Task 2).

Output: <out>/<split>.pt per ogni split richiesto. Uno split già presente viene saltato.

Uso:
  python scripts/cache_embeddings.py --raw-root <Kvasir-SEG> --processed <processed> \
      --checkpoint checkpoints/sam_vit_b_01ec64.pth --out cache/vit_b
"""
import argparse
import json
from pathlib import Path

import torch

from sampeft.cache import compute_embeddings, save_cache
from sampeft.data.kvasir import KvasirDataset
from sampeft.model import load_sam


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--processed", type=Path, required=True)
    p.add_argument("--splits", type=Path, default=Path("data/splits"))
    p.add_argument("--which", nargs="+", default=["train", "val", "test"])
    p.add_argument("--checkpoint", type=Path, default=None, help="None = pesi casuali (solo per i test)")
    p.add_argument("--model-type", default="vit_b")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="solo le prime N immagini (debug)")
    args = p.parse_args()

    split = json.loads((args.splits / "split.json").read_text())
    sam = load_sam(args.checkpoint, args.model_type, args.device)
    for name in args.which:
        path = args.out / f"{name}.pt"
        if path.exists():
            print(f"{name}: già in cache ({path})")
            continue
        # serve solo l'immagine: train=True evita di dover passare un file di prompt (i box non si usano)
        ds = KvasirDataset(args.raw_root, args.processed, split[name], train=True)
        cache = compute_embeddings(sam, ds, args.device, args.num_workers, args.limit)
        cache["checkpoint"] = str(args.checkpoint)
        save_cache(cache, path)
        n = len(cache["ids"])
        print(f"{name}: {n} immagini in {cache['seconds']:.0f} s "
              f"({cache['seconds'] / n * 1000:.0f} ms/immagine) → {path}")


if __name__ == "__main__":
    main()
