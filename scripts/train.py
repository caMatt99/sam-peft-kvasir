"""Fine-tuning di SAM. Task 2: --method decoder (solo il mask decoder, embedding dalla cache).

A ogni epoca: training sul train set (box con jitter nuovo), poi validation con i box fissi.
Si salva il modello dell'epoca con il Dice di validation più alto (box esatti).
L'epoca 0 è la validation PRIMA del training: deve ritrovare lo zero-shot della Task 1.

Output in <out>/:
  trainable.pt    parametri addestrati della migliore epoca (per scripts/evaluate.py --weights)
  history.json    loss e Dice di validation per epoca
  summary.json    migliore epoca, parametri, tempi, memoria, configurazione

Uso:
  python scripts/train.py --raw-root <Kvasir-SEG> --processed <processed> \
      --checkpoint checkpoints/sam_vit_b_01ec64.pth --cache cache/vit_b --out runs/decoder
"""
import argparse
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from sampeft.cache import load_cache
from sampeft.data.kvasir import KvasirDataset
from sampeft.evaluation import evaluate, summarize
from sampeft.model import METHODS, apply_method, count_parameters, load_sam, save_finetuned
from sampeft.training import train_one_epoch


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--processed", type=Path, required=True)
    p.add_argument("--splits", type=Path, default=Path("data/splits"))
    p.add_argument("--checkpoint", type=Path, default=None, help="None = pesi casuali (solo per i test)")
    p.add_argument("--model-type", default="vit_b")
    p.add_argument("--method", choices=METHODS, default="decoder")
    p.add_argument("--cache", type=Path, required=True, help="cartella con train.pt e val.pt")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0, help="ordine delle immagini e jitter dei box")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--jitter", type=float, default=0.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="solo le prime N immagini per epoca (debug)")
    args = p.parse_args()

    set_seed(args.seed)
    split = json.loads((args.splits / "split.json").read_text())
    cache = {**load_cache(args.cache / "train.pt"), **load_cache(args.cache / "val.pt")}
    train_ds = KvasirDataset(args.raw_root, args.processed, split["train"], train=True,
                             jitter_ratio=args.jitter, seed=args.seed, with_image=False)
    val_prompts_file = args.splits / "prompts_val.json"
    val_prompts = json.loads(val_prompts_file.read_text())["prompts"]
    val_ds = KvasirDataset(args.raw_root, args.processed, split["val"], train=False,
                           prompts_file=val_prompts_file, with_image=False)

    sam = load_sam(args.checkpoint, args.model_type, args.device)
    apply_method(sam, args.method)
    params = [p for p in sam.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    embed = lambda x: cache[x["id"]][None].to(args.device).float()

    def validate() -> dict:
        recs = evaluate(sam, val_ds, val_prompts, args.device, num_workers=args.num_workers,
                        limit=args.limit, embeddings=cache)
        g = summarize(recs)["groups"]["all"]
        return {k: g[f"{k}.dice_raw"]["mean"] for k in ("boxes", "boxes_jitter")} | \
               {"fail_rate": g["boxes.fail_rate"]}

    args.out.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    v = validate()
    history = [{"epoch": 0, "train_loss": None, "train_s": 0.0, "lr": args.lr,
                "val_dice": v["boxes"], "val_dice_jitter": v["boxes_jitter"], "val_fail_rate": v["fail_rate"]}]
    print(f"epoca  0  (zero-shot)          val Dice {v['boxes']:.3f}  jitter {v['boxes_jitter']:.3f}")
    best = {"epoch": 0, "val_dice": v["boxes"]}
    save_finetuned(sam, args.method, args.out / "trainable.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        loss = train_one_epoch(sam, train_ds, embed, optimizer, args.device, epoch=epoch, seed=args.seed,
                               batch_size=args.batch_size, num_workers=args.num_workers, limit=args.limit)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        train_s = time.perf_counter() - t0
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        v = validate()
        history.append({"epoch": epoch, "train_loss": loss, "train_s": train_s, "lr": lr,
                        "val_dice": v["boxes"], "val_dice_jitter": v["boxes_jitter"], "val_fail_rate": v["fail_rate"]})
        mark = ""
        if v["boxes"] > best["val_dice"]:
            best = {"epoch": epoch, "val_dice": v["boxes"]}
            save_finetuned(sam, args.method, args.out / "trainable.pt")
            mark = " *"
        print(f"epoca {epoch:2d}  loss {loss:.4f}  {train_s:5.1f} s  "
              f"val Dice {v['boxes']:.3f}  jitter {v['boxes_jitter']:.3f}{mark}")

    summary = {
        "best_epoch": best["epoch"], "best_val_dice": best["val_dice"],
        "parameters": {"trainable": sum(p.numel() for p in params),
                       "total": count_parameters(sam)["total"]},
        "train_s_per_epoch_median": float(np.median([h["train_s"] for h in history[1:]])),
        "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated() / 2**20
                               if args.device.startswith("cuda") else None),
        "config": {k: str(v) for k, v in vars(args).items()} | {
            "git_commit": git_commit(), "torch": torch.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None},
    }
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nmigliore: epoca {best['epoch']}, val Dice {best['val_dice']:.3f} → {args.out / 'trainable.pt'}")


if __name__ == "__main__":
    main()
