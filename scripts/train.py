"""Fine-tuning di SAM.

  --method decoder   (Task 2) solo il mask decoder; embedding letti dalla cache (--cache)
  --method lora      (Task 5) LoRA su q e v dell'encoder (--rank) + mask decoder; l'encoder gira a ogni passo

A ogni epoca: training sul train set (box con jitter nuovo), poi validation con i box fissi.
Si salva il modello dell'epoca con il Dice di validation più alto (box esatti).
L'epoca 0 è la validation PRIMA del training: deve ritrovare lo zero-shot della Task 1 (~0.906).

Protocollo comune a tutti i metodi (stesse epoche, stessa precisione, stessa selezione):
lo fissano gli argomenti --epochs e --amp, uguali per ogni run del confronto.

Output in <out>/:
  trainable.pt    parametri addestrati della migliore epoca (per scripts/evaluate.py --weights)
  history.json    loss e Dice di validation per epoca
  summary.json    migliore epoca, parametri (encoder / decoder / totale), tempi, memoria, configurazione

Uso:
  python scripts/train.py ... --method decoder --cache cache/vit_b --epochs 10 --amp --out runs/decoder
  python scripts/train.py ... --method lora --rank 8 --epochs 10 --amp --out runs/lora_r8
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
from sampeft.model import METHODS, apply_method, encode, load_sam, parameter_report, save_finetuned
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
    p.add_argument("--cache", type=Path, default=None, help="decoder: cartella con train.pt e val.pt")
    p.add_argument("--rank", type=int, default=8, help="lora: rango r")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0, help="ordine delle immagini e jitter dei box")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4, help="learning rate del decoder")
    p.add_argument("--lora-lr", type=float, default=5e-4, help="learning rate dei parametri LoRA")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--jitter", type=float, default=0.1)
    p.add_argument("--amp", action="store_true", help="training e validation in float16 (GPU)")
    p.add_argument("--checkpointing", action="store_true", help="gradient checkpointing nell'encoder (meno memoria)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="solo le prime N immagini per epoca (debug)")
    args = p.parse_args()

    set_seed(args.seed)
    cached = args.method == "decoder"
    if cached and args.cache is None:
        p.error("--method decoder richiede --cache")
    split = json.loads((args.splits / "split.json").read_text())
    cache = {**load_cache(args.cache / "train.pt"), **load_cache(args.cache / "val.pt")} if cached else None
    train_ds = KvasirDataset(args.raw_root, args.processed, split["train"], train=True,
                             jitter_ratio=args.jitter, seed=args.seed, with_image=not cached)
    val_prompts_file = args.splits / "prompts_val.json"
    val_prompts = json.loads(val_prompts_file.read_text())["prompts"]
    val_ds = KvasirDataset(args.raw_root, args.processed, split["val"], train=False,
                           prompts_file=val_prompts_file, with_image=not cached)

    sam = load_sam(args.checkpoint, args.model_type, args.device)
    config = {"r": args.rank} if args.method == "lora" else {}
    apply_method(sam, args.method, **config)
    if args.checkpointing:
        from sampeft.lora import enable_checkpointing
        enable_checkpointing(sam)
    groups = [{"params": [p for p in sam.mask_decoder.parameters() if p.requires_grad], "lr": args.lr}]
    enc_params = [p for p in sam.image_encoder.parameters() if p.requires_grad]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lora_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device.startswith("cuda"))

    if cached:
        embed = lambda x: cache[x["id"]][None].to(args.device).float()
    else:
        embed = lambda x: encode(sam, x["image"][None].to(args.device))

    def validate() -> dict:
        recs = evaluate(sam, val_ds, val_prompts, args.device, num_workers=args.num_workers,
                        limit=args.limit, embeddings=cache, amp=args.amp)
        g = summarize(recs)["groups"]["all"]
        return {k: g[f"{k}.dice_raw"]["mean"] for k in ("boxes", "boxes_jitter")} | \
               {"fail_rate": g["boxes.fail_rate"]}

    args.out.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    v = validate()
    history = [{"epoch": 0, "train_loss": None, "train_s": 0.0, "val_s": time.perf_counter() - t_start,
                "val_dice": v["boxes"], "val_dice_jitter": v["boxes_jitter"], "val_fail_rate": v["fail_rate"]}]
    print(f"epoca  0  (zero-shot)          val Dice {v['boxes']:.3f}  jitter {v['boxes_jitter']:.3f}")
    best = {"epoch": 0, "val_dice": v["boxes"]}
    save_finetuned(sam, args.method, args.out / "trainable.pt", config)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        loss = train_one_epoch(sam, train_ds, embed, optimizer, args.device, epoch=epoch, seed=args.seed,
                               batch_size=args.batch_size, num_workers=args.num_workers, limit=args.limit,
                               amp=args.amp, scaler=scaler)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        train_s = time.perf_counter() - t0
        scheduler.step()
        t0 = time.perf_counter()
        v = validate()
        val_s = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_loss": loss, "train_s": train_s, "val_s": val_s,
                        "val_dice": v["boxes"], "val_dice_jitter": v["boxes_jitter"], "val_fail_rate": v["fail_rate"]})
        mark = ""
        if v["boxes"] > best["val_dice"]:
            best = {"epoch": epoch, "val_dice": v["boxes"]}
            save_finetuned(sam, args.method, args.out / "trainable.pt", config)
            mark = " *"
        print(f"epoca {epoch:2d}  loss {loss:.4f}  train {train_s:6.1f} s  val {val_s:5.1f} s  "
              f"val Dice {v['boxes']:.3f}  jitter {v['boxes_jitter']:.3f}{mark}", flush=True)

    summary = {
        "method": args.method, "method_config": config,
        "best_epoch": best["epoch"], "best_val_dice": best["val_dice"],
        "parameters": parameter_report(sam),
        "train_s_per_epoch_median": float(np.median([h["train_s"] for h in history[1:]])),
        "train_s_total": float(sum(h["train_s"] for h in history)),
        "wall_s_total": time.perf_counter() - t_start,
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
