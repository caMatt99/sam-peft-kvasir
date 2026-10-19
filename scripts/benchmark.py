"""Benchmark del training con l'encoder (Task 5): tempo per immagine e memoria, prima dei run veri.

Per ogni modalità (float32, AMP) addestra LoRA su poche immagini e misura:
  - secondi per immagine nel training (forward + backward dell'encoder);
  - secondi per immagine nella validation (solo forward);
  - picco di memoria GPU.
Se una modalità esaurisce la memoria, la riprova con il gradient checkpointing.
Poi stima la durata dei run completi per scegliere epoche, precisione e rank.

Uso:
  python scripts/benchmark.py --raw-root <Kvasir-SEG> --processed <processed> \
      --checkpoint checkpoints/sam_vit_b_01ec64.pth --out runs/benchmark.json
"""
import argparse
import gc
import json
import time
from pathlib import Path

import torch

from sampeft.data.kvasir import KvasirDataset
from sampeft.evaluation import evaluate
from sampeft.model import apply_method, encode, load_sam
from sampeft.training import train_one_epoch

N_TRAIN, N_VAL = 800, 100


def run_mode(args, split, amp: bool, checkpointing: bool) -> dict:
    sam = load_sam(args.checkpoint, device=args.device)
    apply_method(sam, "lora", r=args.rank)
    if checkpointing:
        from sampeft.lora import enable_checkpointing
        enable_checkpointing(sam)
    opt = torch.optim.AdamW([p for p in sam.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and args.device.startswith("cuda"))
    ids = split["train"][: args.warmup + args.n_images]
    warm = KvasirDataset(args.raw_root, args.processed, ids[: args.warmup], train=True)
    bench = KvasirDataset(args.raw_root, args.processed, ids[args.warmup:], train=True)
    embed = lambda x: encode(sam, x["image"][None].to(args.device))
    kw = dict(batch_size=8, num_workers=args.num_workers, amp=amp, scaler=scaler, seed=0, epoch=1)
    cuda = args.device.startswith("cuda")

    if args.warmup:
        train_one_epoch(sam, warm, embed, opt, args.device, **kw)
    if cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    train_one_epoch(sam, bench, embed, opt, args.device, **kw)
    if cuda:
        torch.cuda.synchronize()
    train_s = (time.perf_counter() - t0) / len(bench)
    peak = torch.cuda.max_memory_allocated() / 2**20 if cuda else None

    val_ids = split["val"][: args.n_val]
    val = KvasirDataset(args.raw_root, args.processed, val_ids, train=False,
                        prompts_file=args.splits / "prompts_val.json")
    prompts = json.loads((args.splits / "prompts_val.json").read_text())["prompts"]
    t0 = time.perf_counter()
    evaluate(sam, val, prompts, args.device, amp=amp, num_workers=args.num_workers)
    val_s = (time.perf_counter() - t0) / len(val)
    del sam, opt
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return {"train_s_per_image": train_s, "val_s_per_image": val_s, "peak_gpu_memory_mb": peak}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--processed", type=Path, required=True)
    p.add_argument("--splits", type=Path, default=Path("data/splits"))
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--n-images", type=int, default=48, help="immagini misurate nel training")
    p.add_argument("--warmup", type=int, default=8, help="immagini di riscaldamento (escluse dalla misura)")
    p.add_argument("--n-val", type=int, default=20)
    p.add_argument("--modes", nargs="+", default=["fp32", "amp"], choices=["fp32", "amp"])
    p.add_argument("--epochs", nargs="+", type=int, default=[5, 10, 15], help="epoche da stimare")
    p.add_argument("--n-ranks", type=int, default=3, help="numero di run LoRA da stimare")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--checkpointing", action="store_true", help="usa sempre il gradient checkpointing")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    split = json.loads((args.splits / "split.json").read_text())
    results = {}
    for mode in args.modes:
        amp = mode == "amp"
        try:
            results[mode] = run_mode(args, split, amp, checkpointing=args.checkpointing) | \
                            {"checkpointing": args.checkpointing}
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            print(f"{mode}: memoria esaurita, riprovo con gradient checkpointing")
            results[mode] = run_mode(args, split, amp, checkpointing=True) | {"checkpointing": True}

    print(f"\nLoRA r={args.rank}, misurato su {args.n_images} immagini di training e {args.n_val} di validation\n")
    print(f"{'modalità':<10}{'train s/img':>12}{'val s/img':>11}{'memoria MB':>12}{'checkpoint':>12}{'min/epoca':>11}")
    for mode, r in results.items():
        epoch_min = (N_TRAIN * r["train_s_per_image"] + N_VAL * r["val_s_per_image"]) / 60
        r["epoch_min"] = epoch_min
        mem = f"{r['peak_gpu_memory_mb']:.0f}" if r["peak_gpu_memory_mb"] else "-"
        print(f"{mode:<10}{r['train_s_per_image']:>12.2f}{r['val_s_per_image']:>11.2f}{mem:>12}"
              f"{str(r['checkpointing']):>12}{epoch_min:>11.1f}")

    print(f"\nStima per {args.n_ranks} run LoRA (ore, solo training + validation):")
    print(f"{'modalità':<10}" + "".join(f"{f'{e} epoche':>12}" for e in args.epochs))
    for mode, r in results.items():
        print(f"{mode:<10}" + "".join(f"{args.n_ranks * e * r['epoch_min'] / 60:>12.1f}" for e in args.epochs))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"rank": args.rank, "n_images": args.n_images, "results": results}, indent=2))


if __name__ == "__main__":
    main()
