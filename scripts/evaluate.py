"""Valutazione di SAM su val o test. Task 1: SAM zero-shot (pesi originali, nessun training).

Output in <out>/<split>/:
  per_image.csv   una riga per immagine (per analizzare i casi difficili)
  metrics.json    riassunto per gruppi + tempi + memoria + configurazione

Uso:
  python scripts/evaluate.py --raw-root <Kvasir-SEG> --processed <processed> \
      --checkpoint checkpoints/sam_vit_b_01ec64.pth --split test --out runs/zeroshot
"""
import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path

import torch

from sampeft.data.kvasir import KvasirDataset
from sampeft.evaluation import PROMPT_KEYS, evaluate, summarize
from sampeft.model import count_parameters, load_sam


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def print_table(summary: dict) -> None:
    print(f"{'gruppo':<14}{'n':>5}  " + "  ".join(f"{k + ' dice':>20}" for k in PROMPT_KEYS))
    for name, g in summary["groups"].items():
        cells = [f"{g[f'{k}.dice_raw']['mean']:.3f} ± {g[f'{k}.dice_raw']['sem']:.3f}" for k in PROMPT_KEYS]
        print(f"{name:<14}{g['n']:>5}  " + "  ".join(f"{c:>20}" for c in cells))
    t = summary["timing_median_ms"]
    print(f"\nencoder {t['encoder_ms']:.0f} ms · decoder {t['decoder_ms']:.1f} ms (mediana per immagine)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--processed", type=Path, required=True)
    p.add_argument("--splits", type=Path, default=Path("data/splits"))
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--checkpoint", type=Path, default=None, help="None = pesi casuali (solo per i test)")
    p.add_argument("--model-type", default="vit_b")
    p.add_argument("--out", type=Path, default=Path("runs/zeroshot"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", help="float16 sulla GPU")
    p.add_argument("--limit", type=int, default=None, help="valuta solo le prime N immagini (debug)")
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    split = json.loads((args.splits / "split.json").read_text())
    prompts_file = args.splits / f"prompts_{args.split}.json"
    prompts = json.loads(prompts_file.read_text())["prompts"]
    dataset = KvasirDataset(args.raw_root, args.processed, split[args.split], train=False,
                            prompts_file=prompts_file)

    sam = load_sam(args.checkpoint, args.model_type, args.device)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    records = evaluate(sam, dataset, prompts, args.device, amp=args.amp, limit=args.limit,
                       num_workers=args.num_workers)
    summary = summarize(records)
    summary["peak_gpu_memory_mb"] = (torch.cuda.max_memory_allocated() / 2**20
                                     if args.device.startswith("cuda") else None)
    summary["parameters"] = {"image_encoder": count_parameters(sam.image_encoder),
                             "mask_decoder": count_parameters(sam.mask_decoder),
                             "prompt_encoder": count_parameters(sam.prompt_encoder)}
    summary["config"] = {k: str(v) for k, v in vars(args).items()} | {
        "git_commit": git_commit(), "torch": torch.__version__, "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "jitter_ratio": json.loads(prompts_file.read_text())["jitter_ratio"],
    }

    out = args.out / args.split
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print_table(summary)
    print(f"\nrisultati in {out}")


if __name__ == "__main__":
    main()
