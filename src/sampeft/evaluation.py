"""Valutazione per immagine e riassunto per gruppi.

Per ogni immagine:
  - l'encoder gira UNA volta;
  - il decoder gira una volta per ogni insieme di box (esatti e con jitter);
  - la maschera predetta è l'unione delle maschere dei singoli box;
  - Dice e IoU si calcolano alla risoluzione originale contro due GT:
      raw   = JPEG binarizzato (GT ufficiale, confrontabile con la letteratura)
      clean = senza i pixel di rumore (per verificare che la pulizia non cambi nulla)

Oltre alla media si riporta il tasso di fallimento (Dice < 0.5): nella Task 1 pochi
fallimenti totali (Dice ~0) spiegano quasi tutta la distanza dalla media dei casi buoni.
"""
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.kvasir import KvasirDataset, collate_list
from .metrics import dice, iou
from .model import decode, encode

PROMPT_KEYS = ("boxes", "boxes_jitter")
METRICS = ("dice_raw", "iou_raw", "dice_clean", "iou_clean")
LARGE_POLYP = 0.25   # area del polipo / area immagine; fissata dopo la Task 1, prima di ogni training
FAIL_DICE = 0.5      # sotto questa soglia la predizione è da considerare fallita


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(sam, dataset: KvasirDataset, prompts: dict, device: str, *,
             amp: bool = False, limit: int | None = None, num_workers: int = 2,
             multimask: bool = False) -> list[dict]:
    """Una riga per immagine: metriche per ogni insieme di box, tempi di encoder e decoder."""
    sam.eval()
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_list, num_workers=num_workers)
    autocast = (torch.autocast("cuda", dtype=torch.float16) if amp and device.startswith("cuda")
                else nullcontext())
    records = []
    for i, (x,) in enumerate(loader):
        if limit is not None and i >= limit:
            break
        meta = dataset.meta[x["id"]]
        gt_raw = x["gt"].numpy()
        gt_clean = x["masks"].any(0).numpy()
        area_frac = meta["raw_pixels"] / (meta["height"] * meta["width"])
        rec = {"id": x["id"], "n_polyps": meta["n_polyps"], "area_frac": round(area_frac, 4),
               "multi_polyp": meta["n_polyps"] > 1, "big_box": meta["big_box_clean"],
               "large_polyp": area_frac > LARGE_POLYP}

        with autocast:
            _sync(device); t0 = time.perf_counter()
            emb = encode(sam, x["image"][None].to(device))
            _sync(device); rec["encoder_ms"] = (time.perf_counter() - t0) * 1000

            for key in PROMPT_KEYS:
                boxes = np.asarray(prompts[x["id"]][key], dtype=np.float32).reshape(-1, 4)
                boxes = torch.from_numpy(dataset.transform.boxes(boxes, x["original_size"])).to(device)
                _sync(device); t0 = time.perf_counter()
                logits, _ = decode(sam, emb[0], boxes, x["input_size"], x["original_size"],
                                   multimask=multimask)
                _sync(device)
                if key == "boxes":
                    rec["decoder_ms"] = (time.perf_counter() - t0) * 1000
                pred = (logits > sam.mask_threshold).any(0).cpu().numpy()
                rec[f"{key}.dice_raw"] = dice(pred, gt_raw)
                rec[f"{key}.iou_raw"] = iou(pred, gt_raw)
                rec[f"{key}.dice_clean"] = dice(pred, gt_clean)
                rec[f"{key}.iou_clean"] = iou(pred, gt_clean)
        records.append(rec)
    return records


def _stats(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    n = len(v)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if n > 1 else 0.0,
            "sem": float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0, "n": n}


def summarize(records: list[dict]) -> dict:
    """Media, deviazione standard ed errore standard della media, per gruppo di immagini.

    sem = std / sqrt(n): misura l'incertezza sulla media. Due metodi la cui differenza
    è dello stesso ordine del sem non sono distinguibili.
    """
    groups = {
        "all": records,
        "single_polyp": [r for r in records if not r["multi_polyp"]],
        "multi_polyp": [r for r in records if r["multi_polyp"]],
        "big_box": [r for r in records if r["big_box"]],
        "large_polyp": [r for r in records if r["large_polyp"]],
    }
    out = {}
    for name, rs in groups.items():
        if not rs:
            continue
        out[name] = {"n": len(rs)}
        for key in PROMPT_KEYS:
            for m in METRICS:
                out[name][f"{key}.{m}"] = _stats([r[f"{key}.{m}"] for r in rs])
            out[name][f"{key}.fail_rate"] = float(np.mean([r[f"{key}.dice_raw"] < FAIL_DICE for r in rs]))
    timing = {k: float(np.median([r[k] for r in records])) for k in ("encoder_ms", "decoder_ms")}
    return {"groups": out, "timing_median_ms": timing}
