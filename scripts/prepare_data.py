"""Task 0, passo 1: pulizia delle maschere e metadati. Gira su CPU.

Per ogni immagine:
  1. binarizza la maschera JPEG (> 127)            → GT raw (non viene salvata: si rilegge dal JPEG)
  2. rimuove le componenti < min_area pixel        → maschera clean, salvata in PNG (senza perdita)
  3. calcola un box per polipo dalla maschera clean → meta.json

Output in <out>/:
  masks_clean/<id>.png   maschere pulite (0/255)
  meta.json              per immagine: dimensioni, box, aree, pixel rimossi
  stats.json             statistiche del dataset (materiale per il README)

Uso:
  python scripts/prepare_data.py --raw-root /kaggle/working/data/Kvasir-SEG --out /kaggle/working/processed
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sampeft.data.masks import MIN_AREA, clean_mask, load_raw_mask, split_components
from sampeft.data.prompts import box_from_component

BIG_BOX = 0.9  # un box che copre più del 90% dell'immagine non indica dove guardare


def box_area_frac(box, hw) -> float:
    x0, y0, x1, y1 = box
    return float((x1 - x0 + 1) * (y1 - y0 + 1) / (hw[0] * hw[1]))


def process_one(mask_path: Path, out_dir: Path, min_area: int) -> dict:
    raw = load_raw_mask(mask_path)
    clean = clean_mask(raw, min_area)
    cv2.imwrite(str(out_dir / f"{mask_path.stem}.png"), clean.astype(np.uint8) * 255)

    hw = raw.shape
    comps = split_components(clean, min_area)
    boxes = [box_from_component(c).tolist() for c in comps]
    n_raw_comp = cv2.connectedComponents(raw.astype(np.uint8), connectivity=8)[0] - 1
    raw_union = box_from_component(raw) if raw.any() else None
    return {
        "height": hw[0],
        "width": hw[1],
        "n_polyps": len(comps),
        "boxes": boxes,
        "areas": [int(c.sum()) for c in comps],
        "raw_pixels": int(raw.sum()),
        "removed_pixels": int(raw.sum() - clean.sum()),
        "removed_components": int(n_raw_comp - len(comps)),
        "big_box_raw_union": raw_union is not None and box_area_frac(raw_union, hw) > BIG_BOX,
        "big_box_clean": any(box_area_frac(b, hw) > BIG_BOX for b in boxes),
    }


def summarize(meta: dict) -> dict:
    m = list(meta.values())
    sizes = np.array([[v["height"], v["width"]] for v in m])
    area_frac = np.array([sum(v["areas"]) / (v["height"] * v["width"]) for v in m])
    removed = np.array([v["removed_pixels"] for v in m])
    n_polyps = np.bincount([v["n_polyps"] for v in m])
    return {
        "n_images": len(m),
        "image_height": {"min": int(sizes[:, 0].min()), "median": int(np.median(sizes[:, 0])),
                         "max": int(sizes[:, 0].max())},
        "image_width": {"min": int(sizes[:, 1].min()), "median": int(np.median(sizes[:, 1])),
                        "max": int(sizes[:, 1].max())},
        "polyps_per_image": {str(k): int(c) for k, c in enumerate(n_polyps) if c},
        "polyp_area_fraction_quantiles": {q: round(float(np.quantile(area_frac, float(q))), 4)
                                          for q in ["0.05", "0.25", "0.5", "0.75", "0.95"]},
        "images_with_removed_pixels": int((removed > 0).sum()),
        "removed_pixels_total": int(removed.sum()),
        "removed_pixels_max_per_image": int(removed.max()),
        "removed_fraction_of_raw_positive": round(float(removed.sum() / sum(v["raw_pixels"] for v in m)), 6),
        "big_box_raw_union": int(sum(v["big_box_raw_union"] for v in m)),
        "big_box_clean": int(sum(v["big_box_clean"] for v in m)),
        "empty_masks": int(sum(v["n_polyps"] == 0 for v in m)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True, help="cartella con images/ e masks/")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-area", type=int, default=MIN_AREA)
    args = p.parse_args()

    out_masks = args.out / "masks_clean"
    out_masks.mkdir(parents=True, exist_ok=True)
    mask_paths = sorted((args.raw_root / "masks").glob("*.jpg"))
    if not mask_paths:
        raise SystemExit(f"nessuna maschera in {args.raw_root / 'masks'}")

    meta = {mp.stem: process_one(mp, out_masks, args.min_area) for mp in mask_paths}
    stats = {"min_area": args.min_area, **summarize(meta)}

    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
