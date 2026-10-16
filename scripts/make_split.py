"""Task 0, passo 2: split train/val/test e prompt fissi per val e test. Gira su CPU.

1. trova i quasi-duplicati (dHash) e li raggruppa: ogni gruppo finisce in un solo split
2. split stratificato per "un polipo" / "più polipi"
3. per val e test salva i box esatti e quelli con jitter: tutti i metodi useranno questi

Output in <out>/ (da versionare con git):
  split.json          id di train/val/test + seed e conteggi
  duplicates.json     gruppi di quasi-duplicati trovati (da controllare a occhio)
  closest_pairs.json  le 12 coppie più simili in assoluto, per scegliere la soglia
  prompts_val.json    box per val
  prompts_test.json   box per test

Uso:
  python scripts/make_split.py --raw-root /kaggle/working/data/Kvasir-SEG \
      --processed /kaggle/working/processed --out data/splits
"""
import argparse
import json
import zlib
from pathlib import Path

import cv2
import numpy as np

from sampeft.data.prompts import jitter_boxes
from sampeft.data.split import (closest_pairs, dhash, duplicate_groups, hamming_matrix,
                                stratified_group_split)


def make_prompts(ids, meta, ratio, seed) -> dict:
    prompts = {}
    for img_id in ids:
        boxes = np.asarray(meta[img_id]["boxes"], dtype=np.float32).reshape(-1, 4)
        hw = (meta[img_id]["height"], meta[img_id]["width"])
        rng = np.random.default_rng([seed, zlib.crc32(img_id.encode())])  # stabile per id
        prompts[img_id] = {
            "boxes": boxes.tolist(),
            "boxes_jitter": jitter_boxes(boxes, ratio, rng, hw).tolist(),
        }
    return {"jitter_ratio": ratio, "seed": seed, "prompts": prompts}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--processed", type=Path, required=True, help="output di prepare_data.py")
    p.add_argument("--out", type=Path, default=Path("data/splits"))
    p.add_argument("--n-val", type=int, default=100)
    p.add_argument("--n-test", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dup-threshold", type=int, default=52,
                   help="bit diversi su 256 per dire 'quasi uguali' (scegli guardando closest_pairs.json)")
    p.add_argument("--jitter", type=float, default=0.1)
    args = p.parse_args()

    meta = json.loads((args.processed / "meta.json").read_text())
    ids = sorted(meta)

    hashes = np.stack([
        dhash(cv2.imread(str(args.raw_root / "images" / f"{i}.jpg"), cv2.IMREAD_GRAYSCALE)) for i in ids
    ])
    dist = hamming_matrix(hashes)
    groups = duplicate_groups(dist, args.dup_threshold)
    dup = [g for g in groups if len(g) > 1]

    strata = np.array([int(meta[i]["n_polyps"] > 1) for i in ids])
    split_idx = stratified_group_split(groups, strata, args.n_val, args.n_test, args.seed)
    split = {k: [ids[i] for i in v] for k, v in split_idx.items()}

    assert sum(len(v) for v in split.values()) == len(ids)
    assert not (set(split["train"]) & set(split["val"]) or set(split["train"]) & set(split["test"])
                or set(split["val"]) & set(split["test"]))

    counts = {k: {"images": len(v), "multi_polyp": int(sum(meta[i]["n_polyps"] > 1 for i in v)),
                  "big_box_clean": int(sum(meta[i]["big_box_clean"] for i in v))}
              for k, v in split.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "split.json").write_text(json.dumps({
        "seed": args.seed, "dup_threshold": args.dup_threshold,
        "n_duplicate_groups": len(dup), "counts": counts, **split,
    }, indent=1))
    (args.out / "duplicates.json").write_text(json.dumps([
        {"ids": [ids[i] for i in g], "max_distance": int(dist[np.ix_(g, g)].max())} for g in dup
    ], indent=1))
    (args.out / "closest_pairs.json").write_text(json.dumps([
        {"ids": [ids[i], ids[j]], "distance": d} for i, j, d in closest_pairs(dist, 12)
    ], indent=1))
    for name in ["val", "test"]:
        (args.out / f"prompts_{name}.json").write_text(
            json.dumps(make_prompts(split[name], meta, args.jitter, args.seed), indent=1))

    print(f"gruppi di quasi-duplicati: {len(dup)} ({sum(len(g) for g in dup)} immagini)")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
