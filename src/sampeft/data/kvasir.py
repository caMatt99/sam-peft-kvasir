"""Dataset Kvasir-SEG nel formato richiesto da SAM.

Ogni elemento è un dizionario (il "contratto" usato da tutto il resto del codice):

    id             str
    image          FloatTensor (3, 1024, 1024)  normalizzata e con padding
    input_size     (H', W')                     dimensione dopo il resize, prima del padding
    original_size  (H, W)
    boxes          FloatTensor (K, 4)           un box per polipo, coordinate nel frame 1024
    boxes_orig     FloatTensor (K, 4)           gli stessi box in coordinate originali
    masks          BoolTensor (K, H, W)         maschera PULITA di ciascun polipo (target della loss)
    gt             BoolTensor (H, W)            maschera RAW binarizzata (GT ufficiale per le metriche)

Box:
- train: box puliti dal meta.json + jitter nuovo a ogni epoca (seed = [seed, epoch, indice]).
- val/test: box letti da un file di prompt precalcolato, identico per tutti i metodi.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .masks import load_image, load_raw_mask, split_components
from .prompts import jitter_boxes
from .transforms import SamTransform


class KvasirDataset(Dataset):
    def __init__(
        self,
        raw_root: str | Path,
        processed_root: str | Path,
        ids: list[str],
        *,
        train: bool,
        jitter_ratio: float = 0.1,
        prompts_file: str | Path | None = None,
        prompt_key: str = "boxes",
        seed: int = 0,
        transform: SamTransform | None = None,
    ):
        self.raw_root = Path(raw_root)
        self.processed_root = Path(processed_root)
        self.ids = list(ids)
        self.train = train
        self.jitter_ratio = jitter_ratio
        self.seed = seed
        self.epoch = 0
        self.transform = transform or SamTransform()
        self.meta = json.loads((self.processed_root / "meta.json").read_text())

        if not train:
            if prompts_file is None:
                raise ValueError("val/test richiedono un prompts_file precalcolato")
            prompts = json.loads(Path(prompts_file).read_text())["prompts"]
            self.boxes = {i: np.asarray(prompts[i][prompt_key], dtype=np.float32).reshape(-1, 4)
                          for i in self.ids}

    def set_epoch(self, epoch: int) -> None:
        """Da chiamare a inizio epoca: cambia il jitter del train in modo riproducibile."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        img_id = self.ids[idx]
        img = load_image(self.raw_root / "images" / f"{img_id}.jpg")
        gt = load_raw_mask(self.raw_root / "masks" / f"{img_id}.jpg")
        clean = cv2.imread(str(self.processed_root / "masks_clean" / f"{img_id}.png"),
                           cv2.IMREAD_GRAYSCALE) > 0
        original_size = img.shape[:2]

        if self.train:
            boxes = np.asarray(self.meta[img_id]["boxes"], dtype=np.float32).reshape(-1, 4)
            rng = np.random.default_rng([self.seed, self.epoch, idx])
            boxes = jitter_boxes(boxes, self.jitter_ratio, rng, original_size)
        else:
            boxes = self.boxes[img_id]

        # componenti già filtrate in prepare_data: min_area=1 le prende tutte, nello stesso ordine dei box
        comps = split_components(clean, min_area=1)
        masks = np.stack(comps) if comps else np.zeros((0, *original_size), dtype=bool)

        image, input_size = self.transform.image(img)
        return {
            "id": img_id,
            "image": image,
            "input_size": input_size,
            "original_size": original_size,
            "boxes": torch.from_numpy(self.transform.boxes(boxes, original_size)),
            "boxes_orig": torch.from_numpy(boxes),
            "masks": torch.from_numpy(masks),
            "gt": torch.from_numpy(gt),
        }


def collate_list(batch: list[dict]) -> list[dict]:
    """Le immagini hanno un numero diverso di polipi e dimensioni diverse: il batch resta una lista."""
    return batch
