"""Cache degli embedding dell'image encoder (Task 2).

Con l'encoder congelato l'embedding di un'immagine non cambia mai: si calcola una volta,
si salva in float16 (256×64×64×2 byte = 2 MB per immagine) e il training usa solo il decoder.
Un file per split: {"ids": [...], "embeddings": (N, 256, 64, 64) float16, ...}.
"""
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.kvasir import KvasirDataset, collate_list
from .model import encode


@torch.no_grad()
def compute_embeddings(sam, dataset: KvasirDataset, device: str, num_workers: int = 2,
                       limit: int | None = None) -> dict:
    """Encoder in float32 (come nella valutazione), poi conversione a float16 per il salvataggio."""
    sam.eval()
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_list, num_workers=num_workers)
    ids, embs = [], []
    t0 = time.perf_counter()
    for i, (x,) in enumerate(loader):
        if limit is not None and i >= limit:
            break
        ids.append(x["id"])
        embs.append(encode(sam, x["image"][None].to(device)).half().cpu())
    return {"ids": ids, "embeddings": torch.cat(embs), "seconds": time.perf_counter() - t0}


def save_cache(cache: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def load_cache(path: str | Path) -> dict[str, torch.Tensor]:
    """id → embedding (256, 64, 64) float16, in RAM. Va convertito in float32 prima del decoder."""
    c = torch.load(path, map_location="cpu")
    return dict(zip(c["ids"], c["embeddings"]))
