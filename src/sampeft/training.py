"""Loss e ciclo di training, comuni a tutti i metodi.

Il ciclo riceve una funzione embed(x) → (1, 256, 64, 64):
- Task 2 (decoder): legge l'embedding dalla cache;
- Task 3–5 (PEFT): passerà l'immagine nell'encoder, con i gradienti.

Ogni box produce una maschera, confrontata con la maschera PULITA del suo polipo.
La loss è calcolata alla risoluzione originale, la stessa delle metriche.
"""
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.kvasir import KvasirDataset, collate_list
from .model import decode


def seg_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE + soft Dice, mediati sui box. logits (K, H, W), target (K, H, W) bool.

    BCE guarda ogni pixel; Dice guarda la sovrapposizione complessiva e non risente
    dello sbilanciamento tra polipo e sfondo.
    """
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = logits.sigmoid().flatten(1)
    t = target.flatten(1)
    soft_dice = (2 * (prob * t).sum(1) + 1) / (prob.sum(1) + t.sum(1) + 1)
    return bce + (1 - soft_dice).mean()


def train_one_epoch(sam, dataset: KvasirDataset, embed: Callable[[dict], torch.Tensor],
                    optimizer: torch.optim.Optimizer, device: str, *, epoch: int, seed: int,
                    batch_size: int = 8, num_workers: int = 2, limit: int | None = None) -> float:
    """Un'epoca; restituisce la loss media. Batch = batch_size immagini, una loss per immagine."""
    sam.train()
    dataset.set_epoch(epoch)                                   # jitter nuovo e riproducibile
    gen = torch.Generator().manual_seed(seed * 10_000 + epoch)  # ordine riproducibile
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=gen,
                        collate_fn=collate_list, num_workers=num_workers)
    losses = []
    for step, batch in enumerate(loader):
        if limit is not None and step * batch_size >= limit:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = 0.0
        for x in batch:   # le immagini hanno dimensioni e numero di polipi diversi
            logits, _ = decode(sam, embed(x), x["boxes"].to(device), x["input_size"], x["original_size"])
            loss = loss + seg_loss(logits, x["masks"].to(device))
        loss = loss / len(batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))
