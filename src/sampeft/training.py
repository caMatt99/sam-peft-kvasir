"""Loss e ciclo di training, comuni a tutti i metodi.

Il ciclo riceve una funzione embed(x) → (1, 256, 64, 64):
- decoder (Task 2): legge l'embedding dalla cache;
- lora / prompt tuning (Task 3–5): passa l'immagine nell'encoder, con i gradienti.

Ogni box produce una maschera, confrontata con la maschera PULITA del suo polipo.
La loss è calcolata alla risoluzione originale, la stessa delle metriche.

Gradient accumulation: forward e backward si fanno UNA immagine alla volta e i gradienti
si sommano; i pesi si aggiornano una volta per batch. Matematicamente è lo stesso batch,
ma in memoria c'è il grafo di una sola immagine (indispensabile con l'encoder nel training).
"""
from contextlib import nullcontext
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


def autocast_ctx(amp: bool, device: str):
    """float16 sulla GPU se amp=True; altrimenti nessun effetto."""
    return torch.autocast("cuda", dtype=torch.float16) if amp and device.startswith("cuda") else nullcontext()


def train_one_epoch(sam, dataset: KvasirDataset, embed: Callable[[dict], torch.Tensor],
                    optimizer: torch.optim.Optimizer, device: str, *, epoch: int, seed: int,
                    batch_size: int = 8, num_workers: int = 2, limit: int | None = None,
                    amp: bool = False, scaler: torch.amp.GradScaler | None = None) -> float:
    """Un'epoca; restituisce la loss media per batch.

    amp=True: forward in float16 (loss in float32); lo scaler evita che i gradienti piccoli
    diventino zero in float16.
    """
    sam.train()
    scaler = scaler or torch.amp.GradScaler("cuda", enabled=False)
    dataset.set_epoch(epoch)                                   # jitter nuovo e riproducibile
    gen = torch.Generator().manual_seed(seed * 10_000 + epoch)  # ordine riproducibile
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=gen,
                        collate_fn=collate_list, num_workers=num_workers)
    losses = []
    for step, batch in enumerate(loader):
        if limit is not None and step * batch_size >= limit:
            break
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for x in batch:   # una immagine alla volta: dimensioni e numero di polipi diversi
            with autocast_ctx(amp, device):
                logits, _ = decode(sam, embed(x), x["boxes"].to(device), x["input_size"], x["original_size"])
            loss = seg_loss(logits.float(), x["masks"].to(device)) / len(batch)
            scaler.scale(loss).backward()
            batch_loss += loss.item()
        scaler.step(optimizer)
        scaler.update()
        losses.append(batch_loss)
    return float(np.mean(losses))
