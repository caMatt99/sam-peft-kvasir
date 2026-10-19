"""Caricamento di SAM e forward "a mano", compatibile con il training.

Sam.forward e SamPredictor sono decorati con @torch.no_grad(): vanno bene per la
predizione ma non per il fine-tuning. Qui il forward è diviso in due pezzi:

    encode(sam, images)            immagini (B, 3, 1024, 1024) → embedding (B, 256, 64, 64)
    decode(sam, emb, boxes, ...)   un embedding + K box → K maschere di logit alla risoluzione originale

La divisione serve anche alla Task 2: con l'encoder congelato, encode si calcola una volta
e si salva su disco, e il training usa solo decode.
"""
from pathlib import Path

import torch
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam


def load_sam(checkpoint: str | Path | None, model_type: str = "vit_b", device: str = "cpu") -> Sam:
    """checkpoint=None costruisce SAM con pesi casuali: utile solo nei test."""
    sam = sam_model_registry[model_type](checkpoint=None if checkpoint is None else str(checkpoint))
    return sam.to(device)


def encode(sam: Sam, images: torch.Tensor) -> torch.Tensor:
    """(B, 3, 1024, 1024) già normalizzate e con padding → (B, 256, 64, 64)."""
    return sam.image_encoder(images)


def decode(
    sam: Sam,
    embedding: torch.Tensor,
    boxes: torch.Tensor,
    input_size: tuple[int, int],
    original_size: tuple[int, int],
    multimask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Un'immagine, K box (coordinate nel frame 1024) → logit (K, H, W) e IoU stimato (K,).

    multimask=False (default, usato nel training): una maschera per box.
    multimask=True (solo valutazione zero-shot): SAM propone 3 maschere per box e si tiene
    quella con l'IoU stimato più alto. Corregge i casi in cui l'uscita singola sceglie
    l'interpretazione sbagliata di un box molto grande (Task 1).
    Per la maschera finale dell'immagine si fa l'unione: (logits > 0).any(0).
    """
    if embedding.dim() == 3:
        embedding = embedding[None]
    sparse, dense = sam.prompt_encoder(points=None, boxes=boxes, masks=None)
    low_res, iou_pred = sam.mask_decoder(
        image_embeddings=embedding,                      # (1, 256, 64, 64): ripetuto per i K box
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=multimask,
    )
    best = iou_pred.argmax(1)                            # (K,): sempre 0 se multimask=False
    k = torch.arange(len(best), device=best.device)
    low_res, iou_pred = low_res[k, best][:, None], iou_pred[k, best]
    logits = sam.postprocess_masks(low_res, input_size, original_size)   # (K, 1, H, W)
    return logits[:, 0], iou_pred


def count_parameters(module: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
