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


METHODS = ("decoder", "lora", "prefix_shallow", "prefix_deep")


def apply_method(sam: Sam, method: str, **config) -> Sam:
    """Prepara SAM per un metodo di fine-tuning: decide quali parametri si addestrano.

    decoder (Task 2): solo il mask decoder; image encoder e prompt encoder congelati.
    lora (Task 5): LoRA su q e v dell'encoder (config: r, alpha) + mask decoder.
    prefix_shallow / prefix_deep (Task 3 / 4): prefix addestrabili nell'attenzione del blocco 0 /
    di tutti i blocchi (config: length) + mask decoder.
    In tutti i metodi il decoder si addestra e parte dai pesi originali di SAM.
    """
    if method not in METHODS:
        raise ValueError(f"metodo sconosciuto: {method}")
    sam.requires_grad_(False)
    sam.mask_decoder.requires_grad_(True)
    if method == "lora":
        from .lora import add_lora
        add_lora(sam, config["r"], config.get("alpha"))
    elif method.startswith("prefix"):
        from .prefix import add_prefix
        add_prefix(sam, config["length"], deep=method == "prefix_deep")
    return sam


def parameter_report(sam: Sam) -> dict[str, int]:
    """Parametri addestrati separati: encoder (quelli che distinguono i metodi PEFT), decoder, totale."""
    enc = count_parameters(sam.image_encoder)["trainable"]
    dec = count_parameters(sam.mask_decoder)["trainable"]
    tot = count_parameters(sam)
    return {"encoder": enc, "decoder": dec, "trainable": tot["trainable"], "total": tot["total"]}


def save_finetuned(sam: Sam, method: str, path: str | Path, config: dict | None = None) -> None:
    """Salva solo i parametri addestrati (decoder ~16 MB, LoRA qualche MB in più) e la configurazione."""
    state = {n: p.detach().cpu() for n, p in sam.named_parameters() if p.requires_grad}
    torch.save({"method": method, "config": config or {}, "state_dict": state}, path)


def load_finetuned(sam: Sam, path: str | Path, merge: bool = True) -> dict:
    """Applica il metodo salvato e carica i suoi parametri sopra il checkpoint originale.

    merge=True (valutazione): con LoRA somma B·A dentro i pesi, così l'inferenza costa come SAM.
    """
    ckpt = torch.load(path, map_location="cpu")
    config = ckpt.get("config", {})
    apply_method(sam, ckpt["method"], **config)
    result = sam.load_state_dict(ckpt["state_dict"], strict=False)
    if result.unexpected_keys:
        raise KeyError(f"parametri non presenti nel modello: {result.unexpected_keys[:5]}")
    info = {"method": ckpt["method"], "config": config, "parameters": parameter_report(sam)}
    if merge and ckpt["method"] == "lora":
        from .lora import merge_lora
        merge_lora(sam)
    return info
