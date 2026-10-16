"""Metriche di segmentazione, calcolate per immagine alla risoluzione originale.

La media sul dataset si fa DOPO, sulle metriche per immagine (non sommando i pixel di tutte le immagini).
"""
import numpy as np

EPS = 1e-8


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:          # entrambe vuote: predizione perfetta
        return 1.0
    return float(2 * np.logical_and(pred, gt).sum() / (denom + EPS))


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / (union + EPS))
