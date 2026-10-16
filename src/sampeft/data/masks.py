"""Lettura di immagini e maschere Kvasir-SEG, e pulizia delle maschere.

Due "viste" della stessa maschera:
- raw:   JPEG originale binarizzato con > 127. È la GT ufficiale, usata SOLO per le metriche.
- clean: raw senza le componenti più piccole di `min_area` pixel (artefatti JPEG).
         Usata per generare i box e come target della loss.
"""
from pathlib import Path

import cv2
import numpy as np

THRESHOLD = 127
MIN_AREA = 100


def load_image(path: str | Path) -> np.ndarray:
    """Immagine RGB uint8 (H, W, 3)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_raw_mask(path: str | Path) -> np.ndarray:
    """Maschera bool (H, W): il JPEG originale binarizzato. GT ufficiale per le metriche."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m > THRESHOLD


def split_components(mask: np.ndarray, min_area: int = MIN_AREA) -> list[np.ndarray]:
    """Una maschera bool per ogni componente connessa con area >= min_area."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [labels == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def clean_mask(mask: np.ndarray, min_area: int = MIN_AREA) -> np.ndarray:
    """Maschera bool senza le componenti più piccole di min_area."""
    out = np.zeros_like(mask, dtype=bool)
    for comp in split_components(mask, min_area):
        out |= comp
    return out
