"""Box prompt: un box per polipo, con jitter opzionale.

Formato SAM: [x0, y0, x1, y1] in pixel dell'immagine ORIGINALE (estremi inclusi).
"""
import numpy as np

from .masks import MIN_AREA, split_components


def box_from_component(comp: np.ndarray) -> np.ndarray:
    ys, xs = np.where(comp)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def boxes_from_mask(mask: np.ndarray, min_area: int = MIN_AREA) -> np.ndarray:
    """(K, 4) float32: un box per ogni componente con area >= min_area.

    Le componenti sono ordinate come le restituisce OpenCV (scansione riga per riga),
    quindi l'ordine è deterministico e coincide con split_components().
    """
    comps = split_components(mask, min_area)
    if not comps:
        return np.zeros((0, 4), dtype=np.float32)
    return np.stack([box_from_component(c) for c in comps])


def jitter_boxes(boxes: np.ndarray, ratio: float, rng: np.random.Generator,
                 hw: tuple[int, int]) -> np.ndarray:
    """Sposta ogni coordinata di U(-ratio, ratio) * lato del box, poi resta dentro l'immagine.

    Simula un box disegnato a mano: ratio=0.1 → ogni lato può spostarsi fino al 10%.
    """
    if ratio == 0 or len(boxes) == 0:
        return boxes.astype(np.float32).copy()
    h, w = hw
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    scale = np.stack([bw, bh, bw, bh], axis=1)
    out = boxes + rng.uniform(-ratio, ratio, boxes.shape) * scale
    out = np.clip(out, 0, [w - 1, h - 1, w - 1, h - 1])
    # dopo il clip un box potrebbe "rovesciarsi" (x0 > x1): riordina gli estremi
    x0, x1 = np.minimum(out[:, 0], out[:, 2]), np.maximum(out[:, 0], out[:, 2])
    y0, y1 = np.minimum(out[:, 1], out[:, 3]), np.maximum(out[:, 1], out[:, 3])
    return np.stack([x0, y0, x1, y1], axis=1).astype(np.float32)
