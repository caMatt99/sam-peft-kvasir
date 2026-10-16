"""Preprocessing identico a quello di SAM (SamPredictor + Sam.preprocess).

Usiamo ResizeLongestSide di segment_anything invece di riscriverlo, e replichiamo Sam.preprocess
(normalizzazione + padding) con le stesse costanti. tests/test_transforms.py verifica l'equivalenza.
"""
import numpy as np
import torch
import torch.nn.functional as F
from segment_anything.utils.transforms import ResizeLongestSide

IMG_SIZE = 1024
PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)


class SamTransform:
    def __init__(self, img_size: int = IMG_SIZE):
        self.img_size = img_size
        self.resize = ResizeLongestSide(img_size)

    def image(self, img: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        """RGB uint8 (H, W, 3) → tensore normalizzato con padding (3, 1024, 1024) e input_size (H', W').

        input_size è la dimensione dopo il resize e prima del padding: serve per togliere il padding
        dalle predizioni prima di riportarle alla risoluzione originale.
        """
        resized = self.resize.apply_image(img)
        x = torch.as_tensor(resized).permute(2, 0, 1).float()
        input_size = tuple(x.shape[-2:])
        x = (x - PIXEL_MEAN) / PIXEL_STD
        x = F.pad(x, (0, self.img_size - input_size[1], 0, self.img_size - input_size[0]))
        return x, input_size

    def boxes(self, boxes: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
        """Box (K, 4) dalle coordinate originali a quelle dell'immagine ridimensionata."""
        return self.resize.apply_boxes(boxes, original_size).astype(np.float32)
