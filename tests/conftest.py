"""Dataset sintetico in stile Kvasir-SEG per testare la pipeline senza scaricare i dati veri."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _make_sample(rng, h, w, n_polyps):
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), np.uint8)
    for k in range(n_polyps):
        cx, cy = int(w * (0.25 + 0.5 * k)), int(h * 0.5)
        cv2.circle(mask, (cx, cy), min(h, w) // 8, 255, -1)
    mask[2, 2] = 255            # pixel di "rumore JPEG" in un angolo
    return img, mask


@pytest.fixture(scope="session")
def synthetic_kvasir(tmp_path_factory):
    """40 immagini (8 con due polipi), elaborate con prepare_data.py e make_split.py."""
    base = tmp_path_factory.mktemp("kvasir")
    raw = base / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "masks").mkdir()
    rng = np.random.default_rng(0)
    for i in range(40):
        h, w = (300, 400) if i % 2 else (500, 380)
        img, mask = _make_sample(rng, h, w, 2 if i < 8 else 1)
        cv2.imwrite(str(raw / "images" / f"img{i:03d}.jpg"), img)
        cv2.imwrite(str(raw / "masks" / f"img{i:03d}.jpg"), mask)

    processed, splits = base / "processed", base / "splits"
    run = lambda *a: subprocess.run([sys.executable, *a], check=True, cwd=ROOT, capture_output=True)
    run("scripts/prepare_data.py", "--raw-root", raw, "--out", processed)
    run("scripts/make_split.py", "--raw-root", raw, "--processed", processed, "--out", splits,
        "--n-val", "5", "--n-test", "5")
    return {"raw": raw, "processed": processed, "splits": splits}
