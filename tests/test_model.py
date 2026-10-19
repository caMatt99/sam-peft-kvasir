"""Il nostro forward deve dare le stesse maschere di SamPredictor, e deve permettere il training.

Si usa SAM con pesi casuali: per verificare che due pipeline facciano gli stessi calcoli
non servono pesi addestrati, basta che usino lo stesso modello.
"""
import numpy as np
import pytest
import torch
from segment_anything import SamPredictor

from sampeft.data.transforms import SamTransform
from sampeft.model import count_parameters, decode, encode, load_sam


@pytest.fixture(scope="module")
def sam():
    torch.manual_seed(0)
    return load_sam(None).eval()


def test_matches_sam_predictor(sam):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    boxes = np.array([[20, 30, 150, 200], [220, 60, 380, 280]], np.float32)

    predictor = SamPredictor(sam)
    predictor.set_image(img)
    ref_logits, _, _ = predictor.predict_torch(
        point_coords=None, point_labels=None,
        boxes=torch.from_numpy(predictor.transform.apply_boxes(boxes, img.shape[:2])),
        multimask_output=False, return_logits=True)

    t = SamTransform()
    x, input_size = t.image(img)
    with torch.no_grad():
        emb = encode(sam, x[None])
        logits, _ = decode(sam, emb[0], torch.from_numpy(t.boxes(boxes, img.shape[:2])),
                           input_size, img.shape[:2])

    assert logits.shape == (2, 300, 400)
    torch.testing.assert_close(logits, ref_logits[:, 0], rtol=1e-4, atol=1e-4)


def test_decode_lets_gradients_flow(sam):
    """Serve per il training: il forward non deve essere sotto no_grad."""
    sam.zero_grad()
    emb = torch.randn(1, 256, 64, 64)
    boxes = torch.tensor([[100.0, 100.0, 400.0, 500.0]])
    logits, iou_pred = decode(sam, emb, boxes, (768, 1024), (300, 400))
    assert logits.shape == (1, 300, 400) and iou_pred.shape == (1,)
    logits.mean().backward()
    grads = [p.grad for p in sam.mask_decoder.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_parameter_counts(sam):
    enc = count_parameters(sam.image_encoder)
    dec = count_parameters(sam.mask_decoder)
    assert 85e6 < enc["total"] < 95e6        # ViT-B: ~89.7M
    assert 3.5e6 < dec["total"] < 4.5e6      # ~4.1M
