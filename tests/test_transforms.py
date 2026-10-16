"""Il nostro preprocessing deve coincidere con quello di SamPredictor + Sam.preprocess."""
import numpy as np
import pytest
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from sampeft.data.transforms import SamTransform


@pytest.fixture(scope="module")
def sam():
    return sam_model_registry["vit_b"](checkpoint=None).eval()   # pesi casuali: preprocess non li usa


@pytest.mark.parametrize("hw", [(529, 622), (1072, 1920), (332, 487)])
def test_image_matches_sam(sam, hw):
    img = np.random.default_rng(0).integers(0, 255, (*hw, 3), dtype=np.uint8)
    ours, input_size = SamTransform().image(img)

    resized = ResizeLongestSide(1024).apply_image(img)                     # come SamPredictor.set_image
    ref = sam.preprocess(torch.as_tensor(resized).permute(2, 0, 1)[None].float())[0]

    assert ours.shape == (3, 1024, 1024)
    assert input_size == resized.shape[:2]
    torch.testing.assert_close(ours, ref)


def test_boxes_scale_and_roundtrip():
    t = SamTransform()
    box = np.array([[10, 20, 300, 400]], np.float32)
    out = t.boxes(box, (500, 1000))                     # lato lungo 1000 → 1024
    np.testing.assert_allclose(out, box * 1.024, rtol=1e-6)
