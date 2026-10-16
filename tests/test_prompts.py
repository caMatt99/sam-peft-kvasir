import numpy as np

from sampeft.data.masks import clean_mask, split_components
from sampeft.data.prompts import boxes_from_mask, jitter_boxes


def two_polyps_with_noise():
    m = np.zeros((100, 120), bool)
    m[10:30, 10:40] = True          # polipo 1: 20x30 = 600 px
    m[60:90, 70:110] = True         # polipo 2: 30x40 = 1200 px
    m[95, 115] = True               # rumore JPEG: 1 px
    return m


def test_clean_removes_only_noise():
    m = two_polyps_with_noise()
    c = clean_mask(m)
    assert c.sum() == m.sum() - 1
    assert not c[95, 115]


def test_one_box_per_polyp_exact():
    boxes = boxes_from_mask(two_polyps_with_noise())
    np.testing.assert_array_equal(boxes, [[10, 10, 39, 29], [70, 60, 109, 89]])


def test_boxes_match_component_order():
    m = clean_mask(two_polyps_with_noise())
    for box, comp in zip(boxes_from_mask(m), split_components(m)):
        x0, y0, x1, y1 = box.astype(int)
        assert comp[y0:y1 + 1, x0:x1 + 1].sum() == comp.sum()   # il box contiene tutta la sua componente


def test_empty_mask_gives_no_boxes():
    assert boxes_from_mask(np.zeros((10, 10), bool)).shape == (0, 4)


def test_jitter_zero_is_identity():
    b = boxes_from_mask(two_polyps_with_noise())
    np.testing.assert_array_equal(jitter_boxes(b, 0.0, np.random.default_rng(0), (100, 120)), b)


def test_jitter_is_reproducible_and_valid():
    b = boxes_from_mask(two_polyps_with_noise())
    j1 = jitter_boxes(b, 0.2, np.random.default_rng(42), (100, 120))
    j2 = jitter_boxes(b, 0.2, np.random.default_rng(42), (100, 120))
    np.testing.assert_array_equal(j1, j2)
    assert (j1[:, 0] <= j1[:, 2]).all() and (j1[:, 1] <= j1[:, 3]).all()
    assert (j1 >= 0).all() and (j1[:, [0, 2]] <= 119).all() and (j1[:, [1, 3]] <= 99).all()
    assert not np.array_equal(j1, b)
