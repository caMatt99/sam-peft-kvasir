import numpy as np
import pytest

from sampeft.metrics import dice, iou


def square(r0, r1, c0, c1, shape=(10, 10)):
    m = np.zeros(shape, bool)
    m[r0:r1, c0:c1] = True
    return m


def test_perfect_overlap():
    a = square(2, 5, 2, 5)
    assert dice(a, a) == pytest.approx(1.0)
    assert iou(a, a) == pytest.approx(1.0)


def test_disjoint():
    assert dice(square(0, 3, 0, 3), square(5, 8, 5, 8)) == 0.0
    assert iou(square(0, 3, 0, 3), square(5, 8, 5, 8)) == 0.0


def test_half_overlap():
    a, b = square(0, 4, 0, 4), square(0, 4, 2, 6)     # 16 px ciascuna, 8 in comune
    assert dice(a, b) == pytest.approx(2 * 8 / 32)
    assert iou(a, b) == pytest.approx(8 / 24)


def test_both_empty_is_perfect():
    e = np.zeros((5, 5), bool)
    assert dice(e, e) == 1.0 and iou(e, e) == 1.0


def test_dice_ge_iou():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.random((16, 16)) > 0.5, rng.random((16, 16)) > 0.5
        assert dice(a, b) >= iou(a, b)
