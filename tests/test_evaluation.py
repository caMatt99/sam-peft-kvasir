import json
import subprocess
import sys
from pathlib import Path

import pytest

from sampeft.evaluation import summarize

ROOT = Path(__file__).resolve().parents[1]


def _rec(i, d, multi=False, big=False, large=False):
    r = {"id": str(i), "n_polyps": 2 if multi else 1, "multi_polyp": multi, "big_box": big,
         "large_polyp": large, "encoder_ms": 400.0, "decoder_ms": 5.0}
    for k in ("boxes", "boxes_jitter"):
        for m in ("dice_raw", "iou_raw", "dice_clean", "iou_clean"):
            r[f"{k}.{m}"] = d
    return r


def test_summarize_groups_and_stats():
    recs = [_rec(0, 0.9), _rec(1, 0.7, multi=True), _rec(2, 0.8, big=True)]
    s = summarize(recs)
    g = s["groups"]
    assert g["all"]["n"] == 3 and g["multi_polyp"]["n"] == 1 and g["big_box"]["n"] == 1
    assert g["all"]["boxes.dice_raw"]["mean"] == pytest.approx(0.8)
    assert g["all"]["boxes.dice_raw"]["sem"] == pytest.approx(0.1 / 3 ** 0.5)
    assert s["timing_median_ms"]["encoder_ms"] == 400.0
    assert "large_polyp" not in g                       # gruppi vuoti omessi


def test_summarize_fail_rate_and_large_polyp():
    recs = [_rec(0, 0.95), _rec(1, 0.01, large=True), _rec(2, 0.9, large=True), _rec(3, 0.4)]
    g = summarize(recs)["groups"]
    assert g["all"]["boxes.fail_rate"] == pytest.approx(0.5)
    assert g["large_polyp"]["n"] == 2 and g["large_polyp"]["boxes.fail_rate"] == pytest.approx(0.5)


def test_evaluate_script_end_to_end(synthetic_kvasir, tmp_path):
    """Smoke test: lo script gira e scrive i file attesi (pesi casuali, 2 immagini, CPU)."""
    d = synthetic_kvasir
    subprocess.run([sys.executable, "scripts/evaluate.py", "--raw-root", d["raw"],
                    "--processed", d["processed"], "--splits", d["splits"], "--split", "test",
                    "--out", tmp_path, "--device", "cpu", "--limit", "2", "--num-workers", "0"],
                   check=True, cwd=ROOT, capture_output=True)
    m = json.loads((tmp_path / "test" / "metrics.json").read_text())
    assert m["groups"]["all"]["n"] == 2
    assert 0.0 <= m["groups"]["all"]["boxes.dice_raw"]["mean"] <= 1.0
    assert (tmp_path / "test" / "per_image.csv").read_text().count("\n") == 3   # header + 2 righe
    assert "boxes.fail_rate" in m["groups"]["all"]
