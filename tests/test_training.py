"""Task 2: cache degli embedding, loss, metodo "decoder", salvataggio dei pesi, script di training.

SAM con pesi casuali e dataset sintetico: si verifica che i pezzi siano collegati correttamente,
non che il modello impari bene.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sampeft.cache import compute_embeddings, load_cache, save_cache
from sampeft.data.kvasir import KvasirDataset
from sampeft.evaluation import evaluate
from sampeft.model import apply_method, count_parameters, load_finetuned, load_sam, save_finetuned
from sampeft.training import seg_loss

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sam():
    torch.manual_seed(0)
    return load_sam(None).eval()


def _split(d):
    return json.loads((d["splits"] / "split.json").read_text())


def test_seg_loss():
    target = torch.zeros(2, 20, 30, dtype=torch.bool)
    target[:, 5:15, 10:20] = True
    perfect = torch.where(target, 20.0, -20.0)
    assert seg_loss(perfect, target) < 1e-3
    logits = torch.zeros(2, 20, 30, requires_grad=True)
    loss = seg_loss(logits, target)
    assert loss > 0.5
    loss.backward()
    assert logits.grad.abs().sum() > 0


def test_apply_method_decoder(sam):
    apply_method(sam, "decoder")
    assert count_parameters(sam)["trainable"] == count_parameters(sam.mask_decoder)["total"]
    assert not any(p.requires_grad for p in sam.image_encoder.parameters())
    assert not any(p.requires_grad for p in sam.prompt_encoder.parameters())
    with pytest.raises(ValueError):
        apply_method(sam, "boh")


def test_save_and_load_finetuned(sam, tmp_path):
    apply_method(sam, "decoder")
    with torch.no_grad():
        for p in sam.mask_decoder.parameters():
            p.add_(0.01)
    save_finetuned(sam, "decoder", tmp_path / "w.pt")
    sd = torch.load(tmp_path / "w.pt")["state_dict"]
    assert all(k.startswith("mask_decoder.") for k in sd)       # solo i parametri addestrati

    torch.manual_seed(1)
    other = load_sam(None)
    info = load_finetuned(other, tmp_path / "w.pt")
    assert info["method"] == "decoder" and info["config"] == {}
    assert info["parameters"]["encoder"] == 0
    assert info["parameters"]["trainable"] == count_parameters(sam.mask_decoder)["total"]
    for (n, p), q in zip(sam.mask_decoder.named_parameters(), other.mask_decoder.parameters()):
        torch.testing.assert_close(p, q, msg=n)


def test_dataset_without_image(synthetic_kvasir):
    d = synthetic_kvasir
    ids = _split(d)["train"][:3]
    full = KvasirDataset(d["raw"], d["processed"], ids, train=True)
    light = KvasirDataset(d["raw"], d["processed"], ids, train=True, with_image=False)
    for i in range(3):
        a, b = full[i], light[i]
        assert "image" not in b
        assert a["input_size"] == b["input_size"]
        torch.testing.assert_close(a["boxes"], b["boxes"])
        assert torch.equal(a["masks"], b["masks"])


def test_cached_evaluation_matches_encoder(sam, synthetic_kvasir):
    """La validation con la cache (float16) deve dare le stesse metriche dell'encoder (float32)."""
    d = synthetic_kvasir
    ids = _split(d)["val"][:2]
    prompts_file = d["splits"] / "prompts_val.json"
    prompts = json.loads(prompts_file.read_text())["prompts"]
    ds = KvasirDataset(d["raw"], d["processed"], ids, train=False, prompts_file=prompts_file)
    cache = compute_embeddings(sam, ds, "cpu", num_workers=0)
    assert cache["ids"] == ids and cache["embeddings"].dtype == torch.float16
    assert cache["embeddings"].shape == (2, 256, 64, 64)

    ref = evaluate(sam, ds, prompts, "cpu", num_workers=0)
    light = KvasirDataset(d["raw"], d["processed"], ids, train=False, prompts_file=prompts_file,
                          with_image=False)
    cached = evaluate(sam, light, prompts, "cpu", num_workers=0,
                      embeddings=dict(zip(cache["ids"], cache["embeddings"])))
    for r, c in zip(ref, cached):
        assert r["id"] == c["id"] and c["encoder_ms"] == 0.0
        assert c["boxes.dice_raw"] == pytest.approx(r["boxes.dice_raw"], abs=0.02)


def test_cache_script(synthetic_kvasir, tmp_path):
    d = synthetic_kvasir
    subprocess.run([sys.executable, "scripts/cache_embeddings.py", "--raw-root", d["raw"],
                    "--processed", d["processed"], "--splits", d["splits"], "--which", "val",
                    "--out", tmp_path, "--device", "cpu", "--limit", "2", "--num-workers", "0"],
                   check=True, cwd=ROOT, capture_output=True)
    cache = load_cache(tmp_path / "val.pt")
    assert len(cache) == 2 and next(iter(cache.values())).shape == (256, 64, 64)


def test_train_script_end_to_end(synthetic_kvasir, tmp_path):
    """2 epoche su poche immagini con embedding casuali, poi valutazione con --weights."""
    d = synthetic_kvasir
    split = _split(d)
    torch.manual_seed(0)
    for name in ("train", "val"):
        ids = split[name]
        save_cache({"ids": ids, "embeddings": torch.randn(len(ids), 256, 64, 64).half()},
                   tmp_path / "cache" / f"{name}.pt")
    out = tmp_path / "run"
    subprocess.run([sys.executable, "scripts/train.py", "--raw-root", d["raw"], "--processed", d["processed"],
                    "--splits", d["splits"], "--cache", tmp_path / "cache", "--out", out,
                    "--epochs", "2", "--batch-size", "2", "--limit", "4", "--device", "cpu",
                    "--num-workers", "0"], check=True, cwd=ROOT, capture_output=True)
    history = json.loads((out / "history.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    assert [h["epoch"] for h in history] == [0, 1, 2]
    assert all(h["train_loss"] > 0 for h in history[1:])
    assert 3.5e6 < summary["parameters"]["trainable"] < 4.5e6 and summary["parameters"]["encoder"] == 0
    assert (out / "trainable.pt").exists()

    subprocess.run([sys.executable, "scripts/evaluate.py", "--raw-root", d["raw"], "--processed", d["processed"],
                    "--splits", d["splits"], "--split", "test", "--out", out, "--device", "cpu",
                    "--limit", "1", "--num-workers", "0", "--weights", out / "trainable.pt"],
                   check=True, cwd=ROOT, capture_output=True)
    m = json.loads((out / "test" / "metrics.json").read_text())
    assert m["finetuned"]["method"] == "decoder"
