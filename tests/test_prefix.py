"""Task 3 e 4: prefix nell'attenzione dell'encoder, ed export dei risultati leggeri.

Verifiche: con prefix "spenti" (chiavi molto negative) l'attenzione coincide con quella di SAM,
quindi il resto del calcolo è identico; conta dei parametri; solo i prefix e il decoder si
addestrano; salvataggio e caricamento; training da script; export.
"""
import base64
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch

from sampeft.export import download_link_html
from sampeft.lora import enable_checkpointing
from sampeft.model import apply_method, encode, load_finetuned, load_sam, parameter_report, save_finetuned
from sampeft.prefix import PrefixAttention

ROOT = Path(__file__).resolve().parents[1]


def _sam():
    torch.manual_seed(0)
    return load_sam(None).eval()


@pytest.fixture(scope="module")
def x():
    torch.manual_seed(0)
    return torch.randn(1, 3, 1024, 1024)


def test_prefix_attention_reproduces_sam():
    """Con 0 prefix PrefixAttention deve dare esattamente l'attenzione originale (blocco a finestre e globale);
    con prefix non nulli l'uscita cambia."""
    sam = _sam()
    for blk in (sam.image_encoder.blocks[0], sam.image_encoder.blocks[2]):   # a finestre, globale
        size = 14 if blk.window_size else 64
        inp = torch.randn(2, size, size, 768)
        with torch.no_grad():
            ref = blk.attn(inp)
            torch.testing.assert_close(PrefixAttention(blk.attn, length=0)(inp), ref)
            assert not torch.allclose(PrefixAttention(blk.attn, length=4)(inp), ref)


def test_prefix_params(x):
    for method, n_blocks in [("prefix_shallow", 1), ("prefix_deep", 12)]:
        sam = _sam()
        apply_method(sam, method, length=8)
        rep = parameter_report(sam)
        assert rep["encoder"] == n_blocks * 2 * 8 * 768
        assert rep["trainable"] == rep["encoder"] + rep["decoder"]
    assert rep["encoder"] == 147_456                                  # deep = LoRA r = 4


def test_prefix_gradients(x):
    sam = _sam()
    apply_method(sam, "prefix_deep", length=4)
    enable_checkpointing(sam)                                         # la RAM locale non basta senza
    sam.train()
    encode(sam, x).mean().backward()
    pa = sam.image_encoder.blocks[0].attn
    assert pa.prefix_k.grad is not None and pa.prefix_k.grad.abs().sum() > 0
    assert pa.prefix_v.grad.abs().sum() > 0
    assert pa.attn.qkv.weight.grad is None                            # attenzione originale congelata


def test_prefix_save_load(x, tmp_path):
    sam = _sam()
    apply_method(sam, "prefix_shallow", length=8)
    with torch.no_grad():
        ref = encode(sam, x)
    save_finetuned(sam, "prefix_shallow", tmp_path / "w.pt", {"length": 8})
    other = _sam()
    info = load_finetuned(other, tmp_path / "w.pt")
    assert info["method"] == "prefix_shallow" and info["parameters"]["encoder"] == 2 * 8 * 768
    with torch.no_grad():
        torch.testing.assert_close(encode(other, x), ref)


def test_train_script_prefix(synthetic_kvasir, tmp_path):
    d = synthetic_kvasir
    out = tmp_path / "run"
    subprocess.run([sys.executable, "scripts/train.py", "--raw-root", d["raw"], "--processed", d["processed"],
                    "--splits", d["splits"], "--method", "prefix_deep", "--length", "4", "--out", out,
                    "--epochs", "1", "--batch-size", "2", "--limit", "2", "--device", "cpu",
                    "--num-workers", "0", "--checkpointing"], check=True, cwd=ROOT, capture_output=True)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["parameters"]["encoder"] == 12 * 2 * 4 * 768
    assert summary["method_config"] == {"length": 4}
    assert summary["config"]["prompt_lr"] == "0.001"


def test_download_link(tmp_path):
    runs = tmp_path / "runs"
    (runs / "a").mkdir(parents=True)
    (runs / "a" / "metrics.json").write_text('{"x": 1}')
    (runs / "a" / "trainable.pt").write_bytes(b"0" * 1000)
    html = download_link_html(runs, "r.zip")
    b64 = re.search(r"base64,([^\"]+)", html).group(1)
    names = zipfile.ZipFile(io.BytesIO(base64.b64decode(b64))).namelist()
    assert names == ["runs/a/metrics.json"]                           # niente pesi
