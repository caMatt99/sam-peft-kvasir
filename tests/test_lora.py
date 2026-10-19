"""Task 5: LoRA su q e v dell'encoder.

Verifiche: all'inizio il modello coincide con SAM (B = 0); conta dei parametri; solo A, B e
il decoder si addestrano; il merge dà le stesse uscite; salvataggio e caricamento; training da script.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sampeft.lora import LoRAQKV, enable_checkpointing
from sampeft.model import apply_method, encode, load_finetuned, load_sam, parameter_report, save_finetuned

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def x():
    torch.manual_seed(0)
    return torch.randn(1, 3, 1024, 1024)


def _sam():
    torch.manual_seed(0)
    return load_sam(None).eval()


def test_lora_qkv_layer():
    torch.manual_seed(0)
    base = torch.nn.Linear(16, 48)
    lora = LoRAQKV(base, r=4)
    inp = torch.randn(2, 5, 16)
    torch.testing.assert_close(lora(inp), base(inp))           # B = 0: nessuna differenza
    with torch.no_grad():
        lora.b_q.weight.normal_(); lora.b_v.weight.normal_()
    out = lora(inp)
    torch.testing.assert_close(out[..., 16:32], base(inp)[..., 16:32])   # k non cambia
    assert not torch.allclose(out[..., :16], base(inp)[..., :16])
    torch.testing.assert_close(lora.merged()(inp), out, rtol=1e-5, atol=1e-5)


def test_apply_lora_params_and_identity(x):
    sam = _sam()
    with torch.no_grad():
        ref = encode(sam, x)
    apply_method(sam, "lora", r=8)
    rep = parameter_report(sam)
    assert rep["encoder"] == 12 * 3072 * 8                     # 294,912
    assert rep["trainable"] == rep["encoder"] + rep["decoder"]
    assert not any(p.requires_grad for p in sam.prompt_encoder.parameters())
    with torch.no_grad():
        torch.testing.assert_close(encode(sam, x), ref)       # all'inizio è SAM originale


def test_lora_gradients_reach_encoder(x):
    sam = _sam()
    apply_method(sam, "lora", r=4)
    enable_checkpointing(sam)                                  # anche con checkpointing
    sam.train()
    encode(sam, x).mean().backward()
    qkv = sam.image_encoder.blocks[0].attn.qkv
    assert qkv.b_q.weight.grad.abs().sum() > 0                 # B riceve gradiente subito
    assert qkv.qkv.weight.grad is None                         # W resta congelata


def test_save_load_and_merge(x, tmp_path):
    sam = _sam()
    apply_method(sam, "lora", r=4)
    with torch.no_grad():
        for blk in sam.image_encoder.blocks:
            blk.attn.qkv.b_q.weight.normal_(std=0.01); blk.attn.qkv.b_v.weight.normal_(std=0.01)
        ref = encode(sam, x)
    save_finetuned(sam, "lora", tmp_path / "w.pt", {"r": 4})

    other = _sam()
    info = load_finetuned(other, tmp_path / "w.pt")            # merge=True di default
    assert info["method"] == "lora" and info["config"] == {"r": 4}
    assert info["parameters"]["encoder"] == 12 * 3072 * 4
    assert not any(isinstance(b.attn.qkv, LoRAQKV) for b in other.image_encoder.blocks)
    with torch.no_grad():
        torch.testing.assert_close(encode(other, x), ref, rtol=1e-4, atol=1e-4)


def test_train_script_lora(synthetic_kvasir, tmp_path):
    """1 epoca su 2 immagini con LoRA (encoder nel training, CPU), poi valutazione con merge."""
    d = synthetic_kvasir
    out = tmp_path / "run"
    subprocess.run([sys.executable, "scripts/train.py", "--raw-root", d["raw"], "--processed", d["processed"],
                    "--splits", d["splits"], "--method", "lora", "--rank", "4", "--out", out,
                    "--epochs", "1", "--batch-size", "2", "--limit", "2", "--device", "cpu",
                    "--num-workers", "0", "--checkpointing"], check=True, cwd=ROOT, capture_output=True)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["parameters"]["encoder"] == 12 * 3072 * 4
    assert summary["method_config"] == {"r": 4}

    subprocess.run([sys.executable, "scripts/evaluate.py", "--raw-root", d["raw"], "--processed", d["processed"],
                    "--splits", d["splits"], "--split", "test", "--out", out, "--device", "cpu",
                    "--limit", "1", "--num-workers", "0", "--weights", out / "trainable.pt"],
                   check=True, cwd=ROOT, capture_output=True)
    m = json.loads((out / "test" / "metrics.json").read_text())
    assert m["finetuned"]["method"] == "lora"


def test_benchmark_script(synthetic_kvasir, tmp_path):
    d = synthetic_kvasir
    subprocess.run([sys.executable, "scripts/benchmark.py", "--raw-root", d["raw"], "--processed", d["processed"],
                    "--splits", d["splits"], "--rank", "4", "--n-images", "1", "--warmup", "0", "--n-val", "1",
                    "--modes", "fp32", "--checkpointing", "--device", "cpu", "--num-workers", "0", "--out", tmp_path / "b.json"],
                   check=True, cwd=ROOT, capture_output=True)
    b = json.loads((tmp_path / "b.json").read_text())
    assert b["results"]["fp32"]["train_s_per_image"] > 0
