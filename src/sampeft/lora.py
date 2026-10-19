"""LoRA sulle proiezioni q e v dell'attenzione dell'image encoder (Task 5).

In ogni blocco del ViT il layer qkv (768 → 3·768) calcola query, key e value insieme.
LoRA lo congela e aggiunge una correzione a basso rango solo a q e v:

    q = W_q x + s · B_q A_q x        v = W_v x + s · B_v A_v x        k invariata

A: (r, 768) inizializzata casuale, B: (768, r) inizializzata a zero → all'inizio la correzione
vale zero e il modello coincide con SAM originale. s = alpha / r (alpha = r → s = 1, come SAM-LoRA).
Parametri: 2 · r · (768 + 768) = 3072·r per blocco, 12 blocchi.

Dopo il training merge_lora() somma s·B·A dentro W: il modello torna alla forma originale
e l'inferenza costa esattamente come SAM.
"""
import math

import torch
from torch import nn


class LoRAQKV(nn.Module):
    def __init__(self, qkv: nn.Linear, r: int, alpha: float | None = None):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.r = r
        self.scale = (alpha if alpha is not None else r) / r
        self.a_q = nn.Linear(self.dim, r, bias=False)
        self.b_q = nn.Linear(r, self.dim, bias=False)
        self.a_v = nn.Linear(self.dim, r, bias=False)
        self.b_v = nn.Linear(r, self.dim, bias=False)
        for a in (self.a_q, self.a_v):
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))   # come nel codice di LoRA
        for b in (self.b_q, self.b_v):
            nn.init.zeros_(b.weight)
        self.to(qkv.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dq = self.b_q(self.a_q(x)) * self.scale
        dv = self.b_v(self.a_v(x)) * self.scale
        return self.qkv(x) + torch.cat([dq, torch.zeros_like(dq), dv], dim=-1)

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """Un nn.Linear normale con s·B·A già sommato dentro i pesi di q e v."""
        out = nn.Linear(self.dim, 3 * self.dim, bias=self.qkv.bias is not None).to(self.qkv.weight.device)
        w = self.qkv.weight.clone()
        w[: self.dim] += self.scale * self.b_q.weight @ self.a_q.weight
        w[2 * self.dim:] += self.scale * self.b_v.weight @ self.a_v.weight
        out.weight.copy_(w)
        if self.qkv.bias is not None:
            out.bias.copy_(self.qkv.bias)
        return out


def add_lora(sam, r: int, alpha: float | None = None) -> None:
    """Sostituisce il qkv di ogni blocco dell'encoder con la versione LoRA (solo A e B addestrabili)."""
    for blk in sam.image_encoder.blocks:
        if not isinstance(blk.attn.qkv, LoRAQKV):
            blk.attn.qkv = LoRAQKV(blk.attn.qkv, r, alpha)
    for n, p in sam.image_encoder.named_parameters():
        p.requires_grad_(any(k in n for k in (".a_q.", ".b_q.", ".a_v.", ".b_v.")))


def enable_checkpointing(sam) -> None:
    """Gradient checkpointing sui blocchi dell'encoder: non salva le attivazioni intermedie e le
    ricalcola nel backward. Meno memoria (~metà o meno), più tempo (~+30%). Solo se la memoria non basta."""
    from torch.utils.checkpoint import checkpoint
    for blk in sam.image_encoder.blocks:
        fwd = blk.forward
        blk.forward = lambda x, fwd=fwd: checkpoint(fwd, x, use_reentrant=False)


def merge_lora(sam) -> None:
    """Dopo il training: ogni LoRAQKV torna un nn.Linear con i pesi aggiornati (inferenza senza costi extra)."""
    for blk in sam.image_encoder.blocks:
        if isinstance(blk.attn.qkv, LoRAQKV):
            blk.attn.qkv = blk.attn.qkv.merged()
