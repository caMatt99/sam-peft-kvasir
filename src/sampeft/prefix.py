"""Prompt tuning nell'encoder di SAM, nella forma "prefix" (Task 3 e 4).

VPT aggiunge token addestrabili alla sequenza di un ViT. In SAM i token formano una griglia 64×64,
8 blocchi su 12 usano attenzione a finestre 14×14 e c'è un bias di posizione relativa sulla griglia:
aggiungere token "in coda" romperebbe tutto questo. Per questo i prompt entrano DENTRO l'attenzione:

    in un blocco scelto, L chiavi e L valori addestrabili si aggiungono a quelli calcolati dalla griglia;
    ogni query (ogni posizione della griglia) può guardare anche i prefix, con una softmax unica.

- La griglia, le finestre e il bias di posizione restano intatti; i prefix non hanno posizione
  e sono condivisi da tutte le finestre.
- shallow (Task 3): prefix solo nel blocco 0.  deep (Task 4): prefix indipendenti in tutti i 12 blocchi.
- Parametri: 2 · L · 768 per blocco → L = 8: 12,288 (shallow), 147,456 (deep, come LoRA r = 4).
- Inizializzazione normale (std 0.02). Con la softmax unica, all'inizio una piccola parte dell'attenzione
  va ai prefix: il modello NON coincide esattamente con SAM (il controllo dell'epoca 0 lo misura).
- A differenza di LoRA non c'è merge: i prefix restano anche in inferenza (costo piccolo ma non nullo).
"""
import torch
from torch import nn
from segment_anything.modeling.image_encoder import add_decomposed_rel_pos


class PrefixAttention(nn.Module):
    def __init__(self, attn: nn.Module, length: int):
        super().__init__()
        self.attn = attn                       # l'attenzione originale di SAM, congelata
        self.length = length
        dim = attn.qkv.in_features
        self.prefix_k = nn.Parameter(torch.empty(length, dim, device=attn.qkv.weight.device))
        self.prefix_v = nn.Parameter(torch.empty(length, dim, device=attn.qkv.weight.device))
        nn.init.normal_(self.prefix_k, std=0.02)
        nn.init.normal_(self.prefix_v, std=0.02)

    def _heads(self, p: torch.Tensor, batch: int, dtype: torch.dtype) -> torch.Tensor:
        """(L, dim) → (batch · heads, L, head_dim), come k e v nell'attenzione di SAM."""
        nh = self.attn.num_heads
        hd = p.shape[1] // nh
        p = p.to(dtype).view(self.length, nh, hd).permute(1, 0, 2)
        return p.unsqueeze(0).expand(batch, -1, -1, -1).reshape(batch * nh, self.length, hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.attn
        B, H, W, _ = x.shape                   # B = immagini × finestre nei blocchi a finestre
        nh = a.num_heads
        qkv = a.qkv(x).reshape(B, H * W, 3, nh, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * nh, H * W, -1).unbind(0)

        logits = (q * a.scale) @ k.transpose(-2, -1)                     # griglia → griglia
        if a.use_rel_pos:
            logits = add_decomposed_rel_pos(logits, q, a.rel_pos_h, a.rel_pos_w, (H, W), (H, W))
        pk = self._heads(self.prefix_k, B, q.dtype)
        pv = self._heads(self.prefix_v, B, v.dtype)
        logits_p = (q * a.scale) @ pk.transpose(-2, -1)                  # griglia → prefix (senza posizione)
        attn = torch.cat([logits, logits_p], dim=-1).softmax(dim=-1)     # softmax unica
        out = attn @ torch.cat([v, pv], dim=1)
        out = out.view(B, nh, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        return a.proj(out)


def add_prefix(sam, length: int, deep: bool) -> None:
    """deep=False: solo il blocco 0 (shallow). deep=True: tutti i blocchi. Solo i prefix si addestrano."""
    blocks = sam.image_encoder.blocks if deep else sam.image_encoder.blocks[:1]
    for blk in blocks:
        if not isinstance(blk.attn, PrefixAttention):
            blk.attn = PrefixAttention(blk.attn, length)
    for n, p in sam.image_encoder.named_parameters():
        p.requires_grad_(".prefix_" in n)
