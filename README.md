# Parameter-Efficient Fine-Tuning of SAM for Polyp Segmentation

How much can parameter-efficient fine-tuning (PEFT) improve the **Segment Anything Model (SAM)** on a specialised medical domain, and at what cost?

This project adapts SAM ViT-B to colonoscopy polyp segmentation on **Kvasir-SEG** and compares PEFT methods along three axes:

- **segmentation accuracy** (Dice, IoU);
- **trainable parameters**;
- **training efficiency** (time, GPU memory).

> **Status:** work in progress. Data pipeline complete; zero-shot evaluation and fine-tuning experiments in progress.

## Methods compared

| Method | What is trained | Encoder |
|---|---|---|
| Zero-shot SAM | nothing | frozen |
| Decoder-only (baseline) | mask decoder | frozen (embeddings cached) |
| LoRA (r = 4, 8, 16) | low-rank adapters on attention `qkv` + decoder | frozen + adapters |
| Prefix tuning, shallow | learnable key/value tokens in the first block + decoder | frozen + prefixes |
| Prefix tuning, deep | learnable key/value tokens in every block + decoder | frozen + prefixes |

All methods share the same data split, prompts, loss, optimiser and early-stopping rule. The decoder is trained in every method, so the gap to the decoder-only baseline isolates the contribution of adapting the image encoder. Trainable parameters are reported both for the encoder and in total.

## Evaluation protocol

- **Split:** 800 / 100 / 100 (train / val / test), seed 0. Stratified by single- vs multi-polyp images. Near-duplicate frames (dHash) are kept within the same split to avoid leakage.
- **Prompts:** one box per polyp, from connected components of the mask. Test boxes are precomputed and stored in `data/splits/`: an exact version, and a jittered version (±10% of box size) to simulate a hand-drawn box.
- **Metrics:** Dice and IoU per image at the original resolution, averaged over images. They are computed on the official masks (JPEG binarised at 127).
- **Mask cleaning:** components smaller than 100 px are removed **only** to derive box prompts and training targets.

## Preliminary finding

Kvasir-SEG masks are stored as JPEG. Compression leaves isolated foreground pixels far from the polyp. A single bounding box over all foreground pixels can therefore span the whole image and give SAM no localisation cue.

On a 20-image sample, zero-shot SAM ViT-B went from **0.825** to **0.947** Dice when we switched to one box per cleaned connected component. The prompt protocol matters as much as the model: results on this dataset are only comparable under an identical prompting scheme.

## Repository structure

```
├── data/splits/          fixed split and val/test prompts (versioned)
├── notebooks/            exploration and Kaggle runner notebooks
├── scripts/
│   ├── prepare_data.py   mask cleaning, per-polyp boxes, dataset statistics
│   └── make_split.py     stratified, duplicate-aware split + fixed prompts
├── src/sampeft/
│   ├── data/             masks, prompts, split, SAM-identical transforms, Dataset
│   └── metrics.py        Dice, IoU
└── tests/                pytest suite (runs on a synthetic dataset, no GPU needed)
```

## Usage

```bash
pip install -e ".[dev]"
pytest -q

python scripts/prepare_data.py --raw-root <Kvasir-SEG> --out <processed>
python scripts/make_split.py   --raw-root <Kvasir-SEG> --processed <processed> --out data/splits
```

Experiments run on Kaggle (NVIDIA T4, 16 GB).

## Results

To be added: accuracy vs trainable parameters, and accuracy vs training time.

## References

- Kirillov et al., *Segment Anything*, ICCV 2023.
- Jha et al., *Kvasir-SEG: A Segmented Polyp Dataset*, MMM 2020.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.
- Li & Liang, *Prefix-Tuning: Optimizing Continuous Prompts for Generation*, ACL 2021.
