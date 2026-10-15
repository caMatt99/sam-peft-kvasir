import json

import torch

from sampeft.data.kvasir import KvasirDataset


def _datasets(d):
    split = json.loads((d["splits"] / "split.json").read_text())
    train = KvasirDataset(d["raw"], d["processed"], split["train"], train=True)
    test = KvasirDataset(d["raw"], d["processed"], split["test"], train=False,
                         prompts_file=d["splits"] / "prompts_test.json")
    return train, test


def test_item_contract(synthetic_kvasir):
    train, test = _datasets(synthetic_kvasir)
    for ds in (train, test):
        for i in range(len(ds)):
            x = ds[i]
            h, w = x["original_size"]
            k = len(x["boxes"])
            assert x["image"].shape == (3, 1024, 1024)
            assert max(x["input_size"]) == 1024
            assert x["boxes"].shape == x["boxes_orig"].shape == (k, 4)
            assert x["masks"].shape == (k, h, w) and x["masks"].dtype == torch.bool
            assert x["gt"].shape == (h, w)
            assert k in (1, 2)


def test_gt_raw_keeps_noise_masks_clean_do_not(synthetic_kvasir):
    _, test = _datasets(synthetic_kvasir)
    x = test[0]
    assert x["gt"][2, 2]                       # la GT ufficiale resta quella originale
    assert not x["masks"].any(0)[2, 2]         # il target della loss è pulito


def test_train_jitter_changes_per_epoch_but_is_reproducible(synthetic_kvasir):
    train, _ = _datasets(synthetic_kvasir)
    a = train[0]["boxes_orig"]
    assert torch.equal(a, train[0]["boxes_orig"])
    train.set_epoch(1)
    assert not torch.equal(a, train[0]["boxes_orig"])


def test_test_prompts_are_fixed(synthetic_kvasir):
    _, test = _datasets(synthetic_kvasir)
    prompts = json.loads((synthetic_kvasir["splits"] / "prompts_test.json").read_text())["prompts"]
    x = test[0]
    assert x["boxes_orig"].tolist() == prompts[x["id"]]["boxes"]
