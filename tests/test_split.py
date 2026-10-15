import json

import numpy as np

from sampeft.data.split import dhash, duplicate_groups, hamming_matrix, stratified_group_split


def test_dhash_identical_and_different():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    b = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    d = hamming_matrix(np.stack([dhash(a), dhash(a.copy()), dhash(b)]))
    assert d[0, 1] == 0 and d[0, 2] > 10


def test_duplicate_groups_transitive():
    dist = np.array([[0, 1, 9, 9],
                     [1, 0, 2, 9],
                     [9, 2, 0, 9],
                     [9, 9, 9, 0]])
    assert duplicate_groups(dist, threshold=2) == [[0, 1, 2], [3]]


def test_stratified_group_split():
    n = 200
    strata = np.array([1] * 20 + [0] * 180)
    groups = [[0, 1]] + [[i] for i in range(2, n)]           # 0 e 1 sono quasi-duplicati
    s = stratified_group_split(groups, strata, n_val=20, n_test=20, seed=0)
    all_ids = s["train"] + s["val"] + s["test"]
    assert sorted(all_ids) == list(range(n))                 # copre tutto, nessuna ripetizione
    assert len(s["val"]) == 20 and len(s["test"]) == 20
    assert sum(strata[s["test"]]) == 2                       # 10% di strato 1, come nel totale
    assert any({0, 1} <= set(v) for v in s.values())          # il gruppo resta unito


def test_split_is_deterministic():
    strata = np.zeros(50, int)
    groups = [[i] for i in range(50)]
    assert stratified_group_split(groups, strata, 5, 5, 1) == stratified_group_split(groups, strata, 5, 5, 1)


def test_split_files(synthetic_kvasir):
    split = json.loads((synthetic_kvasir["splits"] / "split.json").read_text())
    sets = [set(split[k]) for k in ("train", "val", "test")]
    assert sum(map(len, sets)) == 40 and len(set.union(*sets)) == 40
    assert split["counts"]["test"]["images"] == 5
    prompts = json.loads((synthetic_kvasir["splits"] / "prompts_test.json").read_text())
    assert set(prompts["prompts"]) == set(split["test"])
