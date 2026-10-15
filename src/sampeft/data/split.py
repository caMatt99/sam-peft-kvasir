"""Split train/val/test: stratificato per numero di polipi e consapevole dei quasi-duplicati.

Quasi-duplicati: Kvasir-SEG viene da video di colonscopia, quindi due frame quasi identici
potrebbero finire uno nel train e uno nel test (data leakage). Li troviamo con un dHash
(impronta di 64 bit che cambia poco tra immagini simili) e mettiamo ogni gruppo di
quasi-duplicati interamente nello stesso split.
"""
import cv2
import numpy as np


def dhash(img_gray: np.ndarray, size: int = 8) -> np.ndarray:
    """dHash: confronta pixel vicini in una miniatura (size+1) x size → vettore di size*size bit."""
    small = cv2.resize(img_gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def hamming_matrix(hashes: np.ndarray) -> np.ndarray:
    """(N, B) bool → (N, N) numero di bit diversi tra ogni coppia."""
    h = hashes.astype(np.uint8)
    return (h[:, None, :] != h[None, :, :]).sum(-1)


def duplicate_groups(dist: np.ndarray, threshold: int) -> list[list[int]]:
    """Gruppi di indici collegati da distanza <= threshold (union-find, chiusura transitiva)."""
    n = len(dist)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ii, jj = np.where(np.triu(dist <= threshold, k=1))
    for i, j in zip(ii, jj):
        parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def stratified_group_split(groups: list[list[int]], strata: np.ndarray, n_val: int, n_test: int,
                           seed: int) -> dict[str, list[int]]:
    """Assegna gruppi interi a val/test/train, rispettando le proporzioni di ogni strato.

    strata[i] = etichetta dello strato dell'elemento i (es. 0 = un polipo, 1 = più polipi).
    Lo strato di un gruppo è il massimo dei suoi elementi.
    """
    rng = np.random.default_rng(seed)
    n = len(strata)
    group_stratum = np.array([max(strata[i] for i in g) for g in groups])
    labels = sorted(set(group_stratum.tolist()))
    sizes = {s: sum(len(g) for g, gs in zip(groups, group_stratum) if gs == s) for s in labels}

    # quote per strato, proporzionali; l'ultimo strato (il più grande) prende il resto
    order = sorted(labels, key=lambda s: sizes[s])
    targets = {}
    left_val, left_test = n_val, n_test
    for s in order[:-1]:
        targets[s] = (round(n_val * sizes[s] / n), round(n_test * sizes[s] / n))
        left_val -= targets[s][0]
        left_test -= targets[s][1]
    targets[order[-1]] = (left_val, left_test)

    out = {"train": [], "val": [], "test": []}
    for s in labels:
        idx = [k for k, gs in enumerate(group_stratum) if gs == s]
        rng.shuffle(idx)
        t_val, t_test = targets[s]
        c_val = c_test = 0
        for k in idx:
            g = groups[k]
            if c_test + len(g) <= t_test:
                out["test"] += g
                c_test += len(g)
            elif c_val + len(g) <= t_val:
                out["val"] += g
                c_val += len(g)
            else:
                out["train"] += g
    return {k: sorted(v) for k, v in out.items()}
