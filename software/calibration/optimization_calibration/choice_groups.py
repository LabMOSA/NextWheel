import numpy as  np
from typing import Sequence, Callable, Any


def margin_trials(
    trials: list[dict],
    Y: np.ndarray,
    pct: np.ndarray,
    min_tol: np.ndarray,
    *,
    X: np.ndarray | None = None,
    score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Enlève les quasi-doublons selon y_key_from_relative_tolerances:
      - construit une key par trial (tuple d'entiers)
      - regroupe les trials ayant la même key
      - garde 1 trial par key (meilleur score si X+score_fn, sinon aléatoire)

    Retourne:
        kept_idx: np.ndarray d'indices dans [0..n-1] des trials conservés.
    """
    Y = np.asarray(Y, float)
    pct = np.asarray(pct, float).ravel()
    min_tol = np.asarray(min_tol, float).ravel()

    n = len(trials)
    if Y.shape[0] != n:
        raise ValueError("Y must have same number of rows as trials")

    if pct.size != Y.shape[1] or min_tol.size != Y.shape[1]:
        raise ValueError("pct and min_tol must match Y columns")
    if np.any(pct <= 0) or np.any(min_tol <= 0):
        raise ValueError("pct and min_tol must be > 0")

    rng = np.random.default_rng(seed)

    # --- 1) construire les keys EXACTEMENT comme ta fonction ---
    tol_ij = np.maximum(min_tol[None, :], pct[None, :] * np.abs(Y))
    Q = np.rint(Y / tol_ij).astype(np.int64)
    keys = [tuple(int(x) for x in row.tolist()) for row in Q]

    # --- 2) regrouper key -> indices ---
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)

    # --- 3) choisir 1 représentant par groupe ---
    kept: list[int] = []

    use_score = (X is not None) and (score_fn is not None)
    if use_score:
        X = np.asarray(X, float)
        if X.shape[0] != n:
            raise ValueError("X must have same number of rows as trials")

    for idxs_list in groups.values():
        idxs = np.asarray(idxs_list, dtype=int)

        if idxs.size == 1:
            kept.append(int(idxs[0]))
            continue

        if use_score:
            scores = np.asarray(score_fn(X, idxs), float).ravel()
            if scores.size != idxs.size:
                raise ValueError("score_fn must return one score per idx in idxs")
            winner = int(idxs[int(np.argmax(scores))])
        else:
            winner = int(rng.choice(idxs))

        kept.append(winner)

    kept_idx = np.array(sorted(set(kept)), dtype=int)
    return kept_idx



def zone_id_from_Y_ranges(
    Y: np.ndarray,
    edges_per_axis: Sequence[np.ndarray],
    axes: Sequence[int] | None = None,
) -> np.ndarray:
    Y = np.asarray(Y, float)
    n = Y.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    if axes is None:
        axes = list(range(len(edges_per_axis)))
    if len(axes) != len(edges_per_axis):
        raise ValueError("axes and edges_per_axis must have the same length")

    per_axis_bins: list[np.ndarray] = []
    sizes: list[int] = []

    for ax, edges in zip(axes, edges_per_axis):
        edges = np.asarray(edges, float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("Each edges array must be 1D with at least 2 values")

        z = np.digitize(Y[:, ax], edges, right=False) - 1
        z = np.clip(z, 0, edges.size - 2)
        per_axis_bins.append(z.astype(int))
        sizes.append(int(edges.size - 1))

    zone = np.zeros(n, dtype=int)
    mult = 1
    for z, s in zip(per_axis_bins, sizes):
        zone += z * mult
        mult *= s

    return zone

def default_score_fn(X: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    """
    Score "informatif" simple (sans PCA) :
    ||X|| sur X normalisé (z-score par colonne).
    """
    X = np.asarray(X, float)
    idxs = np.asarray(idxs, dtype=int)

    Xs = X.copy()
    mu = Xs.mean(axis=0, keepdims=True)
    sd = Xs.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    Xs = (Xs - mu) / sd

    return np.linalg.norm(Xs[idxs], axis=1)



def build_choice_groups_by_zone(
    trials: list[dict],
    X: np.ndarray,
    Y: np.ndarray,
    *,
    seed: int = 0,
    edges_per_axis: Sequence[np.ndarray] | None = None,
    axes_for_ranges: Sequence[int] | None = None,
    pct: np.ndarray | None = None,
    min_tol: np.ndarray | None = None,
    max_candidates_per_group: int | None = None,
) -> dict[tuple[int], list[int]]:

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = len(trials)

    if X.shape[0] != n or Y.shape[0] != n:
        raise ValueError("X and Y must have same number of rows as trials")

    rng = np.random.default_rng(seed)

    # --- 1) dedup -> indices globaux conservés ---
    if pct is not None:
        if min_tol is None:
            raise ValueError("min_tol must be provided when pct is not None")
        kept_idx = margin_trials(
            trials, Y=Y, pct=pct, min_tol=min_tol,
            X=X, score_fn=default_score_fn, seed=seed
        )
    else:
        kept_idx = np.arange(n, dtype=int)

    kept_idx = np.asarray(kept_idx, dtype=int)

    # --- 2) zones calculées sur tout Y (OK) ---
    if edges_per_axis is None:
        zone_ranges = np.zeros(n, dtype=int)
    else:
        zone_ranges = zone_id_from_Y_ranges(
            Y, edges_per_axis=edges_per_axis, axes=axes_for_ranges
        ).astype(int)

    # --- 3) groupes par zone mais seulement pour kept_idx ---
    groups: dict[tuple[int], list[int]] = {}
    for i in kept_idx.tolist():
        z = int(zone_ranges[i])
        key = (z,)
        groups.setdefault(key, []).append(int(i))

    return groups

def groups_to_list(groups: Any) -> list[list[int]]:
    if isinstance(groups, dict):
        return [list(v) for v in groups.values()]
    return [list(g) for g in groups]

def generate_random_protocols(
    groups: Any,
    n_protocols: int,
    *,
    seed: int = 0,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    groups_list = groups_to_list(groups)

    protos: list[list[int]] = []
    for _ in range(int(n_protocols)):
        idxs = [int(rng.choice(g)) for g in groups_list]
        protos.append(idxs)
    return protos


import math


def count_all_protocols(groups: Any) -> int:
    groups_list = groups_to_list(groups)
    sizes = [len(g) for g in groups_list]
    if any(s == 0 for s in sizes):
        raise ValueError("Il y a au moins un groupe vide.")
    return math.prod(sizes)
