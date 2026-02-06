from typing import Callable, Any, Sequence
import numpy as np

AXES_6 = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")

def make_edges_uniform(min_val: float, max_val: float, n_bins: int) -> np.ndarray:
    """Bins uniformes sur une plage physique (recommandé)."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    # n_bins bins => n_bins+1 edges
    return np.linspace(float(min_val), float(max_val), int(n_bins) + 1)
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
def make_edges_quantile(y: np.ndarray, n_bins: int) -> np.ndarray:
    """Bins par quantiles (utile si tu veux équilibrer les counts)."""
    y = np.asarray(y, float).ravel()
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(y, qs)
    edges = np.unique(edges)
    # fallback si trop de valeurs identiques
    if edges.size < 2:
        v = float(y[0]) if y.size else 0.0
        return np.array([v - 1.0, v + 1.0], float)
    return edges
def coverage_1d_counts(Y: np.ndarray, edges_per_axis) -> dict[str, np.ndarray]:
    """Retourne le comptage par bin sur chaque axe (diagnostic)."""
    Y = np.asarray(Y, float)
    out: dict[str, np.ndarray] = {}
    for ax, edges in enumerate(edges_per_axis):
        edges = np.asarray(edges, float)
        z = np.digitize(Y[:, ax], edges, right=False) - 1
        z = np.clip(z, 0, edges.size - 2)
        counts = np.bincount(z, minlength=edges.size - 1)
        out[AXES_6[ax]] = counts
    return out
def y_key_from_tolerances(Y: np.ndarray, tol: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, float)
    tol = np.asarray(tol, float).ravel()

    if tol.size != Y.shape[1]:
        raise ValueError("tol must match Y columns")
    if np.any(tol <= 0):
        raise ValueError("All tolerances must be > 0")

    Q = np.rint(Y / tol[None, :]).astype(np.int64)

    keys = [tuple(map(int, row)) for row in Q]
    return np.array(keys, dtype=object)
def y_key_from_relative_tolerances(
    Y: np.ndarray,
    pct: np.ndarray,        # ex: [0.05,...] (5%)
    min_tol: np.ndarray,    # plancher absolu par axe
) -> np.ndarray:
    Y = np.asarray(Y, float)
    pct = np.asarray(pct, float).ravel()
    min_tol = np.asarray(min_tol, float).ravel()

    if pct.size != Y.shape[1] or min_tol.size != Y.shape[1]:
        raise ValueError("pct and min_tol must match Y columns")
    if np.any(pct <= 0) or np.any(min_tol <= 0):
        raise ValueError("pct and min_tol must be > 0")

    tol_ij = np.maximum(min_tol[None, :], pct[None, :] * np.abs(Y))

    Q = np.rint(Y / tol_ij).astype(np.int64)
    keys = [tuple(map(int, row)) for row in Q]
    return np.array(keys, dtype=object)


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
def counts_per_axis_bin(Y, edges_per_axis, axes=None):
    Y = np.asarray(Y, float)
    if axes is None:
        axes = list(range(len(edges_per_axis)))

    out = {}
    for ax, edges in zip(axes, edges_per_axis):
        edges = np.asarray(edges, float)
        z = np.digitize(Y[:, ax], edges, right=False) - 1
        z = np.clip(z, 0, len(edges) - 2).astype(int)

        counts = np.bincount(z, minlength=len(edges) - 1)
        out[ax] = counts  # counts[k] = nb essais dans le bin k
    return out
def cap_trials_per_zone_mass_and_Y_ranges_no_pca(
    trials: list[dict],
    X: np.ndarray,
    Y: np.ndarray,
    k_per_key: int | None,
    mass_fn: Callable[[dict], Any],
    seed: int = 0,
    edges_per_axis: Sequence[np.ndarray] | None = None,
    axes_for_ranges: Sequence[int] | None = None,
    zone_ranges: np.ndarray | None = None,
    pct: np.ndarray | None = None,
    min_tol: np.ndarray | None = None,
    score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = default_score_fn,
    quality_fn: Callable[[dict], float] | None = None,
    max_total: int | None = None,
) -> list[dict]:

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = len(trials)
    if X.shape[0] != n or Y.shape[0] != n:
        raise ValueError("X and Y must have same number of rows as trials")

    rng = np.random.default_rng(seed)

    # --- zones Y (ranges/bins) ---
    if edges_per_axis is None:
        zone_ranges = np.zeros(n, dtype=int)
    else:
        if zone_ranges is None:
            zone_ranges = zone_id_from_Y_ranges(Y, edges_per_axis=edges_per_axis, axes=axes_for_ranges)
        zone_ranges = np.asarray(zone_ranges, dtype=int)


        unique, counts = np.unique(zone_ranges, return_counts=True)
        order = np.argsort(counts)[::-1]  # tri décroissant

        for z, c in zip(unique[order], counts[order]):
            print(f"zone_id {z:6d} -> {c} essais")

        print("nb zones utilisées:", len(unique))
        print("total essais:", counts.sum())

        if zone_ranges.shape[0] != n:
            raise ValueError("zone_ranges must match trials length")


    if pct is not None:
        y_keys = y_key_from_relative_tolerances(Y, pct, min_tol)

        for i, k in enumerate(y_keys):
            print(f"{i:4d} -> {k}")
    else:
        y_keys = None

    # # --- buckets = (mass, zone_ranges) ---
    # buckets: dict[tuple[Any, int], list[int]] = {}
    # for i, t in enumerate(trials):
    #     key = (mass_fn(t), int(zone_ranges[i]))
    #     buckets.setdefault(key, []).append(i)
    #
    # print(buckets)
    #
    # kept_idx: list[int] = []
    #
    # for idxs in buckets.values():
    #     idxs = np.asarray(idxs, dtype=int)
    #     rng.shuffle(idxs)
    #
    #     # 1) dédoublonnage dans le bucket
    #     if y_keys is not None:
    #         groups: dict[Any, list[int]] = {}
    #         for i in idxs.tolist():
    #             k = tuple(np.asarray(y_keys[i]).ravel().tolist())
    #             groups.setdefault(k, []).append(i)
    #
    #         winners: list[int] = []
    #         for g in groups.values():
    #             g_idx = np.asarray(g, dtype=int)
    #
    #             scores = score_fn(X, g_idx)
    #
    #             if quality_fn is not None:
    #                 q = np.array([float(quality_fn(trials[i])) for i in g_idx], float)
    #                 # combine (zscore + zscore)
    #                 s = (scores - scores.mean()) / (scores.std() + 1e-12)
    #                 qn = (q - q.mean()) / (q.std() + 1e-12)
    #                 scores = s + qn
    #
    #             winners.append(int(g_idx[int(np.argmax(scores))]))
    #
    #         idxs2 = np.asarray(winners, dtype=int)
    #     else:
    #         idxs2 = idxs
    #
    #     # 2) cap par bucket
    #     if k_per_key is None:
    #         kept_idx.extend(idxs2.tolist())
    #     else:
    #         k = int(k_per_key)
    #         if k <= 0:
    #             continue
    #         rng.shuffle(idxs2)
    #         kept_idx.extend(idxs2[: min(k, len(idxs2))].tolist())
    #
    # kept_idx = sorted(set(kept_idx))
    #
    # # --- cap total ---
    # if max_total is not None:
    #     mt = int(max_total)
    #     if mt <= 0:
    #         return []
    #     if len(kept_idx) > mt:
    #         arr = np.asarray(kept_idx, dtype=int)
    #         rng.shuffle(arr)
    #         kept_idx = sorted(arr[:mt].tolist())
    #
    # return [trials[i] for i in kept_idx]

    # --- buckets = (zone_ranges) ---  (sans masse)
    buckets: dict[int, list[int]] = {}
    for i in range(len(trials)):
        key = int(zone_ranges[i])
        buckets.setdefault(key, []).append(i)

    print(buckets)

    kept_idx: list[int] = []

    for idxs in buckets.values():
        idxs = np.asarray(idxs, dtype=int)
        rng.shuffle(idxs)

        # 1) dédoublonnage dans le bucket
        if y_keys is not None:
            groups: dict[tuple[int, ...], list[int]] = {}
            for i in idxs.tolist():
                k = tuple(np.asarray(y_keys[i]).ravel().astype(int).tolist())
                groups.setdefault(k, []).append(i)

            winners: list[int] = []
            for g in groups.values():
                g_idx = np.asarray(g, dtype=int)

                scores = score_fn(X, g_idx)

                if quality_fn is not None:
                    q = np.array([float(quality_fn(trials[i])) for i in g_idx], float)
                    # combine (zscore + zscore)
                    s = (scores - scores.mean()) / (scores.std() + 1e-12)
                    qn = (q - q.mean()) / (q.std() + 1e-12)
                    scores = s + qn

                winners.append(int(g_idx[int(np.argmax(scores))]))

            idxs2 = np.asarray(winners, dtype=int)
        else:
            idxs2 = idxs

        # 2) cap par bucket
        if k_per_key is None:
            kept_idx.extend(idxs2.tolist())
        else:
            k = int(k_per_key)
            if k <= 0:
                continue
            rng.shuffle(idxs2)
            kept_idx.extend(idxs2[: min(k, len(idxs2))].tolist())

    kept_idx = sorted(set(kept_idx))

    # --- cap total ---
    if max_total is not None:
        mt = int(max_total)
        if mt <= 0:
            return []
        if len(kept_idx) > mt:
            arr = np.asarray(kept_idx, dtype=int)
            rng.shuffle(arr)
            kept_idx = sorted(arr[:mt].tolist())

    return [trials[i] for i in kept_idx]

