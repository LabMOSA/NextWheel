import numpy as np
from typing import Sequence, Callable, Any
import math


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
    Remove near-duplicate trials based on relative/absolute tolerances applied to Y.

    This function builds a discrete key for each trial by quantifying each Y component
    using a per-sample tolerance:

        tol_ij = max(min_tol[j], pct[j] * abs(Y[i, j]))
        Q[i, j] = round(Y[i, j] / tol_ij)

    Trials sharing the same key are considered near-duplicates. For each key/group,
    exactly one representative is kept:
    - If X and score_fn are provided: keep the trial with the best score.
    - Otherwise: keep a random trial from the group.

    Parameters
    ----------
    trials : list[dict]
        Trial objects (not used directly by this function, kept for API consistency).
    Y : np.ndarray
        Array of shape (n_trials, n_outputs) used to define near-duplicates.
    pct : np.ndarray
        Relative tolerance per Y column (shape (n_outputs,)). Must be > 0.
    min_tol : np.ndarray
        Minimum absolute tolerance per Y column (shape (n_outputs,)). Must be > 0.
    X : np.ndarray | None, optional         feature matrix of shape (n_trials, n_features), used for scoring.
    score_fn : Callable[[np.ndarray, np.ndarray], np.ndarray] | None, optional
        Scoring function. Called as score_fn(X, idxes) and must return one score per idx.
        Higher score means "better".
    seed : int, default=0
        Random seed used when selecting a representative randomly.

    Returns
    -------
    kept_idx : np.ndarray
        Sorted array of unique indices (dtype int) in [0...n_trials-1] of retained trials.
    """
    Y = np.asarray(Y, float)
    pct = np.asarray(pct, float).ravel()
    min_tol = np.asarray(min_tol, float).ravel()

    n = len(trials)
    if Y.ndim != 2:
        raise ValueError("Y must be a 2D array of shape (n_trials, n_outputs).")
    if Y.shape[0] != n:
        raise ValueError("Y must have the same number of rows as trials.")
    if pct.size != Y.shape[1] or min_tol.size != Y.shape[1]:
        raise ValueError(
            "pct and min_tol must have the same length as the number of Y columns."
        )
    if np.any(pct <= 0) or np.any(min_tol <= 0):
        raise ValueError("pct and min_tol must be strictly positive (> 0).")

    rng = np.random.default_rng(seed)

    # 1) Build keys exactly as described
    tol_ij = np.maximum(min_tol[None, :], pct[None, :] * np.abs(Y))
    Q = np.rint(Y / tol_ij).astype(np.int64)
    keys = [tuple(int(x) for x in row.tolist()) for row in Q]

    # 2) Group indices by key
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)

    # 3) Pick one representative per group
    kept: list[int] = []

    use_score = (X is not None) and (score_fn is not None)
    if use_score:
        X = np.asarray(X, float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_trials, n_features).")
        if X.shape[0] != n:
            raise ValueError("X must have the same number of rows as trials.")

    for idxes_list in groups.values():
        idxes = np.asarray(idxes_list, dtype=int)

        if idxes.size == 1:
            kept.append(int(idxes[0]))
            continue

        if use_score:
            scores = np.asarray(score_fn(X, idxes), float).ravel()
            if scores.size != idxes.size:
                raise ValueError("score_fn must return one score per index in idxes.")
            winner = int(idxes[int(np.argmax(scores))])
        else:
            winner = int(rng.choice(idxes))

        kept.append(winner)

    kept_idx = np.array(sorted(set(kept)), dtype=int)
    return kept_idx


def zone_id_from_Y_ranges(
    Y: np.ndarray,
    edges_per_axis: Sequence[np.ndarray],
    axes: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Assign a zone ID to each row of Y based on per-axis bin edges.

    For each selected axis, values are binned using np.digitize and then combined
    into a single integer ID (mixed radix encoding).

    Parameters
    ----------
    Y : np.ndarray
        Array of shape (n_samples, n_dims).
    edges_per_axis : Sequence[np.ndarray]
        A sequence of 1D arrays of bin edges for each selected axis.
        Each edges array must have length >= 2.
    axes : Sequence[int] | None, optional
        Indices of Y columns to use. If None, uses [0...len(edges_per_axis)-1].

    Returns
    -------
    zone : np.ndarray
        Array of shape (n_samples,) of integer zone IDs.
    """
    Y = np.asarray(Y, float)
    if Y.ndim != 2:
        raise ValueError("Y must be a 2D array of shape (n_samples, n_dims).")

    n = Y.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    if axes is None:
        axes = list(range(len(edges_per_axis)))
    if len(axes) != len(edges_per_axis):
        raise ValueError("axes and edges_per_axis must have the same length.")

    per_axis_bins: list[np.ndarray] = []
    sizes: list[int] = []

    for ax, edges in zip(axes, edges_per_axis):
        edges = np.asarray(edges, float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("Each edges array must be 1D with at least 2 values.")

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


def default_score_fn(X: np.ndarray, idxes: np.ndarray) -> np.ndarray:
    """
    Simple "informativeness" score without PCA.

    Computes the L2 norm of z-scored features for the given indices:
      score(i) = || zscore(X)[i] ||_2

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    idxes : np.ndarray
        Indices of samples to score.

    Returns
    -------
    scores : np.ndarray
        Array of shape (len(idxes),) with one score per index.
    """
    X = np.asarray(X, float)
    idxes = np.asarray(idxes, dtype=int)

    Xs = X.copy()
    mu = Xs.mean(axis=0, keepdims=True)
    sd = Xs.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    Xs = (Xs - mu) / sd

    return np.linalg.norm(Xs[idxes], axis=1)


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
) -> dict[tuple[int], list[int]]:
    """
        Build candidate-choice groups keyed by a zone ID (and optionally deduplicate first).

        Steps
        -----
        1) Optionally remove near-duplicates using `margin_trials` (based on Y tolerances).
        2) Compute a zone ID for each trial using `zone_id_from_Y_ranges` if edges are provided.
        3) Create groups: (zone_id,) -> list of global indices from the retained set.

        Parameters
        ----------
        trials : list[dict]
            Trial objects (used for length and passed to margin_trials).
        X : np.ndarray
            Feature matrix of shape (n_trials, n_features).
        Y : np.ndarray
            Output/label matrix of shape (n_trials, n_outputs).
        seed : int, default=0
            Random seed used by dedup selection (and any later random selection).
        edges_per_axis : Sequence[np.ndarray] | None, optional
            Bin edges per axis used to build zones. If None, all trials share the same zone.
        axes_for_ranges : Sequence[int] | None, optional
            Which Y columns to use for zoning. If None, uses [0...len(edges_per_axis)-1].
        pct : np.ndarray | None, optional
            Relative tolerance per Y column for deduplication. If provided, `min_tol` is required.
        min_tol : np.ndarray | None, optional
            Minimum absolute tolerance per Y column for deduplication.
        Returns
        -------
        groups : dict[tuple[int], list[int]]
            Mapping from group key (zone_id,) to a list of retained global indices.
        """

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = len(trials)

    if X.shape[0] != n or Y.shape[0] != n:
        raise ValueError("X and Y must have same number of rows as trials")

    if pct is not None:
        if min_tol is None:
            raise ValueError("min_tol must be provided when pct is not None")
        kept_idx = margin_trials(
            trials,
            Y=Y,
            pct=pct,
            min_tol=min_tol,
            X=X,
            score_fn=default_score_fn,
            seed=seed,
        )
    else:
        kept_idx = np.arange(n, dtype=int)

    kept_idx = np.asarray(kept_idx, dtype=int)

    if edges_per_axis is None:
        zone_ranges = np.zeros(n, dtype=int)
    else:
        zone_ranges = zone_id_from_Y_ranges(
            Y, edges_per_axis=edges_per_axis, axes=axes_for_ranges
        ).astype(int)

    groups: dict[tuple[int], list[int]] = {}
    for i in kept_idx.tolist():
        z = int(zone_ranges[i])
        key = (z,)
        groups.setdefault(key, []).append(int(i))

    return groups


def groups_to_list(groups: Any) -> list[list[int]]:
    """
    Normalize a grouping object into a list of index lists.

    Parameters
    ----------
    groups : Any
        Either a dict-like object (values are index lists) or an iterable of index lists.

    Returns
    -------
    groups_list : list[list[int]]
        List of groups, each group is a list of indices.
    """
    if isinstance(groups, dict):
        return [list(v) for v in groups.values()]
    return [list(g) for g in groups]


def generate_random_protocols(
    groups: Any,
    n_protocols: int,
    *,
    seed: int = 0,
) -> list[list[int]]:
    """
    Generate random protocols by selecting one index from each group.

    Parameters
    ----------
    groups : Any
        Grouping (dict or list of groups). Each group is a list of candidate indices.
    n_protocols : int
        Number of protocols to generate.
    seed : int, default=0
        Random seed.

    Returns
    -------
    protos : list[list[int]]
        List of protocols. Each protocol is a list of chosen indices, one per group.
    """
    rng = np.random.default_rng(seed)
    groups_list = groups_to_list(groups)

    protos: list[list[int]] = []
    for i in range(int(n_protocols)):
        idxes = [int(rng.choice(g)) for g in groups_list]
        protos.append(idxes)
    return protos


def count_all_protocols(groups: Any) -> int:
    """
    Count the total number of possible protocols (cartesian product of group sizes).

    Parameters
    ----------
    groups : Any
        Grouping (dict or list of groups). Each group is a list of candidate indices.

    Returns
    -------
    total : int
        Total number of possible protocols.

    Raises
    ------
    ValueError
        If at least one group is empty.
    """
    groups_list = groups_to_list(groups)
    sizes = [len(g) for g in groups_list]
    if any(s == 0 for s in sizes):
        raise ValueError("At least one group is empty.")
    return math.prod(sizes)
