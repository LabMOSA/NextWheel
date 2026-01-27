# from __future__ import annotations
#
# from typing import Callable, Any, Sequence
# import numpy as np
#
#
# # def zone_id_from_X_pca_angle(X: np.ndarray, n_bins: int = 6) -> np.ndarray:
# #     """
# #     PCA -> 2D (via SVD), then angular bins.
# #
# #     Returns zone_id in [0..n_bins-1], one per row of X.
# #     Robust to small n: if n < 2 or PCA can't produce 2D, returns all zeros.
# #     """
# #     X = np.asarray(X, float)
# #     n = X.shape[0]
# #     if n == 0:
# #         return np.array([], dtype=int)
# #     if n < 2:
# #         return np.zeros(n, dtype=int)
# #
# #     Xc = X - X.mean(axis=0, keepdims=True)
# #
# #     _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
# #     if Vt.shape[0] < 2:
# #         return np.zeros(n, dtype=int)
# #
# #     Z = Xc @ Vt[:2].T  # (n,2)
# #     if Z.shape[1] < 2:
# #         return np.zeros(n, dtype=int)
# #
# #     ang = np.arctan2(Z[:, 1], Z[:, 0])         # [-pi, pi]
# #     ang01 = (ang + np.pi) / (2.0 * np.pi)      # [0, 1)
# #     zone_id = np.floor(ang01 * n_bins).astype(int)
# #     zone_id = np.clip(zone_id, 0, n_bins - 1)
# #     return zone_id
# #
# #
# # def zone_id_from_Y_ranges(
# #     Y: np.ndarray,
# #     edges_per_axis: Sequence[np.ndarray],
# #     axes: Sequence[int] | None = None,
# # ) -> np.ndarray:
# #     """
# #     Build a composite zone_id from per-axis numeric ranges (bins) on Y.
# #
# #     edges_per_axis: list of edges arrays, one per axis in `axes`
# #       e.g. [np.linspace(-40,40,9), np.linspace(-40,40,9), np.linspace(0,200,9)]
# #     axes: which columns of Y to use. If None, uses first len(edges_per_axis) axes.
# #
# #     Returns zone_id in [0..prod(n_bins_axis)-1], one per row of Y.
# #     """
# #     Y = np.asarray(Y, float)
# #     n = Y.shape[0]
# #     if n == 0:
# #         return np.array([], dtype=int)
# #
# #     if axes is None:
# #         axes = list(range(len(edges_per_axis)))
# #     if len(axes) != len(edges_per_axis):
# #         raise ValueError("axes and edges_per_axis must have the same length")
# #
# #     # Per-axis bin ids
# #     per_axis_bins: list[np.ndarray] = []
# #     sizes: list[int] = []
# #     for ax, edges in zip(axes, edges_per_axis):
# #         edges = np.asarray(edges, float)
# #         if edges.ndim != 1 or edges.size < 2:
# #             raise ValueError("Each edges array must be 1D with at least 2 values")
# #
# #         z = np.digitize(Y[:, ax], edges, right=False) - 1  # -> [0..len(edges)-2] ideally
# #         z = np.clip(z, 0, edges.size - 2)
# #         per_axis_bins.append(z.astype(int))
# #         sizes.append(int(edges.size - 1))  # number of bins
# #
# #     # Mixed-radix encoding: zone = z0 + z1*s0 + z2*s0*s1 + ...
# #     zone = np.zeros(n, dtype=int)
# #     mult = 1
# #     for z, s in zip(per_axis_bins, sizes):
# #         zone += z * mult
# #         mult *= s
# #
# #     return zone
# #
# #
# # def cap_trials_per_zone_mass_pca_and_ranges(
# #     trials: list[dict],
# #     X: np.ndarray,
# #     Y: np.ndarray,
# #     k_per_key: int,
# #     mass_fn: Callable[[dict], Any],
# #     seed: int = 0,
# #     # PCA zoning (on X):
# #     pca_bins: int = 6,
# #     # Range zoning (on Y):
# #     edges_per_axis: Sequence[np.ndarray] | None = None,
# #     axes_for_ranges: Sequence[int] | None = None,
# #     # Optional: if you already computed zone ids, pass them
# #     zone_pca: np.ndarray | None = None,
# #     zone_ranges: np.ndarray | None = None,
# # ) -> list[dict]:
# #     """
# #     Keep at most k_per_key trials per (mass, PCA_zone, RANGE_zone).
# #
# #     - PCA_zone comes from X via PCA->2D->angular bins (captures "sensor-space geometry").
# #     - RANGE_zone comes from Y via per-axis bins (captures "physical excitation coverage").
# #     - Combining both avoids selecting many redundant points and also forces coverage of desired ranges.
# #
# #     Parameters
# #     ----------
# #     trials : list[dict]
# #         Trials list.
# #     X : np.ndarray
# #         Feature matrix aligned with trials, shape (n_trials, n_features).
# #     Y : np.ndarray
# #         Target matrix aligned with trials, shape (n_trials, n_axes) (e.g., FM).
# #     k_per_key : int
# #         Max number kept per bucket.
# #     mass_fn : Callable
# #         Extracts mass value from a trial (e.g., lambda t: t["Mass"]).
# #     edges_per_axis : Sequence[np.ndarray] | None
# #         Defines the desired ranges/bins on Y. If None, range-zoning is disabled (all in one range zone).
# #     axes_for_ranges : Sequence[int] | None
# #         Which Y axes the edges correspond to. If None, uses first len(edges_per_axis).
# #
# #     Returns
# #     -------
# #     list[dict]
# #         Selected trials.
# #     """
# #     if k_per_key is None:
# #         return trials
# #     k_per_key = int(k_per_key)
# #     if k_per_key <= 0:
# #         return []
# #
# #     X = np.asarray(X, float)
# #     Y = np.asarray(Y, float)
# #     if X.shape[0] != len(trials):
# #         raise ValueError(f"X rows ({X.shape[0]}) must match trials ({len(trials)})")
# #     if Y.shape[0] != len(trials):
# #         raise ValueError(f"Y rows ({Y.shape[0]}) must match trials ({len(trials)})")
# #
# #     rng = np.random.default_rng(seed)
# #
# #     # --- PCA zone on X ---
# #     if zone_pca is None:
# #         zone_pca = zone_id_from_X_pca_angle(X, n_bins=pca_bins)
# #     zone_pca = np.asarray(zone_pca, dtype=int)
# #     if zone_pca.shape[0] != len(trials):
# #         raise ValueError("zone_pca must have shape (n_trials,) and match trials length")
# #
# #     # --- Range zone on Y ---
# #     if edges_per_axis is None:
# #         # single "range zone" if user didn't specify bins
# #         zone_ranges = np.zeros(len(trials), dtype=int)
# #     else:
# #         if zone_ranges is None:
# #             zone_ranges = zone_id_from_Y_ranges(Y, edges_per_axis=edges_per_axis, axes=axes_for_ranges)
# #         zone_ranges = np.asarray(zone_ranges, dtype=int)
# #         if zone_ranges.shape[0] != len(trials):
# #             raise ValueError("zone_ranges must have shape (n_trials,) and match trials length")
# #
# #     # --- Bucket by (mass, pca_zone, range_zone) ---
# #     buckets: dict[tuple[Any, int, int], list[int]] = {}
# #     for i, t in enumerate(trials):
# #         key = (mass_fn(t), int(zone_pca[i]), int(zone_ranges[i]))
# #         buckets.setdefault(key, []).append(i)
# #
# #     kept: list[int] = []
# #     for idxs in buckets.values():
# #         idxs = np.asarray(idxs, dtype=int)
# #         rng.shuffle(idxs)
# #         kept.extend(idxs[: min(k_per_key, len(idxs))].tolist())
# #
# #     kept = sorted(set(kept))
# #     return [trials[i] for i in kept]
#
#
#
from __future__ import annotations

from typing import Callable, Any, Sequence
import numpy as np


def zone_id_from_X_pca_angle_radius(X: np.ndarray, n_ang: int = 6, n_rad: int = 3) -> np.ndarray:
    """
    PCA -> 2D (via SVD), then zoning by:
      - angle bins (n_ang)
      - radius bins (n_rad), using quantiles so bins are populated

    Returns a single zone_id in [0 .. n_ang*n_rad - 1] per row of X.

    Robust to small/degenerate inputs: if PCA can't make 2D or n<2, returns all zeros.
    """
    X = np.asarray(X, float)
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if n < 2:
        return np.zeros(n, dtype=int)

    Xc = X - X.mean(axis=0, keepdims=True)

    # PCA via SVD
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    if Vt.shape[0] < 2:
        return np.zeros(n, dtype=int)

    Z = Xc @ Vt[:2].T  # (n,2)
    if Z.shape[1] < 2:
        return np.zeros(n, dtype=int)

    # --- angle bin ---
    ang = np.arctan2(Z[:, 1], Z[:, 0])               # [-pi, pi]
    ang01 = (ang + np.pi) / (2.0 * np.pi)            # [0,1)
    z_ang = np.floor(ang01 * n_ang).astype(int)
    z_ang = np.clip(z_ang, 0, n_ang - 1)

    # --- radius bin (quantiles) ---
    r = np.sqrt(Z[:, 0] ** 2 + Z[:, 1] ** 2)

    # If all radii equal, everything in the same radial bin
    if np.allclose(r, r[0]):
        z_rad = np.zeros(n, dtype=int)
    else:
        qs = np.linspace(0.0, 1.0, n_rad + 1)
        edges = np.quantile(r, qs)

        # Remove duplicate edges (can happen if many radii are equal)
        edges = np.unique(edges)
        if edges.size < 2:
            z_rad = np.zeros(n, dtype=int)
        else:
            z_rad = np.digitize(r, edges, right=False) - 1
            z_rad = np.clip(z_rad, 0, edges.size - 2)
            # If edges collapsed, keep within [0..n_rad-1]
            z_rad = np.minimum(z_rad, n_rad - 1).astype(int)

    # Combine bins
    return (z_rad * n_ang + z_ang).astype(int)


def zone_id_from_Y_ranges(
    Y: np.ndarray,
    edges_per_axis: Sequence[np.ndarray],
    axes: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Composite zone_id from per-axis numeric ranges (bins) on Y.

    Returns zone_id in [0..prod(n_bins_axis)-1], one per row of Y.
    """
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


# def cap_trials_per_zone_mass_pca_and_ranges(
#     trials: list[dict],
#     X: np.ndarray,
#     Y: np.ndarray,
#     k_per_key: int,
#     mass_fn: Callable[[dict], Any],
#     seed: int = 0,
#     # PCA zoning (on X): angle+radius bins
#     pca_n_ang: int = 6,
#     pca_n_rad: int = 3,
#     # Range zoning (on Y):
#     edges_per_axis: Sequence[np.ndarray] | None = None,
#     axes_for_ranges: Sequence[int] | None = None,
#     # Optional: if you already computed zone ids, pass them
#     zone_pca: np.ndarray | None = None,
#     zone_ranges: np.ndarray | None = None,
# ) -> list[dict]:
#     """
#     Keep at most k_per_key trials per (mass, PCA_zone, RANGE_zone).
#
#     PCA_zone:
#       - computed from X by PCA->2D then (angle bins + radius bins).
#       - better than angle-only for spiral/annulus-shaped clouds.
#
#     RANGE_zone:
#       - computed from Y by per-axis bins (physical excitation coverage).
#
#     Combining both avoids selecting many redundant points AND forces coverage of desired ranges.
#     """
#     if k_per_key is None:
#         return trials
#     k_per_key = int(k_per_key)
#     if k_per_key <= 0:
#         return []
#
#     X = np.asarray(X, float)
#     Y = np.asarray(Y, float)
#
#     if X.shape[0] != len(trials):
#         raise ValueError(f"X rows ({X.shape[0]}) must match trials ({len(trials)})")
#     if Y.shape[0] != len(trials):
#         raise ValueError(f"Y rows ({Y.shape[0]}) must match trials ({len(trials)})")
#
#     rng = np.random.default_rng(seed)
#
#     # --- PCA zone on X (angle+radius) ---
#     if zone_pca is None:
#         zone_pca = zone_id_from_X_pca_angle_radius(X, n_ang=pca_n_ang, n_rad=pca_n_rad)
#     zone_pca = np.asarray(zone_pca, dtype=int)
#     if zone_pca.shape[0] != len(trials):
#         raise ValueError("zone_pca must have shape (n_trials,) and match trials length")
#
#     # --- Range zone on Y ---
#     if edges_per_axis is None:
#         zone_ranges = np.zeros(len(trials), dtype=int)
#     else:
#         if zone_ranges is None:
#             zone_ranges = zone_id_from_Y_ranges(
#                 Y, edges_per_axis=edges_per_axis, axes=axes_for_ranges
#             )
#         zone_ranges = np.asarray(zone_ranges, dtype=int)
#         if zone_ranges.shape[0] != len(trials):
#             raise ValueError("zone_ranges must have shape (n_trials,) and match trials length")
#
#     # --- Bucket by (mass, pca_zone, range_zone) ---
#     buckets: dict[tuple[Any, int, int], list[int]] = {}
#     for i, t in enumerate(trials):
#         key = (mass_fn(t), int(zone_pca[i]), int(zone_ranges[i]))
#         buckets.setdefault(key, []).append(i)
#
#     kept: list[int] = []
#     for idxs in buckets.values():
#         idxs = np.asarray(idxs, dtype=int)
#         rng.shuffle(idxs)
#         kept.extend(idxs[: min(k_per_key, len(idxs))].tolist())
#
#     kept = sorted(set(kept))
#     return [trials[i] for i in kept]

def cap_trials_per_zone_mass_pca_and_ranges(
    trials: list[dict],
    X: np.ndarray,
    Y: np.ndarray,
    k_per_key: int,
    mass_fn: Callable[[dict], Any],
    seed: int = 0,
    pca_n_ang: int = 6,
    pca_n_rad: int = 3,
    edges_per_axis: Sequence[np.ndarray] | None = None,
    axes_for_ranges: Sequence[int] | None = None,
    zone_pca: np.ndarray | None = None,
    zone_ranges: np.ndarray | None = None,
    # NEW caps:
    max_per_pca_zone: int | None = None,
    max_total: int | None = None,
) -> list[dict]:
    if k_per_key is None:
        return trials
    k_per_key = int(k_per_key)
    if k_per_key <= 0:
        return []

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if X.shape[0] != len(trials):
        raise ValueError(f"X rows ({X.shape[0]}) must match trials ({len(trials)})")
    if Y.shape[0] != len(trials):
        raise ValueError(f"Y rows ({Y.shape[0]}) must match trials ({len(trials)})")

    rng = np.random.default_rng(seed)

    # --- PCA zones ---
    if zone_pca is None:
        zone_pca = zone_id_from_X_pca_angle_radius(X, n_ang=pca_n_ang, n_rad=pca_n_rad)
    zone_pca = np.asarray(zone_pca, dtype=int)
    if zone_pca.shape[0] != len(trials):
        raise ValueError("zone_pca must match trials length")

    # --- Range zones ---
    if edges_per_axis is None:
        zone_ranges = np.zeros(len(trials), dtype=int)
    else:
        if zone_ranges is None:
            zone_ranges = zone_id_from_Y_ranges(Y, edges_per_axis=edges_per_axis, axes=axes_for_ranges)
        zone_ranges = np.asarray(zone_ranges, dtype=int)
        if zone_ranges.shape[0] != len(trials):
            raise ValueError("zone_ranges must match trials length")

    # --- Stage 1: fine buckets (mass, pca, ranges) ---
    buckets: dict[tuple[Any, int, int], list[int]] = {}
    for i, t in enumerate(trials):
        key = (mass_fn(t), int(zone_pca[i]), int(zone_ranges[i]))
        buckets.setdefault(key, []).append(i)

    kept_idx: list[int] = []
    for idxs in buckets.values():
        idxs = np.asarray(idxs, dtype=int)
        rng.shuffle(idxs)
        kept_idx.extend(idxs[: min(k_per_key, len(idxs))].tolist())

    kept_idx = sorted(set(kept_idx))

    # --- Stage 2a: cap per PCA zone (ignoring mass/ranges) ---
    if max_per_pca_zone is not None:
        max_per_pca_zone = int(max_per_pca_zone)
        if max_per_pca_zone <= 0:
            return []

        by_zone: dict[int, list[int]] = {}
        for i in kept_idx:
            z = int(zone_pca[i])
            by_zone.setdefault(z, []).append(i)

        new_kept: list[int] = []
        for z, idxs in by_zone.items():
            idxs = np.asarray(idxs, dtype=int)
            rng.shuffle(idxs)
            new_kept.extend(idxs[: min(max_per_pca_zone, len(idxs))].tolist())

        kept_idx = sorted(set(new_kept))

    # --- Stage 2b: cap total ---
    if max_total is not None:
        max_total = int(max_total)
        if max_total <= 0:
            return []
        if len(kept_idx) > max_total:
            idxs = np.asarray(kept_idx, dtype=int)
            rng.shuffle(idxs)
            kept_idx = sorted(idxs[:max_total].tolist())

    return [trials[i] for i in kept_idx]

