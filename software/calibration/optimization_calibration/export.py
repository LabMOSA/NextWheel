from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLAN_DIRS = {"FxFyMz": "FxFyMz", "MxMyFz": "MxMyFz"}


def infer_plane_from_path(file_path: Any) -> Optional[str]:
    """Infer plane name from a file path by checking known folder names."""
    if file_path is None:
        return None
    p = Path(str(file_path))
    parts = set(p.parts)
    for folder, plane in PLAN_DIRS.items():
        if folder in parts:
            return plane
    return None


def deep_get_first(d: Any, keys: Sequence[str]) -> Optional[Any]:
    """
    Get the first matching key from a dict, looking at top-level and common nested dicts.
    """
    if not isinstance(d, dict):
        return None

    # top-level
    for k in keys:
        if k in d:
            return d.get(k)

    # common nested containers
    for container_key in (
        "meta", "Meta", "info", "Info", "parameters", "Parameters", "config", "Config"
    ):
        sub = d.get(container_key, None)
        if isinstance(sub, dict):
            for k in keys:
                if k in sub:
                    return sub.get(k)

    return None


def to_float_or_none(x: Any) -> Optional[float]:
    """Convert a value to float if possible; otherwise return None."""
    if x is None:
        return None
    try:
        xf = float(np.asarray(x).ravel()[0])
        return xf if np.isfinite(xf) else None
    except Exception:
        return None


def extract_trial_meta_row(trial: dict) -> dict:
    """
    Extract (plane, mass_kg, angle_deg, file) from a trial dict.
    Tries common keys: Degree/Angle, Mass/Weight.
    """
    file_path = trial.get("__file__", None)
    plane = infer_plane_from_path(file_path)

    angle = to_float_or_none(deep_get_first(trial, keys=("Degree", "Angle", "angle_deg")))
    mass = to_float_or_none(deep_get_first(trial, keys=("Mass", "Weight", "mass_kg")))

    return {
        "plane": plane,
        "mass_kg": mass,
        "angle_deg": angle,
        "file": str(file_path) if file_path is not None else None,
    }


# -----------------------------------------------------------------------------
# Build a DataFrame from selected protocols
# -----------------------------------------------------------------------------

def best_to_trials_df(
    best: Sequence[tuple[float, Any, list[int]]],
    trial_pool: Sequence[dict],
    *,
    protocol_limit: Optional[int] = None,
    merge: str = "union",
) -> pd.DataFrame:
    """
    Build a pandas DataFrame of selected trials from `best`.

    Parameters
    ----------
    best
        Sequence of (score, eval_obj, indices) tuples.
    trial_pool
        Full pool of trial dicts, indexed by the integers in `indices`.
    protocol_limit
        If provided, only use the first N protocols from `best`.
    merge
        How to merge selected trials across protocols:
        - "union": keep each trial once (unique trial index across protocols)
        - "stack": keep all rows (a trial can appear multiple times across protocols)

    Returns
    -------
    df
        DataFrame with one row per selected trial (or per selected trial occurrence
        if merge="stack"), including protocol info.
    """
    if protocol_limit is not None:
        best = best[: int(protocol_limit)]

    rows: list[dict] = []
    seen: set[int] = set()

    for rank, (score, ev, idxs) in enumerate(best, start=1):
        for t_idx in idxs:
            ti = int(t_idx)

            if merge == "union":
                if ti in seen:
                    continue
                seen.add(ti)

            meta = extract_trial_meta_row(trial_pool[ti])
            rows.append(
                {
                    "protocol_rank": rank,
                    "protocol_score": float(score),
                    "trial_pool_index": ti,
                    **meta,
                }
            )

    df = pd.DataFrame(rows)

    # Useful cleaning: keep angle in [0, 360) if you want (optional)
    if "angle_deg" in df.columns:
        df["angle_deg"] = df["angle_deg"].astype(float, errors="ignore")
        df.loc[df["angle_deg"].notna(), "angle_deg"] = np.mod(df.loc[df["angle_deg"].notna(), "angle_deg"], 360.0)

    return df


def summarize_trials_df(df: pd.DataFrame) -> dict:
    """Quick summary stats for coverage."""
    out = {
        "n_rows": int(len(df)),
        "n_unique_trials": int(df["trial_pool_index"].nunique()) if "trial_pool_index" in df else int(len(df)),
        "n_planes": int(df["plane"].nunique(dropna=True)) if "plane" in df else 0,
        "n_unique_masses": int(df["mass_kg"].nunique(dropna=True)) if "mass_kg" in df else 0,
        "n_unique_angles": int(df["angle_deg"].nunique(dropna=True)) if "angle_deg" in df else 0,
    }
    if "plane" in df:
        out["plane_counts"] = df["plane"].fillna("Unknown").value_counts().to_dict()
    return out


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_plane_counts(df: pd.DataFrame) -> None:
    """Bar plot of plane counts."""
    s = df["plane"].fillna("Unknown").value_counts()
    plt.figure()
    plt.bar(s.index.astype(str), s.values)
    plt.xlabel("Plane")
    plt.ylabel("Count")
    plt.title("Selected trials: plane distribution")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_mass_hist(df: pd.DataFrame, *, bins: Optional[int] = None) -> None:
    """Histogram of masses (kg). If masses are discrete, you can use bins='auto'."""
    x = df["mass_kg"].dropna().to_numpy(dtype=float)
    if x.size == 0:
        print("No mass data to plot.")
        return

    plt.figure()
    plt.hist(x, bins=("auto" if bins is None else bins))
    plt.xlabel("Mass (kg)")
    plt.ylabel("Count")
    plt.title("Selected trials: mass distribution")
    plt.tight_layout()
    plt.show()


def plot_angle_hist(df: pd.DataFrame, *, bin_deg: float = 10.0) -> None:
    """Histogram of angles in degrees (0–360)."""
    x = df["angle_deg"].dropna().to_numpy(dtype=float)
    if x.size == 0:
        print("No angle data to plot.")
        return

    edges = np.arange(0.0, 360.0 + bin_deg, bin_deg)
    plt.figure()
    plt.hist(x, bins=edges)
    plt.xlabel("Angle (deg)")
    plt.ylabel("Count")
    plt.title(f"Selected trials: angle distribution (bin={bin_deg:g}°)")
    plt.xlim(0, 360)
    plt.tight_layout()
    plt.show()


def plot_angle_hist_by_plane(df: pd.DataFrame, *, bin_deg: float = 10.0) -> None:
    """One angle histogram per plane (separate figures, easy to compare)."""
    planes = df["plane"].fillna("Unknown").unique().tolist()
    edges = np.arange(0.0, 360.0 + bin_deg, bin_deg)

    for pl in planes:
        sub = df[df["plane"].fillna("Unknown") == pl]
        x = sub["angle_deg"].dropna().to_numpy(dtype=float)
        if x.size == 0:
            continue

        plt.figure()
        plt.hist(x, bins=edges)
        plt.xlabel("Angle (deg)")
        plt.ylabel("Count")
        plt.title(f"Angle distribution for plane: {pl} (bin={bin_deg:g}°)")
        plt.xlim(0, 360)
        plt.tight_layout()
        plt.show()


def plot_mass_hist_by_plane(df: pd.DataFrame, *, bins: Optional[int] = None) -> None:
    """One mass histogram per plane (separate figures)."""
    planes = df["plane"].fillna("Unknown").unique().tolist()

    for pl in planes:
        sub = df[df["plane"].fillna("Unknown") == pl]
        x = sub["mass_kg"].dropna().to_numpy(dtype=float)
        if x.size == 0:
            continue

        plt.figure()
        plt.hist(x, bins=("auto" if bins is None else bins))
        plt.xlabel("Mass (kg)")
        plt.ylabel("Count")
        plt.title(f"Mass distribution for plane: {pl}")
        plt.tight_layout()
        plt.show()


def plot_angle_mass_2d_hist(df: pd.DataFrame, *, angle_bin_deg: float = 10.0, mass_bins: Optional[int] = None) -> None:
    """
    2D histogram of (angle, mass). Useful to see coverage holes.
    """
    sub = df.dropna(subset=["angle_deg", "mass_kg"])
    if len(sub) == 0:
        print("No (angle, mass) pairs to plot.")
        return

    angles = sub["angle_deg"].to_numpy(dtype=float)
    masses = sub["mass_kg"].to_numpy(dtype=float)

    angle_edges = np.arange(0.0, 360.0 + angle_bin_deg, angle_bin_deg)
    n_mass_bins = 10 if mass_bins is None else mass_bins  # ou le nombre que tu veux


    plt.figure()
    plt.hist2d(angles, masses, bins=[angle_edges, n_mass_bins])
    plt.xlabel("Angle (deg)")
    plt.ylabel("Mass (kg)")
    plt.title("Selected trials: 2D histogram (angle × mass)")
    plt.xlim(0, 360)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# One-call API
# -----------------------------------------------------------------------------

def analyze_selected_protocols(
    best: Sequence[tuple[float, Any, list[int]]],
    trial_pool: Sequence[dict],
    *,
    protocol_limit: Optional[int] = 10,
    merge: str = "union",
    angle_bin_deg: float = 10.0,
) -> pd.DataFrame:
    """
    Build a DataFrame from selected protocols and show basic plots.

    Parameters
    ----------
    protocol_limit
        Use the first N protocols from best. Set None to use all.
    merge
        "union" or "stack".
    angle_bin_deg
        Bin width for angle histograms.

    Returns
    -------
    df
        DataFrame used for analysis (you can save it or further inspect it).
    """
    df = best_to_trials_df(best, trial_pool, protocol_limit=protocol_limit, merge=merge)

    print("Summary:", summarize_trials_df(df))

    plot_plane_counts(df)
    plot_mass_hist(df)
    plot_angle_hist(df, bin_deg=angle_bin_deg)
    plot_angle_hist_by_plane(df, bin_deg=angle_bin_deg)
    plot_mass_hist_by_plane(df)
    plot_angle_mass_2d_hist(df, angle_bin_deg=angle_bin_deg)

    return df



