from pathlib import Path
from typing import Any, Optional, Sequence, Dict, List
import numpy as np
import kineticstoolkit as ktk
from software.calibration.optimization_calibration.types import ProtocolEval, TrialMeta

PLAN_DIRS = {"FxFyMz": "FxFyMz", "MxMyFz": "MxMyFz"}
def infer_plane_from_path(file_path: Any) -> Optional[str]:
    """
    Infer the plane name from a file path.
    Parameters
    ----------
    file_path : Any
        A path-like object (str, Path, or convertible to str). May be None.
    Returns
    -------
    plane : str | None
        "FxFyMz" or "MxMyFz" if a matching folder is found in the path, otherwise None.
    """
    if file_path is None:
        return None
    p = Path(str(file_path))
    parts = set(p.parts)
    for folder, plane in PLAN_DIRS.items():
        if folder in parts:
            return plane
    return None
def _to_float_or_none(x: Any) -> Optional[float]:
    """
    Robustly convert a value to a float.

    Parameters
    ----------
    x : Any
        Value to convert.
    Returns
    -------
    out : float | None
        Converted float value if possible, otherwise None.
        """
    if x is None:
        return None
    try:
        xf = float(np.asarray(x).ravel()[0])
        return xf if np.isfinite(xf) else None
    except Exception:
        return None

def extract_trial_meta(trial: dict) -> TrialMeta:
    """
    Extract plane, angle, and weight metadata from a trial and return a TrialMeta.
    Parameters
    ----------
    trial : dict
        trials
    Returns
    -------
    meta : TrialMeta
        Extracted metadata (file, plane, angle_deg, weight_kg).
    """
    file_path = trial.get("__file__", None)
    plane = infer_plane_from_path(file_path)
    angle = _to_float_or_none(trial["Degree"])
    weight = _to_float_or_none(trial["Mass"])

    return TrialMeta(
        file=str(file_path) if file_path is not None else None,
        plane=plane,
        angle_deg=angle,
        weight_kg=weight,
    )


# ----------------------------
# Pretty formatting
# ----------------------------

def _fmt_angle(angle_deg: Optional[float]) -> str:
    if angle_deg is None:
        return "N/A"
    # angle often integer-ish: 30.0 -> 30
    a = float(angle_deg)
    if abs(a - round(a)) < 1e-9:
        return f"{int(round(a))}°"
    return f"{a:.1f}°"

def _fmt_weight(weight_kg: Optional[float]) -> str:
    if weight_kg is None:
        return "N/A"
    w = float(weight_kg)
    # 1.7 -> 1.70 ; 11.9 -> 11.90
    return f"{w:.2f} kg"

def _fmt_plane(plane: Optional[str]) -> str:
    return plane if plane is not None else "Unknown"


# ----------------------------
# Main: list each trial
# ----------------------------

def list_trials_pretty(trials: Sequence[dict], *, show_file: bool = False) -> List[str]:
    """
    Returns:
        ["Trial 1 : Plan FxFyMz | Poids : 1.73 kg | Angle : 30°", ...]
    """
    cache: Dict[str, Optional[dict]] = {}
    out: List[str] = []

    for i, t in enumerate(trials, start=1):
        meta = extract_trial_meta(t)

        s = (
            f"Trial {i} : "
            f"Plan {_fmt_plane(meta.plane)} | "
            f"Poids : {_fmt_weight(meta.weight_kg)} | "
            f"Angle : {_fmt_angle(meta.angle_deg)}"
        )
        if show_file and meta.file:
            s += f" | file={meta.file}"
        out.append(s)

    return out


def list_trials_rows(trials: Sequence[dict]) -> List[dict]:
    """
    Build a structured representation (one dict per trial), convenient for
    DataFrame/CSV export.

    Parameters
    ----------
    trials : Sequence[dict]
        Collection of trial dictionaries.

    Returns
    -------
    rows : list[dict]
        List of dictionaries with keys:
        - "trial", "plane", "weight_kg", "angle_deg", "file"
    """
    cache: Dict[str, Optional[dict]] = {}
    rows: List[dict] = []
    for i, t in enumerate(trials, start=1):
        meta = extract_trial_meta(t)
        rows.append(
            {
                "trial": i,
                "plane": meta.plane,
                "weight_kg": meta.weight_kg,
                "angle_deg": meta.angle_deg,
                "file": meta.file,
            }
        )
    return rows

def print_trials_table(
    trials: Sequence[dict],
    *,
    show_file: bool = False,
    sort_by: tuple[str, ...] = ("plane", "angle_deg", "weight_kg"),
) -> None:
    """
    Affiche un tableau aligné pour comparer facilement.
    Tri par défaut: plan -> angle -> poids
    """
    rows = list_trials_rows(trials)

    # tri
    def _key(r: dict):
        keys = []
        for k in sort_by:
            v = r.get(k, None)
            # mettre None à la fin
            if v is None:
                keys.append((1, 0))
            else:
                keys.append((0, v))
        return tuple(keys)

    rows.sort(key=_key)

    # formatage champs
    def fmt_plane(x): return x if x is not None else "Unknown"
    def fmt_angle(x):
        if x is None: return "N/A"
        if abs(x - round(x)) < 1e-9: return f"{int(round(x))}°"
        return f"{x:.1f}°"
    def fmt_weight(x):
        if x is None: return "N/A"
        return f"{x:.2f}"
    def fmt_file(x):
        if not x: return ""
        return Path(x).name  # juste le nom du fichier pour alléger

    # construire lignes imprimables
    table = []
    for r in rows:
        table.append({
            "Trial": str(r["trial"]),
            "Plan": fmt_plane(r["plane"]),
            "Angle": fmt_angle(r["angle_deg"]),
            "Poids(kg)": fmt_weight(r["weight_kg"]),
            "Fichier": fmt_file(r["file"]) if show_file else "",
        })

    # colonnes
    cols = ["Trial", "Plan", "Angle", "Poids(kg)"] + (["Fichier"] if show_file else [])
    widths = {c: max(len(c), max(len(row[c]) for row in table)) for c in cols}

    # print header
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)

    # print rows
    for row in table:
        print(" | ".join(row[c].ljust(widths[c]) for c in cols))


def print_best_protocol_trials(
    best: Sequence[tuple[float, ProtocolEval, list[int]]],
    trial_pool: Sequence[dict],
    *,
    max_protocols: int | None = None,
    show_file: bool = False,
) -> None:
    """
    Affiche, pour chaque protocole best, la table des essais (chosen) seulement.
    """
    items = list(best)
    if max_protocols is not None:
        items = items[: int(max_protocols)]

    for rank, (score, ev, indexes) in enumerate(items, start=1):
        chosen = [trial_pool[int(i)] for i in indexes]

        print("\n" + "=" * 80)
        print(f"BEST #{rank} | score={score:.6g} | rmse={ev.rmse_total:.6g} | cv={ev.A_cv_mean:.6g} | n={len(indexes)}")
        print("=" * 80)
        print_trials_table(chosen, show_file=show_file)
