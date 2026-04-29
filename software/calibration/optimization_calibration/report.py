from typing import Sequence, Any
import json
import csv
import numpy as np
from .calibration_types import ReportRow, InfluenceResult

def export_csv(rows: list[ReportRow], path: str) -> None:
    """
    Export ranked trials to CSV.

    Parameters
    ----------
    rows : list[ReportRow]
        Ranked list of trials.
    path : str
        Output CSV file path.
    Returns
    -------
    None
    """
    fieldnames = ["rank", "idx", "score", "deltaA_frob", "cook", "loo_rmse", "leverage", "file", "Mass", "Degree", "position"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for k, r in enumerate(rows, start=1):
            w.writerow({
                "rank": k,
                "idx": r.idx,
                "score": r.score,
                "deltaA_frob": r.deltaA_Frobenius,
                "cook": r.cook,
                "loo_rmse": r.loo_rmse,
                "leverage": r.leverage,
                "file": r.meta.get("__file__"),
                "Mass": r.meta.get("Mass"),
                "Degree": r.meta.get("Degree"),
                "position": r.meta.get("position"),
            })


def export_json(rows: list[ReportRow], path: str) -> None:
    """
    Export ranked trials to JSON.

    Parameters
    ----------
    rows : list[ReportRow]
        Ranked list of trials.
    path : str
        Output JSON file path.
    Returns
    -------
    None
    """
    payload = [
        {
            "rank": k,
            "idx": r.idx,
            "score": r.score,
            "deltaA_frob": r.deltaA_Frobenius,
            "cook": r.cook,
            "loo_rmse": r.loo_rmse,
            "leverage": r.leverage,
            "meta": r.meta,
        }
        for k, r in enumerate(rows, start=1)
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def rank_trials(
    influence: InfluenceResult,
    trials: Sequence[dict],
    axis_weights: np.ndarray | None = None,
    w_deltaA: float = 1.0,
    w_cook: float = 1.0,
    w_loo: float = 1.0,
) -> list[ReportRow]:
    """
    Score combiné simple:
      score = w_deltaA*z(deltaA) + w_cook*z(cook) + w_loo*z(||e_LOO||)
    """
    N = len(trials)
    deltaA = influence.delta_A_Frobenius
    cook = influence.cook_combined

    # LOO rmse (option: pondérer par axes)
    Eloo = influence.loo_residuals  # (N,6)
    if axis_weights is None:
        loo_norm = np.sqrt(np.mean(Eloo**2, axis=1))
    else:
        w = axis_weights[None, :]
        loo_norm = np.sqrt(np.mean((Eloo**2) * w, axis=1))

    def z(x):
        x = np.asarray(x, float)
        s = np.std(x)
        return (x - np.mean(x)) / (s if s > 1e-12 else 1.0)

    score = w_deltaA * z(deltaA) + w_cook * z(cook) + w_loo * z(loo_norm)

    rows: list[ReportRow] = []
    for i in range(N):
        meta = {k: trials[i].get(k) for k in ("Mass", "Degree", "__file__", "position")}
        rows.append(
            ReportRow(
                idx=i,
                score=float(score[i]),
                deltaA_Frobenius = float(deltaA[i]),
                cook=float(cook[i]),
                loo_rmse=float(loo_norm[i]),
                leverage=float(influence.leverage[i]),
                meta=meta,
            )
        )

    rows.sort(key=lambda r: r.score, reverse=True)
    return rows



def select_plateau(
    best: Sequence[tuple[float, object, list[int]]],
    portion_kept: float,
    max_kept: int,
) -> list[tuple[float, object, list[int]]]:
    """
    Select a "plateau" of top protocols based on score.

    This function keeps protocols whose score is close to the best score, using
    a relative threshold derived from `portion_kept`.

    Assumption: lower score is better.

    Parameters
    ----------
    best : Sequence[tuple[float, object, list[int]]]
        Ranked results as (score, eval, indices), sorted by ascending score.
    portion_kept : float
        Fraction of the list used to define the plateau width. Must be in (0, 1].
        The relative tolerance is computed as the score gap between the best score
        and the score at rank floor(len(best) * portion_kept) - 1.
    max_kept : int
        Maximum number of results to return. Must be >= 1.

    Returns
    -------
    plateau : list[tuple[float, object, list[int]]]
        Subset of `best` containing the plateau, capped to `max_kept`.
    """
    if len(best) == 0:
        return []

    if not (0 < portion_kept <= 1.0):
        raise ValueError("portion_kept must be in (0, 1].")
    if max_kept < 1:
        raise ValueError("max_kept must be >= 1.")

    best_score = float(best[0][0])

    # Index that defines the plateau width
    k = max(1, int(round(len(best) * portion_kept)))
    ref_idx = min(len(best) - 1, k - 1)
    ref_score = float(best[ref_idx][0])

    # Relative tolerance derived from that reference score
    tol = max(0.0, ref_score - best_score)

    # Keep everything within [best_score, best_score + tol]
    plateau = [item for item in best if float(item[0]) <= best_score + tol]

    return plateau[:max_kept]



