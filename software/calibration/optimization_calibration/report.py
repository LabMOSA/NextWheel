from typing import Sequence
import json
import csv
import numpy as np
from .types import ReportRow, InfluenceResult

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

def select_plateau(best: Sequence[tuple], portion_kept: float, max_kept: int) -> Sequence[tuple]:
    """
    Select the plateau of top trials based on score.
    """
    if not best:
        return []
    best_score = best[0].score
    kept = [int(len(best) * portion_kept), max_kept]
    return kept

