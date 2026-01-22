from __future__ import annotations
from typing import Sequence
import json
import csv
import numpy as np
from .types import ReportRow, InfluenceResult


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
                deltaA_frob=float(deltaA[i]),
                cook=float(cook[i]),
                loo_rmse=float(loo_norm[i]),
                leverage=float(influence.leverage[i]),
                meta=meta,
            )
        )

    rows.sort(key=lambda r: r.score, reverse=True)
    return rows


def export_csv(rows: list[ReportRow], path: str) -> None:
    fieldnames = ["rank", "idx", "score", "deltaA_frob", "cook", "loo_rmse", "leverage", "file", "Mass", "Degree", "position"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for k, r in enumerate(rows, start=1):
            w.writerow({
                "rank": k,
                "idx": r.idx,
                "score": r.score,
                "deltaA_frob": r.deltaA_frob,
                "cook": r.cook,
                "loo_rmse": r.loo_rmse,
                "leverage": r.leverage,
                "file": r.meta.get("__file__"),
                "Mass": r.meta.get("Mass"),
                "Degree": r.meta.get("Degree"),
                "position": r.meta.get("position"),
            })


def export_json(rows: list[ReportRow], path: str) -> None:
    payload = [
        {
            "rank": k,
            "idx": r.idx,
            "score": r.score,
            "deltaA_frob": r.deltaA_frob,
            "cook": r.cook,
            "loo_rmse": r.loo_rmse,
            "leverage": r.leverage,
            "meta": r.meta,
        }
        for k, r in enumerate(rows, start=1)
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
