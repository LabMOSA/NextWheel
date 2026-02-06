from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Sequence, Any, List
from pathlib import Path
import csv
import json
import numpy as np

from software.calibration.optimization_2_planes.export import export_protocols_csv, export_protocols_json
from software.calibration.optimization_calibration.types import (
    FitConfig,
    MonteCarloConfig,
    ProtocolSpec,
    ProtocolEval, ProtocolSpecZone,
)

from software.calibration.optimization_calibration.features import build_XY, position_from_imu_gravity
from software.calibration.optimization_calibration.fit import fit
from software.calibration.optimization_calibration.metrics import (
    RMSE_total,
    RMSE_per_axis,
    R2_total,
    R2_per_axis,
)
from software.calibration.optimization_calibration.montecarlo import bootstrap_A
from software.calibration.optimization_calibration.influence import influence_analytic
from software.calibration.optimization_calibration.report import rank_trials

from software.calibration.fit_A_train import build_FMs_Channels



# # ----------------------------
# # Protocol evaluation
# # ----------------------------

def _finite_or_nan(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")

def _get(trial: dict, key: str, default=None):
    try:
        return trial.get(key, default)
    except Exception:
        return default

def default_trial_id(trial: dict):
    # identifiant stable (chez toi: __file__)
    return _get(trial, "__file__", None)

def evaluate_protocol(
    trials: Sequence[dict],
    chosen: Sequence[dict],
    spec,
    acc_bias: np.ndarray,
    base: np.ndarray,
    fit_cfg,
    mc_cfg,
    test_ratio: float = 0.2,
    seed: int = 0,
    trial_id_fn: Callable[[dict], Any] = default_trial_id,
) -> ProtocolEval:
    chosen = list(chosen)
    n_total = len(chosen)

    # Guard: enough total
    if n_total < spec.min_trials:
        return ProtocolEval(
            name=spec.name,
            n_total=n_total,
            n_train=0,
            n_test=0,
            rmse_total=float("nan"),
            r2_total=float("nan"),
            rmse_per_axis=[float("nan")] * 6,
            r2_per_axis=[float("nan")] * 6,
            A_std_mean=float("nan"),
            A_std_max=float("nan"),
            A_cv_mean=float("nan"),
            A_cv_max=float("nan"),
            influence_top_score=None,
            influence_top_idx=None,
            influence_top_file=None,
            notes=f"Not enough trials after zone+mass cap: {n_total} < min_trials={spec.min_trials}",
        )

    # --- split ---
    rng = np.random.default_rng(seed)
    idx = np.arange(n_total)

    # --- TRAIN = chosen ---
    train_trials = chosen

    # --- TEST = trials \ chosen ---
    chosen_ids = set()
    for t in train_trials:
        chosen_ids.add(trial_id_fn(t))

    test_trials: list[dict] = []
    for t in trials:
        if trial_id_fn(t) not in chosen_ids:
            test_trials.append(t)

    # Guard: enough train
    min_train = 6
    if len(train_trials) < min_train:
        return ProtocolEval(
            name=spec.name,
            n_total=n_total,
            n_train=len(train_trials),
            n_test=len(test_trials),
            rmse_total=float("nan"),
            r2_total=float("nan"),
            rmse_per_axis=[float("nan")] * 6,
            r2_per_axis=[float("nan")] * 6,
            A_std_mean=float("nan"),
            A_std_max=float("nan"),
            A_cv_mean=float("nan"),
            A_cv_max=float("nan"),
            influence_top_score=None,
            influence_top_idx=None,
            influence_top_file=None,
            notes=f"Not enough TRAIN trials after zone+mass cap: {len(train_trials)} < {min_train}",
        )

    # --- 3) build X/Y ---
    try:
        X_train, Y_train, y_scale_train = build_XY(train_trials, acc_bias, base)
        X_test, Y_test, y_scale_test = build_XY(test_trials, acc_bias, base)
    except TypeError:
        X_train, Y_train, y_scale_train = build_XY(train_trials, acc_bias, base)
        X_test, Y_test, y_scale_test = build_XY(test_trials, acc_bias, base)

    X_train = np.asarray(X_train, float)
    Y_train = np.asarray(Y_train, float)
    X_test = np.asarray(X_test, float)
    Y_test = np.asarray(Y_test, float)

    # --- 4) fit ---
    fr = fit(X_train, Y_train, fit_cfg)
    A = fr.A
    b0 = getattr(fr, "b0", None)
    Y_pred = X_test @ A.T
    if b0 is not None:
        Y_pred = Y_pred + b0[None, :]

    # --- 6) metrics ---
    rmse_total = _finite_or_nan(RMSE_total(Y_test, Y_pred))
    r2_total = _finite_or_nan(R2_total(Y_test, Y_pred))
    rmse_per_axis = [float(x) for x in np.asarray(RMSE_per_axis(Y_test, Y_pred)).ravel().tolist()]
    r2_per_axis = [float(x) for x in np.asarray(R2_per_axis(Y_test, Y_pred)).ravel().tolist()]

    if not np.isfinite(rmse_total):
        return ProtocolEval(
            name=spec.name,
            n_total=n_total,
            n_train=len(train_trials),
            n_test=len(test_trials),
            rmse_total=float("nan"),
            r2_total=float("nan"),
            rmse_per_axis=[float("nan")] * 6,
            r2_per_axis=[float("nan")] * 6,
            A_std_mean=float("nan"),
            A_std_max=float("nan"),
            A_cv_mean=float("nan"),
            A_cv_max=float("nan"),
            influence_top_score=None,
            influence_top_idx=None,
            influence_top_file=None,
            notes="Non-finite RMSE (fit likely ill-conditioned for this protocol).",
        )
    # --- 7) bootstrap stability on train ---
    boot = bootstrap_A(
        train_trials,
        acc_bias,
        base,
        builder=build_FMs_Channels,
        fit_configuration=fit_cfg,
        montecarlo_configuration=mc_cfg,
    )

    A_mean = np.asarray(boot.A_mean, float)
    A_std = np.asarray(boot.A_std, float)

    A_std_mean = float(np.mean(A_std))
    A_std_max = float(np.max(A_std))

    denom = np.abs(A_mean)
    mask = denom > 1e-12
    cv = np.zeros_like(A_std)
    cv[mask] = A_std[mask] / denom[mask]
    A_cv_mean = float(np.mean(cv[mask])) if np.any(mask) else float("nan")
    A_cv_max = float(np.max(cv[mask])) if np.any(mask) else float("nan")

    # --- 8) influence ranking on train ---
    influence = influence_analytic(X=X_train, Y=Y_train, A=A, b0=b0)
    rows = rank_trials(influence, train_trials)

    top_score = top_idx = None
    top_file = None
    if rows:
        top_score = float(rows[0].score)
        top_idx = int(rows[0].idx)
        top_file = str(rows[0].meta.get("__file__")) if rows[0].meta else None

    return ProtocolEval(
        name=spec.name,
        n_total=n_total,
        n_train=len(train_trials),
        n_test=len(test_trials),
        rmse_total=rmse_total,
        r2_total=r2_total,
        rmse_per_axis=rmse_per_axis,
        r2_per_axis=r2_per_axis,
        A_std_mean=A_std_mean,
        A_std_max=A_std_max,
        A_cv_mean=A_cv_mean,
        A_cv_max=A_cv_max,
        influence_top_score=top_score,
        influence_top_idx=top_idx,
        influence_top_file=top_file,
        notes="",
    )

# ----------------------------
# Search / ranking protocols
# ----------------------------


def protocol_score(
    eval_: ProtocolEval,
    alpha_rmse: float = 1.0,
    beta_cv: float = 1.0,
    gamma_dom: float = 0.0,
) -> float:
    """
    Lower is better.

    score = alpha*RMSE_test + beta*A_cv_mean + gamma*(dominance penalty)

    Notes
    -----
    - A_cv_mean can explode if many coefficients have near-zero mean. We already mask small means.
    - If you find CV dominates too much, reduce beta_cv or use A_std_mean instead.
    """
    if not np.isfinite(eval_.rmse_total) or not np.isfinite(eval_.A_cv_mean):
        return float("inf")

    dom = eval_.influence_top_score if eval_.influence_top_score is not None else 0.0
    return float(
        alpha_rmse * eval_.rmse_total
        + beta_cv * eval_.A_cv_mean
        + gamma_dom * float(dom)
    )
AXES_6 = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
def summarize_protocols(best: Sequence[tuple]) -> list[dict]:
    """
    best: liste de (score, ProtocolEval) ou (score, ProtocolEval, idxs)

    Retour: liste de dicts avec score + rmse_total (+ rmse par axe si dispo)
    """
    rows: list[dict] = []
    for rank, item in enumerate(best, start=1):
        if len(item) == 2:
            score, ev = item
            idxs = None
        else:
            score, ev, idxs = item

        row = {
            "rank": rank,
            "score": float(score),
            "name": getattr(ev, "name", ""),
            "n_total": int(getattr(ev, "n_total", -1)),
            "n_train": int(getattr(ev, "n_train", -1)),
            "n_test": int(getattr(ev, "n_test", -1)),
            "rmse_total": float(getattr(ev, "rmse_total", float("nan"))),
            "r2_total": float(getattr(ev, "r2_total", float("nan"))),
            "A_cv_mean": float(getattr(ev, "A_cv_mean", float("nan"))),
        }

        rmse_axes = getattr(ev, "rmse_per_axis", None)
        if rmse_axes is not None:
            ra = np.asarray(rmse_axes, float).ravel()
            if ra.size == 6:
                for name, val in zip(AXES_6, ra):
                    row[f"rmse_{name}"] = float(val)

        if idxs is not None:
            row["protocol_len"] = len(idxs)
        rows.append(row)

    return rows

def search_best_protocols_from_indices(
    trial_pool: Sequence[dict],                 # all_trials
    protocol_indices: Sequence[Sequence[int]],  # [[i0,i1,...], [j0,j1,...], ...]
    spec,
    acc_bias: np.ndarray,
    base: np.ndarray,
    fit_cfg: FitConfig,
    mc_cfg: MonteCarloConfig,
    *,
    top_n: int = 20,
    alpha_rmse: float = 1.0,
    beta_cv: float = 1.0,
    gamma_dom: float = 0.0,
    seed: int = 0,
    test_ratio: float = 0.2,
    split_seed_stride: int = 1,
) -> list[tuple[float, ProtocolEval, list[int]]]:
    results: list[tuple[float, ProtocolEval, list[int]]] = []

    for k, idxs in enumerate(protocol_indices):
        chosen = [trial_pool[int(i)] for i in idxs]

        ev = evaluate_protocol(
            trials= trial_pool,
            chosen=chosen,
            spec=spec,
            acc_bias=acc_bias,
            base=base,
            fit_cfg=fit_cfg,
            mc_cfg=mc_cfg,
            test_ratio=test_ratio,
            seed=seed + split_seed_stride * k,
        )

        if not np.isfinite(ev.rmse_total) or not np.isfinite(ev.A_cv_mean):
            continue

        s = protocol_score(ev, alpha_rmse=alpha_rmse, beta_cv=beta_cv, gamma_dom=gamma_dom)
        if not np.isfinite(s):
            continue

        results.append((float(s), ev, list(map(int, idxs))))

    results.sort(key=lambda x: x[0])
    return results[: int(top_n)]


# # ----------------------------
# # One-call helper
# # ----------------------------
#
#
# def recommend_protocol(
#     trials: Sequence[dict],
#     acc_bias: np.ndarray,
#     base: np.ndarray,
#     fit_cfg: FitConfig,
#     mc_cfg: MonteCarloConfig,
#     out_dir: str | Path | None = None,
#     seed: int = 0,
#     min_trials: int = 20,
#     top_n: int = 5,
#     alpha_rmse: float = 1.0,
#     beta_cv: float = 1.0,
#     gamma_dom: float = 0.0,
# ) -> tuple[list[tuple[float, ProtocolEval]], list[ProtocolSpec]]:
#     """
#     Generate realistic candidate protocols, evaluate them, and return the best ones.
#     Optionally export CSV/JSON.
#
#     Defaults are set to avoid "ubuesque" scenarios:
#       - min_trials=20
#       - invalid protocols are discarded automatically
#     """
#     cands = generate_protocol_candidates(trials, min_trials=min_trials)
#
#     best = search_best_protocols(
#         trials=trials,
#         acc_bias=acc_bias,
#         base=base,
#         fit_cfg=fit_cfg,
#         mc_cfg=mc_cfg,
#         top_n=top_n,
#         alpha_rmse=alpha_rmse,
#         beta_cv=beta_cv,
#         gamma_dom=gamma_dom,
#         candidates=cands,
#         seed=seed,
#     )
#
#     if out_dir is not None:
#         out_dir = Path(out_dir)
#         export_protocols_csv(best, str(out_dir / "protocol_ranking.csv"))
#         export_protocols_json(best, str(out_dir / "protocol_ranking.json"))
#
#     return best, cands
