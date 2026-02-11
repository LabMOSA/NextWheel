from typing import Callable, Sequence, Any
import numpy as np

from software.calibration.optimization_calibration.recommand_protocol import list_trials_pretty, print_trials_table
from software.calibration.optimization_calibration.types import (
    FitConfig,
    MonteCarloConfig,
    ProtocolEval
)
from software.calibration.optimization_calibration.features import build_XY
from software.calibration.optimization_calibration.fit import fit
from software.calibration.optimization_calibration.metrics import rmse_total, r2_total, rmse_per_axis, r2_per_axis
from software.calibration.optimization_calibration.montecarlo import bootstrap_A
from software.calibration.optimization_calibration.influence import influence_analytic
from software.calibration.optimization_calibration.report import rank_trials

AXES_6 = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")

# # ----------------------------
# # Protocol evaluation
# # ----------------------------

def _finite_or_nan(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")

def _get(trial: dict, key: str, default=None):
        return trial.get(key, default)


def default_trial_id(trial: dict):
    return _get(trial, "__file__", None)

def evaluate_protocol(
    trials: Sequence[dict],
    chosen: Sequence[dict],
    spec,
    acc_bias: np.ndarray,
    base: np.ndarray,
    fit_configuration,
    montecarlo_configuration,
    trial_id_fn: Callable[[dict], Any] = default_trial_id,
) -> ProtocolEval:
    """
    Evaluate a protocol defined by the chosen trials.

    Parameters
    ----------
    trials : Sequence[dict]
        The full set of trials to consider for testing (including the chosen ones).
    chosen : Sequence[dict]
        The subset of trials that define the protocol (used for training).
    spec : Any
        A specification object containing at least a 'name' attribute for the protocol.
    acc_bias : np.ndarray
        The accelerometer bias to use when building the feature matrices.
    base : np.ndarray
        The base values to use when building the feature matrices.
    fit_configuration : FitConfig
        Configuration for the fitting procedure.
    montecarlo_configuration : MonteCarloConfig
        Configuration for the Monte Carlo bootstrap procedure.
    trial_id_fn : Callable[[dict], Any], optional
        A function to extract a unique identifier from a trial, by default default_trial_id.

    Returns
    -------
    ProtocolEval
        An object containing the evaluation results for the protocol, including metrics and influence scores.
    """

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
    fr = fit(X_train, Y_train, fit_configuration)
    A = fr.A
    b0 = getattr(fr, "b0", None)
    Y_pred = X_test @ A.T
    if b0 is not None:
        Y_pred = Y_pred + b0[None, :]

    # --- 6) metrics ---
    _rmse_total = _finite_or_nan(rmse_total(Y=Y_test, Y_pred=Y_pred))
    _r2_total = _finite_or_nan(r2_total(Y_test, Y_pred))
    _rmse_per_axis = [float(x) for x in np.asarray(rmse_per_axis(Y_test, Y_pred)).ravel().tolist()]
    _r2_per_axis = [float(x) for x in np.asarray(r2_per_axis(Y_test, Y_pred)).ravel().tolist()]

    if not np.isfinite(_rmse_total):
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
        fit_configuration=fit_configuration,
        montecarlo_configuration=montecarlo_configuration,
    )

    A_mean = np.asarray(boot.A_mean, float)
    A_std = np.asarray(boot.A_std, float)

    A_std_mean = float(np.mean(A_std))
    A_std_max = float(np.max(A_std))

    denominator = np.abs(A_mean)
    mask = denominator > 1e-12
    cv = np.zeros_like(A_std)
    cv[mask] = A_std[mask] / denominator[mask]
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

    notes = "\n".join(list_trials_pretty(chosen, show_file=False))
    return ProtocolEval(
        name=spec.name,
        n_total=n_total,
        n_train=len(train_trials),
        n_test=len(test_trials),
        rmse_total=_rmse_total,
        r2_total=_r2_total,
        rmse_per_axis=_rmse_per_axis,
        r2_per_axis=_r2_per_axis,
        A_std_mean=A_std_mean,
        A_std_max=A_std_max,
        A_cv_mean=A_cv_mean,
        A_cv_max=A_cv_max,
        influence_top_score=top_score,
        influence_top_idx=top_idx,
        influence_top_file=top_file,
        notes= notes,
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
    Compute a composite score for a protocol based on its evaluation metrics.
    The score is a weighted sum of the RMSE, the mean coefficient of variation (CV) of the parameters, and an optional dominance penalty based on the top influence score.
    Parameters
    ----------
    eval_ : ProtocolEval
        The evaluation results for the protocol, containing metrics and influence scores.
    alpha_rmse : float, optional
        Weight for the RMSE in the protocol score, by default 1.0.
    beta_cv : float, optional
        Weight for the A_cv_mean in the protocol score, by default 1.0.
    gamma_dom : float, optional
        Weight for the dominance penalty in the protocol score, by default 0.0.
    Returns
    -------
    float
        The computed protocol score, where lower is better. If the RMSE or A_cv_mean are not finite, returns infinity to indicate an invalid protocol.
    """
    if not np.isfinite(eval_.rmse_total) or not np.isfinite(eval_.A_cv_mean):
        return float("inf")

    dom = eval_.influence_top_score if eval_.influence_top_score is not None else 0.0
    return float(
        alpha_rmse * eval_.rmse_total
        + beta_cv * eval_.A_cv_mean
        + gamma_dom * float(dom)
    )


def search_best_protocols_from_indices(
    trial_pool: Sequence[dict],
    protocol_indices: Sequence[Sequence[int]],
    spec,
    acc_bias: np.ndarray,
    base: np.ndarray,
    fit_configuration: FitConfig,
    montecarlo_configuration: MonteCarloConfig,
    *,
    top_n: int = 20,
    alpha_rmse: float = 1.0,
    beta_cv: float = 1.0,
    gamma_dom: float = 0.0,
) -> list[tuple[float, ProtocolEval, list[int]]]:
    """
    Evaluate and rank protocols defined by their trial indices.
    
    Parameters     
    ----------
    trial_pool : Sequence[dict]
        The full set of trials to consider for testing (including the chosen ones).
    protocol_indices : Sequence[Sequence[int]]
        A sequence of protocols, where each protocol is defined by a sequence of trial indices.
    spec : Any
        A specification object containing at least a 'name' attribute for the protocol.
    acc_bias : np.ndarray
        The accelerometer bias to use when building the feature matrices.
    base : np.ndarray
        The base values to use when building the feature matrices.
    fit_configuration : FitConfig
        Configuration for the fitting procedure.
    montecarlo_configuration : MonteCarloConfig
        Configuration for the Monte Carlo bootstrap procedure.
    top_n : int, optional
        The number of top protocols to return, by default 20.
    alpha_rmse : float, optional
        Weight for the RMSE in the protocol score, by default 1.0.
    beta_cv : float, optional
        Weight for the A_cv_mean in the protocol score, by default 1.0.
    gamma_dom : float, optional
        Weight for the dominance penalty in the protocol score, by default 0.0.
    Returns
    -------
    list[tuple[float, ProtocolEval, list[int]]]
    """

    results: list[tuple[float, ProtocolEval, list[int]]] = []

    for k, indexes in enumerate(protocol_indices):
        chosen = [trial_pool[int(i)] for i in indexes]

        ev = evaluate_protocol(
            trials= trial_pool,
            chosen=chosen,
            spec=spec,
            acc_bias=acc_bias,
            base=base,
            fit_configuration=fit_configuration,
            montecarlo_configuration=montecarlo_configuration
        )

        if not np.isfinite(ev.rmse_total) or not np.isfinite(ev.A_cv_mean):
            continue

        s = protocol_score(ev, alpha_rmse=alpha_rmse, beta_cv=beta_cv, gamma_dom=gamma_dom)
        if not np.isfinite(s):
            continue
        print(f"k={k}: protocol={ev}")
        results.append((float(s), ev, list(map(int, indexes))))

    results.sort(key=lambda x: x[0])
    return results[: int(top_n)]

def summarize_protocols(best: Sequence[tuple]) -> list[dict]:
    """
    Summarize the best protocols into a list of dicts for easier export (CSV/JSON) or display.

        Each dict contains:
        - rank: int
        - score: float
        - name: str
        - n_total: int
        - n_train: int
        - n_test: int
        - rmse_total: float
        - r2_total: float
        - A_cv_mean: float

    Parameters
    ----------
    best : Sequence[tuple]
        A sequence of tuples, where each tuple is either (score, ProtocolEval) or (
        score, ProtocolEval, indexes).

    Returns
    -------
    list[dict]
        A list of dictionaries summarizing the best protocols.
    """
    rows: list[dict] = []
    for rank, item in enumerate(best, start=1):
        if len(item) == 2:
            score, ev = item
            indexes = None
        else:
            score, ev, indexes = item

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
            "notes": str(getattr(ev, "notes", "")),
        }

        rmse_axes = getattr(ev, "rmse_per_axis", None)
        if rmse_axes is not None:
            ra = np.asarray(rmse_axes, float).ravel()
            if ra.size == 6:
                for name, val in zip(AXES_6, ra):
                    row[f"rmse_{name}"] = float(val)

        if indexes is not None:
            row["protocol_len"] = len(indexes)
        rows.append(row)

    return rows


