from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Sequence, Any, List
from pathlib import Path
import csv
import json
import numpy as np

from software.calibration.optimization_calibration.types import (
    FitConfig,
    MonteCarloConfig,
    ProtocolSpec,
    ProtocolEval,
)

from software.calibration.optimization_calibration.features import build_XY
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


# ----------------------------
# Helpers: metadata access
# ----------------------------


def _get(trial: dict, key: str, default=None):
    try:
        return trial.get(key, default)
    except Exception:
        return default


def default_mass(trial: dict):
    return _get(trial, "Mass", None)


def default_degree(trial: dict):
    return _get(trial, "Degree", None)


def default_position(trial: dict):
    # If you don't have position, it'll be None for all trials (fine)
    return _get(trial, "position", None)


def default_trial_id(trial: dict):
    return _get(trial, "__file__", None)


# ----------------------------
# Trial selection
# ----------------------------


def select_trials(
    trials: Sequence[dict],
    spec: ProtocolSpec,
    mass_fn: Callable[[dict], Any] = default_mass,
    degree_fn: Callable[[dict], Any] = default_degree,
    position_fn: Callable[[dict], Any] = default_position,
    shuffle_before_cap: bool = True,
    seed: int = 0,
) -> list[dict]:
    """
    Select trials matching ProtocolSpec and optionally cap repeats per condition.

    Condition key = (mass, degree, position).

    Notes
    -----
    - If spec.max_per_condition is set, we keep at most that many trials per key.
    - By default we shuffle before applying the cap to avoid always keeping the same repeats.
    """
    # 1) Filter by allowed sets
    selected: list[dict] = []
    for t in trials:
        m = mass_fn(t)
        d = degree_fn(t)
        p = position_fn(t)

        if spec.masses is not None and m not in spec.masses:
            continue
        if spec.degrees is not None and d not in spec.degrees:
            continue
        if spec.positions is not None and p not in spec.positions:
            continue
        selected.append(t)

    # 2) Optional cap repeats per condition
    if spec.max_per_condition is not None:
        if shuffle_before_cap:
            rng = np.random.default_rng(seed)
            rng.shuffle(selected)

        cap = int(spec.max_per_condition)
        kept: list[dict] = []
        counts: dict[tuple[Any, Any, Any], int] = {}
        for t in selected:
            key = (mass_fn(t), degree_fn(t), position_fn(t))
            counts[key] = counts.get(key, 0) + 1
            if counts[key] <= cap:
                kept.append(t)
        selected = kept

    return selected


# ----------------------------
# Protocol evaluation
# ----------------------------


def _finite_or_nan(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def evaluate_protocol(
    all_trials: Sequence[dict],
    spec: ProtocolSpec,
    acc_bias: np.ndarray,
    base: np.ndarray,
    fit_cfg: FitConfig,
    mc_cfg: MonteCarloConfig,
    test_ratio: float = 0.2,
    seed: int = 0,
    stratify_by_mass: bool = True,
    mass_fn: Callable[[dict], Any] = default_mass,
    degree_fn: Callable[[dict], Any] = default_degree,
    position_fn: Callable[[dict], Any] = default_position,
    trial_id_fn: Callable[[dict], Any] = default_trial_id,
) -> ProtocolEval:
    """
    Evaluate a protocol scientifically:
      - select trials according to spec
      - split train/test (optionally stratified by mass)
      - fit A on train
      - evaluate RMSE/R2 on test
      - bootstrap stability on train (A_std, A_cv)
      - influence ranking on train (optional but very useful)

    IMPORTANT GUARDS
    ----------------
    - Reject protocol if total selected < spec.min_trials
    - Reject if train size < required minimum (>= 6 for 6x6 OLS without intercept)
    - If any metric becomes NaN/inf, return ProtocolEval with notes (so it can be filtered)
    """
    # --- 1) select ---
    chosen = select_trials(
        all_trials,
        spec,
        mass_fn=mass_fn,
        degree_fn=degree_fn,
        position_fn=position_fn,
        seed=seed,
    )
    n_total = len(chosen)

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
            notes=f"Not enough trials: {n_total} < min_trials={spec.min_trials}",
        )

    # --- 2) split ---
    rng = np.random.default_rng(seed)
    idx = np.arange(n_total)

    if stratify_by_mass:
        groups: dict[Any, list[int]] = {}
        for i, t in enumerate(chosen):
            groups.setdefault(mass_fn(t), []).append(i)

        test_idx: list[int] = []
        for _, g in groups.items():
            g = np.array(g, dtype=int)
            rng.shuffle(g)
            k = int(np.ceil(test_ratio * len(g)))
            test_idx.extend(g[:k].tolist())
        test_idx = np.array(sorted(set(test_idx)), dtype=int)
    else:
        rng.shuffle(idx)
        k = int(np.ceil(test_ratio * n_total))
        test_idx = np.array(sorted(idx[:k]), dtype=int)

    train_idx = np.array(
        [i for i in range(n_total) if i not in set(test_idx.tolist())], dtype=int
    )

    train_trials = [chosen[i] for i in train_idx]
    test_trials = [chosen[i] for i in test_idx]

    # Hard guard: we need enough train trials
    # (6 is the bare minimum for identifying a 6x6 mapping, but in practice you want more)
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
            notes=f"Not enough TRAIN trials: {len(train_trials)} < {min_train}",
        )

    # --- 3) build X/Y ---
    # build_XY is your wrapper that uses build_FMs_Channels internally
    # It should return (X, Y_scaled, y_scale)
    try:
        X_train, Y_train, y_scale_train = build_XY(train_trials, acc_bias, base)
        X_test, Y_test, y_scale_test = build_XY(test_trials, acc_bias, base)
    except TypeError:
        # If your build_XY doesn't accept normalize_y, fallback
        X_train, Y_train, y_scale_train = build_XY(train_trials, acc_bias, base)
        X_test, Y_test, y_scale_test = build_XY(test_trials, acc_bias, base)

    X_train = np.asarray(X_train, float)
    Y_train = np.asarray(Y_train, float)
    X_test = np.asarray(X_test, float)
    Y_test = np.asarray(Y_test, float)

    # --- 4) fit ---
    fr = fit(X_train, Y_train, fit_cfg)

    # --- 5) predict ---
    A = fr.A
    b0 = getattr(fr, "b0", None)
    Y_pred = X_test @ A.T
    if b0 is not None:
        Y_pred = Y_pred + b0[None, :]

    # --- 6) metrics ---
    rmse_total = _finite_or_nan(RMSE_total(Y_test, Y_pred))
    r2_total = _finite_or_nan(R2_total(Y_test, Y_pred))

    rmse_per_axis = [
        float(x) for x in np.asarray(RMSE_per_axis(Y_test, Y_pred)).ravel().tolist()
    ]
    r2_per_axis = [
        float(x) for x in np.asarray(R2_per_axis(Y_test, Y_pred)).ravel().tolist()
    ]

    # If metrics are not finite, return early (avoid garbage “top protocols”)
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
    # bootstrap_A expects builder + fit_configuration + montecarlo_configuration
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

    # CV = std / |mean| (masked to avoid exploding near zeros)
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
# Candidate generation
# ----------------------------


def unique_values(trials: Sequence[dict], fn: Callable[[dict], Any]) -> list[Any]:
    vals: list[Any] = []
    seen = set()
    for t in trials:
        v = fn(t)
        if v not in seen:
            seen.add(v)
            vals.append(v)
    return vals


def _counts_by_condition(
    trials: Sequence[dict],
    mass_fn: Callable[[dict], Any],
    degree_fn: Callable[[dict], Any],
    position_fn: Callable[[dict], Any],
) -> dict[tuple[Any, Any, Any], int]:
    counts: dict[tuple[Any, Any, Any], int] = {}
    for t in trials:
        key = (mass_fn(t), degree_fn(t), position_fn(t))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _estimate_selected_count(
    counts: dict[tuple[Any, Any, Any], int],
    masses: Sequence[Any] | None,
    degrees: Sequence[Any] | None,
    positions: Sequence[Any] | None,
    cap: int | None,
) -> int:
    """
    Predict how many trials will be selected by select_trials (upper bound-ish),
    using precomputed condition counts.
    """
    total = 0
    for (m, d, p), c in counts.items():
        if masses is not None and m not in masses:
            continue
        if degrees is not None and d not in degrees:
            continue
        if positions is not None and p not in positions:
            continue
        total += min(c, cap) if cap is not None else c
    return int(total)


def generate_protocol_candidates(
    trials: Sequence[dict],
    masses_sets: Sequence[Sequence[Any]] | None = None,
    degrees_sets: Sequence[Sequence[Any]] | None = None,
    positions_sets: Sequence[Sequence[Any]] | None = None,
    reps_caps: Sequence[int | None] = (None, 1, 2),
    min_trials: int = 20,
    mass_fn: Callable[[dict], Any] = default_mass,
    degree_fn: Callable[[dict], Any] = default_degree,
    position_fn: Callable[[dict], Any] = default_position,
) -> list[ProtocolSpec]:
    """
    Generate realistic candidate protocols.

    Key improvements vs a naive grid:
    - uses min_trials default 20 (practical, avoids absurd tiny protocols)
    - filters out candidates that cannot reach min_trials based on dataset counts
    """
    masses_all = unique_values(trials, mass_fn)
    degrees_all = unique_values(trials, degree_fn)
    positions_all = unique_values(trials, position_fn)

    # A) Defaults: keep them pragmatic (avoid tiny sets)
    if masses_sets is None:
        masses_sets = []
        # all masses
        if masses_all:
            masses_sets.append(list(masses_all))
        # low/high pair (if possible)
        if len(masses_all) >= 2:
            masses_sets.append([masses_all[0], masses_all[-1]])
        # top 3 masses (if possible)
        if len(masses_all) >= 3:
            masses_sets.append(list(masses_all[:3]))

    if degrees_sets is None:
        degrees_sets = []
        if degrees_all:
            degrees_sets.append(list(degrees_all))
        # 4 roughly spread angles (if you have many degrees)
        if len(degrees_all) >= 4:
            degrees_sets.append(
                [
                    degrees_all[0],
                    degrees_all[len(degrees_all) // 4],
                    degrees_all[len(degrees_all) // 2],
                    degrees_all[-1],
                ]
            )
        # 8 spread angles if available
        if len(degrees_all) >= 8:
            degrees_sets.append(
                [
                    degrees_all[i]
                    for i in np.linspace(0, len(degrees_all) - 1, 8).astype(int)
                ]
            )

    if positions_sets is None:
        positions_sets = [list(positions_all)]  # keep all by default

    # Precompute counts by condition for a fast feasibility filter
    counts = _counts_by_condition(trials, mass_fn, degree_fn, position_fn)

    specs: list[ProtocolSpec] = []
    k = 0
    for ms in masses_sets:
        for ds in degrees_sets:
            for ps in positions_sets:
                for cap in reps_caps:
                    est = _estimate_selected_count(counts, ms, ds, ps, cap)
                    if est < min_trials:
                        continue  # don't even generate absurd candidates

                    k += 1
                    name = f"P{k:02d}_m{len(ms)}_d{len(ds)}_p{len(ps)}_cap{cap}"
                    specs.append(
                        ProtocolSpec(
                            name=name,
                            masses=tuple(ms) if ms is not None else None,
                            degrees=tuple(ds) if ds is not None else None,
                            positions=tuple(ps) if ps is not None else None,
                            max_per_condition=cap,
                            min_trials=min_trials,
                        )
                    )

    return specs


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


def search_best_protocols(
    trials: Sequence[dict],
    acc_bias: np.ndarray,
    base: np.ndarray,
    fit_cfg: FitConfig,
    mc_cfg: MonteCarloConfig,
    candidates: Sequence[ProtocolSpec],
    top_n: int = 20,
    alpha_rmse: float = 1.0,
    beta_cv: float = 1.0,
    gamma_dom: float = 0.0,
    seed: int = 0,
) -> list[tuple[float, ProtocolEval]]:
    """
    Evaluate candidates and return the best ones.

    Critical behavior:
    - Protocols with non-finite metrics are discarded (so no more ubuesque NaNs in top).
    """
    results: list[tuple[float, ProtocolEval]] = []

    for spec in candidates:
        ev = evaluate_protocol(
            all_trials=trials,
            spec=spec,
            acc_bias=acc_bias,
            base=base,
            fit_cfg=fit_cfg,
            mc_cfg=mc_cfg,
            seed=seed,
        )

        if not np.isfinite(ev.rmse_total) or not np.isfinite(ev.A_cv_mean):
            continue  # discard invalid/degenerate candidates

        s = protocol_score(
            ev, alpha_rmse=alpha_rmse, beta_cv=beta_cv, gamma_dom=gamma_dom
        )
        if not np.isfinite(s):
            continue

        results.append((s, ev))

    results.sort(key=lambda x: x[0])
    return results[:top_n]


# ----------------------------
# Exports
# ----------------------------


def export_protocols_csv(
    scored: Sequence[tuple[float, ProtocolEval]], path: str
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "score",
        "name",
        "n_total",
        "n_train",
        "n_test",
        "rmse_total",
        "r2_total",
        "A_std_mean",
        "A_std_max",
        "A_cv_mean",
        "A_cv_max",
        "influence_top_score",
        "influence_top_idx",
        "influence_top_file",
        "rmse_per_axis",
        "r2_per_axis",
        "notes",
    ]

    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for k, (s, ev) in enumerate(scored, start=1):
            w.writerow(
                {
                    "rank": k,
                    "score": float(s),
                    "name": ev.name,
                    "n_total": ev.n_total,
                    "n_train": ev.n_train,
                    "n_test": ev.n_test,
                    "rmse_total": ev.rmse_total,
                    "r2_total": ev.r2_total,
                    "A_std_mean": ev.A_std_mean,
                    "A_std_max": ev.A_std_max,
                    "A_cv_mean": ev.A_cv_mean,
                    "A_cv_max": ev.A_cv_max,
                    "influence_top_score": ev.influence_top_score,
                    "influence_top_idx": ev.influence_top_idx,
                    "influence_top_file": ev.influence_top_file,
                    "rmse_per_axis": json.dumps(ev.rmse_per_axis),
                    "r2_per_axis": json.dumps(ev.r2_per_axis),
                    "notes": ev.notes,
                }
            )


def export_protocols_json(
    scored: Sequence[tuple[float, ProtocolEval]], path: str
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = []
    for k, (s, ev) in enumerate(scored, start=1):
        payload.append({"rank": k, "score": float(s), **asdict(ev)})

    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ----------------------------
# One-call helper
# ----------------------------


def recommend_protocol(
    trials: Sequence[dict],
    acc_bias: np.ndarray,
    base: np.ndarray,
    fit_cfg: FitConfig,
    mc_cfg: MonteCarloConfig,
    out_dir: str | Path | None = None,
    seed: int = 0,
    min_trials: int = 20,
    top_n: int = 5,
    alpha_rmse: float = 1.0,
    beta_cv: float = 1.0,
    gamma_dom: float = 0.0,
) -> tuple[list[tuple[float, ProtocolEval]], list[ProtocolSpec]]:
    """
    Generate realistic candidate protocols, evaluate them, and return the best ones.
    Optionally export CSV/JSON.

    Defaults are set to avoid "ubuesque" scenarios:
      - min_trials=20
      - invalid protocols are discarded automatically
    """
    cands = generate_protocol_candidates(trials, min_trials=min_trials)

    best = search_best_protocols(
        trials=trials,
        acc_bias=acc_bias,
        base=base,
        fit_cfg=fit_cfg,
        mc_cfg=mc_cfg,
        top_n=top_n,
        alpha_rmse=alpha_rmse,
        beta_cv=beta_cv,
        gamma_dom=gamma_dom,
        candidates=cands,
        seed=seed,
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        export_protocols_csv(best, str(out_dir / "protocol_ranking.csv"))
        export_protocols_json(best, str(out_dir / "protocol_ranking.json"))

    return best, cands
