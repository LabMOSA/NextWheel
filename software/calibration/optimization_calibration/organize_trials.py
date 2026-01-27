import numpy as np
from typing import Literal, Optional

IDX = {"Fx": 0, "Fy": 1, "Fz": 2, "Mx": 3, "My": 4, "Mz": 5}

def robust_axis_amplitude(V: np.ndarray) -> np.ndarray:
    """
    V: (T,6) forces/moments bruts (ADC ou unités quelconques)
    amplitude robuste = RMS autour de la médiane (par axe)
    """
    V = np.asarray(V, float)
    med = np.median(V, axis=0)
    A = np.sqrt(np.mean((V - med) ** 2, axis=0))
    return A  # (6,)

def plane_from_analog_force(trial: dict, ratio_margin: float = 1.2, min_total: float = 1e-6):
    """
    Retourne:
      label, path, A, eA, eB
    """
    path = trial.get("__file__", None)

    V = trial.get("Analog", {}).get("Force", None)
    if V is None:
        return "ambiguous", path, None, None, None

    V = np.asarray(V, float)
    if V.ndim != 2 or V.shape[1] != 6:
        return "ambiguous", path, None, None, None

    A = robust_axis_amplitude(V)

    eA = float(np.sum(A[[IDX["Fx"], IDX["Fy"], IDX["Mz"]]] ** 2))  # Fx,Fy,Mz
    eB = float(np.sum(A[[IDX["Mx"], IDX["My"], IDX["Fz"]]] ** 2))  # Mx,My,Fz
    etot = eA + eB

    if etot < min_total:
        return "ambiguous", path, A, eA, eB

    rAB = (eA + 1e-12) / (eB + 1e-12)
    print("rAB:", rAB)
    rBA = (eB + 1e-12) / (eA + 1e-12)
    print(f"rBA: {rBA}")
    if rAB >= ratio_margin:
        return "plane_FxFyMz", path, A, eA, eB
    if rBA >= ratio_margin:
        return "plane_MxMyFz", path, A, eA, eB
    return "ambiguous", path, A, eA, eB

def get_trial_FM(trial: dict):
    # adapte ici selon ton format
    # exemples possibles:
    # - trial["FM"]
    # - trial["ForcesForCalibrationMatrix"]
    # - trial["Kinetics"]["FM"]
    for key in ("FM", "ForcesForCalibrationMatrix", "Kinetics", "EstimatedFM"):
        if key in trial:
            val = trial[key]
            if key == "Kinetics" and isinstance(val, dict) and "FM" in val:
                return val["FM"]
            return val
    return None

def assign_planes(trials: list[dict]) -> dict[str, list[dict]]:
    out = {"plane_FxFyMz": [], "plane_MxMyFz": [], "ambiguous": []}
    for t in trials:
        # si tu utilises get_trial_FM:
        # t2 = dict(t); t2["FM"] = get_trial_FM(t)
        # pl = plane_from_trial_FM(t2, fm_key="FM")
        pl = plane_from_trial_FM(t, fm_key="FM")
        out[pl].append(t)
    return out

def plane_label_from_trial(trial: dict) -> str | None:
    pl = plane_from_trial_FM(trial, fm_key="FM")
    if pl == "ambiguous":
        return None
    return pl
