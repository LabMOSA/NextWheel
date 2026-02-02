import numpy as np
from typing import Literal, Optional

from software.calibration.optimization_2_planes.features import position_from_imu_gravity

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

def plane_from_analog_force(trial: dict):
    """
    """
    path = trial.get("__file__", "unknown")
    planeFxFyMz = ("X+","X-", "Y+", "Y-")
    planeMxMyFz = ("Z+", "Z-")
    position = position_from_imu_gravity(trial= trial)
    if position in planeFxFyMz:
        return "plane_FxFyMz", path
    if position in planeMxMyFz:
        return "plane_MxMyFz", path
    return "ambiguous", path


def assign_planes(trials: list[dict]) -> dict[str, list[dict]]:
    out = {"plane_FxFyMz": [], "plane_MxMyFz": [], "ambiguous": []}
    for t in trials:
        pl = plane_from_analog_force(t)
        out[pl[0]].append(t)
    return out

