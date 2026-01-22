from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import kineticstoolkit as ktk
import software.calibration.wheelcalibration as wc


def latest_by_prefix(folder: Path, prefix: str, n: int = 1) -> list[Path]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    c = []
    for p in folder.iterdir():
        if p.is_file():
            m = pattern.match(p.name)
            if m:
                c.append((int(m.group(1)), p))
    if len(c) < n:
        raise FileNotFoundError(f"Need {n} file(s) '{prefix}<int>' in {folder}, found {len(c)}")
    c.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in c[:n]]

def debug_list(folder: Path):
    print("Folder:", folder)
    print("Exists:", folder.exists())
    if not folder.exists():
        return
    print("\nFiles:")
    for p in sorted(folder.iterdir()):
        if p.is_file():
            print("  ", p.name)

def load_fixed_imu_params(imu_package_dir: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute Base and AccBias from a dedicated IMU calibration package folder.
    Returns (base, acc_bias, info).
    """
    folder = Path(imu_package_dir)

    gyro_bias_file = latest_by_prefix(folder, "GyroForBias", 1)[0]
    gyro_z_file    = latest_by_prefix(folder, "GyroForZAxis", 1)[0]
    acc_files      = latest_by_prefix(folder, "AccForXZPlane", 2)

    gyro_bias_trial = ktk.load(str(gyro_bias_file))
    gyro_z_trial    = ktk.load(str(gyro_z_file))
    acc1            = ktk.load(str(acc_files[0]))
    acc2            = ktk.load(str(acc_files[1]))

    gyro_bias = wc.estimate_gyro_bias(gyro_bias_trial["IMU"]["Gyro"])
    z_axis    = wc.get_z_axis(gyro_bias, gyro_z_trial["IMU"]["Gyro"])
    base      = wc.get_wheel_reference(-acc1["IMU"]["Acc"], -acc2["IMU"]["Acc"], z_axis)

    # AccBias: from the two gravity measurements (simple) OR from more trials if you have them
    grav_vectors = np.vstack([
        -np.mean(acc1["IMU"]["Acc"], axis=0),
        -np.mean(acc2["IMU"]["Acc"], axis=0),
    ])
    acc_bias = wc.estimate_acc_bias(grav_vectors)

    info = {
        "imu_package": str(folder),
        "files_used": {
            "GyroForBias": gyro_bias_file.name,
            "GyroForZAxis": gyro_z_file.name,
            "AccForXZPlane": [p.name for p in acc_files],
        }
    }
    return base, acc_bias, info


if __name__ == "__main__":
    path = Path(__file__).resolve().parent
    imu_dir = path / "E1_E2"

    debug_list(imu_dir)

    base, acc_bias, info = load_fixed_imu_params(imu_dir)

    print("\n--- INFO ---")
    print(info)

    print("\n--- SHAPES ---")
    print("Base shape:", np.asarray(base).shape)       # attendu (3,3)
    print("AccBias shape:", np.asarray(acc_bias).shape) # souvent (3,) ou (1,3) selon wc

    print("\n--- VALUES (quick sanity checks) ---")
    print("Base finite:", np.isfinite(base).all())
    print("AccBias finite:", np.isfinite(acc_bias).all())

    # Vérif rapide: Base devrait ressembler à une rotation (colonnes ~ unitaires)
    col_norms = np.linalg.norm(np.asarray(base), axis=0)
    print("Base column norms:", col_norms)

    # Vérif gravité: après correction bias, la norme devrait être "proche" de g (selon unités)
    # Ici on ne peut pas le garantir sans savoir si acc est en m/s^2 ou g,
    # mais on peut au moins afficher l'ordre de grandeur :
    print("AccBias:", acc_bias)

    # Vérif Base orthogonalité: donc det = 1 et R^T R = I (sauf erreurs numériques)
    R = np.asarray(base)
    print("det(Base) =", np.linalg.det(R))
    print("R^T R =", R.T @ R)
