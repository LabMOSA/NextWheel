import numpy as np
from pathlib import Path

import kineticstoolkit as ktk
from software.calibration.calculate_base_AccBias import latest_by_prefix
import software.calibration.wheelcalibration as wc

def load_fixed_imu_params(imu_package_dir: str | Path) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    Compute Base and AccBias from a dedicated IMU calibration package folder.
    Returns (base, acc_bias, info).
    """
    folder = Path(imu_package_dir)

    gyro_bias_file = latest_by_prefix(folder, "GyroForBias", 1)[0]
    gyro_z_file = latest_by_prefix(folder, "GyroForZAxis", 1)[0]
    acc_files = latest_by_prefix(folder, "AccForXZPlane", 2)

    gyro_bias_trial = ktk.load(str(gyro_bias_file))
    gyro_z_trial = ktk.load(str(gyro_z_file))
    acc1 = ktk.load(str(acc_files[0]))
    acc2 = ktk.load(str(acc_files[1]))

    gyro_bias = wc.estimate_gyro_bias(gyro_bias_trial["IMU"]["Gyro"])
    z_axis = wc.get_z_axis(gyro_bias, gyro_z_trial["IMU"]["Gyro"])
    base = wc.get_wheel_reference(-acc1["IMU"]["Acc"], -acc2["IMU"]["Acc"], z_axis)

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
    return base, acc_bias, info, gyro_bias


def apply_imu_calibration(trial: dict, base: np.ndarray, acc_bias: np.ndarray,
                          gyro_bias: np.ndarray | None = None) -> dict:
    out = dict(trial)
    out["IMU"] = dict(trial["IMU"])

    acc = np.asarray(trial["IMU"]["Acc"], float)
    gyr = np.asarray(trial["IMU"]["Gyro"], float)

    acc_c = acc - acc_bias.reshape(1, 3)
    gyr_c = gyr - gyro_bias.reshape(1, 3) if gyro_bias is not None else gyr

    R = np.asarray(base, float)
    if R.shape != (3, 3):
        raise ValueError(f"'base' should be 3x3 rotation matrix, got {R.shape}")

    acc_w = (R @ acc_c.T).T
    gyr_w = (R @ gyr_c.T).T

    out["IMU"]["AccCal"] = acc_w
    out["IMU"]["GyroCal"] = gyr_w
    return out

def apply_calibration_forces(trial: dict, A: np.ndarray, offset: np.ndarray, y_scale: np.ndarray) -> dict:
    out = dict(trial)
    fm = np.asarray(trial["Analog"]["Force"], dtype=float)

    forces_calibrated = (fm - offset) @ A.T
    forces_calibrated = forces_calibrated * y_scale
    out["Analog"]["ForceCalibrated"] = forces_calibrated[:, 0:3]
    out["Analog"]["MomentsCalibrated"] = forces_calibrated[:, 3:6]
    return out

def convert_ticks_to_radians(trial: dict, encoder_variation: float = 0.087890625):
    out = dict(trial)
    angle = np.asarray(trial["Encoder"]["Angle"], dtype=float)
    print(f"Original angle (ticks): {angle[:5]} ...")
    angle = angle - angle[0]
    angle *= encoder_variation
    angle = np.deg2rad(angle)

    out["Encoder"]["AngleCalibrated"] = angle

    return out

def get_centrifuge_force(ts: np.ndarray) -> np.ndarray:
    """
    """
    forces = np.asarray(ts.data["Forces"], dtype=float)
    print(f"Original forces: {forces[:5]} ...")
    return forces[:, 0:3]


def get_speed_from_gyro():
    return

