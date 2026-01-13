"""Fonctions to organise the instrumented wheelchair wheel calibration."""

import numpy as np
import kineticstoolkit as ktk
import nextwheel
from nextwheel import NextWheel
import wheelcalibration as wc
import limitedinteraction as li
import time
import os
import sys
from pathlib import Path


def interface(dialog: str, title: str):
    """
    Use a cute little interface that describe the actions to do.

    Exemple :
        >> dialog = "Make sure the wheel is not moving and click Ok"
        >> title = "Calibration part 1"
        >> interface(dialog, title)

    Parameters
    ----------
    dialog
        The description of the desired action to accomplish.
    title
        The title of the interface.

    Returns
    -------
    None.

    """
    li.button_dialog(
        dialog,
        choices=["OK"],
        title=title,
        icon="light",
    )

def measure(nw: nextwheel.NextWheel, waiting_time: int = 5) -> dict:
    """
    Make a measurement with the NextWheel module.

    Parameters
    ----------
    nw :
        NextWheel object of nextwheel module.
    waiting_time : optional
        The waiting time of a measure in second. The default is 5 sec.

    Returns
    -------
    dict
        The dictionary fetched from the NextWheel.

    """
    li.message("Please wait a few moments.", title="Measuring...", icon="clock")
    nw.start_streaming()
    nw.fetch()
    time.sleep(5)
    nw.stop_streaming()
    li.message("")
    return nw.fetch()

def safe_measure(next_wheel: NextWheel, waiting_time: int = 5) -> dict:
    """
    Make a measurement with the NextWheel module

    Parameters
    ----------
    next_wheel :
        NextWheel object of nextwheel module.
    waiting_time : optional
        The waiting time of a measure in second. The default is 5 sec.
    Returns
    -------
    dict
        The dictionary fetched from the NextWheel.
    """
    li.message(
        f"Measurement will start now. \n\n"
        f"Duration : {waiting_time} s\n"
        f"Please keep the wheel steady.",
        title="Measuring...",
        icon="clock",
    )
    next_wheel.start_streaming()
    try:
        next_wheel.fetch()
        time.sleep(waiting_time)
        data = next_wheel.fetch()
        return data
    finally:
        try:
            next_wheel.stop_streaming()
        except Exception:
            pass
        li.message("Measurement complete.", title="Measuring...", icon="check")

def save_file(data: dict, file_name: str, path: str):
    """
    Save file with Kinetics Toolskit.

    The function doesn't overwrite any file.

    Parameters
    ----------
    data :
        The fectched data from the measure function.
    file_name :
        The file name of the new created file.
    path :
        The path where the file is saved.

    Returns
    -------
    None.

    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    n_trial = 0
    while os.path.isfile(f"{path}{file_name}{n_trial}"):  # not overwriting
        n_trial += 1
    ktk.save(f"{path}{file_name}{n_trial}", data)

def save_file(data: dict, file_name: str, path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)  # crée le dossier si besoin

    n_trial = 0
    target = path / f"{file_name}{n_trial}"
    while target.exists():                   # évite d’écraser
        n_trial += 1
        target = path / f"{file_name}{n_trial}"

    ktk.save(str(target), data)              # ktk.save préfère souvent un str
    return target



def calibrate_part1(nw: nextwheel.NextWheel, path: str):
    """
    Save two measures to find z-axis on the wheel frame.

    First, the bias is measured with a static measure of the wheel. Since the
    wheel is not moving, the measurement is the bias.

    Second, the wheel is spinning around it's rotation axis (the z-axis in the
    wheel referential). Since the gyroscope measurement is the rotation axis
    with the norm being the angular speed, the normalized gyroscope measure is
    the z-axis in the wheel frame.

    Parameters
    ----------
    nw :
        NextWheel object of nextwheel module.
    path :
        The path where the files are saved.

    Returns
    -------
    None.

    """
    # Measure 1 - Gyro bias measurement
    interface(
        "Make sure the wheel is not moving and click Ok",
        "z-axis calibration - Static trial",
    )
    gyro_bias_measured = safe_measure(nw)
    save_file(gyro_bias_measured, "GyroForBias", path)

    # Measure 2 - Z-axis
    interface(
        "Spin the wheel anticlockwise around the desired z-axis and click OK",
        "z-axis calibration - Dynamic trial",
    )

    z_axis_measured = measure(nw)
    save_file(z_axis_measured, "GyroForZAxis", path)
def calibrate_part2(nw: nextwheel.NextWheel, path: str):
    """
    Save two measures to find the xz-plane in the wheel frame.

    Both measures are the gravity measured by the accelerometer in the desired
    xz-plane in the wheel frame.

    Parameters
    ----------
    nw :
        NextWheel object of nextwheel module.
    path :
        The path where the files are saved.

    Returns
    -------
    None.

    """
    # Measure 1
    interface(
        "Make sure the wheel is not moving and that the x-axis point to the lower part",
        "xz-axis calibration - Static trial 1",
    )
    xz_acc_vector_measured1 = measure(nw)
    save_file(xz_acc_vector_measured1, "AccForXZPlane", path)

    # Measure 2
    interface(
        "Same indication that before, but with an other orientation around y-axis",
        "xz-axis calibration - Static trial 2",
    )

    xz_acc_vector_measured2 = measure(nw)
    save_file(xz_acc_vector_measured2, "AccForXZPlane", path)

def calibrate_part3(nw: nextwheel.NextWheel, path: str):
    """
    Save measure with a known hanging mass on the pushrim.

    The mass and the position of the mass on the push rim (in degree) are
    needed. The degree value is determined with the cylindrical coordinate on
    the wheel frame.

    Parameters
    ----------
    nw :
        NextWheel object of nextwheel module.
    path :
        The path where the files are saved.

    Returns
    -------
    None.

    """
    # Measure 1
    interface("Measure the offset force of the channels", "Offset")
    nw.start_streaming()
    nw.fetch()

    offset_measured = measure(nw)
    offset = np.mean(offset_measured["Analog"]["Force"], axis=0)

    # Measure 2
    mass = float(li.input_dialog("What is the mass you add on rim?", icon="question"))
    degree = float(li.input_dialog("At witch degree on the rim?", icon="question"))
    interface(
        "Measure the gravity and forces on rim",
        "Calibration - Force Measurements",
    )
    forces_on_rim_measured = measure(nw)
    forces_on_rim_measured["Mass"] = mass
    forces_on_rim_measured["Degree"] = degree
    forces_on_rim_measured["ForceOffset"] = offset_measured

    forces_on_rim_measured["Analog"]["Force"] -= np.tile(
        offset, (np.shape(forces_on_rim_measured["Analog"]["Force"])[0], 1)
    )

    save_file(
        forces_on_rim_measured,
        "ForcesForCalibrationMatrix",
        path,
    )

def estimate_calibration_matrix(path: str) -> np.array:
    """
    Calculate the calibration matrix with wheelcalibration.py module.

    Estimate the gyroscope bias -> estimate the z-axis -> estimate the frame
    changing matrix (base) -> estimate the accelerometer bias -> estimate the
    calibration matrix with a least square.

    Note : The count variables (count1, count2, count3) are only assuring that
    the newest measure is being used over older measures. It counts the number
    of file with similar name and takes the higher numbers (most recent
    measures). WARNING IF YOU DELETE OR RENAME FILES!

    Parameters
    ----------
    path :
        The path where the files are saved.

    Returns
    -------
    A :
        Calibration matrix.

    """
    Trials = {}
    grav_measured = np.ndarray((1, 3))
    count1 = 0
    count2 = 0
    count3 = 0

    for file_name in os.listdir(path):
        if file_name.startswith("Forces"):
            Trials[file_name] = ktk.load(f"{path}{file_name}")
            grav = -np.mean(Trials[file_name]["IMU"]["Acc"], axis=0)

            grav_measured = np.vstack(
                (
                    grav_measured,
                    grav,
                )
            )

        elif file_name.startswith("GyroForBias"):
            Trials[file_name] = ktk.load(f"{path}{file_name}")
            count1 += 1

        elif file_name.startswith("GyroForZAxis"):
            Trials[file_name] = ktk.load(f"{path}{file_name}")
            count2 += 1

        elif file_name.startswith("AccForXZPlane"):
            Trials[file_name] = ktk.load(f"{path}{file_name}")
            count3 += 1

    # Estimate the gyroscope bias
    Trials["GyroBias"] = wc.estimate_gyro_bias(
        Trials[f"GyroForBias{count1 - 1}"]["IMU"]["Gyro"]
    )

    # Estimate the z-axis

    Trials["Z-Axis"] = wc.get_z_axis(
        Trials["GyroBias"], Trials[f"GyroForZAxis{count2-1}"]["IMU"]["Gyro"]
    )

    # Estimate the frame changing matrix
    Trials["Base"] = wc.get_wheel_reference(
        -Trials[f"AccForXZPlane{count3-2}"]["IMU"]["Acc"],
        -Trials[f"AccForXZPlane{count3-1}"]["IMU"]["Acc"],
        Trials["Z-Axis"],
    )  # estimate the base change matrix

    Trials["AccBias"] = wc.estimate_acc_bias(
        grav_measured[1:, :]
    )  # estimate accelerometer bias

    forces_channels = np.ndarray((1, 6))
    FMs = np.ndarray((1, 6))
    for trial in Trials:
        if trial.startswith("Forces"):
            FM = wc.make_an_estimation_of_forces_moments(
                Trials[trial],
                Trials["AccBias"],
                Trials["Base"],
            )

            # forces = np.mean(Trials[trial]["Analog"]["Force"], axis=0)
            forces = np.median(Trials[trial]["Analog"]["Force"], axis=0)
            forces_channels = np.vstack(
                (
                    forces_channels,
                    forces,
                )
            )

            FMs = np.vstack((FMs, FM))

    A = wc.calculate_calibration_matrix(
        FMs[1:, :], forces_channels[1:, :]
    )  # estimate the calibration matrix in A*forces_channels.T = FMs.T

    return A, Trials


def gyro_static_ok(data: dict, threshold: float = 0.05) -> bool:
    gyro = np.asarray(data["IMU"]["Gyro"])
    return np.all(np.std(gyro, axis=0) < threshold)


def gyro_z_axis_steady_ok(data: dict,
                         min_rate: float = 0.5,      # vitesse min (rad/s ou deg/s selon ton gyro)
                         max_cv: float = 0.15,       # stabilité: coeff de variation de |ω|
                         min_z_ratio: float = 0.8,   # Z doit représenter >= 80% de la norme moyenne
                         expected_sign: int | None = None  # +1 ou -1 si tu veux imposer le sens
                         ) -> bool:
    gyro = np.asarray(data["IMU"]["Gyro"])           # (N,3)
    mean = gyro.mean(axis=0)
    mean_norm = np.linalg.norm(mean)
    # 1) ça tourne assez vite (rotation non négligeable)
    if mean_norm < min_rate:
        return False

    # 2) rotation "steady" : la norme instantanée varie peu
    inst_norm = np.linalg.norm(gyro, axis=1)
    cv = inst_norm.std() / (inst_norm.mean() + 1e-12)  # coeff variation
    if cv > max_cv:
        return False

    # 3) c'est majoritairement l'axe Z
    z_ratio = abs(mean[2]) / (mean_norm + 1e-12)
    print(z_ratio)
    print("---")
    print(z_ratio < min_z_ratio)
    if z_ratio < min_z_ratio:
        return False

    # 4) sens imposé (optionnel)
    print(expected_sign)
    if expected_sign is not None and np.sign(mean[2]) != expected_sign:
        return False

    return True

def calibrate_gyro_bias(next_wheel: NextWheel, path: str):
     # Gyro bias measurement
    interface(
        f"Keep the wheel COMPLETELY still.\n\nClick OK to start.",
        "Gyroscope Bias Calibration",
    )
    data = safe_measure(next_wheel)
    if gyro_static_ok(data):
        save_file(data, "GyroForBias", path)
        return data

    li.message(
        "Rejecting measurement due to excessive gyroscope variance.\n\n"
        "Please ensure the wheel is completely still during the measurement.",
        title="Measurement Rejected",
        icon="error",
    )
    raise RuntimeError("Gyroscope measurement too variable; measurement rejected.")

def calibrate_rotation_axis(next_wheel: NextWheel, path: str):
    # Z-axis measurement
    interface(
        f"Spin the wheel anticlockwise. \n"
        f"Click OK once the rotation is steady, and keep it spinning during the measurement.",
        "Z-axis (dynamic) - attempt",
    )

    data = safe_measure(next_wheel)
    if gyro_z_axis_steady_ok(data, expected_sign= -1):
        save_file(data, "GyroForZAxis", path)
        return data

    li.message(
        "Rejected: rotation too weak or unstable. Please retry.",
        title="Measurement Rejected",
        icon="error",
    )
    raise RuntimeError("Could not obtain a valid z-axis (dynamic) trial.")

def accelerometer_static_ok(data: dict, stable_threshold: float = 0.05) -> bool:
    """
    Check if the accelerometer data is stable enough.
    std_thresh depends on units (m/s^2 vs g). Tune empirically.
    """
    if not isinstance(data, dict) or "IMU" not in data or "Acc" not in data["IMU"]:
        return False
    accelerometer_data = np.asarray(data["IMU"]["Acc"])
    return np.all(np.std(accelerometer_data, axis=0) < stable_threshold)

def gravity_norm_ok(data: dict, g_min: float = 7.0, g_max: float = 12.0) -> bool:
    """
    Checks mean acceleration magnitude is consistent with gravity.
    Default assumes m/s^2. If your Acc is in g, use g_min=0.8, g_max=1.2.
    """
    acc = np.asarray(data["IMU"]["Acc"])
    g = np.linalg.norm(acc.mean(axis=0))
    return g_min <= g <= g_max

def acc_y_near_zero_ok(data: dict, max_ratio: float = 0.15) -> bool:
    """
    Accept if |mean(ay)| is small relative to ||mean(a)||.
    max_ratio=0.15 means ay must be <= 15% of total gravity magnitude.
    """
    acc = np.asarray(data["IMU"]["Acc"])
    mean = acc.mean(axis=0)                 # [ax, ay, az]
    norm = np.linalg.norm(mean) + 1e-12
    y_ratio = abs(mean[1]) / norm
    return y_ratio <= max_ratio

def accelerometer_trials_are_different(data1: dict, data2: dict,min_angle_deg: float = 20.0) -> bool:

    """
    Check if two accelerometer trials are sufficiently different in orientation.
    min_angle_deg is the minimum angle between the two mean acceleration vectors.
    """
    acceleration_measure_1 = np.asarray(data1["IMU"]["Acc"])
    acceleration_measure_2 = np.asarray(data2["IMU"]["Acc"])
    mean1 = acceleration_measure_1.mean(axis=0)
    mean2 = acceleration_measure_2.mean(axis=0)

    norm1 = np.linalg.norm(mean1)
    norm2 = np.linalg.norm(mean2)
    if norm1 < 1e-12 or norm2 < 1e-12:
        return False

    cos_angle = np.clip(np.dot(mean1, mean2) / (norm1 * norm2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg >= min_angle_deg

def calibrate_wheel_frame_orientation_trial_1(next_wheel: NextWheel, path: str):

    interface(
        "Keep the wheel COMPLETELY still.\n\n"
        ""
    )

    xz_acc_vector_measured1 = measure(nw)
    save_file(xz_acc_vector_measured1, "AccForXZPlane", path)


# if __name__ == "__main__":
#     ip = sys.argv[-1]
#     nw = nextwheel.NextWheel(ip)
#     ROOT = Path(__file__).resolve().parent
#     path = str(ROOT) + "/"
#     trials_dir = "Trials_Wheel/"
#
# # %% Part 1 - Z-axis calculated from gyroscope
#
# calibrate_part1(nw, path + trials_dir)
#
# # %% Part 2 - XZ-Plane determined from acc + base change matrix completion
#
# calibrate_part2(nw, path + trials_dir)
#
# # %% Part 3 - More static Force mesures for calibration matrix
#
# calibrate_part3(nw, path + trials_dir)
# while not li.button_dialog(
#     "Do you want to do another measure ?",
#     choices=["Oui", "Non"],
#     title="Force Measures for Calibration Matrix",
#     icon="gear",
# ):
#     calibrate_part3(nw, path + trials_dir)
#
# # %% Part 4 - Calibration matrix calculation
# A, Trials = estimate_calibration_matrix(path + trials_dir)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python calibrationwizard.py <ip>")

    ip = sys.argv[-1]
    nw = nextwheel.NextWheel(ip)

    ROOT = Path(__file__).resolve().parent
    trials_dir = ROOT / "Trials_Wheel"   # pas besoin de / final
    trials_dir.mkdir(parents=True, exist_ok=True)

    # calibrate_gyro_bias(nw, trials_dir)
    calibrate_rotation_axis(nw, trials_dir)
    # Puis tu appelles tes fonctions avec ce dossier:
    # calibrate_part1(nw, trials_dir)
    # calibrate_part2(nw, trials_dir)
    # calibrate_part3(nw, trials_dir)
