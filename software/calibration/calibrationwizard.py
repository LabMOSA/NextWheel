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
from typing import Iterable

# %% Gyroscope checks and calibrations
def gyro_static_ok(data: dict, threshold: float = 0.05) -> bool:
    """
    Check if the gyroscope data is stable enough.
    threshold depends on units (here is deg/s). Tune empirically.

    Parameters
    ----------
    data : dict
        The data dictionary containing IMU data.
    threshold : float, optional
        The maximum allowed standard deviation for each axis. The default is 0.05.

    Returns
    -------
    bool
        True if the gyroscope data is stable enough, False otherwise.
    """
    gyro = np.asarray(data["IMU"]["Gyro"])
    return np.all(np.std(gyro, axis=0) < threshold)
def calibrate_gyro_bias(next_wheel: NextWheel, path: str):
    """
    Calibrate the gyroscope bias by measuring the wheel while it is still.
    Parameters
    ----------
    next_wheel : NextWheel
        The NextWheel object to perform measurements.
    path : str
        The path where the calibration data will be saved.
    Returns
    -------
    dict
        The gyroscope bias data.
    """
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

# %% Gyroscope Z-axis checks and calibrations
def gyro_z_axis_steady_ok(data: dict,
                         min_rate: float = 0.5,
                         max_cv: float = 0.15,
                         min_z_ratio: float = 0.8,
                         expected_sign: int | None = None
                         ) -> bool:
    """
    Check if the gyroscope data indicates a steady rotation around the Z-axis.
    Parameters
    ----------
    data : dict
        The data dictionary containing IMU data.
    min_rate : float, optional
        Minimum mean rotation rate (norm of mean gyro) to consider it valid.
    max_cv : float, optional
        Maximum coefficient of variation of instantaneous rotation rate.
    min_z_ratio : float, optional
        Minimum ratio of |mean(ωz)| to ||mean(ω)||.
    expected_sign : int | None, optional
        If +1 or -1, enforce the sign of mean(ωz). If None
        no sign enforcement. The default is None.
    Returns
    -------
    bool
        True if the gyroscope data indicates a steady rotation around Z-axis, False otherwise.
    """
    gyro = np.asarray(data["IMU"]["Gyro"])           # (N,3)
    mean = gyro.mean(axis=0)
    mean_norm = np.linalg.norm(mean)


    # 1) rotation speed sufficient
    if mean_norm < min_rate:
        return False

    # 2) stable rotation rate
    inst_norm = np.linalg.norm(gyro, axis=1)
    cv = inst_norm.std() / (inst_norm.mean() + 1e-12)  # coeff variation
    if cv > max_cv:
        return False

    # 3) rotation mainly around z-axis
    z_ratio = abs(mean[2]) / (mean_norm + 1e-12)
    print(z_ratio)
    print("---")
    print(z_ratio < min_z_ratio)
    if z_ratio < min_z_ratio:
        return False

    # 4) expected sign of z-axis rotation
    print(expected_sign)
    if expected_sign is not None and np.sign(mean[2]) != expected_sign:
        return False

    return True


def calibrate_rotation_axis(next_wheel: NextWheel, path: str):
    """
    Calibrate the rotation axis (z-axis) of the wheel by measuring while spinning.
    Parameters
    ----------
    next_wheel : NextWheel
        The NextWheel object to perform measurements.
    path : str
        The path where the calibration data will be saved.
    Returns
    -------
    dict
        The rotation axis data.
    """
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

# %% Accelerometer XZ-plane checks and calibrations
def accelerometer_static_ok(data: dict, stable_threshold: float = 0.05) -> bool:
    """
    Check if the accelerometer data is stable enough.
    std_thresh depends on units (m/s^2 vs g). Tune empirically.
    Parameters
    ----------
    data : dict
        The data dictionary containing IMU data.
    stable_threshold : float, optional
        The maximum allowed standard deviation for each axis. The default is 0.05.
    Returns
    -------
    bool
        True if the accelerometer data is stable enough, False otherwise.
    """
    if not isinstance(data, dict) or "IMU" not in data or "Acc" not in data["IMU"]:
        return False
    accelerometer_data = np.asarray(data["IMU"]["Acc"])
    return np.all(np.std(accelerometer_data, axis=0) < stable_threshold)
def gravity_norm_ok(data: dict, g_min: float = 7.0, g_max: float = 12.0) -> bool:
    """
    Checks mean acceleration magnitude is consistent with gravity.
    Default assumes m/s^2. If your Acc is in g, use g_min=0.8, g_max=1.2.

    Parameters
    ----------
    data : dict
        The data dictionary containing IMU data.
    g_min : float, optional
        Minimum acceptable gravity magnitude. The default is 7.0.
    g_max : float, optional
        Maximum acceptable gravity magnitude. The default is 12.0.
    Returns
    -------
    bool
        True if the mean acceleration magnitude is within the specified range, False otherwise.
    """
    acc = np.asarray(data["IMU"]["Acc"])
    g = np.linalg.norm(acc.mean(axis=0))
    return g_min <= g <= g_max
def acc_y_near_zero_ok(data: dict, max_ratio: float = 0.15) -> bool:
    """
    Check if the y-component of mean acceleration is small compared to total norm.
    This helps ensure the xz-plane is approximately vertical.
    Parameters
    ----------
    data : dict
        The data dictionary containing IMU data.
    max_ratio : float, optional
        Maximum allowed ratio of |mean(ay)| to ||mean(a)||. The default is 0.15.
    Returns
    -------
    bool
        True if the y-component ratio is within the specified limit, False otherwise.
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
    Parameters
    ----------
    data1 : dict
        The first data dictionary containing IMU data.
    data2 : dict
        The second data dictionary containing IMU data.
    min_angle_deg : float, optional
        The minimum angle in degrees between the two mean acceleration vectors. The default is 20.
    Returns
    -------
    bool
        True if the angle between the two mean acceleration vectors is at least min_angle_deg, False otherwise.
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
def calibrate_wheel_frame_orientation_trial(next_wheel: NextWheel, path: str):
    """
    Calibrate the wheel frame orientation using accelerometer data.
    Parameters
    ----------
    next_wheel : NextWheel
        The NextWheel object to perform measurements.
    path : str
        The path where the calibration data will be saved.
    Returns
    -------
    None.
    """
    interface(
        "Keep the wheel COMPLETELY still.\n\n"
        "Click OK to start.",
        "Accelerometer XZ-plane calibration"
    )

    data = safe_measure(next_wheel)
    if not accelerometer_static_ok(data):
        li.message(
            "Rejecting measurement due to excessive accelerometer variance.\n\n"
            "Please ensure the wheel is completely still during the measurement.",
            title="Measurement Rejected",
            icon="error",
        )
        raise RuntimeError("Accelerometer measurement too variable; measurement rejected.")

    if not gravity_norm_ok(data):
        li.message(
            "Rejecting measurement due to inconsistent gravity magnitude.\n\n"
            "Please ensure the wheel is completely still during the measurement.",
            title="Measurement Rejected",
            icon="error",
        )
        raise RuntimeError("Gravity magnitude inconsistent; measurement rejected.")

    if not acc_y_near_zero_ok(data):
        li.message(
            "Rejecting measurement due to excessive y-component of acceleration.\n\n"
            "Please ensure the wheel is oriented correctly during the measurement.",
            title="Measurement Rejected",
            icon="error",
        )
        raise RuntimeError("Y-component of acceleration too large; measurement rejected.")
    save_file(data, "AccForXZPlane", path)
def calibrate_xz_plane_two_trials(next_wheel: NextWheel, path:str, max_tries: int = 5):
    """
    Calibrate the xz-plane of the wheel frame using two distinct accelerometer trials.
    Parameters
    ----------
    next_wheel : NextWheel
        The NextWheel object to perform measurements.
    path : str
        The path where the calibration data will be saved.
    max_tries : int, optional
        Maximum number of attempts to obtain two distinct trials. The default is 5.
    Returns
    -------
    None.
    """
    trials = []
    while len(trials) < 2 and max_tries > 0:
        try:
            calibrate_wheel_frame_orientation_trial(next_wheel, path)
            latest_file = sorted(Path(path).glob("AccForXZPlane*"))[-1]
            data = ktk.load(str(latest_file))
            if len(trials) == 0 or accelerometer_trials_are_different(trials[0], data):
                trials.append(data)
            else:
                li.message(
                    "The new trial is too similar to the previous one.\n\n"
                    "Please change the wheel orientation more significantly.",
                    title="Measurement Rejected",
                    icon="error",
                )
        except RuntimeError as e:
            li.message(str(e), title="Measurement Failed", icon="error")
        max_tries -= 1

    if len(trials) < 2:
        raise RuntimeError("Failed to obtain two distinct accelerometer trials.")


def measure_force_offset(next_wheel: NextWheel, path: str):
    # Measure 1
    interface("Measure the offset force of the channels", "Offset")
    nw.start_streaming()
    nw.fetch()

    offset_measured = measure(nw)
    offset = np.mean(offset_measured["Analog"]["Force"], axis=0)

# %% File picking utilities
def _file_by_name(folder: Path, filename: str):
    file = folder / filename
    if not file.exists():
        raise FileNotFoundError(f"Required file '{filename}' not found in '{folder}'.")
    return file
def _latest_by_prefix(folder: Path, prefix: str, n: int= 1) -> list[Path]:
    """
    Get the n latest files in folder with given prefix.
    """
    candidates: list[tuple[int, Path]] = []
    for file in folder.iterdir():
        if file.name.startswith(prefix):
            suffix = file.name[len(prefix):]
            if suffix.isdigit():
                candidates.append((int(suffix), file))
    if len(candidates)< n:
        raise FileNotFoundError(f"Not enough files with prefix '{prefix}' in '{folder}'. Found {len(candidates)}, needed {n}.")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [file for _, file in candidates[:n]]
def pick_files(folder: Path, prefix: str, n: int, explicit: Iterable[str]):
    if explicit is None:
        return _latest_by_prefix(folder, prefix, n)

    explicit = list(explicit)
    if len(explicit) != n:
        raise ValueError(f"Expected {n} files for prefix '{prefix}', got {len(explicit)}.")
    return [_file_by_name(folder, name) for name in explicit]

# def estimate_calibration_matrix(path: str, gyro_bias_file: str, gyro_z_axis_file:str, acc_xz_files: tuple[str,str], forces_files: list[str]):
#     folder = Path(path)
#     if not folder.exists():
#         raise FileNotFoundError(f"Calibration folder '{folder}' does not exist.")
#
#     gyro_bias_path = pick_files(folder, "GyroForBias", 1, None if gyro_bias_file is None else [gyro_bias_file])[0]
#     gyro_z_axis_path = pick_files(folder, "GyroForZAxis", 1, None if gyro_z_axis_file is None else [gyro_z_axis_file])[0]
#     acc_xz_paths = pick_files(folder, "AccForXZPlane", 2, None if acc_xz_files is None else acc_xz_files)
#     forces_paths = pick_files(folder, "ForcesForCalibrationMatrix", len(forces_files) if forces_files is not None else 0, forces_files)
#
#     if not forces_paths:
#         raise ValueError("At least one forces file is required for calibration matrix estimation.")
#
#     # Load trials
#     gyro_bias_trial = ktk.load(str(gyro_bias_path))
#     gyro_z_axis_trial = ktk.load(str(gyro_z_axis_path))
#     acc_xz_trials = [ktk.load(str(p)) for p in acc_xz_paths]
#     forces_trials = [ktk.load(str(p)) for p in forces_paths]
#
#     # Estimate the gyroscope bias
#     gyro_bias = wc.estimate_gyro_bias(gyro_bias_trial["IMU"]["Gyro"])
#     # Estimate the z-axis
#     z_axis = wc.get_z_axis(gyro_bias, gyro_z_axis_trial["IMU"]["Gyro"])
#
#     # Estimate the frame changing matrix
#     base = wc.get_wheel_reference(
#         -acc_xz_trials[0]["IMU"]["Acc"],
#         -acc_xz_trials[1]["IMU"]["Acc"],
#         z_axis,
#     )
#     # Estimate the accelerometer bias
#     grav_measured = np.vstack([ -np.mean(trial["IMU"]["Acc"], axis=0) for trial in forces_trials ])
#     acc_bias = wc.estimate_acc_bias(grav_measured)
#
#     # Build regression matrices
#     channels = []
#     fms = []
#     for trial in forces_trials:
#         FM = wc.make_an_estimation_of_forces_moments(
#             trial,
#             acc_bias,
#             base,
#         )
#         force = np.median(trial["Analog"]["Force"], axis=0)
#         channels.append(force)
#         fms.append(FM)
#
#     FMs = np.vstack(fms)
#     Channels = np.vstack(channels)
#     A = wc.calculate_calibration_matrix(FMs, Channels)
#     return A, {
#         "GyroBias": gyro_bias,
#         "Z-Axis": z_axis,
#         "Base": base,
#         "AccBias": acc_bias,
#         "Trials": {
#             "GyroForBias": gyro_bias_trial,
#             "GyroForZAxis": gyro_z_axis_trial,
#             "AccForXZPlane": acc_xz_trials,
#             "ForcesForCalibrationMatrix": forces_trials,
#         }
#     }

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





if __name__ == "__main__":
    ip = sys.argv[-1]
    nw = nextwheel.NextWheel(ip)
    ROOT = Path(__file__).resolve().parent
    path = str(ROOT) + "/"
    trials_dir = "package_trials_good/mass_3_03_Z-"

# # %% Part 1 - Z-axis calculated from gyroscope
#
# calibrate_part1(nw, path + trials_dir)
#
# # %% Part 2 - XZ-Plane determined from acc + base change matrix completion
#
# calibrate_part2(nw, path + trials_dir)

# %% Part 3 - More static Force mesures for calibration matrix

calibrate_part3(nw, path + trials_dir)
while not li.button_dialog(
    "Do you want to do another measure ?",
    choices=["Oui", "Non"],
    title="Force Measures for Calibration Matrix",
    icon="gear",
):
    calibrate_part3(nw, path + trials_dir)

# %% Part 4 - Calibration matrix calculation
# A, Trials = estimate_calibration_matrix(path + trials_dir)
# print(A)
# print(f"Calibration matrix estimated and printed above.")
# # print(Trials)
# if __name__ == "__main__":
#
#     # if len(sys.argv) < 2:
#     #     raise SystemExit("Usage: python calibrationwizard.py <ip>")
#     #
#     # ip = sys.argv[-1]
#     # nw = nextwheel.NextWheel(ip)
#     #
#     ROOT = Path(__file__).resolve().parent
#     trials_dir = ROOT / "Mass_1_73"   # pas besoin de / final
#     print(_latest_by_prefix(trials_dir, "GyroForBias", n=9))
#     # trials_dir.mkdir(parents=True, exist_ok=True)
#     #
#     # # calibrate_gyro_bias(nw, trials_dir)
#     # calibrate_rotation_axis(nw, trials_dir)
#     # # Puis tu appelles tes fonctions avec ce dossier:
#     # # calibrate_part1(nw, trials_dir)
#     # # calibrate_part2(nw, trials_dir)
#     # # calibrate_part3(nw, trials_dir)
