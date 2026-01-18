"""Functions to calibrate an instrumented wheelchair wheel."""

import numpy as np
import math


def estimate_gyro_bias(omega_static: np.ndarray) -> np.ndarray:
    """
    Estimate the gyro bias using a static trial.

    Parameters
    ----------
    omega_static
        Angular speed measured with the IMU (gyro) during a static trial, in
        the form of an Nx3 array. Since the wheel does not move during a static
        trial, this is the gyro bias.

    Returns
    -------
    array
        Bias, which is the mean of the omega_static argument.

    """
    return np.mean(omega_static, axis=0)


def estimate_acc_bias(
    acc_statics: np.ndarray,
) -> np.ndarray:
    """
    Estimate the accelerometer bias using three static trials.

    The function needs three different static trials to estimate the bias of
    the accelerometer of the IMU. In short, the accelerometer measures a vector
    m that contain the measurement in x, y, z. If we observe the norm. we
    have::

    m_x^2 + m_y^2 + m_z^2 = (a_x + b_x)^2 + (a_y + b_y)^2 + (a_z + b_z)^2 = N^2.

    Where m_x, m_y and m_z are measure of the accelerometer, ax, ay and az are
    the acceleration of the gravity and b_x, b_y and b_z are the bias. N is the
    norm of the m vector.

    If we develop, we obtain::

    a_x^2 + a_y^2 + a_z^2
    + 2 * (a_x * b_x + a_y * b_y + a_z * b_z)
    + b_x^2 + b_y^2 + b_z^2
    = N^2

    |g|^2
    + 2 * (a_x * b_x + a_y * b_y + a_z * bz_)
    + |b|^2 = N^2

    Using two static trials with two different orientations and subtracting
    these trials (the second trial has primes ' in the following equations),
    the constants cancel and we obtain::

    (N^2 - N'^2) / 2
     = b_x * (a_x - a_x') + b_y * (a_y - a_y') + b_z * (a_z - a_z')

    with::

    (a_x - a_x') = (m_x - m_x') because the bias cancels out;
    (a_y - a_y') = (m_y - m_y') because the bias cancels out;
    (a_z - a_z') = (m_z - m_z') because the bias cancels out.

    Therefore::

    (N^2 - N'^2) / 2
     = b_x * (m_x - m_x') + b_y * (m_y - m_y') + b_z * (m_z - m_z')

    With a minimum of three static trials, we can find the bias with a linear
    system Ax = b::

    [
        [(a_x1 - a_x2), (a_y1 - a_y2), (a_z1 - a_z2)],
        [(a_x1 - a_x3), (a_y1 - a_y3), (a_z1 - a_z3)],
        [(a_x2 - a_x3), (a_y2 - a_y3), (a_z2 - a_z3)]
    ]
    *
    [
        [b_x],
        [b_y],
        [b_z]
    ]
    =
    [
        [(N1^2 - N2^2) / 2],
        [(N1^2 - N3^2) / 2],
        [(N2^2 - N3^2) / 2]
    ]

    This function uses a least-square estimation to get the bias from any
    number of static acquisitions higher than 3.

    Parameters
    ----------
    acc_statics
        Contain at least N>=3 different static accelerometer measurements
        taken in different orientations of the wheel, in the form of an Nx3
        array.

    Returns
    -------
    np.ndarray
        Estimated bias as an array of length 3.

    """
    m, n = np.shape(acc_statics)
    norms = np.linalg.norm(acc_statics, axis=1)

    m_line_matrix = int(math.factorial(m) / (2 * math.factorial(m - 2)))

    delta_grav_matrix = np.zeros((m_line_matrix, 3))
    delta_norm_square = np.zeros((m_line_matrix, 1))

    i = 0
    for j in range(m):
        for k in range(j + 1, m):
            delta_grav_matrix[i, :] = acc_statics[j, :] - acc_statics[k, :]
            delta_norm_square[i, 0] = (norms[j] ** 2 - norms[k] ** 2) / 2
            i += 1

    bias = np.linalg.lstsq(delta_grav_matrix, delta_norm_square, rcond=None)

    return bias[0][:, 0]


def get_z_axis(gyro_bias: np.ndarray, omega_dynamic: np.ndarray) -> np.ndarray:
    """
    Calculate the z-axis of the wheel in the IMU's reference frame.

    The z-axis is calculated by rotating the wheel along its rotation axis.

    Parameters
    ----------
    gyro_bias
        Gyro bias as reported by estimate_gyro_bias().
    omega_dynamic
        Angular speed measured with the IMU (gyro) as the wheel turns around
        its rotation axis, as an Nx3 series of angular speeds v_x, v_y, v_z.

    Returns
    -------
    np.ndarray
        The z-axis of the IMU as an array of length 3.

    """
    omega_dynamic -= np.tile(gyro_bias, (np.shape(omega_dynamic)[0], 1))

    z_axis = omega_dynamic / np.transpose(
        np.tile(np.linalg.norm(omega_dynamic, axis=1), (3, 1))
    )

    z_axis = np.mean(
        z_axis,
        axis=0,
    )
    z_axis = z_axis / np.linalg.norm(z_axis)

    return z_axis


def get_wheel_reference(
    acc_static1: np.ndarray, acc_static2: np.ndarray, z_axis: np.ndarray
) -> np.ndarray:
    """
    Calculate the rotation matrix from the IMU to the wheel's reference frames.

    The IMU measures the acceleration with the accelerometer and the gravity is
    measured as well. The static trials measure two orientations of the gravity
    (with bias). If both static trials correspond to a unique rotation around
    the y axis and both acceleration are purely in the xz-plane, then by
    subtracting one from the other, we cancel the bias and the new vector is in
    the xz-plane. From there, we can apply a cross product to find y and then
    x.

    Parameters
    ----------
    acc_static1
        Gravity measured with the IMU (accelerometer) with a static trial.
    acc_static2
        Similar to acc_static1 but with a different angle in the xz plane.
    z_axis
        The z-axis as given by get_z_axis().

    Returns
    -------
    np.ndarray
        The complete wheel reference rotation matrix.

    """
    grav1 = np.mean(acc_static1, axis=0)
    grav2 = np.mean(acc_static2, axis=0)

    if np.abs(np.dot(grav1, z_axis)) > np.abs(np.dot(grav2, z_axis)):
        delta_grav = grav2 - grav1
    else:
        delta_grav = grav1 - grav2

    y_axis = np.cross(z_axis, delta_grav + z_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    wheel_ref = np.vstack((x_axis, y_axis, z_axis))

    return wheel_ref


def calculate_calibration_matrix(FM: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Calculate the calibration matrix from known kinetics and sensor voltages.

    We resolve the A @ V^T = FM^T to find A, the calibration matrix. Let's
    rewrite the equation with V @ A^T = FM. This can be seen as n linear
    equations to resolve. FM and V must have at least 6 lines (6 trials).

    Exemple :

        [(V @ a1) (V @ a2) (V @ a3) ...] = [f1 f2 f3 ...]

    where ak is the kth line of A and fk is the kth column of FM. We have the n
    linear equations that follows :

        V @ a1 = f1
        V @ a2 = f2
        V @ a3 = f3
            ...
        V @ an = fn

    Parameters
    ----------
    FM : np.ndarray
        Forces and moments matrix (N,6) of N trials. The three first columns
        must be the force in x, y, z and the last three values are the moments
        in x, y, z.
    V : np.ndarray
        Voltages measured with the EMG installed in the instrumented wheel for
        each trials. Must be the same shape as FM.

    Returns
    -------
    np.ndarray
        The calibration matrix A.

    """
    return np.linalg.lstsq(V, FM, rcond=None)[0].T


def make_an_estimation_of_forces_moments(
    trial: dict,
    acc_bias: np.ndarray,
    wheel_ref: np.ndarray,
    d: float = 0.52,
    h: float = 0.05,
):
    """
    Estimate the forces and moments of one static trial with a suspended weight.

    A static trial is expressed as a dictionary with at least the following
    keys::
    {
        "Mass": float,    # Mass of the suspended weight
        "Degree": float,  # Position of the suspended weight on the wheel
        "IMU": {
            "Acc": np.array  # Nx3 array of accelerometer measurements
        }
    }

    Parameters
    ----------
    trial
        Trial data in the form presented above.
    acc_bias
        The accelerometer bias.
    wheel_ref
        The 3x3 Matrix that link the IMU referential to the wheel referential.
    d
        Optional. The diameter of the pushrim. The default is 0.52 m.
    h
        Optional. The perpendicular distance between the wheel plane and the
        pushrim. It is the z-distance in cyclindrical coordinates of the wheel.
        The default is 0.05 m.

    Returns
    -------
    FM : np.ndarray
        The theorical estimate of the forces and moments applied on the
        pushrim.

    """
    force_application_point = np.ndarray((3,))
    force_application_point[0] = (
        0.5 * d * np.cos(np.pi * trial["Degree"] / 180)
    )
    force_application_point[1] = (
        0.5 * d * np.sin(np.pi * trial["Degree"] / 180)
    )
    force_application_point[2] = h
    force_application_point = np.transpose(force_application_point)
    ref_grav = np.transpose(np.mean(trial["IMU"]["Acc"], axis=0))
    ref_grav = (ref_grav - acc_bias) / np.linalg.norm(ref_grav - acc_bias)

    f1 = -1 * trial["Mass"] * 9.81 * np.dot(wheel_ref[:3, :3], ref_grav)
    f1 = np.transpose(f1)

    forces = f1
    moments = -np.cross(f1, force_application_point)

    FM = np.hstack((forces, moments))

    return FM
