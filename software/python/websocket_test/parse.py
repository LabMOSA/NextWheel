"""
Parse file logs from python app as Kinetics Toolkit TimeSeries.

Requires:
    >> conda install -c conda-forge kineticstoolkit

"""

import kineticstoolkit.lab as ktk
import matplotlib.pyplot as plt
import pandas as pd


def read_log(filename: str) -> ktk.TimeSeries:
    """Read log, format time in seconds since beginning."""
    # Read ADC
    df = pd.read_csv(filename)
    df.index = (
        df["h"] * 3600 * 1e6 + df["m"] * 60 * 1e6 + df["s"] * 1e6 + df["us"]
    ) / 1e6

    df.index = df.index - df.index[0]
    df = df.drop(columns=["Y", "M", "D", "h", "m", "s", "us"])
    return ktk.TimeSeries.from_dataframe(df)


# %% ADC

ts_adc = read_log("log_adc.csv")
ts_adc.data["ch"] = ts_adc.data["ch"][:, 0:6]

plt.figure()
ts_adc.plot()
plt.title("ADC")

# %% Sampling frequency histogram

plt.figure()
plt.hist(ts_adc.time[1:] - ts_adc.time[0:-1], bins=100)
plt.title("ADC Sampling rate distribution")

# %% IMU

ts_imu = read_log("log_imu.csv")
plt.figure()
plt.subplot(1, 3, 1)
ts_imu.plot("acc")
plt.subplot(1, 3, 2)
ts_imu.plot("gyro")
plt.subplot(1, 3, 3)
ts_imu.plot("mag")
