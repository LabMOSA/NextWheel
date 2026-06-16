from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from nextwheel import NextWheel

power_values = []
voltage_values = []
current_values = []
power_time = []

encoder_values = []
encoder_time = []

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=["r", "g", "b", "c", "m", "y", "k", "tab:orange"]
)
mpl.rcParams["figure.figsize"] = [10, 5]
mpl.rcParams["figure.dpi"] = 75
mpl.rcParams["lines.linewidth"] = 1

# define and adjust figure
fig = plt.figure(figsize=(12, 8))

adc_plot = plt.subplot(2, 1, 1)  # row, col, index
adc_plot.set_title("ADC")

imu_plot = plt.subplot(2, 1, 2)  # row, col, index
imu_plot.set_title("IMU")
#
# power_plot = plt.subplot(4, 1, 3)  # row, col, index
# power_plot.set_title("POWER")
#
# encoder_plot = plt.subplot(4, 1, 4)  # row, col, index
# encoder_plot.set_title("ENCODER")
plt.tight_layout()


# # funct to update the data
def update_plots(i):
    """
    Update the plots.

    This function updates the plots using global variables adc_values,
    imu_values, power_values and encoder_values. It is normally called
    by Matplotlib's FunctAnimation class.

    """
    data = nw.fetch()

    # ADC
    adc_plot.cla()
    adc_plot.set_title(f"[{i}] - Force Channels (ADC)")
    adc_plot.plot(data["Analog"]["Time"], data["Analog"]["Force"])

    # IMU
    imu_plot.cla()
    imu_plot.set_title("IMU")
    imu_plot.plot(
        # data["IMU"]["Time"],
        # data["IMU"]["Acc"],
        data["IMU"]["Time"],
        data["IMU"]["Gyro"],
        # data["IMU"]["Time"],
        # data["IMU"]["Mag"],
    )
    axes = imu_plot.axis()
    imu_plot.axis([axes[0], axes[1], -1000, 1000])
    #
    # # POWER Sample frequency too slow to just use nw.fetch() alone
    # if len(data["Power"]["Time"]) != 0:
    #     power_time.append(data["Power"]["Time"][0])
    #     power_values.append(data["Power"]["Power"][0])
    #     voltage_values.append(data["Power"]["Voltage"][0])
    #     current_values.append(data["Power"]["Current"][0])
    #
    # if len(power_time) > 10:
    #     power_time.pop(0)
    #     power_values.pop(0)
    #     voltage_values.pop(0)
    #     current_values.pop(0)
    #
    # power_plot.cla()
    # power_plot.set_title("Battery level")
    # power_plot.plot(
    #     power_time,
    #     voltage_values,
    #     power_time,
    #     current_values,
    #     power_time,
    #     power_values,
    # )
    #
    # # ENCODER Sample frequency too slow to just use nw.fetch() alone
    # if len(data["Encoder"]["Time"]) != 0:
    #     encoder_time.append(data["Encoder"]["Time"][0])
    #     encoder_values.append(data["Encoder"]["Angle"][0])
    #
    # if len(power_time) > 10:
    #     encoder_time.pop(0)
    #     encoder_values.pop(0)
    #
    # encoder_plot.cla()
    # encoder_plot.set_title("Encoder")
    # encoder_plot.plot(encoder_time, encoder_values)


if __name__ == "__main__":
    # websocket.enableTrace(True)  # Uncomment to print all received data
    nw = NextWheel("192.168.1.155")
    nw.connect()

    # Set device to current time (unix time)
    nw.set_time(str(int(datetime.now().timestamp())))

    ani = FuncAnimation(fig, update_plots, interval=100)
    plt.show()
