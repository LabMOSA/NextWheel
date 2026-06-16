import matplotlib.pyplot as plt
from nextwheel_parser import (
    parse,
)

filename = "long_avec_wifi_sans_ampli_batterie.dat"

# define and adjust figure
fig = plt.figure(figsize=(12, 8), facecolor="#DEDEDE")

adc_plot = plt.subplot(4, 1, 1)  # row, col, index
adc_plot.set_title("ADC")

imu_plot = plt.subplot(4, 1, 2)  # row, col, index
imu_plot.set_title("IMU")

power_plot = plt.subplot(4, 1, 3)  # row, col, index
power_plot.set_title("POWER")

encoder_plot = plt.subplot(4, 1, 4)  # row, col, index
encoder_plot.set_title("ENCODER")


if __name__ == "__main__":
    with open(filename, "rb") as f:
        data = f.read()
        print("file size : ", len(data))
        results = parse(data)

        x_vals = [x[0] for x in results["adc_values"]]
        y_vals = [x[1] for x in results["adc_values"]]
        adc_plot.plot(x_vals, y_vals, "-")

        # IMU
        imu_plot.cla()
        imu_plot.set_title("IMU")
        x_vals = [x[0] for x in results["imu_values"]]
        y_vals = [x[1] for x in results["imu_values"]]
        imu_plot.plot(x_vals, y_vals)

        # POWER
        power_plot.cla()
        power_plot.set_title("POWER")
        x_vals = [x[0] for x in results["power_values"]]
        y_vals = [x[1][0:3] for x in results["power_values"]]
        power_plot.plot(x_vals, y_vals)

        # ENCODER
        encoder_plot.cla()
        encoder_plot.set_title("ENCODER")
        x_vals = [x[0] for x in results["encoder_values"]]
        y_vals = [x[1] for x in results["encoder_values"]]
        encoder_plot.plot(x_vals, y_vals)

        # plt.show()
        plt.tight_layout()
