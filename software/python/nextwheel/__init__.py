# -*- coding: utf-8 -*-
#
# Copyright 2023 NextWheel Developers

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This Python module provides the NextWheel class that fetches data from an
intrumented wheel.
"""

import websocket
import struct
import threading
import numpy as np
from enum import IntEnum
import requests
import os
import json

# Constants

GRAVITY = 9.80665


class FrameType(IntEnum):
    """Identify frame type from websocket message."""

    CONFIG = 1
    ADC = 2
    IMU = 3
    POWER = 4
    ENCODER = 7
    SUPERFRAME = 255


class GlobalConfig:
    """
    Global configuration of the instrumented wheel.

    This is be used to store the current configuration and calculate
    conversions from raw data.

    """

    def __init__(self):
        # IMU CONFIG
        self.accel_range = 16
        self.gyro_range = 2000
        self.mag_range = 2500
        self.imu_rate = 240

        # ADC CONFIG
        self.adc_rate = 240

        # ENCODER CONFIG
        self.encoder_rate = 240

        # ACCORDING TO ADS8688 DATASHEET
        self._adc_v_ref = 4.096
        # ADC RANGING from [-5,5] V according to PCB design
        self._adc_in_max = 1.25 * self._adc_v_ref
        self._adc_in_min = -1.25 * self._adc_v_ref

        # ACCORDING TO BMX160 DATASHEET
        self._accel_mg_lsb_2g = 0.000061035
        self._accel_mg_lsb_4g = 0.000122070
        self._accel_mg_lsb_8g = 0.000244141
        self._accel_mg_lsb_16g = 0.000488281
        self._gyro_sensitivity_125dps = 0.0038110
        self._gyro_sensitivity_250dps = 0.0076220
        self._gyro_sensitivity_500dps = 0.0152439
        self._gyro_sensitivity_1000dps = 0.0304878
        self._gyro_sensitivity_2000dps = 0.0609756
        self._mag_ut_lsb = 0.3

    def convert_adc_values(self, values: np.ndarray) -> np.ndarray:
        """
        Convert ADC values from ADC integers to volts.

        Parameters
        ----------
        values
            ADC outputs (integer).

        Returns
        -------
        np.ndarray
            ADC inputs in volts.

        """
        # This is according to the 86888 datasheet
        return (
            values * (self._adc_in_max - self._adc_in_min) / 65535
            + self._adc_in_min
        )

    def convert_accel_values(self, values: np.ndarray) -> np.ndarray:
        """
        Convert acceleration values from integers to m/s2.

        Parameters
        ----------
        values
            Integers as reported by the sensor.

        Returns
        -------
        np.ndarray
            Acceleration in m/s2.

        """
        factor = {
            2: self._accel_mg_lsb_2g * GRAVITY,
            4: self._accel_mg_lsb_4g * GRAVITY,
            8: self._accel_mg_lsb_8g * GRAVITY,
            16: self._accel_mg_lsb_16g * GRAVITY,
        }

        try:
            return values * factor[self.accel_range]
        except KeyError:
            raise ValueError("Invalid accel range")

    def convert_gyro_values(self, values: np.ndarray) -> np.ndarray:
        """
        Convert gyro values from integers to deg/s.

        Parameters
        ----------
        values
            Integers as reported by the sensor.

        Returns
        -------
        np.ndarray
            Angular velocity in m/s2.

        """
        factor = {
            125: self._gyro_sensitivity_125dps,
            250: self._gyro_sensitivity_250dps,
            500: self._gyro_sensitivity_500dps,
            1000: self._gyro_sensitivity_1000dps,
            2000: self._gyro_sensitivity_2000dps,
        }

        try:
            return values * factor[self.gyro_range]
        except KeyError:
            raise ValueError("Invalid gyro range")

    def convert_mag_values(self, values: np.ndarray) -> np.ndarray:
        """
        Convert magnetometer integers to ??.

        Parameters
        ----------
        values
            Integers as reported by the sensor.

        Returns
        -------
        np.ndarray
            To be determined.

        """
        # FIXME what is the output unit?
        if self.mag_range == 2500:
            return values * self._mag_ut_lsb
        else:
            raise ValueError("Invalid mag range")


class NextWheel:
    """
    Communicate with the instrumented wheel.

    The class permit to communicate with the instrumented wheel with 3 methods:
        - NextWheel.connect() allows to start a connection with the server via
        the WebSocketApp.
        - NextWheel.fetch() return the streamed data in a nested dictionary
        with all the usefull informations. Also clear the buffer.
        - NextWheel.close() close the connection with the instrumented wheel.
    It needs an IP address like in the example below.

    Exemple
    -------
    >>> from nextwheel import NextWheel
    >>> nw = NextWheel("196.168.1.254")
    >>> nw.connect()

    >>> print(nw.fetch())

    >>> nw.close()

    """

    def __init__(self, IP: str, *, debug: bool = False):
        # General configuration
        self.IP = IP
        self.HEADER_LENGTH = 10
        self.TIME_ZERO = 0
        self.max_imu_samples = 0
        self.max_analog_samples = 0
        self.max_encoder_samples = 0
        self.max_power_samples = 0
        self._config = GlobalConfig()
        self._debug = debug

        # Communication stuff
        self.ws = None
        self._mutex = threading.Lock()
        self._thread_is_running = False

        # Data buffers. In these arrays, column 0 is the time
        self._adc_values = []  # type: list[np.ndarray]
        self._imu_values = []  # type: list[np.ndarray]
        self._power_values = []  # type: list[np.ndarray]
        self._encoder_values = []  # type: list[np.ndarray]

        # Calibration constants

        try:
            self.file_download("Calibration.json")
            with open("Calibration.json", "r") as json_file:
                self.CALIBRATION = json.load(json_file)
            self.CALIBRATION_MATRIX = np.array(self.CALIBRATION["Matrix"])
            self.CALIBRATION_OFFSET = np.array(self.CALIBRATION["Offset"])
        except:
            self.CALIBRATION_MATRIX = np.identity(6)
            self.CALIBRATION_OFFSET = np.zeros((6,))
            print("No Calibration File Detected")

    def _parse_message(self, stream: bytes, offset: int = 0) -> int:
        """
        Parse a series of bytes corresponding to a messages.

        It only parses the first message from the stream, then returns the
        position of the next message in the stream. If the received message
        is a superframe, all messages from the superframe are parsed and
        processed.

        Parameters
        ----------
        stream
            The received message or the data stored in the dat file.
        offset
            Start processing at this position in the stream.

        Returns
        -------
        offset
            The position of the next message in the stream.


        """
        if type(stream) is not bytes:
            return

        if offset >= len(stream):
            return offset

        # Extract the type of message
        (frame_type, timestamp, data_size) = struct.unpack(
            "<BQB", stream[offset:offset+10]
        )
        offset += 10
        time = timestamp / 1e6 - self.TIME_ZERO

        # Process the frame
        if frame_type == FrameType.CONFIG:

            # Config frame (should always be first)
            data = stream[offset:offset + data_size]
            offset += data_size

            self.TIME_ZERO = timestamp / 1e6

            (
                accel_range,
                gyro_range,
                mag_range,
                imu_sampling_rate,
                adc_sampling_rate,
                encoder_sampling_rate,
            ) = struct.unpack("<6I", data)

            # Update configuration
            self._config.accel_range = accel_range
            self._config.gyro_range = gyro_range
            self._config.mag_range = mag_range
            self._config.imu_sampling_rate = imu_sampling_rate
            self._config.adc_sampling_rate = adc_sampling_rate
            self._config.encoder_sampling_rate = encoder_sampling_rate

        elif frame_type == FrameType.ADC:  # frame type of the ADC values
            data = stream[offset:offset + data_size]
            offset += data_size

            self._adc_values.append(
                np.hstack([[time], struct.unpack("<8H", data)])
            )

            if len(self._adc_values) > self.max_analog_samples:
                self._adc_values.pop(0)

        elif frame_type == FrameType.IMU:  # frame type of the IMU
            data = stream[offset:offset + data_size]
            offset += data_size

            self._imu_values.append(
                np.hstack(
                    [
                        [time],
                        self._config.convert_accel_values(
                            np.array(struct.unpack_from("<3h", data[:6]))
                        ),
                        self._config.convert_gyro_values(
                            np.array(struct.unpack_from("<3h", data[6:12]))
                        ),
                        self._config.convert_mag_values(
                            np.array(struct.unpack_from("<3h", data[12:18]))
                        ),
                    ]
                )
            )

            if len(self._imu_values) > self.max_imu_samples:
                self._imu_values.pop(0)

        elif frame_type == FrameType.POWER:  # frame type of the POWER
            data = stream[offset:offset + data_size]
            offset += data_size

            self._power_values.append(
                np.hstack([[time], struct.unpack_from("<3fB", data)])
            )

            if len(self._power_values) > self.max_power_samples:
                self._power_values.pop(0)

        elif frame_type == FrameType.ENCODER:  # frame type of the ENCODER
            data = stream[offset:offset + data_size]
            offset += data_size

            self._encoder_values.append(
                np.hstack([[time], struct.unpack_from("<q", data)])
            )

            if len(self._encoder_values) > self.max_encoder_samples:
                self._encoder_values.pop(0)

        elif frame_type == FrameType.SUPERFRAME:

            for sub_count in range(data_size):  # data_size = number of frames
                offset = self._parse_message(stream, offset)

        else:
            raise ValueError(f"Received an unknown frame type: {frame_type}")

        return offset


    def _on_message(self, ws, message):
        """
        React to WebSocketApp message received.

        Simply a wrapper to the more generic _parse_messages function.

        Parameters
        ----------
        ws : _app.WebSocketApp
            DESCRIPTION.
        message : bytes
            Information containing data in bytes.

        Returns
        -------
        None.

        """
        self._mutex.acquire()
        self._parse_message(message)
        self._mutex.release()

    def _on_open(self, ws):
        """Reaction of the WebSocketApp when the connection is open."""
        if self._debug:
            print("Connected: ", self.ws)

    def _on_error(self, ws, error):
        """Reaction of the WebSocketApp when there is an error."""
        print(self.ws, error)
        self.stop_streaming()

    def _on_close(self, ws, close_status_code, close_msg):
        """Reaction of the WebSocketApp when the connection is close."""
        if self._debug:
            print("Closed: ", self.ws, close_status_code, close_msg)

    def start_streaming(
        self,
        max_imu_samples: int = 100000,
        max_analog_samples: int = 100000,
        max_encoder_samples: int = 100000,
        max_power_samples: int = 10,
    ):
        """
        Start streaming.

        Parameters
        ----------
        max_imu_samples : int, optional
            Maximum IMU data to keep in memory. The default is 1000.
        max_analog_samples : int, optional
            Maximum ADC data to keep in memory. The default is 1000.
        max_encoder_samples : int, optional
            Maximum encoder data to keep in memory. The default is 100.
        max_power_samples : int, optional
            Maximum Power data to keep in memory. The default is 10.

        Returns
        -------
        None.

        """
        if self._thread_is_running:
            return

        self.max_imu_samples = max_imu_samples
        self.max_analog_samples = max_analog_samples
        self.max_encoder_samples = max_encoder_samples
        self.max_power_samples = max_power_samples

        self.ws = websocket.WebSocketApp(
            f"ws://{self.IP}:81/",
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        t = threading.Thread(target=self.ws.run_forever)  # type: ignore
        t.start()
        self._thread_is_running = True

    def stop_streaming(self):
        """
        Stop streaming via websocket.

        Returns
        -------
        None.

        """
        self.ws.close()
        self._thread_is_running = False
        try:
            # If something has crashed somewhere, release the mutex for the
            # next connection.
            self._mutex.release()
        except RuntimeError:
            # No mutex was locked. That's fine.
            pass

    def monitor(self) -> None:
        """
        Blocking function that monitors the sensors measurements.

        This function shows the current sensor states in a simple GUI for
        testing and monitoring. To log data, use NextWheel.fetch() instead.

        """
        # Don't import until needed.
        import nextwheel.monitor as nwm  # noqa

        self.start_streaming()
        nwm.monitor(self)
        self.stop_streaming()

    def graphical_monitor(self) -> None:
        """
        Blocking function that monitors using Matplotlib.

        Under development.

        """
        import matplotlib as mpl  # noqa
        import matplotlib.pyplot as plt  # noqa
        from matplotlib import animation  # noqa

        is_running = [True]

        def on_close(_):
            is_running[0] = False

        self.start_streaming()
        plt.pause(0.5)

        # Matplotlib defaults
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
            color=["r", "g", "b", "c", "m", "y", "k", "tab:orange"]
        )
        mpl.rcParams["figure.figsize"] = [10, 5]
        mpl.rcParams["figure.dpi"] = 75
        mpl.rcParams["lines.linewidth"] = 1

        # Create figure and plots
        fig = plt.figure(figsize=(12, 8))
        fig.canvas.mpl_connect("close_event", on_close)

        adc_plot = plt.subplot(4, 1, 1)
        adc_plot.set_title("ADC")
        adc_lines = []
        for i in range(6):
            adc_lines.append(adc_plot.plot([])[0])
        adc_plot.set_xlim(0, self.max_analog_samples)
        adc_plot.set_ylim(0, 65536)
        adc_plot.set_ylabel("Sampled value")

        accel_plot = plt.subplot(4, 1, 2)
        accel_plot.set_title("Accelerometers")
        accel_lines = []
        for i in range(3):
            accel_lines.append(accel_plot.plot([])[0])
        accel_plot.set_xlim(0, self.max_imu_samples)
        accel_plot.set_ylim(-20, 20)
        accel_plot.set_ylabel("m/s2")

        gyro_plot = plt.subplot(4, 1, 3)
        gyro_plot.set_title("Gyrometers")
        gyro_lines = []
        for i in range(3):
            gyro_lines.append(gyro_plot.plot([])[0])
        gyro_plot.set_xlim(0, self.max_imu_samples)
        gyro_plot.set_ylim(-1000, 1000)
        gyro_plot.set_ylabel("Unit to be defined")

        encoder_plot = plt.subplot(4, 1, 4)
        encoder_plot.set_title("Encoder")
        encoder_line = encoder_plot.plot([])[0]
        encoder_plot.set_xlim(0, self.max_encoder_samples)
        encoder_plot.set_ylim(0, 4000)
        encoder_plot.set_ylabel("Unit to be defined")

        plt.tight_layout()

        def on_timer(_):
            self._mutex.acquire()
            adc_values = np.zeros((self.max_analog_samples, 9))
            adc_values[-len(self._adc_values):] = np.array(self._adc_values)

            imu_values = np.zeros((self.max_imu_samples, 10))
            imu_values[-len(self._imu_values):] = np.array(self._imu_values)

            encoder_values = np.zeros((self.max_encoder_samples, 2))
            encoder_values[-len(self._encoder_values):] = np.array(self._encoder_values) % 4000
            self._mutex.release()

            try:
                for i in range(6):
                    adc_lines[i].set_data(
                        range(self.max_analog_samples), adc_values[:, i + 1]
                    )
            except ValueError:
                pass

            try:
                for i in range(3):
                    accel_lines[i].set_data(
                        range(self.max_imu_samples), imu_values[:, i + 1]
                    )

                for i in range(3):
                    gyro_lines[i].set_data(
                        range(self.max_imu_samples), imu_values[:, i + 4]
                    )
            except ValueError:
                pass

            try:
                encoder_line.set_data(
                    range(self.max_encoder_samples), encoder_values[:, 1]
                )
            except ValueError:
                pass

        anim = animation.FuncAnimation(
            fig,
            on_timer,  # type: ignore
            interval=33,
            cache_frame_data=False,
        )  # 30 ips

        while is_running[0]:
            plt.pause(0.5)

        self.stop_streaming()

    def fetch(self) -> dict[str, dict[str, np.ndarray]]:
        """
        Fetch data and return a nested dictionary. Clear the buffer.

        Returns
        -------
        data : dict[str, dict[str, np.ndarray]]
            A dictionary of multiple dictionaries that contain latest
            informations on:
                - IMU values
                - Analog values
                - Encoder values
                - Power values
        """
        self._mutex.acquire()

        # Concatenate all samples in big numpy arrays
        adc_values = np.array(self._adc_values)
        imu_values = np.array(self._imu_values)
        encoder_values = np.array(self._encoder_values)
        power_values = np.array(self._power_values)

        # Reset the buffers
        self._adc_values = []
        self._imu_values = []
        self._encoder_values = []
        self._power_values = []

        self._mutex.release()

        has_imu = len(imu_values) > 0
        has_adc = len(adc_values) > 0
        has_enc = len(encoder_values) > 0
        has_pow = len(power_values) > 0

        if len(adc_values) > 0:
            adc_values[:, 1:7] = (
                np.dot(self.CALIBRATION_MATRIX, adc_values[:, 1:7].T).T
                - self.CALIBRATION_OFFSET
            )

        # Output
        data = {
            "IMU": {
                "Time": imu_values[:, 0] if has_imu > 0 else np.array([]),
                "Acc": imu_values[:, 1:4] if has_imu > 0 else np.array([]),
                "Gyro": imu_values[:, 4:7] if has_imu > 0 else np.array([]),
                "Mag": imu_values[:, 7:] if has_imu > 0 else np.array([]),
            },
            "Analog": {
                "Time": adc_values[:, 0] if has_adc > 0 else np.array([]),
                "Force": adc_values[:, 1:7] if has_adc > 0 else np.array([]),
                "Spare": adc_values[:, 7:] if has_adc > 0 else np.array([]),
            },
            "Encoder": {
                "Time": encoder_values[:, 0] if has_enc > 0 else np.array([]),
                "Angle": encoder_values[:, 1] if has_enc > 0 else np.array([]),
            },
            "Power": {
                "Time": power_values[:, 0] if has_pow > 0 else np.array([]),
                "Voltage": power_values[:, 1] if has_pow > 0 else np.array([]),
                "Current": power_values[:, 2] if has_pow > 0 else np.array([]),
                "Power": power_values[:, 3] if has_pow > 0 else np.array([]),
            },
        }

        return data

    def set_time(self, unix_time: int) -> dict:
        """
        Set the time of the instrumented wheel.

        Parameters
        ----------
        unix_time : int
            Unix time in seconds.

        Returns
        -------
        requests.Response

        """
        response = requests.post(
            f"http://{self.IP}/config_set_time", params={"time": unix_time}
        )
        return json.loads(response.content)

    def set_sensors_params(
        self,
        adc_sampling_rate: int,
        imu_sampling_rate: int,
        encoder_sampling_rate: int,
        accelerometer_precision: int,
        gyrometer_precision: int,
    ) -> requests.Response:
        """
        Set the parameters of the instrumented wheel.

        Parameters
        ----------
        adc_sampling_rate
            Sampling rate for the forces in Hz. Valid values are 120, 240,
            480, 960, 1000, 2000
        imu_sampling_rate
            Sampling rate for the IMU, in Hz. Valid values are 60, 120, 240.
        encoder_sampling_rate
            Sampling rate for the encoder, in Hz. Valid values are 60, 120, 240.
        accelerometer_precision
            Accelerometer range, in g. Value values are 2, 4, 8, 16.
        gyrometer_precision
            Gyrometer range, in degrees per second. Valid values are 250, 500,
            1000, 2000.

        Returns
        -------
        requests.Response

        """
        # TODO Validate params? We assume it is verified in the ESP32 firmware

        response = requests.post(
            f"http://{self.IP}/config_update",
            params={
                "adc_sampling_rate": adc_sampling_rate,
                "imu_sampling_rate": imu_sampling_rate,
                "encoder_sampling_rate": encoder_sampling_rate,
                "accelerometer_precision": accelerometer_precision,
                "gyrometer_precision": gyrometer_precision,
            },
        )
        return response

    def get_sensors_params(self) -> dict:
        """
        Get the parameters of the instrumented wheel sensors.

        Returns
        -------
        dict
            A dictionary in the form parameter:value.

        """
        response = requests.get(f"http://{self.IP}/config")
        return json.loads(response.content)

    def get_system_state(self) -> dict:
        """
        Get the system state of the instrumented wheel.

        Returns
        -------
        dict
            A dictionary in the form parameter:value

        """
        response = requests.get(f"http://{self.IP}/system_state")
        return json.loads(response.content)

    def start_recording(self) -> dict:
        """
        Start recording data on the instrumented wheel.

        Returns
        -------
        dict

        """
        response = requests.get(f"http://{self.IP}/start_recording")
        return json.loads(response.content)

    def stop_recording(self) -> dict:
        """
        Stop recording data on the instrumented wheel.

        Returns
        -------
        dict

        """
        response = requests.get(f"http://{self.IP}/stop_recording")
        return json.loads(response.content)

    def file_list(self) -> dict:
        """
        Get the list of files on the instrumented wheel.

        Returns
        -------
        dict

        """
        response = requests.get(f"http://{self.IP}/file_list")
        return json.loads(response.content)

    def file_download(self, filename: str, save_path: str = ".") -> int:
        """
        Download a file from the instrumented wheel.

        Parameters
        ----------
        filename
            The name of the file to download
        save_folder
            Optional. Where to save the file. The default is the current
            folder.

        Returns
        -------
        int
            The size of the file

        """
        param = {"file": filename}
        response = requests.get(
            f"http://{self.IP}/file_download", params=param, stream=True
        )
        if response.status_code == 200:
            with open(os.path.join(save_path, filename), "wb") as f:
                f.write(response.content)
                # Return size...
                return f.tell()

        # filed download
        return 0

    def file_delete(self, filename: str) -> dict:
        """
        Delete a file from the instrumented wheel.

        Parameters
        ----------
        filename
            The name of the file to delete

        Returns
        -------
        dict

        """
        response = requests.get(
            f"http://{self.IP}/file_delete", params={"file": filename}
        )
        return json.loads(response.content)


def read_dat(filename) -> dict:
    """
    Read a dat file downloaded from the instrumented wheel.

    Parameters
    ----------
    filename
        The name of the local file to read.

    Returns
    -------
    dict
        A dictionary with the same structure as given by NextWheel.fetch.

    """
    # Create a dummy wheel to parse the data
    nw = NextWheel("0.0.0.0")
    nw.max_analog_samples = np.Inf
    nw.max_encoder_samples = np.Inf
    nw.max_imu_samples = np.Inf
    nw.max_power_samples = np.Inf

    offset = 0
    with open(filename, "rb") as fid:
        stream = fid.read()

    while (new_offset := nw._parse_message(stream, offset)) > offset:
        offset = new_offset


    return nw.fetch()
