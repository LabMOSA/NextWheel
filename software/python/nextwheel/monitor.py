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
This Python module provides the GUI for the NextWheel.monitor method.
"""

import multiprocessing as mp
import tkinter as tk
import os
import json
import nextwheel


def _gui_app(conn, cwd: str):
    """GUI used in NextWheel.monitor."""
    root = tk.Tk()
    root.title("NextWheel")
    root.configure(bg="black")
    # root.geometry("300x500")
    message = tk.Label(
        root,
        text="NextWheel Monitor",
        justify="left",
        anchor="w",
        bg="#222",
        fg="#ddd",
    )
    message.pack()

    def on_closing():
        conn.send("quit")
        root.destroy()

    def update():
        conn.send("update")
        if conn.poll(1.0):  # Timeout after 1 second
            text = conn.recv()
            message.config(text=text)
        root.after(10, update)

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.after(100, update)
    root.mainloop()


def monitor(nw):
    """Implement NextWheel.monitor(). nw is the class instance."""
    data = nw.fetch()  # Non-updated data: show the last data.

    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=_gui_app, args=(child_conn, os.getcwd()))
    p.start()

    # Default
    voltage = None
    current = None
    acc = None
    gyro = None
    mag = None
    encoder = None
    forces = None
    current_state = None

    i_refresh_state = 65535  # Current system state. Start with a refresh.
    while parent_conn.recv() != "quit":
        text = f"IP Address: {nw.IP}\n"

        # # Get current state
        # i_refresh_state += 1
        # if i_refresh_state > 10:
        #     current_state = nw.get_system_state()
        #     i_refresh_state = 0

        # if current_state is not None:
        #     text += "\nCurrent State\n"

        #     if current_state["streaming"] == 0:
        #         text += "    Not streaming\n"
        #     else:
        #         text += "    Streaming\n"

        #     if current_state["recording"] == 0:
        #         text += "    Not recording\n"
        #     else:
        #         text += f"    Recording to {current_state['filename']}\n"

        # Get data
        ts = nw.fetch()

        try:
            voltage = ts["Power"].data["Voltage"][-1]
        except IndexError:
            pass

        try:
            current = ts["Power"].data["Current"][-1]
        except IndexError:
            pass

        try:
            acc = ts["IMU"].data["Acc"][-1]
        except IndexError:
            pass

        try:
            gyro = ts["IMU"].data["Gyro"][-1]
        except IndexError:
            pass

        try:
            mag = ts["IMU"].data["Mag"][-1]
        except IndexError:
            pass

        try:
            encoder = ts["Encoder"].data["Angle"][-1]
        except IndexError:
            pass

        try:
            forces = ts["Analog"].data["Force"][-1]
        except IndexError:
            pass

        # Format data
        if voltage is not None:
            text += (
                "\nPower\n"
                f"    Voltage: {voltage:.3f} V\n"
                f"    Current: {1000*current:.0f} mA\n"
            )

        if acc is not None:
            text += (
                "\nAccelerometer (m/s^2)\n"
                f"    x: {acc[0]:.2f}\n"
                f"    y: {acc[1]:.2f}\n"
                f"    z: {acc[2]:.2f}\n"
            )

        if gyro is not None:
            text += (
                "\nGyrometer (deg/s)\n"
                f"    x: {gyro[0]:.0f}\n"
                f"    y: {gyro[1]:.0f}\n"
                f"    z: {gyro[2]:.0f}\n"
            )

        if mag is not None:
            text += (
                "\nMagnetometer (unknown unit)\n"
                f"    x: {mag[0]:.0f}\n"
                f"    y: {mag[1]:.0f}\n"
                f"    z: {mag[2]:.0f}\n"
            )

        if encoder is not None:
            text += "\nEncoder (ticks)\n" f"    {encoder:.2f}\n"

        if forces is not None:
            text += "\nForces (uncalibrated unit)\n"
            for i in range(6):
                text += f"    channel {i+1}: {forces[i]:.0f}\n"

        parent_conn.send(text)

    p.join()
