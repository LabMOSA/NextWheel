from nextwheel import NextWheel

if __name__ == "__main__":
    # websocket.enableTrace(True)  # Uncomment to print all received data
    nw = NextWheel("10.0.1.3")

    # Read back params from sensors
    ret = nw.get_sensors_params()
    if ret.status_code == 200:
        print(
            f"get_sensors_params returned code: {ret.status_code} json:",
            ret.json(),
        )
