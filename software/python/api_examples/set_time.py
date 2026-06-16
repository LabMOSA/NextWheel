from datetime import datetime

from nextwheel import NextWheel

if __name__ == "__main__":
    # websocket.enableTrace(True)  # Uncomment to print all received data
    nw = NextWheel("10.0.1.2")
    # nw.connect()

    # Set device to current time (unix time)
    ret = nw.set_time(str(int(datetime.now().timestamp())))
    print(f"set_time returned code: {ret.status_code} text: {ret.text}")
    # nw.close()
