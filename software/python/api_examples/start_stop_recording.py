import time

from nextwheel import NextWheel

if __name__ == "__main__":
    # websocket.enableTrace(True)  # Uncomment to print all received data
    nw = NextWheel("10.0.1.2")
    # nw.connect()

    # Start recording
    ret = nw.start_recording()
    if ret.status_code == 200:
        print(
            f"start_recording returned code: {ret.status_code} text:", ret.text
        )

    # Let the system change state to recording
    time.sleep(1)

    # Re-start should fail
    ret = nw.start_recording()
    print(
        f"(re) start_recording returned code: {ret.status_code} text:",
        ret.text,
    )

    # Sleep for 10 secs
    time.sleep(10)
    print("Sleeping for 10 secs...")

    # Stop recording
    ret = nw.stop_recording()
    if ret.status_code == 200:
        print(
            f"stop_recording returned code: {ret.status_code} text:", ret.text
        )

    # Re-stop should fail
    ret = nw.stop_recording()
    print(
        f"(re) stop_recording returned code: {ret.status_code} text:", ret.text
    )

    # nw.close()
