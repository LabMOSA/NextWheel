
from nextwheel import NextWheel

if __name__ == "__main__":
    # websocket.enableTrace(True)  # Uncomment to print all received data
    nw = NextWheel("10.0.1.2")
    # nw.connect()

    # Get system state
    ret = nw.file_list()
    if ret.status_code == 200:
        print(f"file_list returned code: {ret.status_code} json:", ret.json())

        for file in ret.json()["files"]:
            print(file["name"], file["size"])

    # nw.close()
