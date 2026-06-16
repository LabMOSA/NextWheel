from nextwheel import NextWheel

if __name__ == "__main__":
    nw = NextWheel("10.0.1.2")

    ret = nw.file_list()
    if ret.status_code == 200:
        print(f"file_list returned code: {ret.status_code} json:", ret.json())

        for file in ret.json()["files"]:
            print(file["name"], file["size"])

            # Delete file
            ret = nw.file_delete(file["name"])
            print(
                f"file_delete : {file['name']} returned code: {ret.status_code} text: {ret.text}"
            )
