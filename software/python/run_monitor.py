#!/usr/bin/env python3
"""Connect to a wheel and run the monitor from command line."""

import argparse

import nextwheel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", required=True, help="IP address of the wheel")
    parser.add_argument(
        "--port",
        type=int,
        default=80,
        help="HTTP (REST API) port of the wheel (default: 80)",
    )
    parser.add_argument(
        "--broadcast_ip",
        required=False,
        default="255.255.255.255",
        help="IP address where the wheel sends streamed data, e.g. "
        "255.255.255.255 or this computer's IP (default: unchanged)",
    )
    parser.add_argument(
        "--broadcast_port",
        type=int,
        help="UDP port where the wheel sends streamed data "
        "(default: unchanged)",
        default=12345,
    )
    args = parser.parse_args()

    nw = nextwheel.NextWheel(args.ip, port=args.port)

    if args.broadcast_ip is not None or args.broadcast_port is not None:
        params = nw.get_sensors_params()
        nw.set_udp_destination(
            args.broadcast_ip or params["udp_broadcast_address"],
            args.broadcast_port or params["udp_broadcast_port"],
        )

    # monitor() starts and stops streaming
    nw.monitor()
