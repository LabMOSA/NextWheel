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
        "--udp_address",
        default="255.255.255.255",
        help="IP address where the wheel sends streamed data, e.g. "
        "255.255.255.255 or this computer's IP (default: 255.255.255.255)",
    )
    parser.add_argument(
        "--udp_port",
        type=int,
        default=nextwheel.DEFAULT_UDP_PORT,
        help="UDP port where the wheel sends streamed data "
        f"(default: {nextwheel.DEFAULT_UDP_PORT})",
    )
    args = parser.parse_args()

    nw = nextwheel.NextWheel(args.ip, port=args.port)

    # monitor() starts and stops streaming
    nw.monitor(args.udp_address, args.udp_port)
