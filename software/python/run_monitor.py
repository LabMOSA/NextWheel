#!/usr/bin/env python3
"""Connect to a wheel and run the monitor from command line."""

import sys

import nextwheel

if __name__ == "__main__":
    ip = sys.argv[-1]
    nw = nextwheel.NextWheel(ip)
    nw.start_streaming()
    nw.monitor()
    nw.stop_streaming()
