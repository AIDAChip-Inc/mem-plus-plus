#!/usr/bin/env python3
"""Run the MemBench memory benchmark. See eval/_runner.py for the shared CLI.

``--data`` may be a single ``<qatype>.json``, one agent dir, or the
``datasets/membench`` root (recursed over both FirstAgent + ThirdAgent).
"""
from __future__ import annotations

import sys

from ._runner import run
from .adapters import membench


def main(argv: list[str] | None = None) -> int:
    return run(
        benchmark="membench",
        default_data="datasets/membench",
        loader=membench.load,
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
