#!/usr/bin/env python3
"""Run the LongBench (v1) benchmark. See eval/_runner.py for the shared CLI.

``--data`` may be a single ``<config>.jsonl`` or the ``datasets/longbench/``
directory of configs (the fetch layout).
"""
from __future__ import annotations

import sys

from .adapters import longbench
from ._runner import run


def main(argv: list[str] | None = None) -> int:
    return run(
        benchmark="longbench",
        default_data="datasets/longbench",
        loader=longbench.load,
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
