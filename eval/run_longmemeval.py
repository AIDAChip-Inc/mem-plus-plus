#!/usr/bin/env python3
"""Run the LongMemEval memory benchmark. See eval/_runner.py for the shared CLI.

Defaults to the ``_s`` variant; point ``--data`` at ``longmemeval_oracle.json``
or ``longmemeval_m.json`` for the other variants.
"""
from __future__ import annotations

import sys

from .adapters import longmemeval
from ._runner import run


def main(argv: list[str] | None = None) -> int:
    return run(
        benchmark="longmemeval",
        default_data="datasets/longmemeval/longmemeval_s_cleaned.json",
        loader=longmemeval.load,
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
