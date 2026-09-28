#!/usr/bin/env python3
"""Run the LoCoMo memory benchmark. See eval/_runner.py for the shared CLI."""
from __future__ import annotations

import sys

from .adapters import locomo
from ._runner import run


def main(argv: list[str] | None = None) -> int:
    return run(
        benchmark="locomo",
        default_data="datasets/locomo/locomo10.json",
        loader=locomo.load,
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
