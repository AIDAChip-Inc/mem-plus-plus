"""Register the local Mem++ adapter without editing the pinned Bench'd checkout."""
import sys

from benchd_harness.adapters import _BUILTIN_ADAPTERS
from benchd_harness.cli import cli
from integrations.benchd.mempp_adapter import MemppAdapter


def main():
    if sys.argv[1:] and sys.argv[1] not in {"run", "adapter", "list", "--help", "--version"}:
        raise RuntimeError("This local driver supports adapter checks, listing and retrieval-only runs")
    if any(arg == "--judge" or arg.startswith("--judge=") for arg in sys.argv[1:]):
        raise RuntimeError("Paid judging requires the separately reviewed shared-budget guard; this driver is retrieval-only")
    if "mempp" in _BUILTIN_ADAPTERS:
        raise RuntimeError("Refusing to replace an existing Mem++ adapter")
    # Even --no-judge can incur model costs inside other builtin adapters.
    # Limit every CLI/SDK route to Mem++, rather than trusting option spelling.
    _BUILTIN_ADAPTERS.clear()
    _BUILTIN_ADAPTERS["mempp"] = MemppAdapter
    cli()


if __name__ == "__main__":
    main()
