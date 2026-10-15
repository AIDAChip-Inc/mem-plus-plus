"""Expose only the native no-model build-and-save path for Mem++ preparation."""
import os
import sys


def main():
    if "--skip-evaluation" not in sys.argv[1:] or "--save-memory" not in sys.argv[1:]:
        raise RuntimeError("This local preparation driver requires --skip-evaluation --save-memory")
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    from integrations.longmemeval_v2.mempp import MemppMemory
    from memory_modules.memory import MEMORY_TYPES
    from evaluation.harness import main as native_main
    MEMORY_TYPES.clear()
    MEMORY_TYPES["mempp"] = MemppMemory
    native_main()


if __name__ == "__main__":
    main()
