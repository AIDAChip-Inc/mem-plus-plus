"""Machine descriptor recorded in every results file.

A latency number without the machine it was measured on is not a measurement,
it is an anecdote. Mem0's Table 2 latencies are their hardware under their
answerer; ours are this workstation under ours, and the only way a reader can
tell the two apart is if the file says what "this workstation" was.

Everything here is BEST-EFFORT and never raises: a missing ``nvidia-smi`` or a
locked-down WMI just leaves the field ``None``. Called once per run, outside any
timed region.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess

_cached: dict | None = None


def _run(cmd: list[str], timeout: float = 6.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:  # noqa: BLE001 — a missing binary is a normal outcome here
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _gpu() -> dict | None:
    """GPU name / current + max SM clock / memory, via nvidia-smi.

    The CURRENT clock matters on this project's box: the RTX 4060 Laptop idles
    at ~480 MHz and only ramps under sustained load, so a small-batch ONNX
    embedding can be measured on a down-clocked GPU and look slow for reasons
    that have nothing to do with the code (see 17-measurement.md)."""
    raw = _run(["nvidia-smi",
                "--query-gpu=name,clocks.sm,clocks.max.sm,memory.total",
                "--format=csv,noheader,nounits"])
    if not raw:
        return None
    parts = [p.strip() for p in raw.splitlines()[0].split(",")]
    if len(parts) < 4:
        return None
    def _int(v):
        try:
            return int(v)
        except ValueError:
            return None
    return {"name": parts[0], "sm_clock_mhz": _int(parts[1]),
            "sm_clock_max_mhz": _int(parts[2]), "memory_mib": _int(parts[3])}


def _cpu_ram() -> dict:
    info: dict = {"processor": platform.processor() or None,
                  "logical_cores": os.cpu_count()}
    if platform.system() == "Windows":
        raw = _run(["powershell", "-NoProfile", "-Command",
                    "$c=Get-CimInstance Win32_Processor|Select-Object -First 1;"
                    "$s=Get-CimInstance Win32_ComputerSystem;"
                    "'{0}|{1}|{2}|{3}' -f $c.Name,$c.NumberOfCores,"
                    "$c.NumberOfLogicalProcessors,$s.TotalPhysicalMemory"])
        if raw:
            f = raw.splitlines()[-1].split("|")
            if len(f) == 4:
                info["cpu_name"] = f[0].strip()
                for key, idx in (("physical_cores", 1), ("logical_cores", 2)):
                    try:
                        info[key] = int(f[idx])
                    except ValueError:
                        pass
                try:
                    info["ram_gb"] = round(int(f[3]) / 1e9, 1)
                except ValueError:
                    pass
    else:
        try:
            info["ram_gb"] = round(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
        except Exception:  # noqa: BLE001
            pass
    return info


def describe_machine(refresh: bool = False) -> dict:
    """One dict: OS, CPU, RAM, GPU, Python, and the engine's DB/embedding env.

    Cached per process (the hardware does not change mid-run); ``refresh=True``
    re-reads, which is what a profiler wants when it needs the GPU clock DURING
    a run rather than before it."""
    global _cached
    if _cached is not None and not refresh:
        return _cached
    dsn = os.environ.get("MEMORY_DATABASE_URL") or ""
    # Never record credentials: keep only the driver + database name.
    db = dsn.rsplit("/", 1)[-1] if "/" in dsn else None
    driver = dsn.split("://", 1)[0] if "://" in dsn else None
    _cached = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        **_cpu_ram(),
        "gpu": _gpu(),
        "db_driver": driver,
        "db_name": db,
        "embedding_backend": os.environ.get("MEMORY_EMBEDDING_BACKEND", "onnx"),
    }
    return _cached


def concurrent_load() -> dict:
    """What ELSE is running on the box right now, for a latency file to carry.

    A single-stream latency measured while three other benchmark drivers share
    the same Postgres and CPU is a different number from one taken on a quiet
    machine, and the only honest thing to do is write down which it was. Lists
    the other Python processes (command line, trimmed) and the 1-minute load
    where the OS exposes it. Best effort; never raises."""
    out: dict = {"sampled_at": None, "python_processes": [], "note": None}
    try:
        import datetime as _dt
        out["sampled_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    except Exception:  # noqa: BLE001
        pass
    me = os.getpid()
    if platform.system() == "Windows":
        raw = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                    "ForEach-Object { '{0}|{1}' -f $_.ProcessId, $_.CommandLine }"])
        if raw:
            for line in raw.splitlines():
                pid, _, cmd = line.partition("|")
                try:
                    if int(pid) == me:
                        continue
                except ValueError:
                    continue
                cmd = cmd.replace('"', "").strip()
                # keep "<script>.py <args>", drop the interpreter path (which on
                # this box contains spaces, so a naive split is wrong)
                m = re.search(r"([\w\-.]+\.py)(.*)$", cmd)
                entry = (m.group(1) + m.group(2)).strip()[:100] if m else cmd[-100:]
                if entry not in out["python_processes"]:
                    out["python_processes"].append(entry)
    else:
        raw = _run(["ps", "-eo", "pid,args"])
        if raw:
            for line in raw.splitlines()[1:]:
                pid, _, cmd = line.strip().partition(" ")
                if "python" in cmd and pid.isdigit() and int(pid) != me:
                    out["python_processes"].append(cmd[:120])
        try:
            out["load_1m"] = os.getloadavg()[0]
        except Exception:  # noqa: BLE001
            pass
    out["n_other_python"] = len(out["python_processes"])
    out["note"] = ("other Python processes present: the number was NOT taken on an "
                   "idle machine" if out["python_processes"] else "no other Python "
                   "processes: quiet box")
    return out
