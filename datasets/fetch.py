#!/usr/bin/env python3
"""Download the memory-research evaluation benchmarks into local, gitignored folders.

Each benchmark lands under ``datasets/<name>/`` (all gitignored — see
``datasets/.gitignore``). Nothing here is committed; the populated folder is
shared with the student out-of-band.

Usage (run from ``memory-research/``)::

    python datasets/fetch.py locomo
    python datasets/fetch.py longmemeval           # _s + _oracle (default)
    python datasets/fetch.py longmemeval --include-m   # opt-in, multi-GB
    python datasets/fetch.py longbench
    python datasets/fetch.py membench              # both agents (default)
    python datasets/fetch.py membench --third-only # observation scenario only (lighter)
    python datasets/fetch.py all                   # locomo + longmemeval(_s,_oracle) + longbench + membench

Dependencies (declared in the project ``pyproject.toml``, owned elsewhere):
  * ``huggingface_hub`` — LongMemEval file downloads
  * ``datasets``        — LongBench loading
LoCoMo and MemBench need neither (stdlib ``urllib``). If a HF dependency is missing the
relevant sub-fetch prints an actionable message and is skipped; it never
hard-crashes the whole run.

Sizes / licenses are documented in ``datasets/README.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

DATA_ROOT = Path(__file__).parent

# --- LoCoMo -----------------------------------------------------------------
# snap-research/locomo, data/locomo10.json (~10 conversations, ~2K QA pairs).
# LICENSE.txt lives in the repo root — we do NOT redistribute the data, so we
# only fetch it locally. Check the in-repo LICENSE before sharing onward.
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LOCOMO_LICENSE_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/LICENSE.txt"

# --- LongMemEval ------------------------------------------------------------
# xiaowu0162/longmemeval-cleaned (HF dataset, MIT). _s ~115K tokens/instance,
# _m is multi-GB (~500 sessions/instance) — opt-in only. 500 questions each.
LME_REPO = "xiaowu0162/longmemeval-cleaned"
LME_FILES_DEFAULT = ["longmemeval_s_cleaned.json", "longmemeval_oracle.json"]
LME_FILE_M = "longmemeval_m_cleaned.json"

# --- LongBench (v1) ---------------------------------------------------------
# THUDM/LongBench (HF, MIT). 21 datasets across 6 task categories. We pull the
# LongBench-E (evenly-distributed-length) subset by default — smaller and the
# standard reporting slice; pass --full for all 21 v1 configs.
LONGBENCH_REPO = "THUDM/LongBench"
LONGBENCH_E_CONFIGS = [
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report",
    "multi_news", "trec", "triviaqa", "samsum", "passage_count",
    "passage_retrieval_en", "lcc", "repobench-p",
]
LONGBENCH_FULL_CONFIGS = LONGBENCH_E_CONFIGS + [
    "narrativeqa", "musique", "dureader", "qmsum", "vcsum", "lsht",
    "passage_retrieval_zh", "multifieldqa_zh",
]

# --- MemBench ---------------------------------------------------------------
# import-myself/Membench (MemBench, Tan et al., ACL 2025 Findings; arXiv
# 2506.21605). MIT (declared in the repo README badge — there is no standalone
# LICENSE file; we fetch-only and never redistribute). The per-task trajectory
# JSONs are committed IN the repo under MemData/, so raw.githubusercontent.com
# serves them directly (stdlib urllib, no HF dependency). Together they are
# large (~600 MB) — --third-only pulls just the lighter observation agent.
MEMBENCH_RAW = "https://raw.githubusercontent.com/import-myself/Membench/main/MemData"
MEMBENCH_FILES = {
    "FirstAgent": [  # participation (dialogue sessions) — the heavier half
        "simple", "comparative", "conditional", "aggregative", "knowledge_update",
        "post_processing", "noisy", "highlevel", "highlevel_rec", "lowlevel_rec",
        "RecMultiSession",
    ],
    "ThirdAgent": [  # observation (statement streams) — lighter
        "simple", "comparative", "conditional", "aggregative", "knowledge_update",
        "post_processing", "noisy", "highlevel",
    ],
}


def _target(name: str) -> Path:
    d = DATA_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, dest: Path) -> None:
    print(f"  ↓ {url}\n    → {dest}")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (trusted benchmark host)
        dest.write_bytes(resp.read())


def fetch_locomo() -> bool:
    print("LoCoMo (snap-research/locomo)…")
    out = _target("locomo")
    _download(LOCOMO_URL, out / "locomo10.json")
    try:
        _download(LOCOMO_LICENSE_URL, out / "LICENSE.txt")
    except Exception as exc:  # noqa: BLE001 — license fetch is best-effort
        print(f"  ! could not fetch LICENSE.txt ({exc}); check it manually before redistributing")
    n = len(json.loads((out / "locomo10.json").read_text()))
    print(f"  ✓ locomo10.json ({n} conversations)")
    return True


def fetch_longmemeval(include_m: bool = False) -> bool:
    print("LongMemEval (xiaowu0162/longmemeval-cleaned)…")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  ! huggingface_hub not installed — `pip install huggingface_hub`; skipping LongMemEval")
        return False
    out = _target("longmemeval")
    files = list(LME_FILES_DEFAULT)
    if include_m:
        files.append(LME_FILE_M)
        print("  (--include-m set: also pulling the multi-GB longmemeval_m.json)")
    for fname in files:
        path = hf_hub_download(repo_id=LME_REPO, filename=fname, repo_type="dataset")
        dest = out / fname
        dest.write_bytes(Path(path).read_bytes())
        size_mb = dest.stat().st_size / 1e6
        print(f"  ✓ {fname} ({size_mb:.1f} MB)")
    return True


def fetch_longbench(full: bool = False) -> bool:
    configs = LONGBENCH_FULL_CONFIGS if full else LONGBENCH_E_CONFIGS
    print(f"LongBench v1 (THUDM/LongBench, {len(configs)} configs)…")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ! datasets not installed — `pip install datasets`; skipping LongBench")
        return False
    out = _target("longbench")
    ok = 0
    for cfg in configs:
        try:
            ds = load_dataset(LONGBENCH_REPO, cfg, split="test")
        except Exception as exc:  # noqa: BLE001 — one bad config shouldn't kill the run
            print(f"  ! {cfg}: {exc}")
            continue
        dest = out / f"{cfg}.jsonl"
        with dest.open("w") as fh:
            for row in ds:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  ✓ {cfg}.jsonl ({len(ds)} rows)")
        ok += 1
    return ok > 0


def fetch_membench(third_only: bool = False) -> bool:
    print("MemBench (import-myself/Membench)…")
    out = _target("membench")
    agents = ["ThirdAgent"] if third_only else ["FirstAgent", "ThirdAgent"]
    if third_only:
        print("  (--third-only set: skipping the heavier FirstAgent/participation half)")
    ok = 0
    for agent in agents:
        agent_dir = out / agent
        agent_dir.mkdir(parents=True, exist_ok=True)
        for stem in MEMBENCH_FILES[agent]:
            dest = agent_dir / f"{stem}.json"
            try:
                _download(f"{MEMBENCH_RAW}/{agent}/{stem}.json", dest)
                n = len(json.loads(dest.read_text()))  # scenario count (sanity)
                print(f"  ✓ {agent}/{stem}.json ({n} scenario(s), {dest.stat().st_size / 1e6:.1f} MB)")
                ok += 1
            except Exception as exc:  # noqa: BLE001 — one bad file shouldn't kill the run
                print(f"  ! {agent}/{stem}.json: {exc}")
    return ok > 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("target", choices=["locomo", "longmemeval", "longbench", "membench", "all"])
    p.add_argument("--include-m", action="store_true",
                   help="LongMemEval: also fetch the multi-GB longmemeval_m.json (opt-in)")
    p.add_argument("--full", action="store_true",
                   help="LongBench: all 21 v1 configs instead of the LongBench-E subset")
    p.add_argument("--third-only", action="store_true",
                   help="MemBench: fetch only the lighter ThirdAgent (observation) half")
    args = p.parse_args(argv)

    results: dict[str, bool] = {}
    if args.target in ("locomo", "all"):
        results["locomo"] = fetch_locomo()
    if args.target in ("longmemeval", "all"):
        results["longmemeval"] = fetch_longmemeval(include_m=args.include_m)
    if args.target in ("longbench", "all"):
        results["longbench"] = fetch_longbench(full=args.full)
    if args.target in ("membench", "all"):
        results["membench"] = fetch_membench(third_only=args.third_only)

    print("\nSummary:")
    for name, ok in results.items():
        print(f"  {name:12s}: {'OK' if ok else 'SKIPPED/FAILED'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
