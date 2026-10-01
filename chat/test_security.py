"""Regression tests for the local-demo hardening: API-key validation, safe .env
parsing (no execution / no injection), secret-file permissions, loopback-only bind."""
from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

from chat import app
from chat.app import (
    on_save_key, on_submit, parse_dotenv, persist_api_key, require_loopback, validate_api_key,
)

_ROOT = Path(__file__).resolve().parent.parent
_GOOD = "sk-ant-TESTONLY-not-a-real-key"
_BAD_KEYS = [
    "abc\nMEMORY_DATABASE_URL=evil",      # newline -> injects a second entry
    "abc\rX=1",
    "abc def",
    "abc\tdef",
    "abc\x00def",
    "abc\x1b[31m",
    "abc;touch /tmp/pwn",
    "abc$(id)",
    "abc`id`",
    "abc'def",
    'abc"def',
    "abc&&id",
    "abc|id",
    "abc#c",
    "abc=def",
    "abc\\def",
    "abc\n",                              # trailing newline must not slip past `$`
    "é-key",
    "a" * 513,
]


# ── key validation ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", _BAD_KEYS)
def test_validate_rejects_unsafe_keys(key):
    with pytest.raises(ValueError) as ei:
        validate_api_key(key)
    assert key not in str(ei.value)


@pytest.mark.parametrize("key", [_GOOD, "dummy", "a.b_c-D9"])
def test_validate_accepts_plain_tokens(key):
    assert validate_api_key(key) == key


@pytest.mark.parametrize("key", _BAD_KEYS)
def test_persist_rejects_before_any_write(tmp_path, key):
    env = tmp_path / ".env"
    with pytest.raises(ValueError):
        persist_api_key(key, env_path=env)
    assert not env.exists()
    assert list(tmp_path.iterdir()) == []  # no temp file left behind either


@pytest.mark.parametrize("key", _BAD_KEYS)
def test_on_save_key_cannot_bypass_validation(tmp_path, monkeypatch, key):
    env = tmp_path / ".env"
    monkeypatch.setattr(app, "_ENV_PATH", env)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    msg = on_save_key(key)
    assert "Invalid API key" in msg
    assert key not in msg
    assert not env.exists()
    assert "ANTHROPIC_API_KEY" not in os.environ


@pytest.mark.parametrize("key", _BAD_KEYS)
def test_on_submit_cannot_bypass_validation(monkeypatch, key):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    called = []
    monkeypatch.setattr(app, "run_turn", lambda *a, **k: called.append(1))
    hist, msg_box, dbg = on_submit("hello", [], "p", 5, "haiku", key)
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert not called
    assert hist == [] and msg_box == "hello"
    assert "Invalid API key" in dbg


def test_persist_failure_is_not_swallowed(tmp_path):
    with pytest.raises(OSError):
        persist_api_key(_GOOD, env_path=tmp_path / "missing-dir" / ".env")


# ── secret file permissions / atomicity ────────────────────────────────────────

def test_secret_file_is_owner_only_even_over_loose_existing_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OTHER=1\n")
    env.chmod(0o644)
    persist_api_key(_GOOD, env_path=env)
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert env.read_text().splitlines() == ["OTHER=1", f"ANTHROPIC_API_KEY={_GOOD}"]
    assert [p.name for p in tmp_path.iterdir()] == [".env"]  # temp file replaced, not leaked


def test_new_secret_file_is_owner_only(tmp_path):
    env = tmp_path / ".env"
    persist_api_key(_GOOD, env_path=env)
    assert stat.S_IMODE(env.stat().st_mode) == 0o600


def test_write_uses_os_replace_and_failure_keeps_old_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("KEEP=me\n")
    seen = []
    real = os.replace
    monkeypatch.setattr(app.os, "replace", lambda a, b: seen.append((a, b)) or real(a, b))
    persist_api_key(_GOOD, env_path=env)
    assert seen and seen[0][1] == env

    def boom(a, b):
        raise PermissionError("denied")
    monkeypatch.setattr(app.os, "replace", boom)
    before = env.read_text()
    with pytest.raises(PermissionError):
        persist_api_key("another-key", env_path=env)
    assert env.read_text() == before
    assert [p.name for p in tmp_path.iterdir()] == [".env"]


def test_duplicate_key_lines_collapse_to_one(tmp_path):
    env = tmp_path / ".env"
    env.write_text(f"A=1\nexport ANTHROPIC_API_KEY=old1\nANTHROPIC_API_KEY=old2\nB=2\n")
    persist_api_key(_GOOD, env_path=env)
    assert env.read_text().splitlines() == ["A=1", f"ANTHROPIC_API_KEY={_GOOD}", "B=2"]


# ── python .env parser ─────────────────────────────────────────────────────────

_MALICIOUS_ENV = """\
# comment
PLAIN=value
QUOTED="two words"
SINGLE='$(touch /tmp/pwn_py)'
SUB=$(touch /tmp/pwn_py)
TICK=`touch /tmp/pwn_py`
INLINE=abc # trailing comment
export EXPORTED=yes
bad key=1
1BAD=2
; touch /tmp/pwn_py
NOEQUALS
"""


def test_parse_dotenv_is_literal_and_skips_invalid_lines():
    got = parse_dotenv(_MALICIOUS_ENV)
    assert got == {
        "PLAIN": "value",
        "QUOTED": "two words",
        "SINGLE": "$(touch /tmp/pwn_py)",
        "SUB": "$(touch /tmp/pwn_py)",
        "TICK": "`touch /tmp/pwn_py`",
        "INLINE": "abc",
        "EXPORTED": "yes",
    }


def test_load_dotenv_does_not_override_existing_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("SEC_T_A=fromfile\nSEC_T_B=fromfile\n")
    monkeypatch.setattr(app, "_ENV_PATH", env)
    monkeypatch.setenv("SEC_T_A", "preset")
    monkeypatch.delenv("SEC_T_B", raising=False)
    app._load_dotenv()
    assert os.environ["SEC_T_A"] == "preset"
    assert os.environ["SEC_T_B"] == "fromfile"
    monkeypatch.delenv("SEC_T_B")


# ── run_demo.sh loader: extracted verbatim, run in bash ────────────────────────

def _bash_load(tmp_path: Path, env_text: str, names: list[str], preset: dict | None = None):
    src = (_ROOT / "run_demo.sh").read_text()
    m = re.search(r"# >>> load_env_file.*?\n(.*?)# <<< load_env_file", src, re.S)
    assert m, "load_env_file markers missing from run_demo.sh"
    envf = tmp_path / "t.env"
    envf.write_text(env_text)
    script = m.group(1) + f'load_env_file "{envf}"\n' + "".join(
        f'printf "%s=<%s>\\n" {n} "${{{n}-UNSET}}"\n' for n in names)
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), **(preset or {})}
    r = subprocess.run(["bash", "-c", "set -euo pipefail\n" + script],
                       capture_output=True, text=True, env=env, cwd=tmp_path, timeout=30)
    assert r.returncode == 0, r.stderr
    return dict(re.findall(r"^(\w+)=<(.*)>$", r.stdout, re.M)), r


def test_shell_loader_never_executes_or_injects(tmp_path):
    pwn = tmp_path / "pwned"
    text = f"""\
PLAIN=value
QUOTED="two words"
SQ='$(touch {pwn})'
SUB=$(touch {pwn})
TICK=`touch {pwn}`
SEMI=a; touch {pwn}
INLINE=abc # c
export EXP=yes
; touch {pwn}
$(touch {pwn})=1
bad key=1
touch {pwn}
EVIL="x"; touch {pwn}
"""
    got, _ = _bash_load(tmp_path, text, ["PLAIN", "QUOTED", "SQ", "SUB", "TICK", "SEMI", "INLINE", "EXP", "EVIL"])
    assert not pwn.exists()
    assert got["PLAIN"] == "value"
    assert got["QUOTED"] == "two words"
    assert got["SQ"] == f"$(touch {pwn})"
    assert got["SUB"] == f"$(touch {pwn})"
    assert got["TICK"] == f"`touch {pwn}`"
    assert got["SEMI"] == f"a; touch {pwn}"   # literal value, nothing executed
    assert got["INLINE"] == "abc"
    assert got["EXP"] == "yes"
    assert got["EVIL"] == f'"x"; touch {pwn}'  # not a matched-quote value -> literal


def test_shell_loader_matches_python_parser_on_plain_and_quoted(tmp_path):
    text = 'A=1\nB="q v"\nC=\'s v\'\nD=x # c\nexport E=5\n'
    got, _ = _bash_load(tmp_path, text, list("ABCDE"))
    assert got == {k: v for k, v in parse_dotenv(text).items()}


def test_shell_loader_file_overrides_preset_like_sourcing(tmp_path):
    got, _ = _bash_load(tmp_path, "X=fromfile\n", ["X"], preset={"X": "preset"})
    assert got["X"] == "fromfile"


def test_shell_loader_handles_missing_trailing_newline_and_crlf(tmp_path):
    got, _ = _bash_load(tmp_path, "A=1\r\nB=2", ["A", "B"])
    assert got == {"A": "1", "B": "2"}


# ── loopback-only bind ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2"])
def test_loopback_hosts_allowed(host):
    assert require_loopback(host) == host


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "", "192.168.1.5", "example.com", "10.0.0.1", "localhost.evil.com"])
def test_non_loopback_hosts_rejected(host):
    with pytest.raises(ValueError, match="non-loopback"):
        require_loopback(host)


def test_main_refuses_non_loopback_and_never_launches(monkeypatch):
    monkeypatch.setenv("GRADIO_SERVER_NAME", "0.0.0.0")
    monkeypatch.setattr(app, "_load_dotenv", lambda: None)
    built = []
    monkeypatch.setattr(app, "build_demo", lambda: built.append(1))
    with pytest.raises(SystemExit) as ei:
        app.main()
    assert "non-loopback" in str(ei.value)
    assert not built


def test_main_launches_on_loopback_with_share_disabled(monkeypatch):
    monkeypatch.setenv("GRADIO_SERVER_NAME", "127.0.0.1")
    monkeypatch.setattr(app, "_load_dotenv", lambda: None)
    kw = {}

    class _D:
        def launch(self, **k):
            kw.update(k)
    monkeypatch.setattr(app, "build_demo", lambda: _D())
    app.main()
    assert kw["server_name"] == "127.0.0.1" and kw["share"] is False


def test_run_demo_rejects_non_loopback_before_starting_anything(tmp_path):
    r = subprocess.run(["bash", str(_ROOT / "run_demo.sh")], capture_output=True, text=True,
                       env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "GRADIO_SERVER_NAME": "0.0.0.0"},
                       cwd=tmp_path, timeout=30)
    assert r.returncode == 1
    assert "local-only" in r.stderr


def test_compose_publishes_postgres_on_loopback_only():
    text = (_ROOT / "docker-compose.yml").read_text()
    ports = re.findall(r'^\s*-\s*"([^"]*:\d+:\d+)"', text, re.M)
    assert ports and all(p.startswith("127.0.0.1:") for p in ports)
