"""Smoke tests for CLI subcommands in rubbish.py."""
import subprocess
import sys
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cli(*args):
    cmd = [sys.executable, "rubbish.py"] + list(args)
    return subprocess.run(cmd, cwd=BASE, capture_output=True, text=True)


def test_cli_salam():
    r = run_cli("salam")
    assert r.returncode == 0
    assert "Key:" in r.stdout
    assert "Quota used:" in r.stdout


def test_cli_bepar():
    r = run_cli("bepar", "رتبه C1 یعنی چی؟", "--topk", "3")
    assert r.returncode == 0
    assert "--- hits (fused) ---" in r.stdout
    assert "--- resolve ---" in r.stdout


def test_cli_bench():
    r = run_cli("bench")
    assert r.returncode == 0
    assert "pipe=adaptive" in r.stdout
    assert "accuracy" in r.stdout


def test_cli_judge():
    r = run_cli("judge", "رتبه A2 چیست؟", "--pre-only")
    assert r.returncode == 0
    assert "System-1 Pre-Retrieval Decision" in r.stdout
    assert "top_k" in r.stdout


def test_cli_judge_bench():
    r = run_cli("judge-bench", "--split", "dev")
    assert r.returncode == 0
    assert "SYSTEM-1 RAG DECISION JUDGE BENCHMARK" in r.stdout
    assert "Routing Accuracy:" in r.stdout


def test_cli_inspect_trace():
    r = run_cli("inspect-trace", "رتبه")
    assert r.returncode == 0
