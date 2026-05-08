"""Automated experiment runner: sweeps env variants, runs perf_eval per
variant, aggregates at the end.

Each variant gets its own short-lived uvicorn (so env changes take effect
on model load). The script:
    1. Stops nothing — caller must shut down any existing aistack on the
       target port first (default 11500).
    2. For each (label, env) pair:
       a. Spawns uvicorn with the env overlay.
       b. Polls /health until ready.
       c. Sends one warmup request (perf-12min.mp3) so the cold-start
          model-load cost doesn't pollute the measured runs.
       d. Runs perf_eval --label <label> on the requested files.
       e. Terminates uvicorn cleanly.
    3. Calls bench.aggregate at the end.

Usage:
    # Default sweep: ctx-128 / ctx-256 / ctx-512 on 25min + 50min
    python -m bench.run_experiments

    # Custom variants & files
    python -m bench.run_experiments \
        --variant ctx-128:AISTACK_PARAKEET_ATT_CONTEXT_SIZE=128,128 \
        --variant ctx-256:AISTACK_PARAKEET_ATT_CONTEXT_SIZE=256,256 \
        --file bench/audio/perf-25min.mp3
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx


REPO_ROOT = Path(__file__).resolve().parent.parent
PY = REPO_ROOT / "myenv" / "Scripts" / "python.exe"
DEFAULT_PORT = 11500
WARMUP_FILE = "bench/audio/perf-12min.mp3"

DEFAULT_VARIANTS = [
    ("ctx-128", {"AISTACK_PARAKEET_ATT_CONTEXT_SIZE": "128,128"}),
    ("ctx-256", {"AISTACK_PARAKEET_ATT_CONTEXT_SIZE": "256,256"}),
    ("ctx-512", {"AISTACK_PARAKEET_ATT_CONTEXT_SIZE": "512,512"}),
]
DEFAULT_FILES = [
    "bench/audio/perf-25min.mp3",
    "bench/audio/perf-50min.mp3",
]


def parse_variant(spec: str) -> tuple[str, dict[str, str]]:
    label, _, kvs = spec.partition(":")
    env: dict[str, str] = {}
    for kv in kvs.split(","):
        if "=" in kv:
            k, _, v = kv.partition("=")
            env[k.strip()] = v.strip()
    return label, env


def wait_for_health(port: int, timeout: float = 90) -> bool:
    deadline = time.monotonic() + timeout
    last_err = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception as e:
            last_err = e
        time.sleep(1)
    print(f"  health timeout after {timeout}s; last error: {last_err}",
          file=sys.stderr)
    return False


def warmup(port: int, audio: Path) -> None:
    print(f"  warmup: {audio.name} ...", end="", flush=True)
    t0 = time.perf_counter()
    try:
        with open(audio, "rb") as fh:
            r = httpx.post(
                f"http://127.0.0.1:{port}/v1/audio/transcriptions",
                files={"file": (audio.name, fh, "audio/mpeg")},
                data={"model": "parakeet", "language": "en",
                      "response_format": "json"},
                headers={"X-Request-ID": "warmup"},
                timeout=600,
            )
            r.raise_for_status()
        print(f" done in {time.perf_counter() - t0:.1f}s")
    except Exception as e:
        print(f" FAILED: {e}", file=sys.stderr)


def spawn_uvicorn(port: int, env_overlay: dict[str, str]) -> subprocess.Popen:
    env = os.environ.copy()
    # Inherit the same model-cache wiring scripts/dev.bat sets.
    env.setdefault("HF_HOME", r"D:\AI_Models\hf")
    env.setdefault("MODELSCOPE_CACHE", r"D:\AI_Models\modelscope")
    env.setdefault("NEMO_CACHE_DIR", r"D:\AI_Models\nemo")
    env.update(env_overlay)
    cmd = [
        str(PY), "-m", "uvicorn", "aistack.main:app",
        "--host", "127.0.0.1", "--port", str(port),
    ]
    creationflags = 0
    if sys.platform == "win32":
        # CREATE_NEW_PROCESS_GROUP so we can send CTRL_BREAK_EVENT.
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(
        cmd, env=env, cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,  # avoid blocking on full pipe buffer
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )


def stop_uvicorn(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


def run_one_variant(label: str, env_overlay: dict[str, str], port: int,
                    files: list[str]) -> int:
    print(f"\n========== {label}  env={env_overlay} ==========")
    proc = spawn_uvicorn(port, env_overlay)
    try:
        if not wait_for_health(port):
            return 1
        print(f"  uvicorn pid={proc.pid} ready")
        warmup(port, REPO_ROOT / WARMUP_FILE)
        eval_cmd = [str(PY), "-m", "bench.perf_eval", "--label", label,
                    "--base-url", f"http://127.0.0.1:{port}"]
        for f in files:
            eval_cmd += ["--file", f]
        rc = subprocess.call(eval_cmd, cwd=str(REPO_ROOT))
        return rc
    finally:
        stop_uvicorn(proc)
        # Give OS a moment to release the port and CUDA to free memory.
        time.sleep(3)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", action="append",
                   help="<label>:<KEY>=<VALUE>[,<KEY>=<VALUE>...]; repeatable")
    p.add_argument("--file", action="append")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--no-aggregate", action="store_true")
    args = p.parse_args()

    variants = (
        [parse_variant(s) for s in args.variant]
        if args.variant else DEFAULT_VARIANTS
    )
    files = args.file or DEFAULT_FILES

    # Refuse to run if a server is already on the target port.
    try:
        r = httpx.get(f"http://127.0.0.1:{args.port}/health", timeout=1)
        if r.status_code == 200:
            print(
                f"port {args.port} already serving — stop the running aistack "
                f"first (Ctrl+C its terminal). This harness controls its own.",
                file=sys.stderr,
            )
            return 2
    except Exception:
        pass  # nothing listening, expected

    overall = 0
    for label, env_overlay in variants:
        rc = run_one_variant(label, env_overlay, args.port, files)
        if rc != 0:
            print(f"  variant {label} returned rc={rc}", file=sys.stderr)
            overall = rc

    if not args.no_aggregate:
        print("\n========== aggregate ==========")
        subprocess.call([str(PY), "-m", "bench.aggregate"], cwd=str(REPO_ROOT))

    return overall


if __name__ == "__main__":
    raise SystemExit(main())
