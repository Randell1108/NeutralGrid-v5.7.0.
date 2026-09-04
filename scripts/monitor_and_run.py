#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PROFILE_DIR = BASE / "data" / "profile"
PATTERN_PROFILE = PROFILE_DIR / "pattern_profile.json"
PROFILE_MODEL = PROFILE_DIR / "profile_model.json"
GATE_FILE = PROFILE_DIR / "profile_gate.json"
LOG_DIR = BASE / "logs"
LOG_DIR.mkdir(exist_ok=True)

POLL_INTERVAL = 30  # seconds
TIMEOUT_HOURS = 6

timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"monitor_and_run_{timestamp}.log"

python_exe = sys.executable

commands = [
    {
        "name": "recalibrate_utility",
        "cmd": [python_exe, str(BASE / "scripts" / "recalibrate_utility.py"), "--promote"],
    },
    {"name": "run_full_pipeline", "cmd": [python_exe, str(BASE / "run_full_pipeline.py")]},
    {"name": "pytest", "cmd": [python_exe, "-m", "pytest", "tests/"]},
]


def log(msg: str):
    ts = datetime.utcnow().isoformat()
    line = f"{ts} | {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def wait_for_profile_artifacts(timeout_hours: float = TIMEOUT_HOURS) -> bool:
    end_time = datetime.utcnow() + timedelta(hours=timeout_hours)
    log(f"MONITOR: Waiting for profile artifacts in {PROFILE_DIR} (timeout {timeout_hours}h)")
    while datetime.utcnow() < end_time:
        if PATTERN_PROFILE.exists() and PROFILE_MODEL.exists():
            log("MONITOR: Detected profile artifacts")
            return True
        log(f"MONITOR: Not yet available: pattern={PATTERN_PROFILE.exists()} model={PROFILE_MODEL.exists()}")
        time.sleep(POLL_INTERVAL)
    log("MONITOR: Timeout waiting for profile artifacts")
    return False


def run_and_capture(cmd: list[str], name: str) -> dict:
    log(f"MONITOR: Running {name}: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = proc.returncode
        log(f"MONITOR: {name} exit_code={rc}")
        # dump to log file
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n--- OUTPUT: " + name + " ---\n")
            f.write(stdout)
            f.write("\n--- ERR: " + name + " ---\n")
            f.write(stderr)
            f.write("\n--- END ---\n")
        return {"name": name, "rc": rc, "stdout": stdout, "stderr": stderr}
    except Exception as exc:
        log(f"MONITOR: Exception running {name}: {exc}")
        return {"name": name, "rc": -1, "stdout": "", "stderr": str(exc)}


def main():
    ok = wait_for_profile_artifacts()
    summary = {"detected": ok, "runs": []}
    if not ok:
        log("MONITOR: Aborting downstream steps due to missing artifacts")
        with open(LOG_DIR / f"monitor_and_run_summary_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return 2

    for entry in commands:
        res = run_and_capture(entry["cmd"], entry["name"])
        summary["runs"].append(res)
        # If a critical step failed (recalibration or pipeline), continue to next but mark it
        # User can inspect logs for details.

    with open(LOG_DIR / f"monitor_and_run_summary_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log("MONITOR: All commands completed. See logs and summary for details.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
