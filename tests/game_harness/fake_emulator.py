# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

"""Synthetic executable used to test the game harness without a game dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0)
    parser.add_argument("--stdout", default="")
    parser.add_argument("--stderr", default="")
    parser.add_argument("--log", default="")
    parser.add_argument("--emit-bytes", type=int, default=0)
    parser.add_argument("--spawn-child", action="store_true")
    parser.add_argument("game")
    args = parser.parse_args()

    if args.spawn_child:
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])

    if args.sleep:
        time.sleep(args.sleep)

    if args.stdout:
        print(args.stdout)
    if args.stderr:
        print(args.stderr, file=sys.stderr)
    if args.emit_bytes:
        sys.stdout.write("x" * args.emit_bytes)

    log_path = Path("user") / "log" / "shad_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(args.log, encoding="utf-8")

    observation = {
        "cwd": str(Path.cwd()),
        "game": args.game,
        "user_directory_exists": (Path.cwd() / "user").is_dir(),
    }
    (Path.cwd() / "observation.json").write_text(
        json.dumps(observation), encoding="utf-8"
    )
    return args.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
