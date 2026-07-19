# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

"""Synthetic executable used to test the game harness without a game dump."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zlib


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + data).to_bytes(4, "big")
    return len(data).to_bytes(4, "big") + chunk_type + data + checksum


def one_pixel_png() -> bytes:
    ihdr = b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    scanline = b"\x00\x00\x00\x00\xff"
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(scanline))
        + png_chunk(b"IEND", b"")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0)
    parser.add_argument("--stdout", default="")
    parser.add_argument("--stderr", default="")
    parser.add_argument("--log", default="")
    parser.add_argument("--emit-bytes", type=int, default=0)
    parser.add_argument("--spawn-child", action="store_true")
    parser.add_argument("--expect-ipc", action="store_true")
    parser.add_argument("--ignore-screenshots", action="store_true")
    parser.add_argument("--malformed-screenshots", action="store_true")
    parser.add_argument("game")
    args = parser.parse_args()

    ipc_commands: list[str] = []
    if args.expect_ipc:
        ipc_commands.extend(
            (sys.stdin.readline().strip(), sys.stdin.readline().strip())
        )

    if args.spawn_child:
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])

    if args.sleep:
        time.sleep(args.sleep)

    if args.expect_ipc:
        while True:
            command = sys.stdin.readline().strip()
            ipc_commands.append(command)
            if command == "SCREENSHOT" and not args.ignore_screenshots:
                screenshots = Path("user") / "screenshots"
                screenshots.mkdir(parents=True, exist_ok=True)
                index = len(list(screenshots.iterdir()))
                png = (
                    b"\x89PNG\r\n\x1a\nmalformed"
                    if args.malformed_screenshots
                    else one_pixel_png()
                )
                (screenshots / f"fake_{index}.png").write_bytes(png)
            if command == "STOP" or not command:
                break

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
        "ipc_commands": ipc_commands,
        "ipc_enabled": os.environ.get("SHADPS4_ENABLE_IPC"),
    }
    (Path.cwd() / "observation.json").write_text(
        json.dumps(observation), encoding="utf-8"
    )
    return args.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
