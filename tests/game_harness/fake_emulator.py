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


def one_pixel_png(red: int = 0) -> bytes:
    ihdr = b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    scanline = b"\x00" + bytes((red, 0, 0, 255))
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(scanline))
        + png_chunk(b"IEND", b"")
    )


def varying_left_pixel_png(red: int) -> bytes:
    ihdr = b"\x00\x00\x00\x02\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    scanline = b"\x00" + bytes((red, 0, 0, 255, 8, 8, 8, 255))
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(scanline))
        + png_chunk(b"IEND", b"")
    )


def dark_movement_png(index: int) -> bytes:
    ihdr = b"\x00\x00\x00\x04\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    pixels = bytearray((0, 0, 0, 255) * 4)
    detail = 0 if index == 0 else 12
    pixels[detail : detail + 3] = bytes((8, 8, 8))
    scanline = b"\x00" + bytes(pixels)
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
    parser.add_argument("--ipc-handshake-delay", type=float, default=0)
    parser.add_argument("--omit-ipc-handshake", action="store_true")
    parser.add_argument("--omit-control-capability", action="store_true")
    parser.add_argument("--omit-gamepad-capability", action="store_true")
    parser.add_argument("--omit-screenshot-capability", action="store_true")
    parser.add_argument("--omit-renderdoc-capability", action="store_true")
    parser.add_argument("--ignore-screenshots", action="store_true")
    parser.add_argument("--screenshot-delay", type=float, default=0)
    parser.add_argument("--single-screenshot", action="store_true")
    parser.add_argument("--ignore-renderdoc-captures", action="store_true")
    parser.add_argument("--malformed-screenshots", action="store_true")
    parser.add_argument("--vary-screenshots", action="store_true")
    parser.add_argument("--vary-screenshots-after", type=int)
    parser.add_argument("--vary-left-pixel", action="store_true")
    parser.add_argument("--dark-movement-screenshots", action="store_true")
    parser.add_argument("game")
    args = parser.parse_args()

    ipc_commands: list[str] = []
    ipc_command_seconds: list[float] = []
    launched = time.monotonic()
    if args.expect_ipc:
        time.sleep(args.ipc_handshake_delay)
        if not args.omit_ipc_handshake:
            capabilities = []
            if not args.omit_control_capability:
                capabilities.append(";ENABLE_EMU_CONTROL")
            if not args.omit_gamepad_capability:
                capabilities.append(";ENABLE_GAMEPAD")
            if not args.omit_screenshot_capability:
                capabilities.append(";ENABLE_SCREENSHOT")
            if not args.omit_renderdoc_capability:
                capabilities.append(";ENABLE_RENDERDOC_CAPTURE")
            print(
                "\n".join((";#IPC_ENABLED", *capabilities, ";#IPC_END")),
                file=sys.stderr,
            )
            sys.stderr.flush()
        for _ in range(2):
            ipc_commands.append(sys.stdin.readline().strip())
            ipc_command_seconds.append(time.monotonic() - launched)

    if args.spawn_child:
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])

    if args.sleep:
        time.sleep(args.sleep)

    if args.expect_ipc:
        while True:
            command = sys.stdin.readline().strip()
            ipc_commands.append(command)
            ipc_command_seconds.append(time.monotonic() - launched)
            if command == "SCREENSHOT" and not args.ignore_screenshots:
                screenshots = Path("user") / "screenshots"
                if args.single_screenshot and screenshots.exists():
                    continue
                time.sleep(args.screenshot_delay)
                screenshots.mkdir(parents=True, exist_ok=True)
                index = len(list(screenshots.iterdir()))
                if args.malformed_screenshots:
                    png = b"\x89PNG\r\n\x1a\nmalformed"
                elif args.dark_movement_screenshots:
                    png = dark_movement_png(index)
                elif args.vary_left_pixel:
                    png = varying_left_pixel_png(index)
                else:
                    png = one_pixel_png(
                        index
                        if args.vary_screenshots
                        or (
                            args.vary_screenshots_after is not None
                            and index >= args.vary_screenshots_after
                        )
                        else 0
                    )
                (screenshots / f"fake_{index}.png").write_bytes(png)
            if command == "RENDERDOC_CAPTURE" and not args.ignore_renderdoc_captures:
                captures = Path("user") / "captures"
                captures.mkdir(parents=True, exist_ok=True)
                index = len(list(captures.iterdir()))
                (captures / f"fake_{index}.rdc").write_bytes(
                    b"synthetic RenderDoc capture"
                )
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

    config_path = Path("user") / "config.json"
    observation = {
        "cwd": str(Path.cwd()),
        "game": args.game,
        "user_directory_exists": (Path.cwd() / "user").is_dir(),
        "ipc_commands": ipc_commands,
        "ipc_command_seconds": ipc_command_seconds,
        "ipc_enabled": os.environ.get("SHADPS4_ENABLE_IPC"),
        "user_config": (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else None
        ),
    }
    (Path.cwd() / "observation.json").write_text(
        json.dumps(observation), encoding="utf-8"
    )
    return args.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
