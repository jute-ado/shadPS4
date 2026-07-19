# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

"""Run repeatable, isolated shadPS4 game smoke tests from a private manifest."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any, BinaryIO, Sequence


REPORT_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_LIMIT_BYTES = 64 * 1024 * 1024
VALID_OUTCOMES = frozenset({"exited_zero", "exited_nonzero", "timed_out"})


class ManifestError(ValueError):
    """Raised when a game test manifest is invalid."""


@dataclass(frozen=True)
class GameCase:
    name: str
    game_path: Path
    timeout_seconds: float
    use_ipc: bool = False
    args: tuple[str, ...] = ()
    allowed_outcomes: tuple[str, ...] = ("exited_zero",)
    required_log_patterns: tuple[str, ...] = ()
    forbidden_log_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class GameManifest:
    source: Path
    emulator: Path | None
    cases: tuple[GameCase, ...]


@dataclass
class CaseResult:
    name: str
    passed: bool
    outcome: str
    exit_code: int | None
    duration_seconds: float
    artifact_directory: Path
    output_truncated: bool
    failures: list[str]

    def to_report(self) -> dict[str, Any]:
        report = asdict(self)
        report["artifact_directory"] = str(self.artifact_directory)
        return report


@dataclass
class RunSummary:
    cases: list[CaseResult]

    @property
    def failed(self) -> int:
        return sum(not case.passed for case in self.cases)

    @property
    def passed(self) -> int:
        return len(self.cases) - self.failed


def _require_string_list(raw: Any, field: str, case_name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ManifestError(f"{case_name}: {field} must be an array of strings")
    return tuple(raw)


def _require_bool(raw: Any, field: str, case_name: str) -> bool:
    if not isinstance(raw, bool):
        raise ManifestError(f"{case_name}: {field} must be a boolean")
    return raw


def _resolve_existing_path(root: Path, raw: Any, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ManifestError(f"{field} must be a non-empty string")
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.exists():
        raise ManifestError(f"{field} does not exist: {path}")
    return path


def load_manifest(path: str | Path) -> GameManifest:
    source = Path(path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read manifest {source}: {error}") from error

    if not isinstance(raw, dict):
        raise ManifestError("manifest root must be an object")
    if raw.get("schemaVersion") != REPORT_SCHEMA_VERSION:
        raise ManifestError(f"schemaVersion must be {REPORT_SCHEMA_VERSION}")

    root = source.parent
    emulator = None
    if "emulator" in raw:
        emulator = _resolve_existing_path(root, raw["emulator"], "emulator")

    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ManifestError("cases must be a non-empty array")

    cases: list[GameCase] = []
    seen_names: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ManifestError(f"case {index}: must be an object")
        name = raw_case.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ManifestError(f"case {index}: name must be a non-empty string")
        if name in seen_names:
            raise ManifestError(f"duplicate case name: {name}")
        seen_names.add(name)

        timeout = raw_case.get("timeoutSeconds")
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or timeout <= 0
        ):
            raise ManifestError(f"{name}: timeoutSeconds must be positive")

        args = _require_string_list(raw_case.get("args"), "args", name)
        outcomes = _require_string_list(
            raw_case.get("allowedOutcomes", ["exited_zero"]),
            "allowedOutcomes",
            name,
        )
        if not outcomes or any(outcome not in VALID_OUTCOMES for outcome in outcomes):
            raise ManifestError(
                f"{name}: allowedOutcomes must contain only {sorted(VALID_OUTCOMES)}"
            )

        cases.append(
            GameCase(
                name=name,
                game_path=_resolve_existing_path(
                    root, raw_case.get("gamePath"), f"{name}: gamePath"
                ),
                timeout_seconds=float(timeout),
                use_ipc=_require_bool(raw_case.get("useIpc", False), "useIpc", name),
                args=args,
                allowed_outcomes=outcomes,
                required_log_patterns=_require_string_list(
                    raw_case.get("requiredLogPatterns"),
                    "requiredLogPatterns",
                    name,
                ),
                forbidden_log_patterns=_require_string_list(
                    raw_case.get("forbiddenLogPatterns"),
                    "forbiddenLogPatterns",
                    name,
                ),
            )
        )

    return GameManifest(source=source, emulator=emulator, cases=tuple(cases))


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._").lower()
    return safe or "case"


def _drain_stream(
    stream: BinaryIO,
    destination: Path,
    limit: int,
    truncated: threading.Event,
) -> None:
    written = 0
    with destination.open("wb") as output:
        while chunk := stream.read(64 * 1024):
            remaining = max(0, limit - written)
            if remaining:
                kept = chunk[:remaining]
                output.write(kept)
                written += len(kept)
            if len(chunk) > remaining:
                truncated.set()


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.kill()
    except OSError:
        pass


def _read_capped(path: Path, limit: int) -> tuple[str, bool]:
    if not path.exists():
        return "", False
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    return content[:limit].decode("utf-8", errors="replace"), len(content) > limit


def run_case(
    case: GameCase,
    *,
    emulator_command: Sequence[str],
    artifacts_root: str | Path,
    output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
    artifact_name: str | None = None,
) -> CaseResult:
    if not emulator_command:
        raise ValueError("emulator_command must not be empty")
    if output_limit_bytes <= 0:
        raise ValueError("output_limit_bytes must be positive")

    artifact_directory = Path(artifacts_root).resolve() / (
        artifact_name or _safe_name(case.name)
    )
    artifact_directory.mkdir(parents=True, exist_ok=False)
    (artifact_directory / "user").mkdir()

    command = [*map(str, emulator_command), *case.args, str(case.game_path)]
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    environment = None
    if case.use_ipc:
        environment = os.environ.copy()
        environment["SHADPS4_ENABLE_IPC"] = "true"
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=artifact_directory,
            stdin=subprocess.PIPE if case.use_ipc else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            creationflags=creation_flags,
            start_new_session=os.name != "nt",
        )
    except OSError as error:
        failure = f"failed to launch emulator: {error}"
        (artifact_directory / "stdout.log").touch()
        (artifact_directory / "stderr.log").write_text(
            f"{failure}\ncommand: {command!r}\n", encoding="utf-8"
        )
        return CaseResult(
            name=case.name,
            passed=False,
            outcome="launch_failed",
            exit_code=None,
            duration_seconds=time.monotonic() - started,
            artifact_directory=artifact_directory,
            output_truncated=False,
            failures=[failure],
        )
    assert process.stdout is not None
    assert process.stderr is not None
    if case.use_ipc:
        assert process.stdin is not None
        process.stdin.write(b"RUN\nSTART\n")
        process.stdin.flush()

    truncated = threading.Event()
    readers = [
        threading.Thread(
            target=_drain_stream,
            args=(
                process.stdout,
                artifact_directory / "stdout.log",
                output_limit_bytes,
                truncated,
            ),
        ),
        threading.Thread(
            target=_drain_stream,
            args=(
                process.stderr,
                artifact_directory / "stderr.log",
                output_limit_bytes,
                truncated,
            ),
        ),
    ]
    for reader in readers:
        reader.start()

    timed_out = False
    try:
        process.wait(timeout=case.timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        if case.use_ipc:
            assert process.stdin is not None
            try:
                process.stdin.write(b"STOP\n")
                process.stdin.flush()
                process.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                _kill_process_tree(process)
                process.wait(timeout=5)
        else:
            _kill_process_tree(process)
            process.wait(timeout=5)
    finally:
        if process.stdin is not None:
            process.stdin.close()
        for reader in readers:
            reader.join(timeout=5)
        process.stdout.close()
        process.stderr.close()
    duration = time.monotonic() - started

    if timed_out:
        outcome = "timed_out"
    elif process.returncode == 0:
        outcome = "exited_zero"
    else:
        outcome = "exited_nonzero"

    stdout, stdout_truncated = _read_capped(
        artifact_directory / "stdout.log", output_limit_bytes
    )
    stderr, stderr_truncated = _read_capped(
        artifact_directory / "stderr.log", output_limit_bytes
    )
    emulator_log, log_truncated = _read_capped(
        artifact_directory / "user" / "log" / "shad_log.txt",
        output_limit_bytes,
    )
    combined_log = "\n".join((stdout, stderr, emulator_log))

    failures: list[str] = []
    if outcome not in case.allowed_outcomes:
        failures.append(
            f"outcome {outcome!r} is not allowed; expected "
            f"{list(case.allowed_outcomes)!r}"
        )
    for pattern in case.required_log_patterns:
        if pattern not in combined_log:
            failures.append(f"required log pattern not found: {pattern!r}")
    for pattern in case.forbidden_log_patterns:
        if pattern in combined_log:
            failures.append(f"forbidden log pattern found: {pattern!r}")

    return CaseResult(
        name=case.name,
        passed=not failures,
        outcome=outcome,
        exit_code=process.returncode,
        duration_seconds=duration,
        artifact_directory=artifact_directory,
        output_truncated=(
            truncated.is_set() or stdout_truncated or stderr_truncated or log_truncated
        ),
        failures=failures,
    )


def run_manifest(
    manifest: GameManifest,
    *,
    emulator_command: Sequence[str] | None = None,
    artifacts_root: str | Path,
    output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
) -> RunSummary:
    command = list(emulator_command or ())
    if not command:
        if manifest.emulator is None:
            raise ManifestError("manifest has no emulator; pass an emulator command")
        command = [str(manifest.emulator)]

    root = Path(artifacts_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = [
        run_case(
            case,
            emulator_command=command,
            artifacts_root=root,
            output_limit_bytes=output_limit_bytes,
            artifact_name=f"{index:03d}-{_safe_name(case.name)}",
        )
        for index, case in enumerate(manifest.cases, start=1)
    ]
    summary = RunSummary(cases=results)
    report = {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "manifest": str(manifest.source),
        "passed": summary.passed,
        "failed": summary.failed,
        "cases": [result.to_report() for result in results],
    }
    (root / "game-test-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=os.environ.get("SHADPS4_GAME_TEST_MANIFEST"),
        help="private game manifest (or SHADPS4_GAME_TEST_MANIFEST)",
    )
    parser.add_argument("--emulator", help="override the emulator executable")
    parser.add_argument(
        "--artifacts",
        default="artifacts/game-tests",
        help="directory for logs and JSON results",
    )
    parser.add_argument(
        "--output-limit-bytes",
        type=int,
        default=DEFAULT_OUTPUT_LIMIT_BYTES,
        help="maximum bytes retained for each output stream",
    )
    args = parser.parse_args(argv)
    if not args.manifest:
        parser.error("--manifest or SHADPS4_GAME_TEST_MANIFEST is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        command = [args.emulator] if args.emulator else None
        summary = run_manifest(
            manifest,
            emulator_command=command,
            artifacts_root=args.artifacts,
            output_limit_bytes=args.output_limit_bytes,
        )
    except (ManifestError, OSError, ValueError) as error:
        print(f"game test error: {error}", file=sys.stderr)
        return 2

    print(f"{summary.passed} passed, {summary.failed} failed")
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
