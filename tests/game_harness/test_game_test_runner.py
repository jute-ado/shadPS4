# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.game_test_runner import (
    ManifestError,
    load_manifest,
    main,
    run_case,
    run_manifest,
)


FIXTURE = Path(__file__).with_name("fake_emulator.py")


class ManifestTests(unittest.TestCase):
    def write_manifest(self, root: Path, content: dict) -> Path:
        path = root / "games.json"
        path.write_text(json.dumps(content), encoding="utf-8")
        return path

    def test_load_manifest_resolves_paths_relative_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            game = root / "games" / "CUSA00001"
            emulator = root / "bin" / "shadPS4.exe"
            game.mkdir(parents=True)
            emulator.parent.mkdir()
            emulator.touch()
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "emulator": "bin/shadPS4.exe",
                    "cases": [
                        {
                            "name": "boot",
                            "gamePath": "games/CUSA00001",
                            "timeoutSeconds": 5,
                        }
                    ],
                },
            )

            manifest = load_manifest(path)

            self.assertEqual(manifest.emulator, emulator.resolve())
            self.assertEqual(manifest.cases[0].game_path, game.resolve())

    def test_load_manifest_rejects_unsupported_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_manifest(
                Path(directory), {"schemaVersion": 2, "cases": []}
            )

            with self.assertRaisesRegex(ManifestError, "schemaVersion"):
                load_manifest(path)

    def test_load_manifest_rejects_duplicate_case_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            game = root / "game"
            game.mkdir()
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {"name": "same", "gamePath": "game", "timeoutSeconds": 1},
                        {"name": "same", "gamePath": "game", "timeoutSeconds": 1},
                    ],
                },
            )

            with self.assertRaisesRegex(ManifestError, "duplicate"):
                load_manifest(path)

    def test_load_manifest_rejects_nonpositive_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            game = root / "game"
            game.mkdir()
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {"name": "boot", "gamePath": "game", "timeoutSeconds": 0}
                    ],
                },
            )

            with self.assertRaisesRegex(ManifestError, "timeoutSeconds"):
                load_manifest(path)

    def test_load_manifest_rejects_missing_game_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_manifest(
                Path(directory),
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "missing",
                            "gamePath": "not-installed",
                            "timeoutSeconds": 1,
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(ManifestError, "gamePath"):
                load_manifest(path)

    def test_load_manifest_rejects_malformed_shapes_and_values(self) -> None:
        invalid_manifests = [
            ("not JSON", "cannot read manifest"),
            (json.dumps([]), "root"),
            (json.dumps({"schemaVersion": 1, "cases": []}), "cases"),
            (
                json.dumps({"schemaVersion": 1, "cases": ["case"]}),
                "must be an object",
            ),
            (
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "cases": [
                            {
                                "name": "",
                                "gamePath": "game",
                                "timeoutSeconds": 1,
                            }
                        ],
                    }
                ),
                "name",
            ),
            (
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "cases": [
                            {
                                "name": "bad args",
                                "gamePath": "game",
                                "timeoutSeconds": 1,
                                "args": "not-an-array",
                            }
                        ],
                    }
                ),
                "args",
            ),
            (
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "cases": [
                            {
                                "name": "bad outcome",
                                "gamePath": "game",
                                "timeoutSeconds": 1,
                                "allowedOutcomes": ["exploded"],
                            }
                        ],
                    }
                ),
                "allowedOutcomes",
            ),
        ]
        for content, expected in invalid_manifests:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / "game").mkdir()
                    path = root / "games.json"
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ManifestError, expected):
                        load_manifest(path)


class RunnerTests(unittest.TestCase):
    def make_manifest(
        self, root: Path, *, case: dict, emulator: str | None = None
    ) -> Path:
        game = root / "game"
        game.mkdir()
        content = {
            "schemaVersion": 1,
            "cases": [{"gamePath": "game", "timeoutSeconds": 2, **case}],
        }
        if emulator is not None:
            content["emulator"] = emulator
        path = root / "games.json"
        path.write_text(json.dumps(content), encoding="utf-8")
        return path

    def test_run_case_isolates_working_directory_and_collects_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = load_manifest(
                self.make_manifest(
                    root,
                    case={
                        "name": "clean boot",
                        "args": [
                            "--stdout",
                            "stdout marker",
                            "--stderr",
                            "stderr marker",
                            "--log",
                            "log marker",
                        ],
                        "requiredLogPatterns": [
                            "stdout marker",
                            "stderr marker",
                            "log marker",
                        ],
                    },
                )
            )

            result = run_case(
                manifest.cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertTrue(result.passed, result.failures)
            self.assertEqual(result.outcome, "exited_zero")
            observation = json.loads(
                (result.artifact_directory / "observation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(observation["user_directory_exists"])
            self.assertEqual(Path(observation["cwd"]), result.artifact_directory)
            self.assertEqual(Path(observation["game"]), (root / "game").resolve())

    def test_run_case_reports_nonzero_exit_when_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = load_manifest(
                self.make_manifest(
                    root,
                    case={
                        "name": "known stop",
                        "args": ["--exit-code", "7"],
                        "allowedOutcomes": ["exited_nonzero"],
                    },
                )
            )

            result = run_case(
                manifest.cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertTrue(result.passed, result.failures)
            self.assertEqual(result.exit_code, 7)

    def test_run_case_fails_on_forbidden_log_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = load_manifest(
                self.make_manifest(
                    root,
                    case={
                        "name": "crash marker",
                        "args": ["--log", "critical guest crash"],
                        "forbiddenLogPatterns": ["guest crash"],
                    },
                )
            )

            result = run_case(
                manifest.cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertFalse(result.passed)
            self.assertIn("forbidden log pattern", result.failures[0])

    def test_run_case_reports_unexpected_outcome_and_missing_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = load_manifest(
                self.make_manifest(
                    root,
                    case={
                        "name": "bad boot",
                        "args": ["--exit-code", "3"],
                        "requiredLogPatterns": ["ready marker"],
                    },
                )
            )

            result = run_case(
                manifest.cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertFalse(result.passed)
            self.assertEqual(len(result.failures), 2)
            self.assertIn("not allowed", result.failures[0])
            self.assertIn("required log pattern", result.failures[1])

    def test_run_case_times_out_and_kills_process_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "survival",
                    "timeoutSeconds": 0.2,
                    "args": ["--spawn-child", "--sleep", "60"],
                    "allowedOutcomes": ["timed_out"],
                },
            )
            manifest = load_manifest(manifest_path)

            result = run_case(
                manifest.cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertTrue(result.passed, result.failures)
            self.assertEqual(result.outcome, "timed_out")
            self.assertLess(result.duration_seconds, 5)

    def test_run_case_caps_output_and_marks_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = load_manifest(
                self.make_manifest(
                    root,
                    case={
                        "name": "noisy",
                        "args": ["--emit-bytes", "4096"],
                    },
                )
            )

            result = run_case(
                manifest.cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
                output_limit_bytes=1024,
            )

            self.assertTrue(result.output_truncated)
            self.assertLessEqual(
                (result.artifact_directory / "stdout.log").stat().st_size, 1024
            )

    def test_run_manifest_writes_machine_readable_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = load_manifest(
                self.make_manifest(root, case={"name": "summary case"})
            )
            artifacts = root / "artifacts"

            summary = run_manifest(
                manifest,
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=artifacts,
            )

            report = json.loads(
                (artifacts / "game-test-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary.failed, 0)
            self.assertEqual(report["schemaVersion"], 1)
            self.assertEqual(report["cases"][0]["name"], "summary case")
            self.assertTrue(report["cases"][0]["passed"])

    def test_run_manifest_uses_emulator_and_arguments_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = load_manifest(
                self.make_manifest(
                    root,
                    emulator=sys.executable,
                    case={
                        "name": "manifest command",
                        "args": [str(FIXTURE), "--stdout", "from manifest"],
                        "requiredLogPatterns": ["from manifest"],
                    },
                )
            )

            summary = run_manifest(
                manifest,
                artifacts_root=root / "artifacts",
            )

            self.assertEqual(summary.failed, 0)

    def test_run_manifest_requires_an_emulator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = load_manifest(
                self.make_manifest(root, case={"name": "no emulator"})
            )

            with self.assertRaisesRegex(ManifestError, "no emulator"):
                run_manifest(manifest, artifacts_root=root / "artifacts")

    def test_run_case_rejects_invalid_runner_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = load_manifest(
                self.make_manifest(root, case={"name": "invalid runner"})
            )
            for command, limit in (([], 1), ([sys.executable], 0)):
                with self.subTest(command=command, limit=limit):
                    with self.assertRaises(ValueError):
                        run_case(
                            manifest.cases[0],
                            emulator_command=command,
                            artifacts_root=root / "artifacts",
                            output_limit_bytes=limit,
                        )

    def test_run_case_records_emulator_launch_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = load_manifest(
                self.make_manifest(root, case={"name": "missing executable"})
            )

            result = run_case(
                manifest.cases[0],
                emulator_command=[str(root / "does-not-exist.exe")],
                artifacts_root=root / "artifacts",
            )

            self.assertFalse(result.passed)
            self.assertEqual(result.outcome, "launch_failed")
            self.assertIsNone(result.exit_code)
            self.assertIn("failed to launch emulator", result.failures[0])
            self.assertIn(
                "does-not-exist.exe",
                (result.artifact_directory / "stderr.log").read_text(encoding="utf-8"),
            )

    def test_run_manifest_writes_report_when_emulator_cannot_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = load_manifest(
                self.make_manifest(root, case={"name": "missing executable"})
            )
            artifacts = root / "artifacts"

            summary = run_manifest(
                manifest,
                emulator_command=[str(root / "does-not-exist.exe")],
                artifacts_root=artifacts,
            )

            report = json.loads(
                (artifacts / "game-test-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary.failed, 1)
            self.assertEqual(report["cases"][0]["outcome"], "launch_failed")

    def test_case_artifact_names_are_safe_and_collision_resistant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            game = root / "game"
            game.mkdir()
            path = root / "games.json"
            path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "cases": [
                            {
                                "name": "Boot / menu",
                                "gamePath": "game",
                                "timeoutSeconds": 1,
                            },
                            {
                                "name": "Boot : menu",
                                "gamePath": "game",
                                "timeoutSeconds": 1,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = load_manifest(path)

            summary = run_manifest(
                manifest,
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            directories = [case.artifact_directory.name for case in summary.cases]
            self.assertEqual(len(set(directories)), 2)
            self.assertTrue(
                all("/" not in name and ":" not in name for name in directories)
            )


class CommandLineTests(unittest.TestCase):
    def make_cli_manifest(
        self, root: Path, *, exit_code: int = 0, include_emulator: bool = False
    ) -> Path:
        (root / "game").mkdir()
        content = {
            "schemaVersion": 1,
            "cases": [
                {
                    "name": "cli",
                    "gamePath": "game",
                    "timeoutSeconds": 2,
                    "args": [str(FIXTURE), "--exit-code", str(exit_code)],
                }
            ],
        }
        if include_emulator:
            content["emulator"] = sys.executable
        path = root / "games.json"
        path.write_text(json.dumps(content), encoding="utf-8")
        return path

    def run_cli(
        self, *args: str, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        script = Path(__file__).parents[2] / "scripts" / "game_test_runner.py"
        return subprocess.run(
            [sys.executable, str(script), *args],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

    def test_cli_runs_from_manifest_and_returns_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_cli_manifest(root, include_emulator=True)

            result = self.run_cli(
                "--manifest",
                str(manifest),
                "--artifacts",
                str(root / "artifacts"),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("1 passed, 0 failed", result.stdout)

    def test_cli_returns_failure_when_a_case_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_cli_manifest(root, exit_code=9)

            result = self.run_cli(
                "--manifest",
                str(manifest),
                "--emulator",
                sys.executable,
                "--artifacts",
                str(root / "artifacts"),
            )

            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("0 passed, 1 failed", result.stdout)

    def test_cli_accepts_manifest_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_cli_manifest(root, include_emulator=True)
            env = os.environ.copy()
            env["SHADPS4_GAME_TEST_MANIFEST"] = str(manifest)

            result = self.run_cli(
                "--artifacts",
                str(root / "artifacts"),
                env=env,
            )

            self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_reports_manifest_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "invalid.json"
            invalid.write_text("{}", encoding="utf-8")

            result = self.run_cli("--manifest", str(invalid))

            self.assertEqual(result.returncode, 2)
            self.assertIn("game test error", result.stderr)

    def test_main_return_codes_and_messages_are_covered_in_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            passing = self.make_cli_manifest(root, include_emulator=True)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                success = main(
                    [
                        "--manifest",
                        str(passing),
                        "--artifacts",
                        str(root / "passing-artifacts"),
                    ]
                )
            self.assertEqual(success, 0)
            self.assertIn("1 passed, 0 failed", stdout.getvalue())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failing = self.make_cli_manifest(root, exit_code=4, include_emulator=True)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                failure = main(
                    [
                        "--manifest",
                        str(failing),
                        "--artifacts",
                        str(root / "failing-artifacts"),
                    ]
                )
            self.assertEqual(failure, 1)
            self.assertIn("0 passed, 1 failed", stdout.getvalue())

        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "invalid.json"
            invalid.write_text("{}", encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                error = main(["--manifest", str(invalid)])
            self.assertEqual(error, 2)
            self.assertIn("game test error", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
