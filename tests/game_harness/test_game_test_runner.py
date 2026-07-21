# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
import zlib

from scripts.game_test_runner import (
    ManifestError,
    ScreenshotRegion,
    _decode_png_rgb,
    _screenshot_difference,
    load_manifest,
    main,
    run_case,
    run_manifest,
)

FIXTURE = Path(__file__).with_name("fake_emulator.py")


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + data).to_bytes(4, "big")
    return len(data).to_bytes(4, "big") + chunk_type + data + checksum


def test_png(width: int, height: int, color_type: int, scanlines: bytes) -> bytes:
    header = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes((8, color_type, 0, 0, 0))
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(scanlines))
        + png_chunk(b"IEND", b"")
    )


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

    def test_load_manifest_resolves_portable_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game").mkdir()
            config = root / "profiles" / "config.json"
            config.parent.mkdir()
            config.write_text('{"GPU":{"readbackLinearImages":true}}', encoding="utf-8")
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "configured boot",
                            "gamePath": "game",
                            "timeoutSeconds": 1,
                            "userConfig": "profiles/config.json",
                        }
                    ],
                },
            )

            self.assertEqual(load_manifest(path).cases[0].user_config, config.resolve())

    def test_load_manifest_resolves_portable_user_data_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game").mkdir()
            seed = root / "profiles" / "checkpoint"
            seed.mkdir(parents=True)
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "seeded boot",
                            "gamePath": "game",
                            "timeoutSeconds": 1,
                            "userDataSeed": "profiles/checkpoint",
                        }
                    ],
                },
            )

            self.assertEqual(
                load_manifest(path).cases[0].user_data_seed, seed.resolve()
            )

    def test_load_manifest_rejects_user_data_seed_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game").mkdir()
            (root / "checkpoint").write_bytes(b"not a directory")
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "bad seed",
                            "gamePath": "game",
                            "timeoutSeconds": 1,
                            "userDataSeed": "checkpoint",
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(
                ManifestError, "userDataSeed must be a directory"
            ):
                load_manifest(path)

    def test_load_manifest_rejects_invalid_portable_user_config(self) -> None:
        for content in ("not json", "[]"):
            with (
                self.subTest(content=content),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                (root / "game").mkdir()
                (root / "config.json").write_text(content, encoding="utf-8")
                path = self.write_manifest(
                    root,
                    {
                        "schemaVersion": 1,
                        "cases": [
                            {
                                "name": "bad config",
                                "gamePath": "game",
                                "timeoutSeconds": 1,
                                "userConfig": "config.json",
                            }
                        ],
                    },
                )

                with self.assertRaisesRegex(
                    ManifestError, "userConfig must contain a JSON object"
                ):
                    load_manifest(path)

    def test_load_manifest_rejects_config_ignored_by_clean_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game").mkdir()
            (root / "config.json").write_text("{}", encoding="utf-8")
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "ignored config",
                            "gamePath": "game",
                            "timeoutSeconds": 1,
                            "userConfig": "config.json",
                            "args": ["--config-clean"],
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(
                ManifestError, "userConfig cannot be combined with --config-clean"
            ):
                load_manifest(path)

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

    def test_load_manifest_accepts_explicit_ipc_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game").mkdir()
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "controlled boot",
                            "gamePath": "game",
                            "timeoutSeconds": 1,
                            "useIpc": True,
                        }
                    ],
                },
            )

            manifest = load_manifest(path)

            self.assertTrue(manifest.cases[0].use_ipc)

    def test_load_manifest_accepts_scheduled_screenshots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game").mkdir()
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "visual boot",
                            "gamePath": "game",
                            "timeoutSeconds": 2,
                            "useIpc": True,
                            "screenshotSeconds": [0.25, 1.5],
                            "minimumDistinctScreenshots": 2,
                        }
                    ],
                },
            )

            manifest = load_manifest(path)

            self.assertEqual(manifest.cases[0].screenshot_seconds, (0.25, 1.5))
            self.assertEqual(manifest.cases[0].minimum_distinct_screenshots, 2)

    def test_load_manifest_accepts_scheduled_renderdoc_captures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game").mkdir()
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "frame diagnosis",
                            "gamePath": "game",
                            "timeoutSeconds": 2,
                            "useIpc": True,
                            "renderdocCaptureSeconds": [0.25, 1.5],
                        }
                    ],
                },
            )

            manifest = load_manifest(path)

            self.assertEqual(manifest.cases[0].renderdoc_capture_seconds, (0.25, 1.5))

    def test_load_manifest_rejects_renderdoc_captures_without_ipc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game").mkdir()
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "manual capture",
                            "gamePath": "game",
                            "timeoutSeconds": 2,
                            "renderdocCaptureSeconds": [0.25],
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(
                ManifestError, "renderdocCaptureSeconds requires useIpc"
            ):
                load_manifest(path)

    def test_load_manifest_accepts_screenshot_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game").mkdir()
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "causal visual",
                            "gamePath": "game",
                            "timeoutSeconds": 2,
                            "useIpc": True,
                            "screenshotSeconds": [0.25, 0.5, 1.0],
                            "screenshotComparisons": [
                                {
                                    "firstScreenshot": 0,
                                    "secondScreenshot": 1,
                                    "maximumDifference": 0.01,
                                },
                                {
                                    "firstScreenshot": 1,
                                    "secondScreenshot": 2,
                                    "minimumDifference": 0.2,
                                    "differenceMode": "cosine",
                                },
                            ],
                        }
                    ],
                },
            )

            case = load_manifest(path).cases[0]

            self.assertEqual(
                [
                    (
                        comparison.first_screenshot,
                        comparison.second_screenshot,
                        comparison.minimum_difference,
                        comparison.maximum_difference,
                        comparison.difference_mode,
                    )
                    for comparison in case.screenshot_comparisons
                ],
                [
                    (0, 1, None, 0.01, "mean_absolute"),
                    (1, 2, 0.2, None, "cosine"),
                ],
            )

    def test_load_manifest_rejects_unknown_screenshot_difference_mode(self) -> None:
        for value in ("unknown", []):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "game").mkdir()
                path = self.write_manifest(
                    root,
                    {
                        "schemaVersion": 1,
                        "cases": [
                            {
                                "name": "bad comparison mode",
                                "gamePath": "game",
                                "timeoutSeconds": 2,
                                "useIpc": True,
                                "screenshotSeconds": [0.25, 0.5],
                                "screenshotComparisons": [
                                    {
                                        "firstScreenshot": 0,
                                        "secondScreenshot": 1,
                                        "minimumDifference": 0.1,
                                        "differenceMode": value,
                                    }
                                ],
                            }
                        ],
                    },
                )

                with self.assertRaisesRegex(
                    ManifestError, "differenceMode must be one of"
                ):
                    load_manifest(path)

    def test_load_manifest_rejects_screenshot_comparison_out_of_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game").mkdir()
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "bad comparison",
                            "gamePath": "game",
                            "timeoutSeconds": 2,
                            "useIpc": True,
                            "screenshotSeconds": [0.25],
                            "screenshotComparisons": [
                                {
                                    "firstScreenshot": 0,
                                    "secondScreenshot": 1,
                                    "minimumDifference": 0.1,
                                }
                            ],
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(
                ManifestError, "secondScreenshot must index screenshotSeconds"
            ):
                load_manifest(path)

    def test_load_manifest_rejects_invalid_screenshot_comparisons(self) -> None:
        invalid_values = (
            ({"screenshotComparisons": {}}, "must be an array"),
            ({"screenshotComparisons": ["bad"]}, "must be an object"),
            (
                {
                    "screenshotComparisons": [
                        {
                            "firstScreenshot": 0,
                            "secondScreenshot": 0,
                            "minimumDifference": 0.1,
                        }
                    ]
                },
                "must compare two different screenshots",
            ),
            (
                {
                    "screenshotComparisons": [
                        {
                            "firstScreenshot": 0,
                            "secondScreenshot": 1,
                            "minimumDifference": -0.1,
                        }
                    ]
                },
                "minimumDifference must be between 0 and 1",
            ),
            (
                {
                    "screenshotComparisons": [
                        {"firstScreenshot": 0, "secondScreenshot": 1}
                    ]
                },
                "must specify minimumDifference or maximumDifference",
            ),
            (
                {
                    "screenshotComparisons": [
                        {
                            "firstScreenshot": 0,
                            "secondScreenshot": 1,
                            "minimumDifference": 0.8,
                            "maximumDifference": 0.2,
                        }
                    ]
                },
                "minimumDifference cannot exceed maximumDifference",
            ),
        )
        for fields, message in invalid_values:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                (root / "game").mkdir()
                case = {
                    "name": "bad comparison",
                    "gamePath": "game",
                    "timeoutSeconds": 2,
                    "useIpc": True,
                    "screenshotSeconds": [0.25, 0.5],
                    **fields,
                }
                path = self.write_manifest(root, {"schemaVersion": 1, "cases": [case]})

                with self.assertRaisesRegex(ManifestError, message):
                    load_manifest(path)

    def test_load_manifest_accepts_scheduled_button_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game").mkdir()
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "automated menu",
                            "gamePath": "game",
                            "timeoutSeconds": 2,
                            "useIpc": True,
                            "buttonEvents": [
                                {
                                    "seconds": 0.25,
                                    "button": "cross",
                                    "pressed": True,
                                },
                                {
                                    "seconds": 0.35,
                                    "button": "cross",
                                    "pressed": False,
                                },
                            ],
                        }
                    ],
                },
            )

            manifest = load_manifest(path)

            self.assertEqual(
                [
                    (event.seconds, event.button, event.pressed)
                    for event in manifest.cases[0].button_events
                ],
                [(0.25, "cross", True), (0.35, "cross", False)],
            )

    def test_load_manifest_accepts_screenshot_driven_button_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game").mkdir()
            expected_hash = "a" * 64
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "state-driven menu",
                            "gamePath": "game",
                            "timeoutSeconds": 2,
                            "useIpc": True,
                            "screenshotButtonEvents": [
                                {
                                    "screenshotSha256": expected_hash,
                                    "button": "cross",
                                    "timeoutSeconds": 0.75,
                                    "pollSeconds": 0.05,
                                    "holdSeconds": 0.1,
                                }
                            ],
                        }
                    ],
                },
            )

            event = load_manifest(path).cases[0].screenshot_button_events[0]

            self.assertEqual(event.screenshot_sha256, expected_hash)
            self.assertEqual(event.button, "cross")
            self.assertEqual(event.timeout_seconds, 0.75)
            self.assertEqual(event.poll_seconds, 0.05)
            self.assertEqual(event.hold_seconds, 0.1)

    def test_load_manifest_accepts_visual_checkpoint_comparison_region(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game").mkdir()
            (root / "reference.png").write_bytes(
                test_png(2, 1, 2, bytes((0, 0, 0, 0, 0, 0, 0)))
            )
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "stable controls",
                            "gamePath": "game",
                            "timeoutSeconds": 2,
                            "useIpc": True,
                            "screenshotButtonEvents": [
                                {
                                    "referenceScreenshot": "reference.png",
                                    "comparisonRegion": {
                                        "left": 1,
                                        "top": 0,
                                        "width": 1,
                                        "height": 1,
                                    },
                                    "button": "cross",
                                    "timeoutSeconds": 0.75,
                                }
                            ],
                        }
                    ],
                },
            )

            event = load_manifest(path).cases[0].screenshot_button_events[0]

            self.assertEqual(
                event.comparison_region,
                ScreenshotRegion(left=1, top=0, width=1, height=1),
            )

    def test_load_manifest_rejects_mixed_timed_and_screenshot_driven_events(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game").mkdir()
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "ambiguous menu",
                            "gamePath": "game",
                            "timeoutSeconds": 2,
                            "useIpc": True,
                            "buttonEvents": [
                                {"seconds": 0.25, "button": "cross", "pressed": True}
                            ],
                            "screenshotButtonEvents": [
                                {
                                    "screenshotSha256": "a" * 64,
                                    "button": "cross",
                                    "timeoutSeconds": 0.75,
                                }
                            ],
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(ManifestError, "cannot be combined"):
                load_manifest(path)

    def test_load_manifest_rejects_failure_capture_without_visual_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game").mkdir()
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "missing visual event",
                            "gamePath": "game",
                            "timeoutSeconds": 1,
                            "useIpc": True,
                            "renderdocCaptureOnVisualFailure": True,
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(
                ManifestError, "requires screenshotButtonEvents"
            ):
                load_manifest(path)

    def test_load_manifest_rejects_invalid_visual_checkpoint_regions(self) -> None:
        scenarios = (
            ([], "comparisonRegion must be an object"),
            (
                {"left": 0, "top": 0, "width": 1},
                "comparisonRegion.height must be an integer",
            ),
            (
                {"left": -1, "top": 0, "width": 1, "height": 1},
                "left and top cannot be negative",
            ),
            (
                {"left": 0, "top": 0, "width": 0, "height": 1},
                "width and height must be positive",
            ),
        )
        for region, message in scenarios:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                (root / "game").mkdir()
                (root / "reference.png").write_bytes(
                    test_png(1, 1, 2, bytes((0, 0, 0, 0)))
                )
                path = self.write_manifest(
                    root,
                    {
                        "schemaVersion": 1,
                        "cases": [
                            {
                                "name": "invalid stable region",
                                "gamePath": "game",
                                "timeoutSeconds": 2,
                                "useIpc": True,
                                "screenshotButtonEvents": [
                                    {
                                        "referenceScreenshot": "reference.png",
                                        "comparisonRegion": region,
                                        "button": "cross",
                                        "timeoutSeconds": 0.5,
                                    }
                                ],
                            }
                        ],
                    },
                )

                with self.assertRaisesRegex(ManifestError, message):
                    load_manifest(path)

    def test_load_manifest_rejects_invalid_screenshot_driven_button_events(
        self,
    ) -> None:
        valid = {
            "screenshotSha256": "a" * 64,
            "button": "cross",
            "timeoutSeconds": 0.75,
        }
        scenarios = (
            ({"screenshotButtonEvents": {}}, "must be an array"),
            ({"screenshotButtonEvents": ["bad"]}, "must be an object"),
            (
                {"screenshotButtonEvents": [{**valid, "screenshotSha256": "bad"}]},
                "must be a SHA-256 hex digest",
            ),
            (
                {"screenshotButtonEvents": [{**valid, "button": "start"}]},
                "unsupported button",
            ),
            (
                {"screenshotButtonEvents": [{**valid, "timeoutSeconds": 0}]},
                "must be a finite positive number",
            ),
            (
                {"screenshotButtonEvents": [{**valid, "pollSeconds": 1.0}]},
                "pollSeconds cannot exceed timeoutSeconds",
            ),
            (
                {
                    "screenshotButtonEvents": [
                        {**valid, "comparisonRegion": {"left": 0}}
                    ]
                },
                "comparisonRegion requires referenceScreenshot",
            ),
            (
                {"useIpc": False, "screenshotButtonEvents": [valid]},
                "requires useIpc",
            ),
            (
                {
                    "timeoutSeconds": 0.8,
                    "screenshotButtonEvents": [valid],
                },
                "must complete before timeoutSeconds",
            ),
        )
        for fields, message in scenarios:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                (root / "game").mkdir()
                case = {
                    "name": "invalid state-driven menu",
                    "gamePath": "game",
                    "timeoutSeconds": 2,
                    "useIpc": True,
                    **fields,
                }
                path = self.write_manifest(root, {"schemaVersion": 1, "cases": [case]})

                with self.assertRaisesRegex(ManifestError, message):
                    load_manifest(path)

    def test_load_manifest_accepts_scheduled_axis_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game").mkdir()
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "analog movement",
                            "gamePath": "game",
                            "timeoutSeconds": 2,
                            "useIpc": True,
                            "axisEvents": [
                                {"seconds": 0.25, "axis": "left_x", "value": 255},
                                {"seconds": 0.75, "axis": "left_x", "value": 128},
                                {"seconds": 1.25, "axis": "r2", "value": 255},
                                {"seconds": 1.5, "axis": "r2", "value": 0},
                            ],
                        }
                    ],
                },
            )

            manifest = load_manifest(path)

            self.assertEqual(
                [
                    (event.seconds, event.axis, event.value)
                    for event in manifest.cases[0].axis_events
                ],
                [
                    (0.25, "left_x", 255),
                    (0.75, "left_x", 128),
                    (1.25, "r2", 255),
                    (1.5, "r2", 0),
                ],
            )

    def test_load_manifest_rejects_invalid_axis_events(self) -> None:
        invalid_events = [
            (None, "axisEvents must be an array"),
            (["not-an-object"], "must be an object"),
            (
                [{"seconds": True, "axis": "left_x", "value": 128}],
                "before timeoutSeconds",
            ),
            ([{"seconds": 0.25, "axis": "left_z", "value": 128}], "unsupported axis"),
            ([{"seconds": 0.25, "axis": "left_x", "value": -1}], "between 0 and 255"),
            ([{"seconds": 0.25, "axis": "left_x", "value": 256}], "between 0 and 255"),
            ([{"seconds": 0.25, "axis": "left_x", "value": 1.5}], "must be an integer"),
            (
                [{"seconds": 0.25, "axis": "left_x", "value": True}],
                "must be an integer",
            ),
            ([{"seconds": 2, "axis": "left_x", "value": 128}], "before timeoutSeconds"),
            (
                [
                    {"seconds": 0.5, "axis": "left_x", "value": 255},
                    {"seconds": 0.25, "axis": "left_x", "value": 128},
                ],
                "in increasing time order",
            ),
        ]
        for events, expected in invalid_events:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / "game").mkdir()
                    axis_events = events if events is not None else "not-an-array"
                    path = self.write_manifest(
                        root,
                        {
                            "schemaVersion": 1,
                            "cases": [
                                {
                                    "name": "invalid analog input",
                                    "gamePath": "game",
                                    "timeoutSeconds": 2,
                                    "useIpc": True,
                                    "axisEvents": axis_events,
                                }
                            ],
                        },
                    )
                    with self.assertRaisesRegex(ManifestError, expected):
                        load_manifest(path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game").mkdir()
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "analog input needs IPC",
                            "gamePath": "game",
                            "timeoutSeconds": 2,
                            "axisEvents": [
                                {"seconds": 0.25, "axis": "left_x", "value": 255}
                            ],
                        }
                    ],
                },
            )
            with self.assertRaisesRegex(ManifestError, "axisEvents requires useIpc"):
                load_manifest(path)

    def test_load_manifest_accepts_scheduled_touch_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game").mkdir()
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "touch gesture",
                            "gamePath": "game",
                            "timeoutSeconds": 2,
                            "useIpc": True,
                            "touchEvents": [
                                {
                                    "seconds": 0.25,
                                    "finger": 0,
                                    "down": True,
                                    "x": 960,
                                    "y": 470,
                                },
                                {
                                    "seconds": 0.75,
                                    "finger": 0,
                                    "down": False,
                                    "x": 960,
                                    "y": 470,
                                },
                            ],
                        }
                    ],
                },
            )

            manifest = load_manifest(path)

            self.assertEqual(
                [
                    (event.seconds, event.finger, event.down, event.x, event.y)
                    for event in manifest.cases[0].touch_events
                ],
                [(0.25, 0, True, 960, 470), (0.75, 0, False, 960, 470)],
            )

    def test_load_manifest_rejects_invalid_touch_events(self) -> None:
        valid = {"seconds": 0.25, "finger": 0, "down": True, "x": 960, "y": 470}
        invalid_events = (
            ("not-an-array", "touchEvents must be an array"),
            (["not-an-object"], "must be an object"),
            ([{**valid, "seconds": True}], "before timeoutSeconds"),
            ([{**valid, "finger": True}], "finger must be an integer"),
            ([{**valid, "finger": 2}], "finger must be 0 or 1"),
            ([{**valid, "down": 1}], "down must be a boolean"),
            ([{**valid, "x": True}], "x must be an integer"),
            ([{**valid, "x": 1920}], "x must be between 0 and 1919"),
            ([{**valid, "y": -1}], "y must be between 0 and 941"),
            (
                [{**valid, "seconds": 0.5}, {**valid, "seconds": 0.25}],
                "in increasing time order",
            ),
        )
        for events, expected in invalid_events:
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                (root / "game").mkdir()
                path = self.write_manifest(
                    root,
                    {
                        "schemaVersion": 1,
                        "cases": [
                            {
                                "name": "invalid touch input",
                                "gamePath": "game",
                                "timeoutSeconds": 2,
                                "useIpc": True,
                                "touchEvents": events,
                            }
                        ],
                    },
                )
                with self.assertRaisesRegex(ManifestError, expected):
                    load_manifest(path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game").mkdir()
            path = self.write_manifest(
                root,
                {
                    "schemaVersion": 1,
                    "cases": [
                        {
                            "name": "touch input needs IPC",
                            "gamePath": "game",
                            "timeoutSeconds": 2,
                            "touchEvents": [valid],
                        }
                    ],
                },
            )
            with self.assertRaisesRegex(ManifestError, "touchEvents requires useIpc"):
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
            (
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "cases": [
                            {
                                "name": "bad IPC",
                                "gamePath": "game",
                                "timeoutSeconds": 1,
                                "useIpc": "yes",
                            }
                        ],
                    }
                ),
                "useIpc",
            ),
            (
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "cases": [
                            {
                                "name": "screenshots need IPC",
                                "gamePath": "game",
                                "timeoutSeconds": 1,
                                "screenshotSeconds": [0.5],
                            }
                        ],
                    }
                ),
                "screenshotSeconds requires useIpc",
            ),
            (
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "cases": [
                            {
                                "name": "late screenshot",
                                "gamePath": "game",
                                "timeoutSeconds": 1,
                                "useIpc": True,
                                "screenshotSeconds": [1],
                            }
                        ],
                    }
                ),
                "before timeoutSeconds",
            ),
            (
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "cases": [
                            {
                                "name": "non-finite screenshot",
                                "gamePath": "game",
                                "timeoutSeconds": 1,
                                "useIpc": True,
                                "screenshotSeconds": [float("nan")],
                            }
                        ],
                    }
                ),
                "finite numbers",
            ),
            (
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "cases": [
                            {
                                "name": "duplicate screenshots",
                                "gamePath": "game",
                                "timeoutSeconds": 1,
                                "useIpc": True,
                                "screenshotSeconds": [0.25, 0.25],
                            }
                        ],
                    }
                ),
                "unique and increasing",
            ),
            (
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "cases": [
                            {
                                "name": "too many distinct screenshots",
                                "gamePath": "game",
                                "timeoutSeconds": 1,
                                "useIpc": True,
                                "screenshotSeconds": [0.25],
                                "minimumDistinctScreenshots": 2,
                            }
                        ],
                    }
                ),
                "cannot exceed screenshotSeconds",
            ),
            (
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "cases": [
                            {
                                "name": "boolean distinct screenshots",
                                "gamePath": "game",
                                "timeoutSeconds": 1,
                                "useIpc": True,
                                "screenshotSeconds": [0.25],
                                "minimumDistinctScreenshots": True,
                            }
                        ],
                    }
                ),
                "must be an integer",
            ),
            (
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "cases": [
                            {
                                "name": "button events need IPC",
                                "gamePath": "game",
                                "timeoutSeconds": 1,
                                "buttonEvents": [
                                    {
                                        "seconds": 0.25,
                                        "button": "cross",
                                        "pressed": True,
                                    }
                                ],
                            }
                        ],
                    }
                ),
                "buttonEvents requires useIpc",
            ),
            (
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "cases": [
                            {
                                "name": "invalid button",
                                "gamePath": "game",
                                "timeoutSeconds": 1,
                                "useIpc": True,
                                "buttonEvents": [
                                    {
                                        "seconds": 0.25,
                                        "button": "start",
                                        "pressed": True,
                                    }
                                ],
                            }
                        ],
                    }
                ),
                "unsupported button",
            ),
            (
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "cases": [
                            {
                                "name": "late button",
                                "gamePath": "game",
                                "timeoutSeconds": 1,
                                "useIpc": True,
                                "buttonEvents": [
                                    {
                                        "seconds": 1,
                                        "button": "options",
                                        "pressed": True,
                                    }
                                ],
                            }
                        ],
                    }
                ),
                "before timeoutSeconds",
            ),
            (
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "cases": [
                            {
                                "name": "unordered buttons",
                                "gamePath": "game",
                                "timeoutSeconds": 1,
                                "useIpc": True,
                                "buttonEvents": [
                                    {
                                        "seconds": 0.5,
                                        "button": "cross",
                                        "pressed": True,
                                    },
                                    {
                                        "seconds": 0.25,
                                        "button": "cross",
                                        "pressed": False,
                                    },
                                ],
                            }
                        ],
                    }
                ),
                "in increasing time order",
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


class PngComparisonTests(unittest.TestCase):
    def test_screenshot_difference_can_ignore_pixels_outside_a_region(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.png"
            second = root / "second.png"
            first.write_bytes(test_png(2, 1, 2, bytes((0, 255, 0, 0, 8, 8, 8))))
            second.write_bytes(test_png(2, 1, 2, bytes((0, 0, 255, 0, 8, 8, 8))))

            self.assertGreater(_screenshot_difference(first, second), 0)
            self.assertEqual(
                _screenshot_difference(
                    first,
                    second,
                    region=ScreenshotRegion(left=1, top=0, width=1, height=1),
                ),
                0.0,
            )
            with self.assertRaisesRegex(ValueError, "outside the image"):
                _screenshot_difference(
                    first,
                    second,
                    region=ScreenshotRegion(left=2, top=0, width=1, height=1),
                )

    def test_decode_png_supports_all_standard_scanline_filters(self) -> None:
        rows = (
            bytes((0, 10, 20, 30, 40, 50, 60))
            + bytes((1, 20, 30, 40, 40, 40, 40))
            + bytes((2, 10, 10, 10, 10, 10, 10))
            + bytes((3, 25, 30, 35, 25, 25, 25))
            + bytes((4, 10, 10, 10, 10, 10, 10))
        )
        expected = bytes(
            (
                10,
                20,
                30,
                40,
                50,
                60,
                20,
                30,
                40,
                60,
                70,
                80,
                30,
                40,
                50,
                70,
                80,
                90,
                40,
                50,
                60,
                80,
                90,
                100,
                50,
                60,
                70,
                90,
                100,
                110,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "filtered.png"
            path.write_bytes(test_png(2, 5, 2, rows))

            self.assertEqual(_decode_png_rgb(path), (2, 5, expected))

    def test_decode_png_expands_grayscale_and_ignores_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grayscale = root / "grayscale.png"
            grayscale.write_bytes(test_png(1, 1, 4, bytes((0, 42, 7))))
            rgba = root / "rgba.png"
            rgba.write_bytes(test_png(1, 1, 6, bytes((0, 1, 2, 3, 4))))

            self.assertEqual(_decode_png_rgb(grayscale), (1, 1, bytes((42, 42, 42))))
            self.assertEqual(_decode_png_rgb(rgba), (1, 1, bytes((1, 2, 3))))

    def test_screenshot_difference_rejects_different_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.png"
            second = root / "second.png"
            first.write_bytes(test_png(1, 1, 2, bytes((0, 0, 0, 0))))
            second.write_bytes(test_png(2, 1, 2, bytes((0, 0, 0, 0, 0, 0, 0))))

            with self.assertRaisesRegex(ValueError, "different dimensions"):
                _screenshot_difference(first, second)

    def test_cosine_screenshot_difference_detects_moved_dark_detail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.png"
            second = root / "second.png"
            first.write_bytes(
                test_png(4, 1, 2, bytes((0, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
            )
            second.write_bytes(test_png(4, 1, 2, bytes((0, *([0] * 9), 8, 8, 8))))

            self.assertLess(_screenshot_difference(first, second), 0.02)
            self.assertEqual(_screenshot_difference(first, second, mode="cosine"), 1.0)

    def test_cosine_screenshot_difference_handles_black_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            black = root / "black.png"
            detail = root / "detail.png"
            black.write_bytes(test_png(1, 1, 2, bytes((0, 0, 0, 0))))
            detail.write_bytes(test_png(1, 1, 2, bytes((0, 1, 1, 1))))

            self.assertEqual(_screenshot_difference(black, black, mode="cosine"), 0.0)
            self.assertEqual(_screenshot_difference(black, detail, mode="cosine"), 1.0)


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

    def test_run_case_installs_private_config_in_portable_user_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed = root / "checkpoint"
            seed.mkdir()
            (seed / "config.json").write_text(
                json.dumps({"GPU": {"readbackLinearImages": False}}),
                encoding="utf-8",
            )
            config = root / "profile.json"
            expected = {"GPU": {"readbackLinearImages": True}}
            config.write_text(json.dumps(expected), encoding="utf-8")
            manifest = load_manifest(
                self.make_manifest(
                    root,
                    case={
                        "name": "configured boot",
                        "userDataSeed": "checkpoint",
                        "userConfig": "profile.json",
                    },
                )
            )

            result = run_case(
                manifest.cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertTrue(result.passed, result.failures)
            copied = result.artifact_directory / "user" / "config.json"
            self.assertEqual(json.loads(copied.read_text(encoding="utf-8")), expected)
            observation = json.loads(
                (result.artifact_directory / "observation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(observation["user_config"], expected)

    def test_run_case_copies_user_data_seed_without_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed = root / "checkpoint"
            savedata = seed / "home" / "1000" / "savedata" / "CUSA00001"
            savedata.mkdir(parents=True)
            source_save = savedata / "state.bin"
            source_save.write_bytes(b"known checkpoint")
            source_save.chmod(stat.S_IREAD)
            manifest = load_manifest(
                self.make_manifest(
                    root,
                    case={
                        "name": "seeded boot",
                        "userDataSeed": "checkpoint",
                    },
                )
            )

            result = run_case(
                manifest.cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertTrue(result.passed, result.failures)
            copied_save = (
                result.artifact_directory
                / "user"
                / "home"
                / "1000"
                / "savedata"
                / "CUSA00001"
                / "state.bin"
            )
            self.assertEqual(copied_save.read_bytes(), b"known checkpoint")
            self.assertEqual(source_save.read_bytes(), b"known checkpoint")
            self.assertTrue(copied_save.stat().st_mode & stat.S_IWRITE)
            self.assertFalse(source_save.stat().st_mode & stat.S_IWRITE)
            self.assertFalse((seed / "log").exists())
            source_save.chmod(stat.S_IWRITE)

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

    def test_run_case_uses_ipc_to_start_and_stop_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "controlled survival",
                    "timeoutSeconds": 0.2,
                    "args": [
                        "--expect-ipc",
                        "--log",
                        "gracefully stopped",
                    ],
                    "useIpc": True,
                    "allowedOutcomes": ["timed_out"],
                    "requiredLogPatterns": ["gracefully stopped"],
                },
            )
            manifest = load_manifest(manifest_path)

            result = run_case(
                manifest.cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertTrue(result.passed, result.failures)
            observation = json.loads(
                (result.artifact_directory / "observation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(observation["ipc_commands"], ["RUN", "START", "STOP"])
            self.assertEqual(observation["ipc_enabled"], "true")

    def test_run_case_schedules_and_requires_screenshots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "visual survival",
                    "timeoutSeconds": 0.25,
                    "args": ["--expect-ipc"],
                    "useIpc": True,
                    "screenshotSeconds": [0.05, 0.1],
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
            self.assertEqual(len(result.screenshots), 2)
            self.assertTrue(all(path.suffix == ".png" for path in result.screenshots))
            self.assertEqual(
                result.to_report()["screenshots"],
                [str(path) for path in result.screenshots],
            )
            self.assertEqual(len(result.screenshot_hashes), 2)
            self.assertEqual(
                result.to_report()["screenshot_hashes"], result.screenshot_hashes
            )
            observation = json.loads(
                (result.artifact_directory / "observation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                observation["ipc_commands"],
                ["RUN", "START", "SCREENSHOT", "SCREENSHOT", "STOP"],
            )

    def test_run_case_presses_button_only_after_reference_screenshot_matches(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_png = test_png(1, 1, 6, bytes((0, 1, 0, 0, 255)))
            reference = root / "expected.png"
            reference.write_bytes(expected_png)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "state-driven visual survival",
                    "timeoutSeconds": 2.5,
                    "args": ["--expect-ipc", "--vary-screenshots"],
                    "useIpc": True,
                    "screenshotButtonEvents": [
                        {
                            "referenceScreenshot": "expected.png",
                            "maximumDifference": 0,
                            "button": "cross",
                            "timeoutSeconds": 1.5,
                            "pollSeconds": 0.1,
                            "holdSeconds": 0.02,
                        }
                    ],
                    "allowedOutcomes": ["timed_out"],
                },
            )

            result = run_case(
                load_manifest(manifest_path).cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertTrue(result.passed, result.failures)
            observation = json.loads(
                (result.artifact_directory / "observation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                observation["ipc_commands"],
                [
                    "RUN",
                    "START",
                    "SCREENSHOT",
                    "SCREENSHOT",
                    "GAMEPAD_BUTTON",
                    "cross",
                    "1",
                    "GAMEPAD_BUTTON",
                    "cross",
                    "0",
                    "STOP",
                ],
            )
            screenshot_times = [
                seconds
                for command, seconds in zip(
                    observation["ipc_commands"], observation["ipc_command_seconds"]
                )
                if command == "SCREENSHOT"
            ]
            self.assertGreaterEqual(screenshot_times[1] - screenshot_times[0], 0.08)
            self.assertEqual(
                [attempt.matched for attempt in result.visual_checkpoint_attempts],
                [False, True],
            )
            self.assertTrue(
                all(
                    attempt.difference is not None
                    for attempt in result.visual_checkpoint_attempts
                )
            )
            json.dumps(result.to_report())

    def test_run_case_matches_visual_checkpoint_inside_comparison_region(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "expected.png"
            reference.write_bytes(test_png(2, 1, 2, bytes((0, 255, 0, 0, 8, 8, 8))))
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "stable control region",
                    "timeoutSeconds": 1.0,
                    "args": ["--expect-ipc", "--vary-left-pixel"],
                    "useIpc": True,
                    "screenshotButtonEvents": [
                        {
                            "referenceScreenshot": "expected.png",
                            "comparisonRegion": {
                                "left": 1,
                                "top": 0,
                                "width": 1,
                                "height": 1,
                            },
                            "maximumDifference": 0,
                            "button": "cross",
                            "timeoutSeconds": 0.5,
                            "pollSeconds": 0.1,
                            "holdSeconds": 0.02,
                        }
                    ],
                    "allowedOutcomes": ["timed_out"],
                },
            )

            result = run_case(
                load_manifest(manifest_path).cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertTrue(result.passed, result.failures)
            self.assertEqual(len(result.visual_checkpoint_attempts), 1)
            self.assertTrue(result.visual_checkpoint_attempts[0].matched)

    def test_run_case_attributes_screenshot_arriving_after_poll_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_png = test_png(1, 1, 6, bytes((0, 0, 0, 0, 255)))
            reference = root / "expected.png"
            reference.write_bytes(expected_png)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "late visual response",
                    "timeoutSeconds": 0.8,
                    "args": [
                        "--expect-ipc",
                        "--screenshot-delay",
                        "0.08",
                        "--single-screenshot",
                    ],
                    "useIpc": True,
                    "screenshotButtonEvents": [
                        {
                            "referenceScreenshot": "expected.png",
                            "maximumDifference": 0,
                            "button": "cross",
                            "timeoutSeconds": 0.4,
                            "pollSeconds": 0.02,
                            "holdSeconds": 0.01,
                        }
                    ],
                    "allowedOutcomes": ["timed_out"],
                },
            )

            result = run_case(
                load_manifest(manifest_path).cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertTrue(result.passed, result.failures)
            self.assertTrue(result.visual_checkpoint_attempts[-1].matched)

    def test_run_case_fails_without_pressing_when_screenshot_never_matches(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "missing visual state",
                    "timeoutSeconds": 0.8,
                    "args": ["--expect-ipc"],
                    "useIpc": True,
                    "renderdocCaptureOnVisualFailure": True,
                    "screenshotButtonEvents": [
                        {
                            "screenshotSha256": "a" * 64,
                            "button": "cross",
                            "timeoutSeconds": 0.4,
                            "pollSeconds": 0.05,
                            "holdSeconds": 0.01,
                        }
                    ],
                    "allowedOutcomes": ["timed_out"],
                },
            )

            result = run_case(
                load_manifest(manifest_path).cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertFalse(result.passed)
            self.assertTrue(
                any(
                    "screenshotButtonEvents[0] did not match" in failure
                    for failure in result.failures
                ),
                result.failures,
            )
            observation = json.loads(
                (result.artifact_directory / "observation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotIn("GAMEPAD_BUTTON", observation["ipc_commands"])
            self.assertIn("RENDERDOC_CAPTURE", observation["ipc_commands"])
            self.assertEqual(len(result.renderdoc_captures), 1)
            self.assertGreaterEqual(len(result.visual_checkpoint_attempts), 1)
            attempt = result.visual_checkpoint_attempts[0]
            self.assertEqual(attempt.event_index, 0)
            self.assertTrue(attempt.screenshot.is_file())
            self.assertEqual(len(attempt.screenshot_sha256), 64)
            self.assertIsNone(attempt.difference)
            self.assertEqual(attempt.mean_intensity, 0.0)
            self.assertEqual(attempt.non_black_fraction, 0.0)
            self.assertFalse(attempt.matched)
            report_attempt = result.to_report()["visual_checkpoint_attempts"][0]
            self.assertEqual(report_attempt["screenshot"], str(attempt.screenshot))
            self.assertNotIn("reference_screenshot", report_attempt)

    def test_run_case_schedules_and_requires_renderdoc_captures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "frame diagnosis",
                    "timeoutSeconds": 0.25,
                    "args": ["--expect-ipc"],
                    "useIpc": True,
                    "renderdocCaptureSeconds": [0.05, 0.1],
                    "allowedOutcomes": ["timed_out"],
                },
            )

            result = run_case(
                load_manifest(manifest_path).cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertTrue(result.passed, result.failures)
            self.assertEqual(len(result.renderdoc_captures), 2)
            self.assertTrue(
                all(path.suffix == ".rdc" for path in result.renderdoc_captures)
            )
            self.assertEqual(
                result.to_report()["renderdoc_captures"],
                [str(path) for path in result.renderdoc_captures],
            )
            self.assertEqual(len(result.renderdoc_capture_hashes), 2)
            observation = json.loads(
                (result.artifact_directory / "observation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                observation["ipc_commands"],
                [
                    "RUN",
                    "START",
                    "RENDERDOC_CAPTURE",
                    "RENDERDOC_CAPTURE",
                    "STOP",
                ],
            )

    def test_run_case_starts_action_timeline_after_ipc_handshake(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "delayed IPC startup",
                    "timeoutSeconds": 0.35,
                    "args": [
                        "--expect-ipc",
                        "--ipc-handshake-delay",
                        "0.1",
                    ],
                    "useIpc": True,
                    "screenshotSeconds": [0.05],
                    "allowedOutcomes": ["timed_out"],
                },
            )

            result = run_case(
                load_manifest(manifest_path).cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertTrue(result.passed, result.failures)
            observation = json.loads(
                (result.artifact_directory / "observation.json").read_text(
                    encoding="utf-8"
                )
            )
            commands = observation["ipc_commands"]
            command_seconds = observation["ipc_command_seconds"]
            screenshot = commands.index("SCREENSHOT")
            start = commands.index("START")
            self.assertGreaterEqual(
                command_seconds[screenshot] - command_seconds[start], 0.04
            )

    def test_run_case_fails_when_ipc_handshake_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "missing IPC handshake",
                    "timeoutSeconds": 0.15,
                    "args": ["--expect-ipc", "--omit-ipc-handshake"],
                    "useIpc": True,
                    "allowedOutcomes": ["timed_out"],
                },
            )

            result = run_case(
                load_manifest(manifest_path).cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertFalse(result.passed)
            self.assertIn("IPC handshake was not observed", result.failures)

    def test_run_case_requires_advertised_ipc_capabilities(self) -> None:
        scenarios = (
            ("control", "--omit-control-capability", {}, "ENABLE_EMU_CONTROL"),
            (
                "screenshot",
                "--omit-screenshot-capability",
                {"screenshotSeconds": [0.05]},
                "ENABLE_SCREENSHOT",
            ),
            (
                "renderdoc",
                "--omit-renderdoc-capability",
                {"renderdocCaptureSeconds": [0.05]},
                "ENABLE_RENDERDOC_CAPTURE",
            ),
            (
                "visual failure renderdoc",
                "--omit-renderdoc-capability",
                {
                    "renderdocCaptureOnVisualFailure": True,
                    "screenshotButtonEvents": [
                        {
                            "screenshotSha256": "a" * 64,
                            "button": "cross",
                            "timeoutSeconds": 0.4,
                        }
                    ],
                },
                "ENABLE_RENDERDOC_CAPTURE",
            ),
            (
                "button",
                "--omit-gamepad-capability",
                {
                    "buttonEvents": [
                        {"seconds": 0.05, "button": "cross", "pressed": True}
                    ]
                },
                "ENABLE_GAMEPAD",
            ),
            (
                "axis",
                "--omit-gamepad-capability",
                {"axisEvents": [{"seconds": 0.05, "axis": "left_x", "value": 255}]},
                "ENABLE_GAMEPAD",
            ),
        )
        for action, omitted_flag, actions, capability in scenarios:
            with (
                self.subTest(action=action),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                manifest_path = self.make_manifest(
                    root,
                    case={
                        "name": f"missing {capability}",
                        "timeoutSeconds": 1.0,
                        "args": ["--expect-ipc", omitted_flag],
                        "useIpc": True,
                        "allowedOutcomes": ["timed_out"],
                        **actions,
                    },
                )

                result = run_case(
                    load_manifest(manifest_path).cases[0],
                    emulator_command=[sys.executable, str(FIXTURE)],
                    artifacts_root=root / "artifacts",
                )

                self.assertFalse(result.passed)
                self.assertIn(
                    f"IPC capability {capability} was not advertised",
                    result.failures,
                )

    def test_run_case_interleaves_button_events_and_screenshots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "automated visual survival",
                    "timeoutSeconds": 0.3,
                    "args": ["--expect-ipc"],
                    "useIpc": True,
                    "buttonEvents": [
                        {"seconds": 0.05, "button": "cross", "pressed": True},
                        {"seconds": 0.15, "button": "cross", "pressed": False},
                    ],
                    "screenshotSeconds": [0.1, 0.2],
                    "allowedOutcomes": ["timed_out"],
                },
            )

            result = run_case(
                load_manifest(manifest_path).cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertTrue(result.passed, result.failures)
            observation = json.loads(
                (result.artifact_directory / "observation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                observation["ipc_commands"],
                [
                    "RUN",
                    "START",
                    "GAMEPAD_BUTTON",
                    "cross",
                    "1",
                    "SCREENSHOT",
                    "GAMEPAD_BUTTON",
                    "cross",
                    "0",
                    "SCREENSHOT",
                    "STOP",
                ],
            )

    def test_run_case_interleaves_axis_events_with_other_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "analog visual survival",
                    "timeoutSeconds": 0.35,
                    "args": ["--expect-ipc"],
                    "useIpc": True,
                    "axisEvents": [
                        {"seconds": 0.05, "axis": "left_y", "value": 255},
                        {"seconds": 0.2, "axis": "left_y", "value": 128},
                    ],
                    "buttonEvents": [
                        {"seconds": 0.15, "button": "cross", "pressed": True}
                    ],
                    "screenshotSeconds": [0.1, 0.25],
                    "allowedOutcomes": ["timed_out"],
                },
            )

            result = run_case(
                load_manifest(manifest_path).cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertTrue(result.passed, result.failures)
            observation = json.loads(
                (result.artifact_directory / "observation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                observation["ipc_commands"],
                [
                    "RUN",
                    "START",
                    "GAMEPAD_AXIS",
                    "left_y",
                    "255",
                    "SCREENSHOT",
                    "GAMEPAD_BUTTON",
                    "cross",
                    "1",
                    "GAMEPAD_AXIS",
                    "left_y",
                    "128",
                    "SCREENSHOT",
                    "STOP",
                ],
            )

    def test_run_case_interleaves_touch_events_with_visual_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "touch visual survival",
                    "timeoutSeconds": 0.3,
                    "args": ["--expect-ipc"],
                    "useIpc": True,
                    "touchEvents": [
                        {
                            "seconds": 0.05,
                            "finger": 0,
                            "down": True,
                            "x": 960,
                            "y": 470,
                        },
                        {
                            "seconds": 0.2,
                            "finger": 0,
                            "down": False,
                            "x": 960,
                            "y": 470,
                        },
                    ],
                    "screenshotSeconds": [0.1, 0.25],
                    "allowedOutcomes": ["timed_out"],
                },
            )

            result = run_case(
                load_manifest(manifest_path).cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertTrue(result.passed, result.failures)
            observation = json.loads(
                (result.artifact_directory / "observation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                observation["ipc_commands"],
                [
                    "RUN",
                    "START",
                    "GAMEPAD_TOUCH",
                    "0",
                    "1",
                    "960",
                    "470",
                    "SCREENSHOT",
                    "GAMEPAD_TOUCH",
                    "0",
                    "0",
                    "960",
                    "470",
                    "SCREENSHOT",
                    "STOP",
                ],
            )

    def test_run_case_fails_when_requested_screenshot_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "missing visual",
                    "timeoutSeconds": 2.0,
                    "args": ["--expect-ipc", "--ignore-screenshots"],
                    "useIpc": True,
                    "screenshotSeconds": [0.2],
                    "allowedOutcomes": ["timed_out"],
                },
            )

            result = run_case(
                load_manifest(manifest_path).cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertFalse(result.passed)
            self.assertTrue(
                any(
                    "captured 0 valid screenshots" in failure
                    for failure in result.failures
                )
            )

    def test_run_case_fails_when_requested_renderdoc_capture_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "missing frame capture",
                    "timeoutSeconds": 1.0,
                    "args": ["--expect-ipc", "--ignore-renderdoc-captures"],
                    "useIpc": True,
                    "renderdocCaptureSeconds": [0.05],
                    "allowedOutcomes": ["timed_out"],
                },
            )

            result = run_case(
                load_manifest(manifest_path).cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertFalse(result.passed)
            self.assertIn(
                "captured 0 valid RenderDoc frames; expected 1", result.failures
            )

    def test_run_case_requires_visual_failure_renderdoc_capture_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "missing visual failure capture",
                    "timeoutSeconds": 0.8,
                    "args": ["--expect-ipc", "--ignore-renderdoc-captures"],
                    "useIpc": True,
                    "renderdocCaptureOnVisualFailure": True,
                    "screenshotButtonEvents": [
                        {
                            "screenshotSha256": "a" * 64,
                            "button": "cross",
                            "timeoutSeconds": 0.4,
                            "pollSeconds": 0.05,
                        }
                    ],
                    "allowedOutcomes": ["timed_out"],
                },
            )

            result = run_case(
                load_manifest(manifest_path).cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertFalse(result.passed)
            self.assertIn(
                "captured 0 valid RenderDoc frames; expected 1", result.failures
            )

    def test_run_case_rejects_malformed_screenshot_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "malformed visual",
                    "timeoutSeconds": 2.0,
                    "args": ["--expect-ipc", "--malformed-screenshots"],
                    "useIpc": True,
                    "screenshotSeconds": [0.2],
                    "allowedOutcomes": ["timed_out"],
                },
            )

            result = run_case(
                load_manifest(manifest_path).cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertTrue(
                list((result.artifact_directory / "user" / "screenshots").glob("*.png"))
            )
            self.assertEqual(result.screenshots, [])
            self.assertIn("captured 0 valid screenshots; expected 1", result.failures)

    def test_run_case_rejects_identical_screenshots_when_progress_is_required(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "stuck visual",
                    "timeoutSeconds": 0.8,
                    "args": ["--expect-ipc"],
                    "useIpc": True,
                    "screenshotSeconds": [0.15, 0.35],
                    "minimumDistinctScreenshots": 2,
                    "allowedOutcomes": ["timed_out"],
                },
            )

            result = run_case(
                load_manifest(manifest_path).cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertFalse(result.passed)
            self.assertEqual(len(set(result.screenshot_hashes)), 1)
            self.assertIn(
                "captured 1 distinct screenshots; expected at least 2",
                result.failures,
            )

    def test_run_case_accepts_required_visual_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "progressing visual",
                    "timeoutSeconds": 0.8,
                    "args": ["--expect-ipc", "--vary-screenshots"],
                    "useIpc": True,
                    "screenshotSeconds": [0.15, 0.35],
                    "minimumDistinctScreenshots": 2,
                    "allowedOutcomes": ["timed_out"],
                },
            )

            result = run_case(
                load_manifest(manifest_path).cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertTrue(result.passed, result.failures)
            self.assertEqual(len(set(result.screenshot_hashes)), 2)

    def test_run_case_accepts_stable_then_changed_screenshot_relationships(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "causal visual",
                    "timeoutSeconds": 0.25,
                    "args": ["--expect-ipc", "--vary-screenshots-after", "2"],
                    "useIpc": True,
                    "screenshotSeconds": [0.05, 0.1, 0.15],
                    "screenshotComparisons": [
                        {
                            "firstScreenshot": 0,
                            "secondScreenshot": 1,
                            "maximumDifference": 0.0,
                        },
                        {
                            "firstScreenshot": 1,
                            "secondScreenshot": 2,
                            "minimumDifference": 0.001,
                        },
                    ],
                    "allowedOutcomes": ["timed_out"],
                },
            )

            result = run_case(
                load_manifest(manifest_path).cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertTrue(result.passed, result.failures)
            self.assertEqual(len(result.screenshot_differences), 2)
            self.assertEqual(result.screenshot_differences[0], 0.0)
            self.assertGreaterEqual(result.screenshot_differences[1], 0.001)

    def test_run_case_uses_cosine_difference_for_dark_movement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "dark causal visual",
                    "timeoutSeconds": 0.2,
                    "args": ["--expect-ipc", "--dark-movement-screenshots"],
                    "useIpc": True,
                    "screenshotSeconds": [0.05, 0.1],
                    "screenshotComparisons": [
                        {
                            "firstScreenshot": 0,
                            "secondScreenshot": 1,
                            "minimumDifference": 0.9,
                            "differenceMode": "cosine",
                        }
                    ],
                    "allowedOutcomes": ["timed_out"],
                },
            )

            result = run_case(
                load_manifest(manifest_path).cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertTrue(result.passed, result.failures)
            self.assertGreaterEqual(result.screenshot_differences[0], 0.9)

    def test_run_case_rejects_unmet_screenshot_relationship(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_manifest(
                root,
                case={
                    "name": "false visual claim",
                    "timeoutSeconds": 0.2,
                    "args": ["--expect-ipc"],
                    "useIpc": True,
                    "screenshotSeconds": [0.05, 0.1],
                    "screenshotComparisons": [
                        {
                            "firstScreenshot": 0,
                            "secondScreenshot": 1,
                            "minimumDifference": 0.1,
                        }
                    ],
                    "allowedOutcomes": ["timed_out"],
                },
            )

            result = run_case(
                load_manifest(manifest_path).cases[0],
                emulator_command=[sys.executable, str(FIXTURE)],
                artifacts_root=root / "artifacts",
            )

            self.assertFalse(result.passed)
            self.assertEqual(result.screenshot_differences, [0.0])
            self.assertIn(
                "screenshot comparison 0 difference 0.000000 is below minimum 0.100000",
                result.failures,
            )

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
