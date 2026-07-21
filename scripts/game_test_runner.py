# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

"""Run repeatable, isolated shadPS4 game smoke tests from a private manifest."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import stat
import subprocess
import sys
import threading
import time
from typing import Any, BinaryIO, Sequence
import zlib

REPORT_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_LIMIT_BYTES = 64 * 1024 * 1024
VALID_OUTCOMES = frozenset({"exited_zero", "exited_nonzero", "timed_out"})
VALID_SCREENSHOT_DIFFERENCE_MODES = frozenset({"cosine", "mean_absolute"})
VALID_SCREENSHOT_SOURCES = frozenset({"game_frame", "presented_frame"})
SUPPORTED_BUTTONS = frozenset(
    {
        "circle",
        "cross",
        "dpad_down",
        "dpad_left",
        "dpad_right",
        "dpad_up",
        "l1",
        "l3",
        "options",
        "r1",
        "r3",
        "square",
        "touchpad",
        "triangle",
    }
)
SUPPORTED_AXES = frozenset({"left_x", "left_y", "right_x", "right_y", "l2", "r2"})


class ManifestError(ValueError):
    """Raised when a game test manifest is invalid."""


@dataclass(frozen=True)
class ButtonEvent:
    seconds: float
    button: str
    pressed: bool


@dataclass(frozen=True)
class ScreenshotRegion:
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class ScreenshotButtonEvent:
    screenshot_sha256: str | None
    reference_screenshot: Path | None
    maximum_difference: float | None
    difference_mode: str
    comparison_region: ScreenshotRegion | None
    scale_reference_to_capture: bool
    button: str
    timeout_seconds: float
    poll_seconds: float = 0.25
    hold_seconds: float = 0.1


@dataclass(frozen=True)
class VisualCheckpointAttempt:
    event_index: int
    screenshot: Path
    screenshot_sha256: str
    difference: float | None
    mean_intensity: float | None
    non_black_fraction: float | None
    matched: bool


@dataclass(frozen=True)
class AxisEvent:
    seconds: float
    axis: str
    value: int


@dataclass(frozen=True)
class TouchEvent:
    seconds: float
    finger: int
    down: bool
    x: int
    y: int


@dataclass(frozen=True)
class ScreenshotComparison:
    first_screenshot: int
    second_screenshot: int
    minimum_difference: float | None = None
    maximum_difference: float | None = None
    difference_mode: str = "mean_absolute"


@dataclass(frozen=True)
class GameCase:
    name: str
    game_path: Path
    timeout_seconds: float
    user_config: Path | None = None
    user_data_seed: Path | None = None
    use_ipc: bool = False
    screenshot_source: str = "game_frame"
    screenshot_seconds: tuple[float, ...] = ()
    renderdoc_capture_seconds: tuple[float, ...] = ()
    minimum_distinct_screenshots: int = 0
    screenshot_comparisons: tuple[ScreenshotComparison, ...] = ()
    renderdoc_capture_on_visual_failure: bool = False
    button_events: tuple[ButtonEvent, ...] = ()
    screenshot_button_events: tuple[ScreenshotButtonEvent, ...] = ()
    axis_events: tuple[AxisEvent, ...] = ()
    touch_events: tuple[TouchEvent, ...] = ()
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
    screenshots: list[Path]
    screenshot_hashes: list[str]
    screenshot_differences: list[float]
    visual_checkpoint_attempts: list[VisualCheckpointAttempt]
    renderdoc_captures: list[Path]
    renderdoc_capture_hashes: list[str]
    failures: list[str]

    def to_report(self) -> dict[str, Any]:
        report = asdict(self)
        report["artifact_directory"] = str(self.artifact_directory)
        report["screenshots"] = [str(path) for path in self.screenshots]
        report["visual_checkpoint_attempts"] = [
            {**asdict(attempt), "screenshot": str(attempt.screenshot)}
            for attempt in self.visual_checkpoint_attempts
        ]
        report["renderdoc_captures"] = [str(path) for path in self.renderdoc_captures]
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


def _require_ipc_schedule(
    raw: Any, *, field: str, case_name: str, timeout: float, use_ipc: bool
) -> tuple[float, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or any(
        not isinstance(item, (int, float)) or isinstance(item, bool) for item in raw
    ):
        raise ManifestError(f"{case_name}: {field} must be an array of numbers")
    schedule = tuple(float(item) for item in raw)
    if any(not math.isfinite(item) for item in schedule):
        raise ManifestError(f"{case_name}: {field} entries must be finite numbers")
    if any(item <= 0 or item >= timeout for item in schedule):
        raise ManifestError(
            f"{case_name}: {field} entries must be positive and before "
            "timeoutSeconds"
        )
    if tuple(sorted(set(schedule))) != schedule:
        raise ManifestError(f"{case_name}: {field} must be unique and increasing")
    if schedule and not use_ipc:
        raise ManifestError(f"{case_name}: {field} requires useIpc")
    return schedule


def _require_screenshot_schedule(
    raw: Any, *, case_name: str, timeout: float, use_ipc: bool
) -> tuple[float, ...]:
    return _require_ipc_schedule(
        raw,
        field="screenshotSeconds",
        case_name=case_name,
        timeout=timeout,
        use_ipc=use_ipc,
    )


def _require_renderdoc_capture_schedule(
    raw: Any, *, case_name: str, timeout: float, use_ipc: bool
) -> tuple[float, ...]:
    return _require_ipc_schedule(
        raw,
        field="renderdocCaptureSeconds",
        case_name=case_name,
        timeout=timeout,
        use_ipc=use_ipc,
    )


def _require_minimum_distinct_screenshots(
    raw: Any, *, case_name: str, screenshot_count: int
) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ManifestError(
            f"{case_name}: minimumDistinctScreenshots must be an integer"
        )
    if raw < 0:
        raise ManifestError(
            f"{case_name}: minimumDistinctScreenshots cannot be negative"
        )
    if raw > screenshot_count:
        raise ManifestError(
            f"{case_name}: minimumDistinctScreenshots cannot exceed screenshotSeconds"
        )
    return raw


def _require_screenshot_comparisons(
    raw: Any, *, case_name: str, screenshot_count: int
) -> tuple[ScreenshotComparison, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ManifestError(f"{case_name}: screenshotComparisons must be an array")

    comparisons: list[ScreenshotComparison] = []
    for index, item in enumerate(raw):
        field = f"{case_name}: screenshotComparisons[{index}]"
        if not isinstance(item, dict):
            raise ManifestError(f"{field} must be an object")

        screenshot_indexes: list[int] = []
        for json_field in ("firstScreenshot", "secondScreenshot"):
            value = item.get(json_field)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or value >= screenshot_count
            ):
                raise ManifestError(
                    f"{field}.{json_field} must index screenshotSeconds"
                )
            screenshot_indexes.append(value)
        if screenshot_indexes[0] == screenshot_indexes[1]:
            raise ManifestError(f"{field} must compare two different screenshots")

        bounds: list[float | None] = []
        for json_field in ("minimumDifference", "maximumDifference"):
            value = item.get(json_field)
            if value is None:
                bounds.append(None)
                continue
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
                or value > 1
            ):
                raise ManifestError(f"{field}.{json_field} must be between 0 and 1")
            bounds.append(float(value))
        minimum, maximum = bounds
        if minimum is None and maximum is None:
            raise ManifestError(
                f"{field} must specify minimumDifference or maximumDifference"
            )
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ManifestError(
                f"{field}.minimumDifference cannot exceed maximumDifference"
            )
        difference_mode = item.get("differenceMode", "mean_absolute")
        if (
            not isinstance(difference_mode, str)
            or difference_mode not in VALID_SCREENSHOT_DIFFERENCE_MODES
        ):
            supported = ", ".join(sorted(VALID_SCREENSHOT_DIFFERENCE_MODES))
            raise ManifestError(f"{field}.differenceMode must be one of: {supported}")
        comparisons.append(
            ScreenshotComparison(
                first_screenshot=screenshot_indexes[0],
                second_screenshot=screenshot_indexes[1],
                minimum_difference=minimum,
                maximum_difference=maximum,
                difference_mode=difference_mode,
            )
        )
    return tuple(comparisons)


def _require_button_events(
    raw: Any, *, case_name: str, timeout: float, use_ipc: bool
) -> tuple[ButtonEvent, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ManifestError(f"{case_name}: buttonEvents must be an array")
    events: list[ButtonEvent] = []
    for index, item in enumerate(raw):
        field = f"{case_name}: buttonEvents[{index}]"
        if not isinstance(item, dict):
            raise ManifestError(f"{field} must be an object")
        seconds = item.get("seconds")
        if (
            not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or not math.isfinite(seconds)
            or seconds <= 0
            or seconds >= timeout
        ):
            raise ManifestError(
                f"{field}.seconds must be a finite positive number before timeoutSeconds"
            )
        button = item.get("button")
        if button not in SUPPORTED_BUTTONS:
            raise ManifestError(
                f"{field} has unsupported button {button!r}; expected one of "
                f"{sorted(SUPPORTED_BUTTONS)}"
            )
        pressed = item.get("pressed")
        if not isinstance(pressed, bool):
            raise ManifestError(f"{field}.pressed must be a boolean")
        events.append(ButtonEvent(float(seconds), button, pressed))
    if any(
        later.seconds <= earlier.seconds for earlier, later in zip(events, events[1:])
    ):
        raise ManifestError(
            f"{case_name}: buttonEvents must be in increasing time order"
        )
    if events and not use_ipc:
        raise ManifestError(f"{case_name}: buttonEvents requires useIpc")
    return tuple(events)


def _require_screenshot_button_events(
    raw: Any, *, case_name: str, timeout: float, use_ipc: bool, root: Path
) -> tuple[ScreenshotButtonEvent, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ManifestError(f"{case_name}: screenshotButtonEvents must be an array")
    events: list[ScreenshotButtonEvent] = []
    for index, item in enumerate(raw):
        field = f"{case_name}: screenshotButtonEvents[{index}]"
        if not isinstance(item, dict):
            raise ManifestError(f"{field} must be an object")
        screenshot_sha256 = item.get("screenshotSha256")
        reference_value = item.get("referenceScreenshot")
        if (screenshot_sha256 is None) == (reference_value is None):
            raise ManifestError(
                f"{field} requires exactly one of screenshotSha256 or referenceScreenshot"
            )
        reference_screenshot: Path | None = None
        maximum_difference: float | None = None
        difference_mode = item.get("differenceMode", "mean_absolute")
        comparison_region_value = item.get("comparisonRegion")
        comparison_region: ScreenshotRegion | None = None
        scale_reference_to_capture = item.get("scaleReferenceToCapture", False)
        if not isinstance(scale_reference_to_capture, bool):
            raise ManifestError(f"{field}.scaleReferenceToCapture must be a boolean")
        if screenshot_sha256 is not None:
            if scale_reference_to_capture:
                raise ManifestError(
                    f"{field}.scaleReferenceToCapture requires referenceScreenshot"
                )
            if comparison_region_value is not None:
                raise ManifestError(
                    f"{field}.comparisonRegion requires referenceScreenshot"
                )
            if (
                not isinstance(screenshot_sha256, str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", screenshot_sha256) is None
            ):
                raise ManifestError(
                    f"{field}.screenshotSha256 must be a SHA-256 hex digest"
                )
            screenshot_sha256 = screenshot_sha256.lower()
        else:
            reference_screenshot = _resolve_existing_path(
                root, reference_value, f"{field}.referenceScreenshot"
            )
            maximum_difference = item.get("maximumDifference", 0.01)
            if (
                not isinstance(maximum_difference, (int, float))
                or isinstance(maximum_difference, bool)
                or not math.isfinite(maximum_difference)
                or not 0 <= maximum_difference <= 1
            ):
                raise ManifestError(
                    f"{field}.maximumDifference must be between 0 and 1"
                )
            maximum_difference = float(maximum_difference)
            if difference_mode not in VALID_SCREENSHOT_DIFFERENCE_MODES:
                raise ManifestError(
                    f"{field}.differenceMode must be one of "
                    f"{sorted(VALID_SCREENSHOT_DIFFERENCE_MODES)}"
                )
            if comparison_region_value is not None:
                if not isinstance(comparison_region_value, dict):
                    raise ManifestError(f"{field}.comparisonRegion must be an object")
                region_values: dict[str, int] = {}
                for component in ("left", "top", "width", "height"):
                    value = comparison_region_value.get(component)
                    if not isinstance(value, int) or isinstance(value, bool):
                        raise ManifestError(
                            f"{field}.comparisonRegion.{component} must be an integer"
                        )
                    region_values[component] = value
                if region_values["left"] < 0 or region_values["top"] < 0:
                    raise ManifestError(
                        f"{field}.comparisonRegion left and top cannot be negative"
                    )
                if region_values["width"] <= 0 or region_values["height"] <= 0:
                    raise ManifestError(
                        f"{field}.comparisonRegion width and height must be positive"
                    )
                comparison_region = ScreenshotRegion(**region_values)
        button = item.get("button")
        if button not in SUPPORTED_BUTTONS:
            raise ManifestError(
                f"{field} has unsupported button {button!r}; expected one of "
                f"{sorted(SUPPORTED_BUTTONS)}"
            )

        values: dict[str, float] = {}
        for json_field, default in (
            ("timeoutSeconds", None),
            ("pollSeconds", 0.25),
            ("holdSeconds", 0.1),
        ):
            value = item.get(json_field, default)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ManifestError(
                    f"{field}.{json_field} must be a finite positive number"
                )
            values[json_field] = float(value)
        if values["pollSeconds"] > values["timeoutSeconds"]:
            raise ManifestError(f"{field}.pollSeconds cannot exceed timeoutSeconds")
        events.append(
            ScreenshotButtonEvent(
                screenshot_sha256=screenshot_sha256,
                reference_screenshot=reference_screenshot,
                maximum_difference=maximum_difference,
                difference_mode=difference_mode,
                comparison_region=comparison_region,
                scale_reference_to_capture=scale_reference_to_capture,
                button=button,
                timeout_seconds=values["timeoutSeconds"],
                poll_seconds=values["pollSeconds"],
                hold_seconds=values["holdSeconds"],
            )
        )
    if events and not use_ipc:
        raise ManifestError(f"{case_name}: screenshotButtonEvents requires useIpc")
    if sum(event.timeout_seconds + event.hold_seconds for event in events) >= timeout:
        raise ManifestError(
            f"{case_name}: screenshotButtonEvents must complete before timeoutSeconds"
        )
    return tuple(events)


def _require_axis_events(
    raw: Any, *, case_name: str, timeout: float, use_ipc: bool
) -> tuple[AxisEvent, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ManifestError(f"{case_name}: axisEvents must be an array")
    events: list[AxisEvent] = []
    for index, item in enumerate(raw):
        field = f"{case_name}: axisEvents[{index}]"
        if not isinstance(item, dict):
            raise ManifestError(f"{field} must be an object")
        seconds = item.get("seconds")
        if (
            not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or not math.isfinite(seconds)
            or seconds <= 0
            or seconds >= timeout
        ):
            raise ManifestError(
                f"{field}.seconds must be a finite positive number before timeoutSeconds"
            )
        axis = item.get("axis")
        if axis not in SUPPORTED_AXES:
            raise ManifestError(
                f"{field} has unsupported axis {axis!r}; expected one of "
                f"{sorted(SUPPORTED_AXES)}"
            )
        value = item.get("value")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ManifestError(f"{field}.value must be an integer")
        if value < 0 or value > 255:
            raise ManifestError(f"{field}.value must be between 0 and 255")
        events.append(AxisEvent(float(seconds), axis, value))
    if any(
        later.seconds <= earlier.seconds for earlier, later in zip(events, events[1:])
    ):
        raise ManifestError(f"{case_name}: axisEvents must be in increasing time order")
    if events and not use_ipc:
        raise ManifestError(f"{case_name}: axisEvents requires useIpc")
    return tuple(events)


def _require_touch_events(
    raw: Any, *, case_name: str, timeout: float, use_ipc: bool
) -> tuple[TouchEvent, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ManifestError(f"{case_name}: touchEvents must be an array")
    events: list[TouchEvent] = []
    for index, item in enumerate(raw):
        field = f"{case_name}: touchEvents[{index}]"
        if not isinstance(item, dict):
            raise ManifestError(f"{field} must be an object")
        seconds = item.get("seconds")
        if (
            not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or not math.isfinite(seconds)
            or seconds <= 0
            or seconds >= timeout
        ):
            raise ManifestError(
                f"{field}.seconds must be a finite positive number before timeoutSeconds"
            )
        finger = item.get("finger")
        if not isinstance(finger, int) or isinstance(finger, bool):
            raise ManifestError(f"{field}.finger must be an integer")
        if finger not in (0, 1):
            raise ManifestError(f"{field}.finger must be 0 or 1")
        down = item.get("down")
        if not isinstance(down, bool):
            raise ManifestError(f"{field}.down must be a boolean")
        coordinates: list[int] = []
        for name, maximum in (("x", 1919), ("y", 941)):
            value = item.get(name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ManifestError(f"{field}.{name} must be an integer")
            if value < 0 or value > maximum:
                raise ManifestError(f"{field}.{name} must be between 0 and {maximum}")
            coordinates.append(value)
        events.append(
            TouchEvent(float(seconds), finger, down, coordinates[0], coordinates[1])
        )
    if any(
        later.seconds <= earlier.seconds for earlier, later in zip(events, events[1:])
    ):
        raise ManifestError(
            f"{case_name}: touchEvents must be in increasing time order"
        )
    if events and not use_ipc:
        raise ManifestError(f"{case_name}: touchEvents requires useIpc")
    return tuple(events)


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


def _resolve_existing_file(root: Path, raw: Any, field: str) -> Path:
    path = _resolve_existing_path(root, raw, field)
    if not path.is_file():
        raise ManifestError(f"{field} must be a file: {path}")
    return path


def _resolve_existing_directory(root: Path, raw: Any, field: str) -> Path:
    path = _resolve_existing_path(root, raw, field)
    if not path.is_dir():
        raise ManifestError(f"{field} must be a directory: {path}")
    return path


def _resolve_user_config(root: Path, raw: Any, field: str) -> Path:
    path = _resolve_existing_file(root, raw, field)
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"{field} must contain a JSON object: {error}") from error
    if not isinstance(content, dict):
        raise ManifestError(f"{field} must contain a JSON object")
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
        user_config = (
            _resolve_user_config(root, raw_case["userConfig"], f"{name}: userConfig")
            if "userConfig" in raw_case
            else None
        )
        user_data_seed = (
            _resolve_existing_directory(
                root, raw_case["userDataSeed"], f"{name}: userDataSeed"
            )
            if "userDataSeed" in raw_case
            else None
        )
        if user_config is not None and "--config-clean" in args:
            raise ManifestError(
                f"{name}: userConfig cannot be combined with --config-clean"
            )
        outcomes = _require_string_list(
            raw_case.get("allowedOutcomes", ["exited_zero"]),
            "allowedOutcomes",
            name,
        )
        if not outcomes or any(outcome not in VALID_OUTCOMES for outcome in outcomes):
            raise ManifestError(
                f"{name}: allowedOutcomes must contain only {sorted(VALID_OUTCOMES)}"
            )

        use_ipc = _require_bool(raw_case.get("useIpc", False), "useIpc", name)
        screenshot_source = raw_case.get("screenshotSource", "game_frame")
        if screenshot_source not in VALID_SCREENSHOT_SOURCES:
            raise ManifestError(
                f"{name}: screenshotSource must be one of "
                f"{sorted(VALID_SCREENSHOT_SOURCES)}"
            )
        screenshot_seconds = _require_screenshot_schedule(
            raw_case.get("screenshotSeconds"),
            case_name=name,
            timeout=float(timeout),
            use_ipc=use_ipc,
        )
        renderdoc_capture_seconds = _require_renderdoc_capture_schedule(
            raw_case.get("renderdocCaptureSeconds"),
            case_name=name,
            timeout=float(timeout),
            use_ipc=use_ipc,
        )
        button_events = _require_button_events(
            raw_case.get("buttonEvents"),
            case_name=name,
            timeout=float(timeout),
            use_ipc=use_ipc,
        )
        screenshot_button_events = _require_screenshot_button_events(
            raw_case.get("screenshotButtonEvents"),
            case_name=name,
            timeout=float(timeout),
            use_ipc=use_ipc,
            root=root,
        )
        renderdoc_capture_on_visual_failure = _require_bool(
            raw_case.get("renderdocCaptureOnVisualFailure", False),
            "renderdocCaptureOnVisualFailure",
            name,
        )
        if renderdoc_capture_on_visual_failure and not screenshot_button_events:
            raise ManifestError(
                f"{name}: renderdocCaptureOnVisualFailure requires "
                "screenshotButtonEvents"
            )
        axis_events = _require_axis_events(
            raw_case.get("axisEvents"),
            case_name=name,
            timeout=float(timeout),
            use_ipc=use_ipc,
        )
        touch_events = _require_touch_events(
            raw_case.get("touchEvents"),
            case_name=name,
            timeout=float(timeout),
            use_ipc=use_ipc,
        )
        if screenshot_button_events and any(
            (
                screenshot_seconds,
                renderdoc_capture_seconds,
                button_events,
                axis_events,
                touch_events,
            )
        ):
            raise ManifestError(
                f"{name}: screenshotButtonEvents cannot be combined with timed events"
            )
        cases.append(
            GameCase(
                name=name,
                game_path=_resolve_existing_path(
                    root, raw_case.get("gamePath"), f"{name}: gamePath"
                ),
                timeout_seconds=float(timeout),
                user_config=user_config,
                user_data_seed=user_data_seed,
                use_ipc=use_ipc,
                screenshot_source=screenshot_source,
                screenshot_seconds=screenshot_seconds,
                renderdoc_capture_seconds=renderdoc_capture_seconds,
                minimum_distinct_screenshots=_require_minimum_distinct_screenshots(
                    raw_case.get("minimumDistinctScreenshots", 0),
                    case_name=name,
                    screenshot_count=len(screenshot_seconds),
                ),
                screenshot_comparisons=_require_screenshot_comparisons(
                    raw_case.get("screenshotComparisons"),
                    case_name=name,
                    screenshot_count=len(screenshot_seconds),
                ),
                renderdoc_capture_on_visual_failure=(
                    renderdoc_capture_on_visual_failure
                ),
                button_events=button_events,
                screenshot_button_events=screenshot_button_events,
                axis_events=axis_events,
                touch_events=touch_events,
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
    observed_patterns: tuple[tuple[bytes, threading.Event], ...] = (),
) -> None:
    written = 0
    search_tail = b""
    longest_pattern = max((len(pattern) for pattern, _ in observed_patterns), default=0)
    read_chunk = getattr(stream, "read1", stream.read)
    with destination.open("wb") as output:
        while chunk := read_chunk(64 * 1024):
            if observed_patterns:
                searchable = search_tail + chunk
                for pattern, observed in observed_patterns:
                    if pattern in searchable:
                        observed.set()
                search_tail = searchable[-(longest_pattern - 1) :]
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


def _close_process_streams(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is None:
            continue
        try:
            stream.close()
        except OSError:
            # Windows pipes may already be invalid after an abrupt child exit.
            pass


def _read_capped(path: Path, limit: int) -> tuple[str, bool]:
    if not path.exists():
        return "", False
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    return content[:limit].decode("utf-8", errors="replace"), len(content) > limit


def _is_valid_png(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                return False
            saw_ihdr = False
            saw_idat = False
            while True:
                length_bytes = stream.read(4)
                if len(length_bytes) != 4:
                    return False
                length = int.from_bytes(length_bytes, "big")
                chunk_type = stream.read(4)
                chunk_data = stream.read(length)
                checksum = stream.read(4)
                if (
                    len(chunk_type) != 4
                    or len(chunk_data) != length
                    or len(checksum) != 4
                    or zlib.crc32(chunk_type + chunk_data)
                    != int.from_bytes(checksum, "big")
                ):
                    return False
                if not saw_ihdr:
                    if (
                        chunk_type != b"IHDR"
                        or length != 13
                        or int.from_bytes(chunk_data[:4], "big") == 0
                        or int.from_bytes(chunk_data[4:8], "big") == 0
                    ):
                        return False
                    saw_ihdr = True
                elif chunk_type == b"IDAT":
                    saw_idat = True
                elif chunk_type == b"IEND":
                    return length == 0 and saw_idat and stream.read(1) == b""
    except OSError:
        return False


def _find_valid_screenshots(artifact_directory: Path) -> list[Path]:
    screenshots = sorted((artifact_directory / "user" / "screenshots").glob("*.png"))
    return [screenshot for screenshot in screenshots if _is_valid_png(screenshot)]


def _find_renderdoc_captures(artifact_directory: Path) -> list[Path]:
    captures = sorted((artifact_directory / "user" / "captures").glob("*.rdc"))
    return [
        capture for capture in captures if capture.is_file() and capture.stat().st_size
    ]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _paeth_predictor(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def _decode_png_rgb(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG image")

    position = 8
    header: bytes | None = None
    compressed = bytearray()
    while position < len(data):
        if position + 12 > len(data):
            raise ValueError("truncated PNG chunk")
        length = int.from_bytes(data[position : position + 4], "big")
        chunk_type = data[position + 4 : position + 8]
        end = position + 12 + length
        if end > len(data):
            raise ValueError("truncated PNG chunk")
        chunk_data = data[position + 8 : position + 8 + length]
        if chunk_type == b"IHDR":
            header = chunk_data
        elif chunk_type == b"IDAT":
            compressed.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
        position = end

    if header is None or len(header) != 13 or not compressed:
        raise ValueError("PNG is missing required image data")
    width = int.from_bytes(header[:4], "big")
    height = int.from_bytes(header[4:8], "big")
    bit_depth, color_type, compression, filtering, interlace = header[8:13]
    channels_by_color_type = {0: 1, 2: 3, 4: 2, 6: 4}
    channels = channels_by_color_type.get(color_type)
    if (
        bit_depth != 8
        or channels is None
        or compression != 0
        or filtering != 0
        or interlace != 0
    ):
        raise ValueError("PNG must be non-interlaced 8-bit grayscale or RGB")

    row_bytes = width * channels
    try:
        encoded = zlib.decompress(compressed)
    except zlib.error as error:
        raise ValueError(f"invalid compressed PNG data: {error}") from error
    if len(encoded) != height * (row_bytes + 1):
        raise ValueError("PNG scanline data has an unexpected size")

    decoded = bytearray(height * row_bytes)
    encoded_position = 0
    for row in range(height):
        filter_type = encoded[encoded_position]
        encoded_position += 1
        if filter_type > 4:
            raise ValueError(f"unsupported PNG filter {filter_type}")
        row_start = row * row_bytes
        for column in range(row_bytes):
            value = encoded[encoded_position]
            encoded_position += 1
            left = decoded[row_start + column - channels] if column >= channels else 0
            above = decoded[row_start - row_bytes + column] if row else 0
            upper_left = (
                decoded[row_start - row_bytes + column - channels]
                if row and column >= channels
                else 0
            )
            if filter_type == 1:
                value += left
            elif filter_type == 2:
                value += above
            elif filter_type == 3:
                value += (left + above) // 2
            elif filter_type == 4:
                value += _paeth_predictor(left, above, upper_left)
            decoded[row_start + column] = value & 0xFF

    if color_type == 2:
        return width, height, bytes(decoded)
    rgb = bytearray(width * height * 3)
    output = 0
    for source in range(0, len(decoded), channels):
        if color_type in (0, 4):
            rgb[output : output + 3] = bytes((decoded[source],)) * 3
        else:
            rgb[output : output + 3] = decoded[source : source + 3]
        output += 3
    return width, height, bytes(rgb)


def _screenshot_difference(
    first: Path,
    second: Path,
    *,
    mode: str = "mean_absolute",
    region: ScreenshotRegion | None = None,
    scale_first_to_second: bool = False,
) -> float:
    second_width, second_height, second_pixels = _decode_png_rgb(second)
    if scale_first_to_second:
        reference_stat = first.stat()
        first_width, first_height, first_pixels = _scaled_reference_rgb(
            str(first.resolve()),
            reference_stat.st_mtime_ns,
            reference_stat.st_size,
            second_width,
            second_height,
        )
    else:
        first_width, first_height, first_pixels = _decode_png_rgb(first)
        if (first_width, first_height) != (second_width, second_height):
            raise ValueError("screenshots have different dimensions")
    if region is not None:
        if (
            region.left < 0
            or region.top < 0
            or region.width <= 0
            or region.height <= 0
            or region.left + region.width > first_width
            or region.top + region.height > first_height
        ):
            raise ValueError("screenshot comparison region is outside the image")

        def crop(pixels: bytes) -> bytes:
            row_bytes = region.width * 3
            return b"".join(
                pixels[
                    ((region.top + row) * first_width + region.left)
                    * 3 : ((region.top + row) * first_width + region.left)
                    * 3
                    + row_bytes
                ]
                for row in range(region.height)
            )

        first_pixels = crop(first_pixels)
        second_pixels = crop(second_pixels)
    if mode == "mean_absolute":
        return sum(
            abs(left - right) for left, right in zip(first_pixels, second_pixels)
        ) / (len(first_pixels) * 255)
    if mode != "cosine":
        raise ValueError(f"unsupported screenshot difference mode: {mode}")

    dot_product = sum(left * right for left, right in zip(first_pixels, second_pixels))
    first_squared = sum(value * value for value in first_pixels)
    second_squared = sum(value * value for value in second_pixels)
    if first_squared == 0 and second_squared == 0:
        return 0.0
    if first_squared == 0 or second_squared == 0:
        return 1.0
    similarity = dot_product / math.sqrt(first_squared * second_squared)
    return max(0.0, min(1.0, 1.0 - similarity))


@lru_cache(maxsize=16)
def _scaled_reference_rgb(
    path: str,
    modified_nanoseconds: int,
    file_size: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int, bytes]:
    del modified_nanoseconds, file_size
    source_width, source_height, pixels = _decode_png_rgb(Path(path))
    if (source_width, source_height) == (target_width, target_height):
        return source_width, source_height, pixels
    if source_width * target_height != target_width * source_height:
        raise ValueError("screenshots have different aspect ratios")
    return (
        target_width,
        target_height,
        _resize_rgb_linear(
            pixels,
            source_width,
            source_height,
            target_width,
            target_height,
        ),
    )


def _resize_rgb_linear(
    pixels: bytes,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> bytes:
    if (source_width, source_height) == (target_width, target_height):
        return pixels
    x_samples: list[tuple[int, int, float]] = []
    for target_x in range(target_width):
        source_x = (target_x + 0.5) * source_width / target_width - 0.5
        first_x = max(0, min(source_width - 1, math.floor(source_x)))
        second_x = min(source_width - 1, first_x + 1)
        x_samples.append((first_x * 3, second_x * 3, source_x - first_x))
    resized = bytearray(target_width * target_height * 3)
    output = 0
    for target_y in range(target_height):
        source_y = (target_y + 0.5) * source_height / target_height - 0.5
        first_y = max(0, min(source_height - 1, math.floor(source_y)))
        second_y = min(source_height - 1, first_y + 1)
        y_weight = source_y - first_y
        first_row = first_y * source_width * 3
        second_row = second_y * source_width * 3
        for first_x, second_x, x_weight in x_samples:
            for channel in range(3):
                top = pixels[first_row + first_x + channel] * (1 - x_weight) + pixels[
                    first_row + second_x + channel
                ] * x_weight
                bottom = pixels[
                    second_row + first_x + channel
                ] * (1 - x_weight) + pixels[
                    second_row + second_x + channel
                ] * x_weight
                resized[output] = round(top * (1 - y_weight) + bottom * y_weight)
                output += 1
    return bytes(resized)


def _screenshot_statistics(path: Path) -> tuple[float, float]:
    _, _, pixels = _decode_png_rgb(path)
    pixel_count = len(pixels) // 3
    mean_intensity = sum(pixels) / (len(pixels) * 255)
    non_black_pixels = sum(
        any(pixels[offset : offset + 3]) for offset in range(0, len(pixels), 3)
    )
    return mean_intensity, non_black_pixels / pixel_count


def _make_tree_owner_writable(root: Path) -> None:
    root.chmod(root.stat().st_mode | stat.S_IWUSR)
    for path in root.rglob("*"):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)


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
    user_directory = artifact_directory / "user"
    if case.user_data_seed is None:
        user_directory.mkdir()
    else:
        shutil.copytree(case.user_data_seed, user_directory)
        _make_tree_owner_writable(user_directory)
    if case.user_config is not None:
        shutil.copyfile(case.user_config, user_directory / "config.json")

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
            screenshots=[],
            screenshot_hashes=[],
            screenshot_differences=[],
            visual_checkpoint_attempts=[],
            renderdoc_captures=[],
            renderdoc_capture_hashes=[],
            failures=[failure],
        )
    assert process.stdout is not None
    assert process.stderr is not None
    truncated = threading.Event()
    ipc_handshake = threading.Event()
    ipc_capabilities = {
        capability: threading.Event()
        for capability in (
            "ENABLE_EMU_CONTROL",
            "ENABLE_GAMEPAD",
            "ENABLE_RENDERDOC_CAPTURE",
            "ENABLE_SCREENSHOT",
        )
    }
    ipc_observers = (
        *(
            (f";{capability}".encode("ascii"), observed)
            for capability, observed in ipc_capabilities.items()
        ),
        (b";#IPC_END", ipc_handshake),
    )
    readers = [
        threading.Thread(
            target=_drain_stream,
            args=(
                process.stdout,
                artifact_directory / "stdout.log",
                output_limit_bytes,
                truncated,
                ipc_observers if case.use_ipc else (),
            ),
        ),
        threading.Thread(
            target=_drain_stream,
            args=(
                process.stderr,
                artifact_directory / "stderr.log",
                output_limit_bytes,
                truncated,
                ipc_observers if case.use_ipc else (),
            ),
        ),
    ]
    for reader in readers:
        reader.start()

    ipc_handshake_seen = not case.use_ipc
    missing_ipc_capabilities: tuple[str, ...] = ()
    ipc_ready = not case.use_ipc
    hard_deadline = started + case.timeout_seconds
    if case.use_ipc:
        while process.poll() is None:
            remaining = hard_deadline - time.monotonic()
            if remaining <= 0 or ipc_handshake.wait(min(0.05, remaining)):
                break
        ipc_handshake_seen = ipc_handshake.is_set()
        required_capabilities = ["ENABLE_EMU_CONTROL"]
        if case.screenshot_seconds or case.screenshot_button_events:
            required_capabilities.append("ENABLE_SCREENSHOT")
        if case.renderdoc_capture_seconds or case.renderdoc_capture_on_visual_failure:
            required_capabilities.append("ENABLE_RENDERDOC_CAPTURE")
        if (
            case.button_events
            or case.screenshot_button_events
            or case.axis_events
            or case.touch_events
        ):
            required_capabilities.append("ENABLE_GAMEPAD")
        missing_ipc_capabilities = tuple(
            capability
            for capability in required_capabilities
            if not ipc_capabilities[capability].is_set()
        )
        ipc_ready = ipc_handshake_seen and not missing_ipc_capabilities
        if ipc_ready:
            assert process.stdin is not None
            process.stdin.write(b"RUN\nSTART\n")
            process.stdin.flush()
    timeline_started = time.monotonic()

    timed_out = False
    runtime_failures: list[str] = []
    visual_checkpoint_attempts: list[VisualCheckpointAttempt] = []
    visual_failure_capture_requests = 0
    visual_checkpoint_failed = False
    try:
        process_exited = False
        if case.screenshot_button_events and ipc_ready:
            assert process.stdin is not None
            for index, event in enumerate(case.screenshot_button_events):
                event_deadline = min(
                    hard_deadline, time.monotonic() + event.timeout_seconds
                )
                matched = False
                known_screenshots = set(_find_valid_screenshots(artifact_directory))
                while process.poll() is None and time.monotonic() < event_deadline:
                    request_started = time.monotonic()
                    process.stdin.write(
                        b"SCREENSHOT_WITH_OVERLAYS\n"
                        if case.screenshot_source == "presented_frame"
                        else b"SCREENSHOT\n"
                    )
                    process.stdin.flush()

                    screenshot: Path | None = None
                    # Do not send another request until this one produces a file.
                    # Slow guest frames can otherwise leave many screenshots queued,
                    # delaying both the visual match and the input that follows it.
                    poll_deadline = event_deadline
                    while process.poll() is None and time.monotonic() < poll_deadline:
                        candidates = [
                            path
                            for path in _find_valid_screenshots(artifact_directory)
                            if path not in known_screenshots
                        ]
                        if candidates:
                            screenshot = candidates[0]
                            known_screenshots.add(screenshot)
                            break
                        time.sleep(min(0.01, max(0, poll_deadline - time.monotonic())))

                    screenshot_matches = False
                    screenshot_hash: str | None = None
                    screenshot_difference: float | None = None
                    if screenshot is not None:
                        screenshot_hash = _hash_file(screenshot)
                        if event.screenshot_sha256 is not None:
                            screenshot_matches = (
                                screenshot_hash == event.screenshot_sha256
                            )
                        else:
                            assert event.reference_screenshot is not None
                            assert event.maximum_difference is not None
                            try:
                                screenshot_difference = _screenshot_difference(
                                    event.reference_screenshot,
                                    screenshot,
                                    mode=event.difference_mode,
                                    region=event.comparison_region,
                                    scale_first_to_second=event.scale_reference_to_capture,
                                )
                                screenshot_matches = (
                                    screenshot_difference <= event.maximum_difference
                                )
                            except (OSError, ValueError):
                                screenshot_matches = False
                        try:
                            mean_intensity, non_black_fraction = _screenshot_statistics(
                                screenshot
                            )
                        except (OSError, ValueError):
                            mean_intensity = None
                            non_black_fraction = None
                        visual_checkpoint_attempts.append(
                            VisualCheckpointAttempt(
                                event_index=index,
                                screenshot=screenshot,
                                screenshot_sha256=screenshot_hash,
                                difference=screenshot_difference,
                                mean_intensity=mean_intensity,
                                non_black_fraction=non_black_fraction,
                                matched=screenshot_matches,
                            )
                        )
                    if not screenshot_matches:
                        remaining_poll = (
                            min(event_deadline, request_started + event.poll_seconds)
                            - time.monotonic()
                        )
                        if remaining_poll > 0:
                            try:
                                process.wait(timeout=remaining_poll)
                                process_exited = True
                                break
                            except subprocess.TimeoutExpired:
                                pass
                        continue

                    process.stdin.write(
                        (f"GAMEPAD_BUTTON\n{event.button}\n1\n").encode("ascii")
                    )
                    process.stdin.flush()
                    try:
                        process.wait(timeout=event.hold_seconds)
                        process_exited = True
                    except subprocess.TimeoutExpired:
                        process.stdin.write(
                            (f"GAMEPAD_BUTTON\n{event.button}\n0\n").encode("ascii")
                        )
                        process.stdin.flush()
                    matched = True
                    break

                if process_exited:
                    break
                if not matched:
                    if case.renderdoc_capture_on_visual_failure:
                        process.stdin.write(b"RENDERDOC_CAPTURE\n")
                        process.stdin.flush()
                        visual_failure_capture_requests += 1
                    runtime_failures.append(
                        f"screenshotButtonEvents[{index}] did not match "
                        f"within {event.timeout_seconds:g} seconds"
                    )
                    visual_checkpoint_failed = True
                    break

        if visual_checkpoint_failed and not process_exited:
            if visual_failure_capture_requests:
                capture_deadline = min(hard_deadline, time.monotonic() + 15.0)
                previous_sizes: tuple[tuple[Path, int], ...] = ()
                stable_since: float | None = None
                while process.poll() is None and time.monotonic() < capture_deadline:
                    captures = _find_renderdoc_captures(artifact_directory)
                    sizes = tuple((path, path.stat().st_size) for path in captures)
                    if len(captures) >= visual_failure_capture_requests:
                        if sizes == previous_sizes:
                            if (
                                stable_since is not None
                                and time.monotonic() - stable_since >= 0.25
                            ):
                                break
                        else:
                            stable_since = time.monotonic()
                    previous_sizes = sizes
                    time.sleep(0.05)

                cooldown_seconds = min(2.0, max(0, hard_deadline - time.monotonic()))
                if cooldown_seconds:
                    try:
                        process.wait(timeout=cooldown_seconds)
                        process_exited = True
                    except subprocess.TimeoutExpired:
                        pass

            timed_out = True
            if not process_exited:
                assert process.stdin is not None
                try:
                    process.stdin.write(b"STOP\n")
                    process.stdin.flush()
                    process.wait(timeout=5)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    _kill_process_tree(process)
                    process.wait(timeout=5)
            process_exited = True

        timeline = (
            [(seconds, "screenshot", None) for seconds in case.screenshot_seconds]
            + [
                (seconds, "renderdoc_capture", None)
                for seconds in case.renderdoc_capture_seconds
            ]
            + [(event.seconds, "button", event) for event in case.button_events]
            + [(event.seconds, "axis", event) for event in case.axis_events]
            + [(event.seconds, "touch", event) for event in case.touch_events]
        )
        timeline.sort(key=lambda entry: entry[0])
        for event_second, event_type, payload in timeline:
            if process_exited:
                break
            try:
                process.wait(
                    timeout=max(0, timeline_started + event_second - time.monotonic())
                )
                process_exited = True
                break
            except subprocess.TimeoutExpired:
                assert process.stdin is not None
                if event_type == "screenshot":
                    process.stdin.write(
                        b"SCREENSHOT_WITH_OVERLAYS\n"
                        if case.screenshot_source == "presented_frame"
                        else b"SCREENSHOT\n"
                    )
                elif event_type == "renderdoc_capture":
                    process.stdin.write(b"RENDERDOC_CAPTURE\n")
                elif event_type == "button":
                    assert isinstance(payload, ButtonEvent)
                    process.stdin.write(
                        (
                            "GAMEPAD_BUTTON\n"
                            f"{payload.button}\n"
                            f"{int(payload.pressed)}\n"
                        ).encode("ascii")
                    )
                elif event_type == "axis":
                    assert isinstance(payload, AxisEvent)
                    process.stdin.write(
                        (f"GAMEPAD_AXIS\n{payload.axis}\n{payload.value}\n").encode(
                            "ascii"
                        )
                    )
                else:
                    assert isinstance(payload, TouchEvent)
                    process.stdin.write(
                        (
                            "GAMEPAD_TOUCH\n"
                            f"{payload.finger}\n"
                            f"{int(payload.down)}\n"
                            f"{payload.x}\n"
                            f"{payload.y}\n"
                        ).encode("ascii")
                    )
                process.stdin.flush()
        if not process_exited:
            process.wait(timeout=max(0, hard_deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out = True
        if case.use_ipc and ipc_ready:
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
        for reader in readers:
            reader.join(timeout=5)
        _close_process_streams(process)
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
    screenshots = _find_valid_screenshots(artifact_directory)
    screenshot_hashes = [_hash_file(path) for path in screenshots]
    renderdoc_captures = _find_renderdoc_captures(artifact_directory)
    renderdoc_capture_hashes = [_hash_file(path) for path in renderdoc_captures]

    failures: list[str] = list(runtime_failures)
    if not ipc_handshake_seen:
        failures.append("IPC handshake was not observed")
    elif missing_ipc_capabilities:
        failures.extend(
            f"IPC capability {capability} was not advertised"
            for capability in missing_ipc_capabilities
        )
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
    if len(screenshots) < len(case.screenshot_seconds):
        failures.append(
            f"captured {len(screenshots)} valid screenshots; expected "
            f"{len(case.screenshot_seconds)}"
        )
    expected_renderdoc_captures = (
        len(case.renderdoc_capture_seconds) + visual_failure_capture_requests
    )
    if len(renderdoc_captures) < expected_renderdoc_captures:
        failures.append(
            f"captured {len(renderdoc_captures)} valid RenderDoc frames; expected "
            f"{expected_renderdoc_captures}"
        )
    distinct_screenshots = len(set(screenshot_hashes))
    if distinct_screenshots < case.minimum_distinct_screenshots:
        failures.append(
            f"captured {distinct_screenshots} distinct screenshots; expected at least "
            f"{case.minimum_distinct_screenshots}"
        )
    screenshot_differences: list[float] = []
    if len(screenshots) >= len(case.screenshot_seconds):
        for index, comparison in enumerate(case.screenshot_comparisons):
            try:
                difference = _screenshot_difference(
                    screenshots[comparison.first_screenshot],
                    screenshots[comparison.second_screenshot],
                    mode=comparison.difference_mode,
                )
            except (OSError, ValueError) as error:
                failures.append(f"screenshot comparison {index} failed: {error}")
                continue
            screenshot_differences.append(difference)
            if (
                comparison.minimum_difference is not None
                and difference < comparison.minimum_difference
            ):
                failures.append(
                    f"screenshot comparison {index} difference {difference:.6f} is "
                    f"below minimum {comparison.minimum_difference:.6f}"
                )
            if (
                comparison.maximum_difference is not None
                and difference > comparison.maximum_difference
            ):
                failures.append(
                    f"screenshot comparison {index} difference {difference:.6f} is "
                    f"above maximum {comparison.maximum_difference:.6f}"
                )

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
        screenshots=screenshots,
        screenshot_hashes=screenshot_hashes,
        screenshot_differences=screenshot_differences,
        visual_checkpoint_attempts=visual_checkpoint_attempts,
        renderdoc_captures=renderdoc_captures,
        renderdoc_capture_hashes=renderdoc_capture_hashes,
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
