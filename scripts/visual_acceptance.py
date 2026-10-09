# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

"""CPU-only, role-aware screenshot acceptance for gameplay checkpoints.

This module deliberately does not identify game objects from pixels.  A private
route owner supplies normalized regions for a checkpoint where the terrain,
protagonist, and horse are expected to be on screen.  The analyzer then checks
that each declared region has non-trivial content and local structure over
multiple frames.  This prevents a global non-black or smoke screenshot from
being treated as proof that gameplay rendered correctly.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any, Sequence

try:
    from scripts.game_test_runner import _decode_png_rgb
except ModuleNotFoundError:  # Direct ``python scripts/visual_acceptance.py`` use.
    from game_test_runner import _decode_png_rgb


CONTRACT_SCHEMA_VERSION = 1
REGION_KINDS = frozenset({"sky", "terrain", "actor", "horse", "other"})
REQUIRED_GAMEPLAY_KINDS = frozenset({"terrain", "actor", "horse"})
CONTRACT_FIELDS = frozenset(
    {
        "schemaVersion",
        "minimumFramesPerRole",
        "minimumFramesWithAllRequiredRoles",
        "minimumDistinctViews",
        "regions",
    }
)
REGION_FIELDS = frozenset(
    {
        "name",
        "kind",
        "left",
        "top",
        "width",
        "height",
        "required",
        "minimumMeanIntensity",
        "minimumNonBlackFraction",
        "minimumLocalContrast",
        "minimumEdgeFraction",
    }
)
SPATIAL_EDGE_LUMINANCE_THRESHOLD = 8


class ContractError(ValueError):
    """Raised when a visual acceptance contract is invalid."""


@dataclass(frozen=True)
class VisualRoleRegion:
    name: str
    kind: str
    left: float
    top: float
    width: float
    height: float
    required: bool
    minimum_mean_intensity: float
    minimum_non_black_fraction: float
    minimum_local_contrast: float
    minimum_edge_fraction: float


@dataclass(frozen=True)
class SceneContract:
    regions: tuple[VisualRoleRegion, ...]
    minimum_frames_per_role: int = 1
    minimum_frames_with_all_required_roles: int = 1
    minimum_distinct_views: int = 0

    @property
    def required_regions(self) -> tuple[VisualRoleRegion, ...]:
        return tuple(region for region in self.regions if region.required)

    @property
    def required_kinds(self) -> frozenset[str]:
        return frozenset(region.kind for region in self.required_regions)


@dataclass(frozen=True)
class RegionMetrics:
    name: str
    kind: str
    required: bool
    mean_intensity: float
    non_black_fraction: float
    local_contrast: float
    edge_fraction: float
    visible: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class FrameMetrics:
    screenshot: str
    width: int
    height: int
    regions: tuple[RegionMetrics, ...]


@dataclass(frozen=True)
class VisualAcceptanceReport:
    passed: bool
    classification: str
    frames: tuple[FrameMetrics, ...]
    role_frame_counts: dict[str, int]
    frames_with_all_required_roles: int
    missing_required_roles: tuple[str, ...]
    findings: tuple[str, ...]

    def to_report(self) -> dict[str, Any]:
        report = asdict(self)
        report["schemaVersion"] = CONTRACT_SCHEMA_VERSION
        return report


def _reject_unknown_fields(raw: dict[str, Any], allowed: frozenset[str], field: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        names = ", ".join(repr(value) for value in unknown)
        raise ContractError(f"{field}: unknown field(s): {names}")


def _unit_interval(raw: Any, *, field: str) -> float:
    if (
        not isinstance(raw, (int, float))
        or isinstance(raw, bool)
        or not math.isfinite(raw)
        or raw < 0
        or raw > 1
    ):
        raise ContractError(f"{field} must be between 0 and 1")
    return float(raw)


def _normalized_region(raw: dict[str, Any], *, field: str) -> VisualRoleRegion:
    _reject_unknown_fields(raw, REGION_FIELDS, field)
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ContractError(f"{field}.name must be a non-empty string")
    kind = raw.get("kind")
    if kind not in REGION_KINDS:
        raise ContractError(
            f"{field}.kind must be one of {sorted(REGION_KINDS)}"
        )
    coordinates: list[float] = []
    for component in ("left", "top", "width", "height"):
        value = raw.get(component)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
            or value > 1
        ):
            raise ContractError(f"{field}.{component} must be between 0 and 1")
        coordinates.append(float(value))
    left, top, width, height = coordinates
    if width <= 0 or height <= 0:
        raise ContractError(f"{field}.width and height must be positive")
    if left + width > 1 or top + height > 1:
        raise ContractError(f"{field} must stay within normalized frame bounds")
    required = raw.get("required", True)
    if not isinstance(required, bool):
        raise ContractError(f"{field}.required must be a boolean")
    return VisualRoleRegion(
        name=name,
        kind=kind,
        left=left,
        top=top,
        width=width,
        height=height,
        required=required,
        minimum_mean_intensity=_unit_interval(
            raw.get("minimumMeanIntensity", 0),
            field=f"{field}.minimumMeanIntensity",
        ),
        minimum_non_black_fraction=_unit_interval(
            raw.get("minimumNonBlackFraction", 0),
            field=f"{field}.minimumNonBlackFraction",
        ),
        minimum_local_contrast=_unit_interval(
            raw.get("minimumLocalContrast", 0),
            field=f"{field}.minimumLocalContrast",
        ),
        minimum_edge_fraction=_unit_interval(
            raw.get("minimumEdgeFraction", 0),
            field=f"{field}.minimumEdgeFraction",
        ),
    )


def load_contract(path: str | Path) -> SceneContract:
    source = Path(path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read contract {source}: {error}") from error
    if not isinstance(raw, dict):
        raise ContractError("contract root must be an object")
    _reject_unknown_fields(raw, CONTRACT_FIELDS, "contract")
    if raw.get("schemaVersion") != CONTRACT_SCHEMA_VERSION:
        raise ContractError(f"schemaVersion must be {CONTRACT_SCHEMA_VERSION}")

    regions_raw = raw.get("regions")
    if not isinstance(regions_raw, list) or not regions_raw:
        raise ContractError("regions must be a non-empty array")
    regions = tuple(
        _normalized_region(item, field=f"regions[{index}]")
        for index, item in enumerate(regions_raw)
    )
    names = [region.name for region in regions]
    if len(set(names)) != len(names):
        raise ContractError("region names must be unique")
    present_kinds = frozenset(region.kind for region in regions)
    missing_kinds = sorted(REQUIRED_GAMEPLAY_KINDS - present_kinds)
    if missing_kinds:
        raise ContractError(
            "required gameplay roles must include "
            + ", ".join(sorted(REQUIRED_GAMEPLAY_KINDS))
            + "; missing "
            + ", ".join(missing_kinds)
        )
    required_kinds = frozenset(region.kind for region in regions if region.required)
    missing_required = sorted(REQUIRED_GAMEPLAY_KINDS - required_kinds)
    if missing_required:
        raise ContractError(
            "terrain, actor, and horse gameplay regions must be required; missing "
            + ", ".join(missing_required)
        )
    by_kind = {region.kind: region for region in regions}
    actor = by_kind["actor"]
    horse = by_kind["horse"]
    if (
        actor.left < horse.left + horse.width
        and horse.left < actor.left + actor.width
        and actor.top < horse.top + horse.height
        and horse.top < actor.top + actor.height
    ):
        raise ContractError("actor and horse regions must not overlap")
    for region in regions:
        if region.kind in REQUIRED_GAMEPLAY_KINDS and (
            region.minimum_local_contrast <= 0 or region.minimum_edge_fraction <= 0
        ):
            raise ContractError(
                f"{region.kind} requires positive local contrast and edge floors"
            )

    minimum_frames_per_role = raw.get("minimumFramesPerRole", 1)
    minimum_frames_with_all = raw.get("minimumFramesWithAllRequiredRoles", 1)
    for field, value in (
        ("minimumFramesPerRole", minimum_frames_per_role),
        ("minimumFramesWithAllRequiredRoles", minimum_frames_with_all),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ContractError(f"{field} must be a positive integer")
    minimum_distinct_views = raw.get("minimumDistinctViews", 0)
    if (
        not isinstance(minimum_distinct_views, int)
        or isinstance(minimum_distinct_views, bool)
        or minimum_distinct_views < 0
    ):
        raise ContractError("minimumDistinctViews must be a non-negative integer")
    return SceneContract(
        regions=regions,
        minimum_frames_per_role=minimum_frames_per_role,
        minimum_frames_with_all_required_roles=minimum_frames_with_all,
        minimum_distinct_views=minimum_distinct_views,
    )


def _crop_region(
    pixels: bytes,
    image_width: int,
    image_height: int,
    region: VisualRoleRegion,
) -> tuple[int, int, bytes]:
    left = min(image_width - 1, math.floor(region.left * image_width))
    top = min(image_height - 1, math.floor(region.top * image_height))
    right = min(image_width, math.ceil((region.left + region.width) * image_width))
    bottom = min(image_height, math.ceil((region.top + region.height) * image_height))
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        raise ValueError(
            f"region {region.name!r} is empty at {image_width}x{image_height}"
        )
    row_size = width * 3
    cropped = b"".join(
        pixels[
            ((top + row) * image_width + left)
            * 3 : ((top + row) * image_width + left) * 3
            + row_size
        ]
        for row in range(height)
    )
    return width, height, cropped


def _decode_bmp_rgb(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if len(data) < 54 or data[:2] != b"BM":
        raise ValueError("unsupported BMP header")
    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    dib_size = struct.unpack_from("<I", data, 14)[0]
    if dib_size < 40 or len(data) < 14 + dib_size:
        raise ValueError("unsupported BMP information header")
    width, signed_height, planes, bits_per_pixel, compression = struct.unpack_from(
        "<iiHHI", data, 18
    )
    if width <= 0 or signed_height == 0:
        raise ValueError("BMP dimensions must be positive")
    if planes != 1 or compression != 0 or bits_per_pixel not in (24, 32):
        raise ValueError("only uncompressed 24-bit and 32-bit BMPs are supported")
    height = abs(signed_height)
    bytes_per_pixel = bits_per_pixel // 8
    row_stride = (width * bytes_per_pixel + 3) & ~3
    required_bytes = pixel_offset + row_stride * height
    if pixel_offset < 14 + dib_size or required_bytes > len(data):
        raise ValueError("BMP pixel data is truncated")
    rgb = bytearray(width * height * 3)
    output = 0
    for y in range(height):
        source_y = y if signed_height < 0 else height - 1 - y
        row_start = pixel_offset + source_y * row_stride
        for x in range(width):
            source = row_start + x * bytes_per_pixel
            rgb[output : output + 3] = bytes(
                (data[source + 2], data[source + 1], data[source])
            )
            output += 3
    return width, height, bytes(rgb)


def _decode_image_rgb(path: Path) -> tuple[int, int, bytes]:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return _decode_png_rgb(path)
    if suffix == ".bmp":
        return _decode_bmp_rgb(path)
    raise ValueError(f"unsupported screenshot format {suffix!r}; expected .png or .bmp")


def _region_metrics(
    pixels: bytes,
    image_width: int,
    image_height: int,
    region: VisualRoleRegion,
) -> RegionMetrics:
    width, height, cropped = _crop_region(pixels, image_width, image_height, region)
    luminance = [
        (54 * cropped[offset] + 183 * cropped[offset + 1] + 19 * cropped[offset + 2])
        // 256
        for offset in range(0, len(cropped), 3)
    ]
    pixel_count = len(luminance)
    mean_intensity = sum(cropped) / (len(cropped) * 255)
    non_black_fraction = sum(
        any(cropped[offset + channel] > 0 for channel in range(3))
        for offset in range(0, len(cropped), 3)
    ) / pixel_count
    horizontal_differences = [
        abs(luminance[index] - luminance[index + 1])
        for row in range(height)
        for index in range(row * width, (row + 1) * width - 1)
    ]
    vertical_differences = [
        abs(luminance[index] - luminance[index + width])
        for index in range((height - 1) * width)
    ]
    differences = horizontal_differences + vertical_differences
    local_contrast = (
        sum(differences) / (len(differences) * 255) if differences else 0.0
    )
    edge_fraction = (
        sum(value >= SPATIAL_EDGE_LUMINANCE_THRESHOLD for value in differences)
        / len(differences)
        if differences
        else 0.0
    )
    failures: list[str] = []
    threshold_values = (
        ("mean intensity", mean_intensity, region.minimum_mean_intensity),
        ("non-black fraction", non_black_fraction, region.minimum_non_black_fraction),
        ("local contrast", local_contrast, region.minimum_local_contrast),
        ("edge fraction", edge_fraction, region.minimum_edge_fraction),
    )
    for label, actual, minimum in threshold_values:
        if actual < minimum:
            failures.append(f"{label} {actual:.6f} < minimum {minimum:.6f}")
    return RegionMetrics(
        name=region.name,
        kind=region.kind,
        required=region.required,
        mean_intensity=mean_intensity,
        non_black_fraction=non_black_fraction,
        local_contrast=local_contrast,
        edge_fraction=edge_fraction,
        visible=not failures,
        failures=tuple(failures),
    )


def _classify(
    frames: Sequence[FrameMetrics],
    role_frame_counts: dict[str, int],
    required_kinds: frozenset[str],
) -> str:
    visible_kinds = {
        region.kind
        for frame in frames
        for region in frame.regions
        if region.visible
    }
    terrain_visible = role_frame_counts.get("terrain", 0) > 0
    actor_horse_visible = all(
        role_frame_counts.get(kind, 0) > 0 for kind in ("actor", "horse")
    )
    if terrain_visible and actor_horse_visible:
        return "terrain_and_actor_horse"
    if terrain_visible and not actor_horse_visible:
        return (
            "actor_horse_missing"
            if required_kinds.intersection({"actor", "horse"})
            else "terrain_only"
        )
    if "sky" in visible_kinds and not (terrain_visible or actor_horse_visible):
        return "sky_only"
    if actor_horse_visible and not terrain_visible:
        return "terrain_missing"
    if not visible_kinds or not required_kinds.intersection(visible_kinds):
        return "blank_or_inconclusive"
    return "partial_scene"


def analyze_screenshots(
    screenshots: Sequence[str | Path | dict[str, Any]], contract: SceneContract
) -> VisualAcceptanceReport:
    frame_metrics: list[FrameMetrics] = []
    frame_view_ids: list[Any] = []
    findings: list[str] = []
    for screenshot in screenshots:
        metadata = screenshot if isinstance(screenshot, dict) else {}
        path = Path(metadata.get("screenshot", screenshot))
        try:
            width, height, pixels = _decode_image_rgb(path)
            region_metrics = tuple(
                _region_metrics(pixels, width, height, region)
                for region in contract.regions
            )
        except (OSError, ValueError) as error:
            findings.append(f"{path}: failed to analyze screenshot: {error}")
            continue
        frame_metrics.append(
            FrameMetrics(
                screenshot=str(path),
                width=width,
                height=height,
                regions=region_metrics,
            )
        )
        frame_view_ids.append(metadata.get("viewId"))

    role_frame_counts = {
        region.kind: sum(
            any(
                metric.kind == region.kind and metric.visible
                for metric in frame.regions
            )
            for frame in frame_metrics
        )
        for region in contract.regions
    }
    frames_with_all_required_roles = sum(
        all(
            any(
                metric.kind == region.kind
                and metric.required
                and metric.visible
                for metric in frame.regions
            )
            for region in contract.required_regions
        )
        for frame in frame_metrics
    )
    missing_required_roles = tuple(
        sorted(
            kind
            for kind in contract.required_kinds
            if role_frame_counts.get(kind, 0) < contract.minimum_frames_per_role
        )
    )
    classification = _classify(
        frame_metrics, role_frame_counts, contract.required_kinds
    )
    if not frame_metrics:
        findings.append("no valid screenshots were analyzed")
    for role in missing_required_roles:
        findings.append(
            f"required role {role!r} visible in {role_frame_counts.get(role, 0)} "
            f"frames; minimum {contract.minimum_frames_per_role}"
        )
    if frames_with_all_required_roles < contract.minimum_frames_with_all_required_roles:
        findings.append(
            f"frames with all required roles {frames_with_all_required_roles}; "
            f"minimum {contract.minimum_frames_with_all_required_roles}"
        )
    if contract.minimum_distinct_views:
        complete_views = {
            frame_view_ids[index]
            for index, frame in enumerate(frame_metrics)
            if frame_view_ids[index] is not None
            and all(
                any(
                    metric.kind == region.kind and metric.required and metric.visible
                    for metric in frame.regions
                )
                for region in contract.required_regions
            )
        }
        if len(complete_views) < contract.minimum_distinct_views:
            findings.append(
                f"distinct complete views {len(complete_views)}; minimum "
                f"{contract.minimum_distinct_views}"
            )
    passed = not findings and classification == "terrain_and_actor_horse"
    return VisualAcceptanceReport(
        passed=passed,
        classification=classification,
        frames=tuple(frame_metrics),
        role_frame_counts=role_frame_counts,
        frames_with_all_required_roles=frames_with_all_required_roles,
        missing_required_roles=missing_required_roles,
        findings=tuple(findings),
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, help="role-region contract JSON")
    parser.add_argument(
        "--output",
        type=Path,
        help="write the JSON diagnostic here; stdout by default",
    )
    parser.add_argument("screenshots", nargs="+", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = analyze_screenshots(args.screenshots, load_contract(args.contract))
    except (ContractError, OSError, ValueError) as error:
        print(f"visual acceptance error: {error}", file=sys.stderr)
        return 2
    payload = json.dumps(report.to_report(), indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
