# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from scripts.visual_acceptance import (
    ContractError,
    VisualEvidence,
    analyze_screenshots,
    load_contract,
)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + data).to_bytes(4, "big")
    return len(data).to_bytes(4, "big") + chunk_type + data + checksum


def _rgb_png(width: int, height: int, pixels: list[tuple[int, int, int]]) -> bytes:
    rows = b"".join(
        b"\x00"
        + bytes(
            channel
            for pixel in pixels[row * width : (row + 1) * width]
            for channel in pixel
        )
        for row in range(height)
    )
    header = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes((8, 2, 0, 0, 0))
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows))
        + _png_chunk(b"IEND", b"")
    )


def _rgb_bmp(width: int, height: int, pixels: list[tuple[int, int, int]]) -> bytes:
    row_stride = (width * 3 + 3) & ~3
    rows = b"".join(
        bytes(
            channel
            for pixel in pixels[row * width : (row + 1) * width]
            for channel in (pixel[2], pixel[1], pixel[0])
        )
        + b"\x00" * (row_stride - width * 3)
        for row in reversed(range(height))
    )
    dib = struct.pack(
        "<IiiHHIIiiII", 40, width, height, 1, 24, 0, len(rows), 0, 0, 0, 0
    )
    return b"BM" + struct.pack("<IHHI", 54 + len(rows), 0, 0, 54) + dib + rows


def _shadow_contract() -> dict:
    return {
        "schemaVersion": 1,
        "minimumFramesPerRole": 2,
        "minimumFramesWithAllRequiredRoles": 2,
        "regions": [
            {
                "name": "sky",
                "kind": "sky",
                "left": 0,
                "top": 0,
                "width": 1,
                "height": 0.45,
                "required": False,
                "minimumMeanIntensity": 0.01,
                "minimumNonBlackFraction": 0.9,
            },
            {
                "name": "terrain",
                "kind": "terrain",
                "left": 0,
                "top": 0.55,
                "width": 1,
                "height": 0.45,
                "minimumMeanIntensity": 0.01,
                "minimumNonBlackFraction": 0.3,
                "minimumLocalContrast": 0.02,
                "minimumEdgeFraction": 0.05,
            },
            {
                "name": "actor",
                "kind": "actor",
                "left": 0.35,
                "top": 0.52,
                "width": 0.14,
                "height": 0.35,
                "minimumMeanIntensity": 0.01,
                "minimumNonBlackFraction": 0.15,
                "minimumLocalContrast": 0.02,
                "minimumEdgeFraction": 0.05,
            },
            {
                "name": "horse",
                "kind": "horse",
                "left": 0.55,
                "top": 0.52,
                "width": 0.2,
                "height": 0.35,
                "minimumMeanIntensity": 0.01,
                "minimumNonBlackFraction": 0.15,
                "minimumLocalContrast": 0.02,
                "minimumEdgeFraction": 0.05,
            },
        ],
    }


def _complete_frame() -> list[tuple[int, int, int]]:
    width = height = 20
    pixels = [(16, 20, 28)] * (width * height)
    for y in range(height):
        for x in range(width):
            if y < 10:
                pixels[y * width + x] = (24, 32, 46)
            else:
                pixels[y * width + x] = ((x * 17 + y * 9) % 160 + 20,) * 3
    # Synthetic actor and horse silhouettes have sharp local contrast in the
    # declared checkpoint regions while remaining CPU-only test data.
    for y in range(11, 18):
        for x in range(7, 10):
            pixels[y * width + x] = (230, 230, 230) if (x + y) % 2 else (12, 12, 12)
    for y in range(11, 18):
        for x in range(11, 15):
            pixels[y * width + x] = (210, 210, 210) if (x + y) % 2 else (10, 10, 10)
    return pixels


class VisualAcceptanceContractTests(unittest.TestCase):
    def test_load_contract_requires_terrain_actor_and_horse_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            contract = _shadow_contract()
            contract["regions"] = contract["regions"][:2]
            path.write_text(json.dumps(contract), encoding="utf-8")

            with self.assertRaisesRegex(ContractError, "actor.*horse"):
                load_contract(path)

    def test_complete_checkpoint_requires_all_roles_in_same_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(_shadow_contract()), encoding="utf-8")
            frames = []
            for index in range(2):
                frame = root / f"frame-{index}.png"
                frame.write_bytes(_rgb_png(20, 20, _complete_frame()))
                frames.append(frame)

            report = analyze_screenshots(frames, load_contract(contract_path))

            self.assertTrue(report.passed)
            self.assertEqual(report.classification, "terrain_and_actor_horse")
            self.assertEqual(report.frames_with_all_required_roles, 2)
            self.assertEqual(report.role_frame_counts["terrain"], 2)
            self.assertEqual(report.role_frame_counts["actor"], 2)
            self.assertEqual(report.role_frame_counts["horse"], 2)

    def test_complete_checkpoint_accepts_test_lab_bmp_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(_shadow_contract()), encoding="utf-8")
            frame = root / "frame.bmp"
            frame.write_bytes(_rgb_bmp(20, 20, _complete_frame()))

            report = analyze_screenshots([frame, frame], load_contract(contract_path))

            self.assertTrue(report.passed)
            self.assertEqual(report.frames[0].width, 20)

    def test_transition_frame_may_miss_role_when_required_window_still_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _shadow_contract()
            contract["minimumFramesPerRole"] = 2
            contract["minimumFramesWithAllRequiredRoles"] = 2
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            complete = _complete_frame()
            terrain_only = list(complete)
            for y in range(10, 19):
                for x in range(7, 15):
                    terrain_only[y * 20 + x] = (70, 70, 70)
            frames = []
            for index, pixels in enumerate((complete, terrain_only, complete)):
                frame = root / f"frame-{index}.png"
                frame.write_bytes(_rgb_png(20, 20, pixels))
                frames.append(frame)

            report = analyze_screenshots(frames, load_contract(contract_path))

            self.assertTrue(report.passed)
            self.assertEqual(report.frames_with_all_required_roles, 2)

    def test_terrain_without_actor_or_horse_is_not_a_visual_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _shadow_contract()
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            frame = root / "terrain-only.png"
            pixels = _complete_frame()
            for y in range(10, 19):
                for x in range(7, 15):
                    pixels[y * 20 + x] = (70, 70, 70)
            frame.write_bytes(_rgb_png(20, 20, pixels))

            report = analyze_screenshots([frame, frame], load_contract(contract_path))

            self.assertFalse(report.passed)
            self.assertEqual(report.classification, "actor_horse_missing")
            self.assertIn("actor", report.missing_required_roles)
            self.assertIn("horse", report.missing_required_roles)
            self.assertTrue(any("actor" in finding for finding in report.findings))

    def test_load_contract_rejects_optional_gameplay_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _shadow_contract()
            for region in contract["regions"]:
                if region["kind"] in {"terrain", "actor", "horse"}:
                    region["required"] = False
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")

            with self.assertRaisesRegex(ContractError, "must be required"):
                load_contract(contract_path)

    def test_load_contract_rejects_zero_spatial_floors_for_gameplay_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _shadow_contract()
            for region in contract["regions"]:
                if region["kind"] in {"terrain", "actor", "horse"}:
                    region["minimumLocalContrast"] = 0
                    region["minimumEdgeFraction"] = 0
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")

            with self.assertRaisesRegex(ContractError, "positive local contrast"):
                load_contract(contract_path)

    def test_load_contract_rejects_overlapping_actor_and_horse_regions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _shadow_contract()
            actor = next(
                region for region in contract["regions"] if region["kind"] == "actor"
            )
            horse = next(
                region for region in contract["regions"] if region["kind"] == "horse"
            )
            horse.update(
                left=actor["left"],
                top=actor["top"],
                width=actor["width"],
                height=actor["height"],
            )
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")

            with self.assertRaisesRegex(ContractError, "must not overlap"):
                load_contract(contract_path)

    def test_load_contract_rejects_overlap_across_multiple_role_regions(self) -> None:
        """Every actor/horse evidence pair must be disjoint, not only the last."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _shadow_contract()
            contract["regions"].append(
                {
                    "name": "actor-secondary",
                    "kind": "actor",
                    "left": 0.75,
                    "top": 0.52,
                    "width": 0.10,
                    "height": 0.20,
                    "minimumMeanIntensity": 0.01,
                    "minimumNonBlackFraction": 0.15,
                    "minimumLocalContrast": 0.02,
                    "minimumEdgeFraction": 0.05,
                },
            )
            horse = next(
                region for region in contract["regions"] if region["kind"] == "horse"
            )
            # The first actor overlaps the horse, while the last actor is
            # disjoint.  A last-value dict lookup therefore misses this.
            horse.update(left=0.36, top=0.55, width=0.10, height=0.20)
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")

            with self.assertRaisesRegex(ContractError, "must not overlap"):
                load_contract(contract_path)

    def test_load_contract_rejects_non_object_region_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _shadow_contract()
            contract["regions"][1] = "terrain"
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")

            with self.assertRaisesRegex(ContractError, r"regions\[1\].*object"):
                load_contract(contract_path)

    def test_minimum_distinct_views_rejects_duplicate_view_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _shadow_contract()
            contract["minimumDistinctViews"] = 2
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            frame = root / "frame.png"
            frame.write_bytes(_rgb_png(20, 20, _complete_frame()))
            loaded = load_contract(contract_path)

            report = analyze_screenshots(
                [
                    VisualEvidence(frame, view_id="view-a"),
                    VisualEvidence(frame, view_id="view-a"),
                ],
                loaded,
            )

            self.assertFalse(report.passed)
            self.assertEqual(report.classification, "insufficient_distinct_views")
            self.assertTrue(
                any("distinct complete views 1" in finding for finding in report.findings)
            )

    def test_minimum_distinct_views_requires_complete_roles_per_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _shadow_contract()
            contract["minimumDistinctViews"] = 2
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            complete = root / "complete.png"
            complete.write_bytes(_rgb_png(20, 20, _complete_frame()))
            partial_pixels = _complete_frame()
            for y in range(10, 19):
                for x in range(7, 15):
                    partial_pixels[y * 20 + x] = (70, 70, 70)
            partial = root / "partial.png"
            partial.write_bytes(_rgb_png(20, 20, partial_pixels))
            loaded = load_contract(contract_path)

            report = analyze_screenshots(
                [
                    VisualEvidence(partial, view_id="view-a"),
                    VisualEvidence(complete, view_id="view-b"),
                ],
                loaded,
            )

            self.assertFalse(report.passed)
            self.assertEqual(report.classification, "insufficient_distinct_views")
            self.assertTrue(
                any("distinct complete views 1" in finding for finding in report.findings)
            )

    def test_minimum_distinct_views_accepts_complete_distinct_view_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _shadow_contract()
            contract["minimumDistinctViews"] = 2
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            frames = []
            for view_id in ("view-a", "view-b"):
                frame = root / f"{view_id}.png"
                frame.write_bytes(_rgb_png(20, 20, _complete_frame()))
                frames.append(VisualEvidence(frame, view_id=view_id))

            report = analyze_screenshots(frames, load_contract(contract_path))

            self.assertTrue(report.passed)
            self.assertEqual(report.distinct_complete_views, 2)

    def test_minimum_distinct_views_rejects_path_only_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _shadow_contract()
            contract["minimumDistinctViews"] = 2
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            frame = root / "frame.png"
            frame.write_bytes(_rgb_png(20, 20, _complete_frame()))

            report = analyze_screenshots([frame, frame], load_contract(contract_path))

            self.assertFalse(report.passed)
            self.assertEqual(report.distinct_complete_views, 0)
            self.assertIn("view ID", " ".join(report.findings))

    def test_uniform_fog_does_not_satisfy_spatial_content_floors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = _shadow_contract()
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            frame = root / "fog.png"
            frame.write_bytes(_rgb_png(20, 20, [(80, 80, 80)] * 400))

            report = analyze_screenshots([frame, frame], load_contract(contract_path))

            self.assertFalse(report.passed)
            self.assertEqual(report.classification, "sky_only")
            self.assertIn("terrain", report.missing_required_roles)


if __name__ == "__main__":
    unittest.main()
