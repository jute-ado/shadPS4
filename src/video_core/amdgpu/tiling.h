// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <string_view>

#include "common/types.h"

namespace AmdGpu {

struct Image;

static constexpr size_t NUM_TILE_MODES = 32;

enum class PipeConfig : u32 {
    P2 = 0,
    P4_8x16 = 4,
    P4_16x16 = 5,
    P4_16x32 = 6,
    P4_32x32 = 7,
    P8_16x16_8x16 = 8,
    P8_16x32_8x16 = 9,
    P8_32x32_8x16 = 10,
    P8_16x32_16x16 = 11,
    P8_32x32_16x16 = 12,
    P8_32x32_16x32 = 13,
    P8_32x64_32x32 = 14,
    P16_32x32_8x16 = 16,
    P16_32x32_16x16 = 17,
    P16 = 18,
};

enum class MicroTileMode : u32 {
    Display = 0,
    Thin = 1,
    Depth = 2,
    Rotated = 3,
    Thick = 4,
};

enum class MacroTileMode : u32 {
    Mode_1x4_16 = 0,
    Mode_1x2_16 = 1,
    Mode_1x1_16 = 2,
    Mode_1x1_16_Dup = 3,
    Mode_1x1_8 = 4,
    Mode_1x1_4 = 5,
    Mode_1x1_2 = 6,
    Mode_1x1_2_Dup = 7,
    Mode_1x8_16 = 8,
    Mode_1x4_16_Dup = 9,
    Mode_1x2_16_Dup = 10,
    Mode_1x1_16_Dup2 = 11,
    Mode_1x1_8_Dup = 12,
    Mode_1x1_4_Dup = 13,
    Mode_1x1_2_Dup2 = 14,
    Mode_1x1_2_Dup3 = 15,
};

/// Returns whether a value can be used to index one of the exhaustive macro-tile lookup
/// tables.  Image descriptors are guest-controlled, and the arithmetic in
/// CalculateMacrotileMode can underflow for sub-64-byte tiles (for example a packed/BC format),
/// yielding an enum value outside the hardware table.  Admission callers must check this before
/// dispatching to a lookup that intentionally asserts on unknown modes.
[[nodiscard]] constexpr bool IsValidMacroTileMode(const MacroTileMode mode) noexcept {
    return static_cast<u32>(mode) <= static_cast<u32>(MacroTileMode::Mode_1x1_2_Dup3);
}

enum class ArrayMode : u32 {
    ArrayLinearGeneral = 0,
    ArrayLinearAligned = 1,
    Array1DTiledThin1 = 2,
    Array1DTiledThick = 3,
    Array2DTiledThin1 = 4,
    ArrayPrtTiledThin1 = 5,
    ArrayPrt2DTiledThin1 = 6,
    Array2DTiledThick = 7,
    Array2DTiledXThick = 8,
    ArrayPrtTiledThick = 9,
    ArrayPrt2DTiledThick = 10,
    ArrayPrt3DTiledThin1 = 11,
    Array3DTiledThin1 = 12,
    Array3DTiledThick = 13,
    Array3DTiledXThick = 14,
    ArrayPrt3DTiledThick = 15,
};

enum class TileMode : u32 {
    Depth2DThin64 = 0,
    Depth2DThin128 = 1,
    Depth2DThin256 = 2,
    Depth2DThin512 = 3,
    Depth2DThin1K = 4,
    Depth1DThin = 5,
    Depth2DThinPrt256 = 6,
    Depth2DThinPrt1K = 7,
    DisplayLinearAligned = 8,
    Display1DThin = 9,
    Display2DThin = 10,
    DisplayThinPrt = 11,
    Display2DThinPrt = 12,
    Thin1DThin = 13,
    Thin2DThin = 14,
    Thin3DThin = 15,
    ThinThinPrt = 16,
    Thin2DThinPrt = 17,
    Thin3DThinPrt = 18,
    Thick1DThick = 19,
    Thick2DThick = 20,
    Thick3DThick = 21,
    ThickThickPrt = 22,
    Thick2DThickPrt = 23,
    Thick3DThickPrt = 24,
    Thick2DXThick = 25,
    Thick3DXThick = 26,
    // Rotated entries in the GFX8/Tonga tile-mode table. Mode 30 is
    // profile-dependent (for example, rotated PRT on some GFX8 parts), so it
    // remains unknown until the GPU profile is carried through layout decode.
    Rotated1DThin = 27,
    Rotated2DThin = 28,
    RotatedThinPrt = 29,
    DisplayLinearGeneral = 31,
};

std::string_view NameOf(TileMode tile_mode);

/// Returns the array layout represented by a guest tile mode, or no value for reserved/unknown
/// five-bit encodings.  Guest descriptors are untrusted; callers at an admission boundary must
/// use this form instead of calling GetArrayMode directly.
[[nodiscard]] std::optional<ArrayMode> TryGetArrayMode(TileMode tile_mode);

ArrayMode GetArrayMode(TileMode tile_mode);

MicroTileMode GetMicroTileMode(TileMode tile_mode);

PipeConfig GetPipeConfig(TileMode tile_mode);

PipeConfig GetAltPipeConfig(TileMode tile_mode);

u32 GetSampleSplit(TileMode tile_mode);

u32 GetTileSplitHw(TileMode tile_mode);

u32 GetBankWidth(MacroTileMode mode);

u32 GetBankHeight(MacroTileMode mode);

u32 GetNumBanks(MacroTileMode mode);

u32 GetMacrotileAspect(MacroTileMode mode);

u32 GetAltBankHeight(MacroTileMode mode);

u32 GetAltNumBanks(MacroTileMode mode);

u32 GetAltMacrotileAspect(MacroTileMode mode);

bool IsMacroTiled(ArrayMode array_mode);

bool IsPrt(ArrayMode array_mode);

u32 GetMicroTileThickness(ArrayMode array_mode);

u32 GetPipeCount(PipeConfig pipe_cfg);

u32 CalculateTileSplit(TileMode tile_mode, ArrayMode array_mode, MicroTileMode micro_tile_mode,
                       u32 bpp);

MacroTileMode CalculateMacrotileMode(TileMode tile_mode, u32 bpp, u32 num_samples);

} // namespace AmdGpu
