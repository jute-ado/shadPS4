// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>

#include "common/types.h"

namespace VideoCore {

u32 QuadListTriangleIndexCount(u32 input_count);

void GenerateQuadListTriangleIndices(std::span<u32> output, u32 input_count);

void ConvertQuadListTriangleIndices(std::span<u16> output, std::span<const u16> input);

void ConvertQuadListTriangleIndices(std::span<u32> output, std::span<const u32> input);

} // namespace VideoCore
