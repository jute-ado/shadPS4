// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Common {

enum class X64Gpr {
    Rdi,
    R10,
};

void* GetXmmPointer(void* ctx, u8 index);

void* GetRip(void* ctx);

void IncrementRip(void* ctx, u64 length);

void SetX64GprToZero(void* ctx, X64Gpr reg);

bool IsWriteError(void* ctx);

} // namespace Common
