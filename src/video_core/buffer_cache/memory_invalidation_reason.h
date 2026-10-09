// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace VideoCore {

enum class MemoryInvalidationReason {
    CpuWrite,
    MappingChange,
};

} // namespace VideoCore
