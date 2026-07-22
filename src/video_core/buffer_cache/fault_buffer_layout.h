// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace VideoCore {

constexpr u32 FaultDownloadEntries = 1024;
constexpr u32 MaxStoredPageFaults = FaultDownloadEntries - 1;
constexpr u64 PageFaultAreaSize = u64{FaultDownloadEntries} * sizeof(u64);

[[nodiscard]] constexpr u32 BoundFaultCount(u64 reported_count) {
    return reported_count <= MaxStoredPageFaults ? static_cast<u32>(reported_count)
                                                 : MaxStoredPageFaults;
}

} // namespace VideoCore
