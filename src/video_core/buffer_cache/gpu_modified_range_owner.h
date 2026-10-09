// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/slot_vector.h"
#include "common/types.h"

namespace VideoCore {

// A slot index alone is not a stable GPU-write identity: SlotVector reuses indices after a
// deferred erase, and a live buffer can receive a newer write before an older readback completes.
// The monotonically increasing generation distinguishes both cases.
struct GpuModifiedRangeOwner {
    Common::SlotId buffer_id{};
    u64 generation{};

    constexpr bool operator==(const GpuModifiedRangeOwner&) const noexcept = default;
};

} // namespace VideoCore
