// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <span>

#include "common/types.h"

namespace VideoCore {

inline constexpr u32 GpuFaultSlotBytes = 8192;
inline constexpr u32 MaxGpuFaultRecords = GpuFaultSlotBytes / sizeof(u64) - 1;

struct GpuFaultRecordPlan {
    u32 read_count;
    u32 reported_count;
    bool truncated;
};

[[nodiscard]] constexpr std::optional<GpuFaultRecordPlan> PlanGpuFaultRecords(u64 slot_size,
                                                                              u32 reported_count) {
    if (slot_size != GpuFaultSlotBytes) {
        return std::nullopt;
    }
    return GpuFaultRecordPlan{std::min(reported_count, MaxGpuFaultRecords), reported_count,
                              reported_count > MaxGpuFaultRecords};
}

struct GpuFaultRecordSnapshot {
    GpuFaultRecordPlan plan;
    std::array<u64, MaxGpuFaultRecords> records{};

    [[nodiscard]] std::span<const u64> Records() const {
        return {records.data(), plan.read_count};
    }
};

template <typename Invalidate>
[[nodiscard]] std::optional<GpuFaultRecordSnapshot> CaptureGpuFaultRecords(
    std::span<const u8> slot, Invalidate&& invalidate) {
    if (slot.size() != GpuFaultSlotBytes || !invalidate()) {
        return std::nullopt;
    }
    // The shader increments only the low DWORD of the reserved first QWORD. Copy rather than
    // dereference typed pointers so the decoder also accepts an unaligned byte view.
    u32 reported_count;
    std::memcpy(&reported_count, slot.data(), sizeof(reported_count));
    GpuFaultRecordSnapshot snapshot{*PlanGpuFaultRecords(slot.size(), reported_count)};
    std::memcpy(snapshot.records.data(), slot.data() + sizeof(u64),
                snapshot.plan.read_count * sizeof(u64));
    return snapshot;
}

} // namespace VideoCore
