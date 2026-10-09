// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace VideoCore {

enum class PhysicalBackingMapDispatch {
    Inline,
    GpuQueueOrdered,
    GpuQueueOrderedAndWait,
};

[[nodiscard]] constexpr PhysicalBackingMapDispatch PlanPhysicalBackingMapDispatch(
    bool has_publication_coordinator, bool requires_completion) noexcept {
    if (!has_publication_coordinator) {
        return PhysicalBackingMapDispatch::Inline;
    }
    return requires_completion ? PhysicalBackingMapDispatch::GpuQueueOrderedAndWait
                               : PhysicalBackingMapDispatch::GpuQueueOrdered;
}

} // namespace VideoCore
