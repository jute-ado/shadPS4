// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace VideoCore {

enum class ImageRefreshResult : u8 {
    Clean,
    MultisampledDirty,
    MaybeCpuDirtyUnchanged,
    GpuModifiedUnchanged,
    Uploaded,
};

struct ImageRefreshObservation {
    ImageRefreshResult result{ImageRefreshResult::Clean};
    bool gpu_modified_before{};
};

struct ImageRefreshStartPlan {
    ImageRefreshResult result{ImageRefreshResult::Clean};
    bool refresh{};
};

[[nodiscard]] constexpr ImageRefreshStartPlan PlanImageRefreshStart(bool dirty,
                                                                    u32 samples) noexcept {
    if (!dirty) {
        return {};
    }
    if (samples > 1) {
        return {.result = ImageRefreshResult::MultisampledDirty};
    }
    return {.refresh = true};
}

[[nodiscard]] constexpr ImageRefreshResult CompleteImageRefresh(bool copies_empty) noexcept {
    return copies_empty ? ImageRefreshResult::GpuModifiedUnchanged : ImageRefreshResult::Uploaded;
}

} // namespace VideoCore
