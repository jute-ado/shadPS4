// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <utility>

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

struct ImageTextureObservation {
    ImageRefreshObservation refresh{};
    bool view_owner_exact{};
};

struct ImageRefreshStartPlan {
    ImageRefreshResult result{ImageRefreshResult::Clean};
    bool refresh{};
};

template <typename Refresh, typename FindView>
decltype(auto) RefreshAndFindTextureOnce(Refresh&& refresh, FindView&& find_view,
                                         ImageTextureObservation* observation) {
    const ImageRefreshObservation refresh_observation =
        std::invoke(std::forward<Refresh>(refresh));
    bool view_owner_exact = false;
    decltype(auto) view = std::invoke(std::forward<FindView>(find_view),
                                     observation != nullptr ? &view_owner_exact : nullptr);
    if (observation != nullptr) {
        *observation = {
            .refresh = refresh_observation,
            .view_owner_exact = view_owner_exact,
        };
    }
    return view;
}

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
