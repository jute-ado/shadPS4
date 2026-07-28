// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <span>

#include "common/types.h"

namespace VideoCore {

struct ImageCandidateMatch {
    bool exact{};
    s32 parent_mip{-1};
    s32 parent_slice{-1};
};

struct ImageCandidateSelection {
    size_t index{};
    s32 mip{-1};
    s32 slice{-1};
};

inline std::optional<ImageCandidateSelection> SelectCanonicalImageCandidate(
    std::span<const ImageCandidateMatch> candidates) {
    for (size_t index = 0; index < candidates.size(); ++index) {
        const auto& candidate = candidates[index];
        if (candidate.parent_mip >= 0 && candidate.parent_slice >= 0) {
            return ImageCandidateSelection{
                .index = index,
                .mip = candidate.parent_mip,
                .slice = candidate.parent_slice,
            };
        }
    }
    for (size_t index = 0; index < candidates.size(); ++index) {
        if (candidates[index].exact) {
            return ImageCandidateSelection{.index = index};
        }
    }
    return std::nullopt;
}

} // namespace VideoCore
