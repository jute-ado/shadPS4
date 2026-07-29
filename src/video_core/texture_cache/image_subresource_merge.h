// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace VideoCore {

struct ImageSubresourceMergePlan {
    bool copy_contents;
    bool rebind_source;
    bool propagate_target;
    bool release_source;
};

constexpr ImageSubresourceMergePlan PlanImageSubresourceMerge(const bool source_is_bound,
                                                              const bool source_is_target,
                                                              const bool parent_available) {
    if (source_is_target) {
        return {
            .copy_contents = false,
            .rebind_source = true,
            .propagate_target = parent_available,
            .release_source = true,
        };
    }

    if (parent_available) {
        return {
            .copy_contents = true,
            .rebind_source = false,
            .propagate_target = false,
            .release_source = true,
        };
    }

    return {};
}

} // namespace VideoCore
