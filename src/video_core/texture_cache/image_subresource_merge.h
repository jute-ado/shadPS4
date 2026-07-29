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
    if (!parent_available) {
        return {};
    }

    return {
        .copy_contents = true,
        .rebind_source = source_is_bound || source_is_target,
        .propagate_target = source_is_target,
        .release_source = true,
    };
}

} // namespace VideoCore
