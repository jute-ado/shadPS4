// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>

namespace VideoCore {

template <typename BindPipeline, typename CommitDynamicState, typename Draw>
void IssueDrawWithVertexInputState(bool uses_dynamic_state, BindPipeline&& bind_pipeline,
                                   CommitDynamicState&& commit_dynamic_state, Draw&& draw) {
    std::forward<BindPipeline>(bind_pipeline)();
    if (uses_dynamic_state) {
        std::forward<CommitDynamicState>(commit_dynamic_state)();
    }
    std::forward<Draw>(draw)();
}

} // namespace VideoCore
