// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>
#include <queue>

#include "common/unique_function.h"

namespace AmdGpu {

class SubmissionBoundaryQueue {
public:
    void Push(Common::UniqueFunction<void>&& completion) {
        completions.push(std::move(completion));
    }

    [[nodiscard]] bool Empty() const noexcept {
        return completions.empty();
    }

    Common::UniqueFunction<void> Pop() {
        auto completion = std::move(completions.front());
        completions.pop();
        return completion;
    }

private:
    std::queue<Common::UniqueFunction<void>> completions;
};

} // namespace AmdGpu
