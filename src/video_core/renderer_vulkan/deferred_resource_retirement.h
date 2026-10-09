// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "common/types.h"
#include "common/unique_function.h"

namespace Vulkan {

/// Holds cleanup callbacks while their command buffer is still being recorded. The scheduler
/// publishes them only after that command buffer has been assigned its exact submission tick.
class SubmissionResourceRetirements {
public:
    void Retain(Common::UniqueFunction<void>&& callback) {
        recording.emplace_back(std::move(callback));
    }

    template <typename PublishCallback>
    void Publish(u64 submission_tick, PublishCallback&& publish_callback) {
        for (auto& callback : recording) {
            publish_callback(std::move(callback), submission_tick);
        }
        recording.clear();
    }

    [[nodiscard]] std::size_t Size() const noexcept {
        return recording.size();
    }

private:
    std::vector<Common::UniqueFunction<void>> recording;
};

} // namespace Vulkan
