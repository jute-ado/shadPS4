// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>

namespace Vulkan {

enum class PipelineBindPoint {
    Graphics,
    Compute,
};

template <typename Handle>
class PipelineBindTracker {
public:
    [[nodiscard]] bool NeedsBind(PipelineBindPoint point, Handle handle) {
        auto& current = point == PipelineBindPoint::Graphics ? graphics : compute;
        if (current && *current == handle) {
            return false;
        }
        current = handle;
        return true;
    }

    void Reset() {
        graphics.reset();
        compute.reset();
    }

private:
    std::optional<Handle> graphics;
    std::optional<Handle> compute;
};

} // namespace Vulkan
