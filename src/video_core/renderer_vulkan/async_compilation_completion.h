// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>

namespace Vulkan {

// Drain only on the renderer owner. A failed requested result must survive publication:
// publishing a failure logs it but deliberately creates no renderer-visible pipeline.
template <typename Queue, typename Key, typename Publish>
[[nodiscard]] bool DrainAsyncCompilationResults(Queue& queue, const Key* requested_key,
                                                 Publish&& publish) {
    bool requested_failed = false;
    while (auto completion = queue.TryTake()) {
        requested_failed |= requested_key && completion->key == *requested_key && !completion->result;
        publish(std::move(*completion));
    }
    return requested_failed;
}

} // namespace Vulkan
