// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>

namespace Core {

// The actual VMM calls this seam with copied range identities, not map iterators. Fresh mappings
// take a non-waiting owner lease; destructive operations drop allocator locks while retiring.
template <typename Lock, typename Capture, typename Acquire, typename Matches>
auto AcquireOptimisticMemoryRetirement(Lock& allocator, Capture&& capture, Acquire&& acquire,
                                      Matches&& matches) {
    for (;;) {
        const auto snapshot = capture();
        if (!snapshot.requires_retirement) {
            return acquire(false);
        }
        allocator.unlock();
        auto admission = acquire(true);
        allocator.lock();
        if (!admission || matches(snapshot)) {
            return admission;
        }
        // A replaced allocation or changed alias set is a different transaction. Release the
        // final submission seal before recapturing; unrelated address changes do not retry.
        admission.reset();
    }
}

} // namespace Core
