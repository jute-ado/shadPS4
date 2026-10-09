// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <utility>

namespace VideoCore {

enum class ImageGcResult { Stop, Retained, Deleted };

template <typename Download, typename Free>
ImageGcResult CollectImageForGc(std::size_t& remaining, bool download, bool tiled, bool pressured,
                                Download&& download_image, Free&& free_image) {
    if (remaining == 0) {
        return ImageGcResult::Stop;
    }
    // Retained GPU-written images must not hide reclaimable entries later in the LRU.
    // Non-linear downloads remain unsupported, and unpressured downloads stay retained.
    if ((tiled && download) || (download && !pressured)) {
        return ImageGcResult::Retained;
    }
    if (download) {
        std::forward<Download>(download_image)();
    }
    std::forward<Free>(free_image)();
    --remaining;
    return ImageGcResult::Deleted;
}

} // namespace VideoCore
