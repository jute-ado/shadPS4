// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>

#include <boost/container/small_vector.hpp>

#include "common/types.h"
#include "video_core/buffer_cache/region_definitions.h"

namespace VideoCore {

// A normal SRT read captures at most one 4 KiB host-tracking page. Include a couple of words for
// the alignment padding used by PlanCpuSnapshotCapture. Larger public ReadMemorySnapshot requests
// transparently use small_vector's heap fallback rather than truncating the capture.
inline constexpr std::size_t SrtSnapshotCaptureInlineBytes =
    static_cast<std::size_t>(TRACKER_BYTES_PER_PAGE + 2 * sizeof(u32));

using SrtSnapshotCaptureBuffer =
    boost::container::small_vector<u8, SrtSnapshotCaptureInlineBytes>;

} // namespace VideoCore
