// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstring>

#include "common/types.h"

namespace VideoCore {

template <typename StreamBuffer>
u64 AllocateZeroedStreamRegion(StreamBuffer& buffer, u64 size, u64 alignment) {
    const auto [data, offset] = buffer.Map(size, alignment);
    std::memset(data, 0, size);
    buffer.Commit();
    return offset;
}

} // namespace VideoCore
