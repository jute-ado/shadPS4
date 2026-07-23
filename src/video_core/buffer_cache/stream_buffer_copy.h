// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstring>
#include <type_traits>

#include "common/types.h"

namespace VideoCore {

template <typename Memory, typename Source>
void CopyStreamBufferSource(Memory& memory, Source source, u8* destination, size_t size) {
    if constexpr (std::is_pointer_v<std::remove_cvref_t<Source>>) {
        std::memcpy(destination, source, size);
    } else {
        const VAddr source_address = static_cast<VAddr>(source);
        if (memory.IsValidMapping(source_address)) {
            memory.CopySparseMemory(source_address, destination, size);
        } else {
            std::memcpy(destination, reinterpret_cast<const void*>(source_address), size);
        }
    }
}

} // namespace VideoCore
