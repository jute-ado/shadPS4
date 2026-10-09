// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/buffer_cache/buffer.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

UniqueBuffer::UniqueBuffer(vk::Device device_, VmaAllocator allocator_)
    : device{device_}, allocator{allocator_} {}

UniqueBuffer::~UniqueBuffer() {
    Reset();
}

void UniqueBuffer::Reset() noexcept {
    if (buffer) {
        vmaDestroyBuffer(allocator, buffer, allocation);
    }
    buffer = VK_NULL_HANDLE;
    allocation = VK_NULL_HANDLE;
    bda_addr = 0;
}

} // namespace VideoCore
