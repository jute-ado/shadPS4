// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace VideoCore {

enum class PhysicalBackingTextureConsumer {
    ComputeShaderRead,
    TransferRead,
};

[[nodiscard]] constexpr PhysicalBackingTextureConsumer PhysicalBackingTextureUploadConsumer(
    bool is_tiled) {
    return is_tiled ? PhysicalBackingTextureConsumer::ComputeShaderRead
                    : PhysicalBackingTextureConsumer::TransferRead;
}

template <typename Buffer, typename Synchronize, typename Publish>
void PublishDmaBufferAfterSynchronization(Buffer& buffer, Synchronize&& synchronize,
                                          Publish&& publish) {
    synchronize(buffer, buffer.CpuAddr(), static_cast<u32>(buffer.SizeBytes()));
    publish();
}

template <typename Touch>
void TouchBufferAfterUploadIfRegistered(bool is_registered, Touch&& touch) {
    if (is_registered) {
        touch();
    }
}

} // namespace VideoCore
