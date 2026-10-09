// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>

#include "common/types.h"

namespace VideoCore {

template <typename Handle>
struct BufferCopyEndpoint {
    Handle handle;
    u64 offset;
};

template <typename PrepareSource, typename PrepareDestination,
          typename ResolveSource, typename ResolveDestination>
auto PrepareBufferCopyEndpoints(VAddr source_address, VAddr destination_address,
                                PrepareSource&& prepare_source,
                                PrepareDestination&& prepare_destination,
                                ResolveSource&& resolve_source,
                                ResolveDestination&& resolve_destination) {
    // Preparation may relocate slot storage, merge either buffer, and wait for a
    // submission that retires the superseded Vulkan handles. Resolve the current
    // registered owners only after both preparations, without allocating or waiting.
    prepare_source();
    prepare_destination();
    const auto& source = resolve_source();
    const auto source_endpoint = BufferCopyEndpoint{source.Handle(), source.Offset(source_address)};
    const auto& destination = resolve_destination();
    return std::pair{
        source_endpoint,
        BufferCopyEndpoint{destination.Handle(), destination.Offset(destination_address)}};
}

} // namespace VideoCore
