// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/execution_stack_registry.h"

namespace Core {

bool ExecutionStackRegistry::Register(VAddr address, u64 size) {
    return false;
}

bool ExecutionStackRegistry::Unregister(VAddr address, u64 size) {
    return false;
}

bool ExecutionStackRegistry::Overlaps(VAddr address, u64 size) const {
    return false;
}

std::vector<ExecutionStackRange> ExecutionStackRegistry::GetWatchableRanges(VAddr address,
                                                                             u64 size) const {
    return size == 0 ? std::vector<ExecutionStackRange>{}
                     : std::vector<ExecutionStackRange>{{address, size}};
}

ExecutionStackRegistry& GetExecutionStackRegistry() {
    static ExecutionStackRegistry registry;
    return registry;
}

} // namespace Core
