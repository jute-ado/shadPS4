// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>

#include "common/types.h"

namespace Core {

struct ExecutionStackRange {
    VAddr address;
    u64 size;

    auto operator<=>(const ExecutionStackRange&) const = default;
};

class ExecutionStackRegistry {
public:
    [[nodiscard]] bool Register(VAddr address, u64 size);
    [[nodiscard]] bool Unregister(VAddr address, u64 size);
    [[nodiscard]] bool Overlaps(VAddr address, u64 size) const;
    [[nodiscard]] std::vector<ExecutionStackRange> GetWatchableRanges(VAddr address,
                                                                      u64 size) const;
};

ExecutionStackRegistry& GetExecutionStackRegistry();

} // namespace Core
