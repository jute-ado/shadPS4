// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <span>

#include "common/types.h"

namespace AmdGpu {

struct Gfx7CondExecPacket {
    u64 address{};
    u32 exec_count{};
    bool reserved_control_nonzero{};
    bool legacy_count_matches{};
};

constexpr std::optional<Gfx7CondExecPacket> DecodeGfx7CondExec(std::span<const u32> body) {
    if (body.size() != 4) {
        return std::nullopt;
    }
    constexpr u32 ExecCountMask = (1U << 14) - 1;
    const u64 address = static_cast<u64>(body[1]) << 32 | (body[0] & ~u64{3});
    const u32 exec_count = body[3] & ExecCountMask;
    return Gfx7CondExecPacket{
        .address = address,
        .exec_count = exec_count,
        .reserved_control_nonzero = body[2] != 0,
        .legacy_count_matches = (body[2] & ExecCountMask) == exec_count,
    };
}

struct CondExecDiagnosticReport {
    u32 total_packets{};
    u32 retained_records{};
    u32 truncated_records{};
    u32 false_conditions{};
    u32 layout_mismatches{};
};

class CondExecDiagnostic {
public:
    static constexpr u32 MaxRecords = 32;

    void Observe(const Gfx7CondExecPacket& packet, bool condition) {
        ++report.total_packets;
        report.false_conditions += !condition;
        report.layout_mismatches += !packet.legacy_count_matches;
        if (report.retained_records < MaxRecords) {
            ++report.retained_records;
        } else {
            ++report.truncated_records;
        }
    }

    const CondExecDiagnosticReport& Report() const {
        return report;
    }

private:
    CondExecDiagnosticReport report{};
};

} // namespace AmdGpu
