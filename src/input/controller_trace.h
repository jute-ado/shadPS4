// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <iosfwd>
#include <optional>

#include "input/controller.h"

namespace Input {

class ControllerTraceWriter {
public:
    static constexpr int ProtocolVersion = 1;

    explicit ControllerTraceWriter(std::ostream& output);

    void Record(const State& state, u64 elapsed_milliseconds);

private:
    struct Snapshot {
        u32 buttons{};
        std::array<s32, std::to_underlying(Axis::AxisMax)> axes{};
        std::array<TouchpadEntry, 2> touches{};

        bool operator==(const Snapshot&) const = default;
    };

    static Snapshot Capture(const State& state);
    void WriteSample(const Snapshot& state, u64 offset);

    std::ostream& output;
    std::optional<Snapshot> previous;
    std::optional<u64> previous_offset;
    u32 sample_count{};
};

void RecordPrimaryControllerState(const State& state);

} // namespace Input
