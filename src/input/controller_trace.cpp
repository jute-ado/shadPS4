// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "controller_trace.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>

namespace Input {
namespace {

using namespace std::string_view_literals;

constexpr u64 MaximumSamples = 1'000'000;
constexpr u64 MaximumElapsedMilliseconds = 7ULL * 24 * 60 * 60 * 1000;

constexpr std::array CanonicalButtons{
    std::pair{Libraries::Pad::OrbisPadButtonDataOffset::Up, "dpad_up"sv},
    std::pair{Libraries::Pad::OrbisPadButtonDataOffset::Right, "dpad_right"sv},
    std::pair{Libraries::Pad::OrbisPadButtonDataOffset::Down, "dpad_down"sv},
    std::pair{Libraries::Pad::OrbisPadButtonDataOffset::Left, "dpad_left"sv},
    std::pair{Libraries::Pad::OrbisPadButtonDataOffset::Square, "square"sv},
    std::pair{Libraries::Pad::OrbisPadButtonDataOffset::Cross, "cross"sv},
    std::pair{Libraries::Pad::OrbisPadButtonDataOffset::Circle, "circle"sv},
    std::pair{Libraries::Pad::OrbisPadButtonDataOffset::Triangle, "triangle"sv},
    std::pair{Libraries::Pad::OrbisPadButtonDataOffset::L1, "l1"sv},
    std::pair{Libraries::Pad::OrbisPadButtonDataOffset::L2, "l2"sv},
    std::pair{Libraries::Pad::OrbisPadButtonDataOffset::R1, "r1"sv},
    std::pair{Libraries::Pad::OrbisPadButtonDataOffset::R2, "r2"sv},
    std::pair{Libraries::Pad::OrbisPadButtonDataOffset::L3, "l3"sv},
    std::pair{Libraries::Pad::OrbisPadButtonDataOffset::R3, "r3"sv},
    std::pair{Libraries::Pad::OrbisPadButtonDataOffset::Options, "options"sv},
    std::pair{Libraries::Pad::OrbisPadButtonDataOffset::TouchPad, "touchpad"sv},
};

int NormalizeStick(s32 value) {
    value = std::clamp(value, 0, 255);
    if (value <= 128) {
        return (value - 128) * 256;
    }
    const auto delta = value - 128;
    return (delta * 32'767 + 63) / 127;
}

u32 NormalizeCoordinate(u16 value, u32 maximum) {
    const auto clamped = std::min<u32>(value, maximum);
    return (clamped * 65'535 + maximum / 2) / maximum;
}

class PrimaryControllerTrace {
public:
    PrimaryControllerTrace() {
        const auto* configured_path = std::getenv("SHADPS4_TEST_LAB_INPUT_TRACE");
        if (configured_path == nullptr || *configured_path == '\0') {
            return;
        }

        const std::filesystem::path path{configured_path};
        if (std::filesystem::exists(path)) {
            throw std::runtime_error("Controller input trace already exists.");
        }
        const auto parent = path.parent_path();
        if (!parent.empty() && !std::filesystem::is_directory(parent)) {
            throw std::runtime_error("Controller input trace directory does not exist.");
        }

        file.open(path, std::ios::out | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Controller input trace could not be created.");
        }
        writer = std::make_unique<ControllerTraceWriter>(file);
        started = std::chrono::steady_clock::now();
    }

    void Record(const State& state) {
        std::scoped_lock lock{mutex};
        if (!writer) {
            return;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
        writer->Record(state, static_cast<u64>(std::max<int64_t>(0, elapsed)));
    }

private:
    std::mutex mutex;
    std::ofstream file;
    std::unique_ptr<ControllerTraceWriter> writer;
    std::chrono::steady_clock::time_point started;
};

PrimaryControllerTrace& GetPrimaryTrace() {
    static PrimaryControllerTrace trace;
    return trace;
}

} // namespace

ControllerTraceWriter::ControllerTraceWriter(std::ostream& output_) : output{output_} {
    output << R"({"kind":"header","protocolVersion":1,"profile":"dualshock4",)"
              R"("clock":"elapsed_milliseconds"})"
           << '\n';
    output.flush();
}

void ControllerTraceWriter::Record(const State& state, u64 elapsed_milliseconds) {
    const auto snapshot = Capture(state);
    if (previous == snapshot) {
        return;
    }
    if (sample_count == MaximumSamples) {
        throw std::runtime_error("Controller input trace sample bound exceeded.");
    }

    auto offset = elapsed_milliseconds;
    if (previous_offset && offset <= *previous_offset) {
        offset = *previous_offset + 1;
    }
    if (offset > MaximumElapsedMilliseconds) {
        throw std::runtime_error("Controller input trace time bound exceeded.");
    }

    WriteSample(snapshot, offset);
    previous = snapshot;
    previous_offset = offset;
    sample_count++;
}

ControllerTraceWriter::Snapshot ControllerTraceWriter::Capture(const State& state) {
    Snapshot snapshot{
        .buttons = std::to_underlying(state.buttonsState),
        .axes = state.axes,
    };
    std::copy(std::begin(state.touchpad), std::end(state.touchpad), snapshot.touches.begin());
    return snapshot;
}

void ControllerTraceWriter::WriteSample(const Snapshot& state, u64 offset) {
    output << R"({"kind":"sample","protocolVersion":1,"offset":)" << offset
           << R"(,"state":{"buttons":[)";
    bool first = true;
    for (const auto& [button, name] : CanonicalButtons) {
        if ((state.buttons & std::to_underlying(button)) == 0) {
            continue;
        }
        if (!first) {
            output << ',';
        }
        first = false;
        output << '"' << name << '"';
    }
    output << R"(],"leftX":)" << NormalizeStick(state.axes[std::to_underlying(Axis::LeftX)])
           << R"(,"leftY":)" << NormalizeStick(state.axes[std::to_underlying(Axis::LeftY)])
           << R"(,"rightX":)" << NormalizeStick(state.axes[std::to_underlying(Axis::RightX)])
           << R"(,"rightY":)" << NormalizeStick(state.axes[std::to_underlying(Axis::RightY)])
           << R"(,"leftTrigger":)"
           << std::clamp(state.axes[std::to_underlying(Axis::TriggerLeft)], 0, 255) * 257
           << R"(,"rightTrigger":)"
           << std::clamp(state.axes[std::to_underlying(Axis::TriggerRight)], 0, 255) * 257
           << R"(,"touches":[)";
    first = true;
    for (u32 index = 0; index < state.touches.size(); index++) {
        const auto& touch = state.touches[index];
        if (!touch.state) {
            continue;
        }
        if (!first) {
            output << ',';
        }
        first = false;
        output << R"({"id":)" << index << R"(,"x":)" << NormalizeCoordinate(touch.x, 1919)
               << R"(,"y":)" << NormalizeCoordinate(touch.y, 941) << '}';
    }
    output << "]}}\n";
    output.flush();
}

void RecordPrimaryControllerState(const State& state) {
    GetPrimaryTrace().Record(state);
}

} // namespace Input
