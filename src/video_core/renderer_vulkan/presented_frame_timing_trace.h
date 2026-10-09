// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace Vulkan {

struct PresentedFrameTimingTraceRequest {
    std::filesystem::path path;
    std::int64_t start_frame;
    std::int32_t sample_count;

    static std::optional<PresentedFrameTimingTraceRequest> Parse(
        const std::string_view path_value, const std::string_view start_value,
        const std::string_view count_value) {
        constexpr std::int64_t MaxPresentedFrame = 50'000'000;
        constexpr std::int32_t MaxSampleCount = 1'000'001;
        std::int64_t start{};
        std::int32_t count{};
        const auto start_result =
            std::from_chars(start_value.data(), start_value.data() + start_value.size(), start);
        const auto count_result =
            std::from_chars(count_value.data(), count_value.data() + count_value.size(), count);
        if (path_value.empty() || start_result.ec != std::errc{} ||
            start_result.ptr != start_value.data() + start_value.size() ||
            count_result.ec != std::errc{} ||
            count_result.ptr != count_value.data() + count_value.size() || start < 1 ||
            start > MaxPresentedFrame || count < 2 || count > MaxSampleCount ||
            start + count - 1 > MaxPresentedFrame) {
            return std::nullopt;
        }
        return PresentedFrameTimingTraceRequest{
            .path = std::filesystem::path{path_value},
            .start_frame = start,
            .sample_count = count,
        };
    }
};

class PresentedFrameTimingTrace {
public:
    explicit PresentedFrameTimingTrace(const PresentedFrameTimingTraceRequest& request)
        : start_frame{request.start_frame},
          final_frame{request.start_frame + request.sample_count - 1} {
        if (std::filesystem::exists(request.path)) {
            throw std::runtime_error{"Performance trace already exists"};
        }
        output.open(request.path, std::ios::out | std::ios::binary);
        if (!output) {
            throw std::runtime_error{"Cannot create performance trace"};
        }
        output << R"({"kind":"header","protocolVersion":1,"source":"presented_frame","clock":"monotonic_nanoseconds"})"
               << '\n';
    }

    PresentedFrameTimingTrace(const PresentedFrameTimingTrace&) = delete;
    PresentedFrameTimingTrace& operator=(const PresentedFrameTimingTrace&) = delete;

    static std::unique_ptr<PresentedFrameTimingTrace> CreateFromEnvironment() {
        const auto* path = std::getenv("SHADPS4_PERFORMANCE_TRACE_PATH");
        const auto* start = std::getenv("SHADPS4_PERFORMANCE_TRACE_START_FRAME");
        const auto* count = std::getenv("SHADPS4_PERFORMANCE_TRACE_SAMPLE_COUNT");
        if (path == nullptr && start == nullptr && count == nullptr) {
            return {};
        }
        if (path == nullptr || start == nullptr || count == nullptr) {
            std::cerr << "[TestLab] Invalid shadPS4 performance trace request.\n";
            return {};
        }
        const auto request = PresentedFrameTimingTraceRequest::Parse(path, start, count);
        if (!request.has_value()) {
            std::cerr << "[TestLab] Invalid shadPS4 performance trace request.\n";
            return {};
        }
        try {
            return std::make_unique<PresentedFrameTimingTrace>(*request);
        } catch (const std::exception&) {
            std::cerr << "[TestLab] Cannot create shadPS4 performance trace.\n";
            return {};
        }
    }

    void Record(const std::int64_t presented_frame,
                const std::int64_t timestamp_nanoseconds) {
        if (completed || presented_frame < start_frame || presented_frame > final_frame) {
            return;
        }
        if (last_written_frame >= 0 && presented_frame != last_written_frame + 1) {
            return;
        }
        output << R"({"kind":"sample","protocolVersion":1,"presentedFrame":)"
               << presented_frame << R"(,"timestampNanoseconds":)" << timestamp_nanoseconds
               << "}\n";
        last_written_frame = presented_frame;
        if (presented_frame == final_frame) {
            output.flush();
            completed = true;
        }
    }

    static std::int64_t MonotonicNanoseconds() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

private:
    std::ofstream output;
    std::int64_t start_frame;
    std::int64_t final_frame;
    std::int64_t last_written_frame{-1};
    bool completed{};
};

} // namespace Vulkan
