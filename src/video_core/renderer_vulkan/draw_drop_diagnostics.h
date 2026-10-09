// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

#include "video_core/renderer_vulkan/descriptor_image_finalization_receipt.h"

namespace Vulkan {

[[nodiscard]] constexpr bool ParseDrawDropDiagnostics(std::string_view value) noexcept {
    return value == "1";
}

struct DrawDropDiagnosticsPolicy {
    bool enabled{};
    bool valid{true};
    std::uint64_t start_ms{};
    std::uint64_t duration_ms{};

    [[nodiscard]] static DrawDropDiagnosticsPolicy Parse(
        std::optional<std::string_view> enable, std::optional<std::string_view> start,
        std::optional<std::string_view> duration) noexcept {
        if (!enable || *enable == "0") {
            return {};
        }

        DrawDropDiagnosticsPolicy result{.valid = false};
        const auto number = [](std::optional<std::string_view> text,
                               std::uint64_t& value) noexcept {
            if (!text || text->empty() || text->front() < '0' || text->front() > '9') {
                return false;
            }
            const auto [end, error] =
                std::from_chars(text->data(), text->data() + text->size(), value);
            return error == std::errc{} && end == text->data() + text->size();
        };
        if (!ParseDrawDropDiagnostics(*enable) || !number(start, result.start_ms) ||
            !number(duration, result.duration_ms) || result.start_ms > 600000 ||
            result.duration_ms == 0 || result.duration_ms > 5000) {
            return result;
        }
        result.enabled = result.valid = true;
        return result;
    }
};

enum class DrawDropReason : std::uint32_t {
    NullPipeline,
    ResourceBinding,
};

[[nodiscard]] constexpr std::string_view DrawDropReasonName(DrawDropReason reason) noexcept {
    switch (reason) {
    case DrawDropReason::NullPipeline:
        return "null_pipeline";
    case DrawDropReason::ResourceBinding:
        return "resource_binding";
    }
    return "unknown";
}

struct DrawDropDiagnosticsSummary {
    static constexpr std::size_t ReasonCount = 2;
    static constexpr std::size_t CommandCount =
        static_cast<std::size_t>(DescriptorImageCommandKind::IndirectDispatch) + 1;

    std::uint64_t total{};
    bool saturated{};
    std::array<std::array<std::uint64_t, CommandCount>, ReasonCount> counts{};

    [[nodiscard]] constexpr std::uint64_t Count(
        DrawDropReason reason, DescriptorImageCommandKind command) const noexcept {
        const auto reason_index = static_cast<std::size_t>(reason);
        const auto command_index = static_cast<std::size_t>(command);
        if (reason_index >= counts.size() || command_index >= counts[reason_index].size()) {
            return 0;
        }
        return counts[reason_index][command_index];
    }
};

// A launch-time diagnostic only. It retains no guest pointers and stops after a fixed number of
// events, while preserving enough command-kind counts to distinguish missing pipelines from
// bindings that could not be published. Production behavior is unchanged when disabled.
class DrawDropDiagnostics {
public:
    static constexpr std::size_t MaxEvents = 4096;

    explicit DrawDropDiagnostics(DrawDropDiagnosticsPolicy policy_ = {}) : policy{policy_} {}

    [[nodiscard]] bool Enabled() const noexcept {
        return policy.enabled && policy.valid && !closed;
    }

    [[nodiscard]] bool Armed() const noexcept {
        return policy.enabled && policy.valid;
    }

    [[nodiscard]] bool Observe(std::uint64_t elapsed_ms, DrawDropReason reason,
                               DescriptorImageCommandContext command) noexcept {
        if (!Enabled() || elapsed_ms < policy.start_ms) {
            return false;
        }
        if (elapsed_ms >= policy.start_ms + policy.duration_ms) {
            closed = true;
            return false;
        }
        if (total >= MaxEvents) {
            saturated = true;
            return false;
        }
        const auto reason_index = static_cast<std::size_t>(reason);
        const auto command_index = static_cast<std::size_t>(command.kind);
        if (reason_index >= counts.size() || command_index >= counts[reason_index].size()) {
            return false;
        }
        ++total;
        ++counts[reason_index][command_index];
        return true;
    }

    [[nodiscard]] DrawDropDiagnosticsSummary Close(std::uint64_t /*elapsed_ms*/) noexcept {
        closed = true;
        return {
            .total = total,
            .saturated = saturated,
            .counts = counts,
        };
    }

private:
    DrawDropDiagnosticsPolicy policy;
    std::uint64_t total{};
    std::array<std::array<std::uint64_t, DrawDropDiagnosticsSummary::CommandCount>,
               DrawDropDiagnosticsSummary::ReasonCount>
        counts{};
    bool saturated{};
    bool closed{};
};

} // namespace Vulkan
