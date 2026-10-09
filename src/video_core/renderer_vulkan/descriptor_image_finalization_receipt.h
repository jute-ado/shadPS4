// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

namespace Vulkan {

[[nodiscard]] constexpr bool ParseDescriptorImageFinalizationReceipt(
    std::string_view value) noexcept {
    return value == "1";
}

struct DescriptorImageFinalizationReceiptPolicy {
    bool enabled{};
    bool valid{true};
    std::uint64_t start_ms{};
    std::uint64_t duration_ms{};

    [[nodiscard]] static DescriptorImageFinalizationReceiptPolicy Parse(
        std::optional<std::string_view> enable, std::optional<std::string_view> start,
        std::optional<std::string_view> duration) noexcept {
        if (!enable || *enable == "0") {
            return {};
        }
        DescriptorImageFinalizationReceiptPolicy result{.valid = false};
        const auto number = [](std::optional<std::string_view> text, std::uint64_t& value) {
            if (!text || text->empty() || text->front() < '0' || text->front() > '9') {
                return false;
            }
            const auto [end, error] =
                std::from_chars(text->data(), text->data() + text->size(), value);
            return error == std::errc{} && end == text->data() + text->size();
        };
        if (!ParseDescriptorImageFinalizationReceipt(*enable) || !number(start, result.start_ms) ||
            !number(duration, result.duration_ms) || result.start_ms > 600000 ||
            result.duration_ms == 0 || result.duration_ms > 5000) {
            return result;
        }
        result.enabled = result.valid = true;
        return result;
    }
};

enum class DescriptorImageFinalizationFailure : std::uint32_t {
    StaleGeneration,
    NeedsRebind,
    ViewMissing,
    DrawDropped,
    PendingIndexOutOfBounds,
};

[[nodiscard]] constexpr std::string_view DescriptorImageFinalizationFailureName(
    DescriptorImageFinalizationFailure failure) noexcept {
    switch (failure) {
    case DescriptorImageFinalizationFailure::StaleGeneration:
        return "stale_generation";
    case DescriptorImageFinalizationFailure::NeedsRebind:
        return "needs_rebind";
    case DescriptorImageFinalizationFailure::ViewMissing:
        return "view_missing";
    case DescriptorImageFinalizationFailure::DrawDropped:
        return "draw_dropped";
    case DescriptorImageFinalizationFailure::PendingIndexOutOfBounds:
        return "pending_index_oob";
    }
    return "unknown";
}

// Ordered to report the earliest point at which final descriptor resolution cannot
// establish a usable current view. This is classification only; it does not repair.
[[nodiscard]] constexpr std::optional<DescriptorImageFinalizationFailure>
ClassifyDescriptorImageFinalization(bool pending_index_in_bounds, bool generation_current,
                                    bool needs_rebind, bool view_present) noexcept {
    if (!pending_index_in_bounds) {
        return DescriptorImageFinalizationFailure::PendingIndexOutOfBounds;
    }
    if (!generation_current) {
        return DescriptorImageFinalizationFailure::StaleGeneration;
    }
    if (needs_rebind) {
        return DescriptorImageFinalizationFailure::NeedsRebind;
    }
    if (!view_present) {
        return DescriptorImageFinalizationFailure::ViewMissing;
    }
    return std::nullopt;
}

enum class DescriptorImageCommandKind : std::uint32_t {
    DirectDraw,
    IndexedDraw,
    IndirectDraw,
    IndexedIndirectDraw,
    IndirectCountDraw,
    IndexedIndirectCountDraw,
    DirectDispatch,
    IndirectDispatch,
};

[[nodiscard]] constexpr std::string_view DescriptorImageCommandKindName(
    DescriptorImageCommandKind kind) noexcept {
    switch (kind) {
    case DescriptorImageCommandKind::DirectDraw:
        return "draw";
    case DescriptorImageCommandKind::IndexedDraw:
        return "draw_indexed";
    case DescriptorImageCommandKind::IndirectDraw:
        return "draw_indirect";
    case DescriptorImageCommandKind::IndexedIndirectDraw:
        return "draw_indexed_indirect";
    case DescriptorImageCommandKind::IndirectCountDraw:
        return "draw_indirect_count";
    case DescriptorImageCommandKind::IndexedIndirectCountDraw:
        return "draw_indexed_indirect_count";
    case DescriptorImageCommandKind::DirectDispatch:
        return "dispatch";
    case DescriptorImageCommandKind::IndirectDispatch:
        return "dispatch_indirect";
    }
    return "unknown";
}

struct DescriptorImageCommandContext {
    DescriptorImageCommandKind kind{};
    std::uint64_t vertex_hash{};
    std::uint64_t fragment_hash{};
    std::uint64_t compute_hash{};

    bool operator==(const DescriptorImageCommandContext&) const = default;
};

struct DescriptorImageFinalizationObservation {
    static constexpr std::uint32_t InvalidIndex = std::numeric_limits<std::uint32_t>::max();

    DescriptorImageFinalizationFailure failure{};
    DescriptorImageCommandContext command{};
    std::uint32_t descriptor_index{InvalidIndex};
    std::uint32_t pending_index{InvalidIndex};
    std::uint32_t image_id{InvalidIndex};
    std::uint64_t image_uid{};

    bool operator==(const DescriptorImageFinalizationObservation&) const = default;
};

// The rasterizer GPU-owner path is serialized. Keep the first unique owned tuples,
// never guest pointers, and stop all builder work at a deterministic fixed cap.
class DescriptorImageFinalizationReceipt {
public:
    static constexpr std::size_t Capacity = 32;
    static constexpr std::size_t FailureKindCount = 5;

    explicit DescriptorImageFinalizationReceipt(
        DescriptorImageFinalizationReceiptPolicy policy_ = {})
        : policy{policy_} {}

    [[nodiscard]] bool Enabled() const noexcept {
        return policy.enabled && policy.valid && !closed && count < Capacity;
    }

    template <typename BuildObservation>
    [[nodiscard]] std::optional<DescriptorImageFinalizationObservation> Observe(
        std::uint64_t elapsed_ms, BuildObservation&& build) {
        if (!Enabled() || elapsed_ms < policy.start_ms) {
            return std::nullopt;
        }
        if (elapsed_ms >= policy.start_ms + policy.duration_ms) {
            closed = true;
            return std::nullopt;
        }
        if (count >= Capacity) {
            return std::nullopt;
        }
        const DescriptorImageFinalizationObservation observation = build();
        const auto failure_index = static_cast<std::size_t>(observation.failure);
        if (failure_index < FailureKindCount) {
            ++failure_occurrences[failure_index];
        }
        for (std::size_t index = 0; index < count; ++index) {
            if (observations[index] == observation) {
                return std::nullopt;
            }
        }
        observations[count++] = observation;
        return observation;
    }

    [[nodiscard]] std::size_t RecordedCount() const noexcept {
        return count;
    }

    // Counts every classified failure in the bounded observation window, including duplicate
    // tuples that are intentionally coalesced in the retained receipt observations.
    [[nodiscard]] std::uint64_t FailureOccurrences(
        DescriptorImageFinalizationFailure failure) const noexcept {
        const auto index = static_cast<std::size_t>(failure);
        return index < FailureKindCount ? failure_occurrences[index] : 0;
    }

private:
    DescriptorImageFinalizationReceiptPolicy policy;
    std::size_t count{};
    std::array<DescriptorImageFinalizationObservation, Capacity> observations{};
    std::array<std::uint64_t, FailureKindCount> failure_occurrences{};
    bool closed{};
};

} // namespace Vulkan
