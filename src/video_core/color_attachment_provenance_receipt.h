// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace VideoCore {

struct ColorAttachmentProvenancePoint {
    std::uint16_t x_permille{};
    std::uint16_t y_permille{};
    auto operator<=>(const ColorAttachmentProvenancePoint&) const = default;
};

struct ColorAttachmentProvenancePolicy {
    static constexpr std::size_t MaxPoints = 4;
    static constexpr std::size_t MaxSelectors = 8;
    static constexpr std::uint32_t MaxCommandCapacity = 64;

    bool enabled{};
    bool valid{true};
    std::uint64_t start_ms{};
    std::uint64_t duration_ms{};
    std::uint32_t command_capacity{};
    std::array<ColorAttachmentProvenancePoint, MaxPoints> points{};
    std::size_t point_count{};
    std::array<std::uint64_t, MaxSelectors> selectors{};
    std::size_t selector_count{};

    [[nodiscard]] bool Selects(const std::uint64_t selector) const noexcept {
        for (std::size_t index = 0; index < selector_count; ++index) {
            if (selectors[index] == selector) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] static ColorAttachmentProvenancePolicy Parse(
        const std::optional<std::string_view> enable, const std::optional<std::string_view> start,
        const std::optional<std::string_view> duration,
        const std::optional<std::string_view> point_list,
        const std::optional<std::string_view> selector_list,
        const std::optional<std::string_view> capacity) noexcept {
        // The disabled path deliberately ignores every other field. This keeps the
        // diagnostic completely inert unless it is explicitly requested.
        if (!enable || *enable == "0") {
            return {};
        }

        ColorAttachmentProvenancePolicy result{.valid = false};
        if (*enable != "1" || !ParseDecimal(start, result.start_ms) ||
            !ParseDecimal(duration, result.duration_ms) ||
            !ParseDecimal(capacity, result.command_capacity) || result.start_ms > 600000 ||
            result.duration_ms == 0 || result.duration_ms > 5000 || result.command_capacity == 0 ||
            result.command_capacity > MaxCommandCapacity || !ParsePoints(point_list, result) ||
            !ParseSelectors(selector_list, result)) {
            return result;
        }

        result.enabled = true;
        result.valid = true;
        return result;
    }

private:
    template <typename Integer>
    [[nodiscard]] static bool ParseDecimal(const std::optional<std::string_view> text,
                                           Integer& value) noexcept {
        if (!text || text->empty() || text->front() < '0' || text->front() > '9') {
            return false;
        }
        const auto [end, error] =
            std::from_chars(text->data(), text->data() + text->size(), value, 10);
        return error == std::errc{} && end == text->data() + text->size();
    }

    template <typename Integer>
    [[nodiscard]] static bool ParseDecimal(const std::string_view text, Integer& value) noexcept {
        return ParseDecimal(std::optional<std::string_view>{text}, value);
    }

    [[nodiscard]] static bool ParsePoints(const std::optional<std::string_view> text,
                                          ColorAttachmentProvenancePolicy& result) noexcept {
        if (!text || text->empty()) {
            return false;
        }
        std::size_t begin{};
        std::size_t entry_count{};
        while (begin < text->size()) {
            if (++entry_count > MaxPoints) {
                return false;
            }
            const std::size_t comma = text->find(',', begin);
            const std::size_t end = comma == std::string_view::npos ? text->size() : comma;
            const auto item = text->substr(begin, end - begin);
            const std::size_t colon = item.find(':');
            if (item.empty() || colon == std::string_view::npos || colon == 0 ||
                colon + 1 >= item.size() || item.find(':', colon + 1) != std::string_view::npos) {
                return false;
            }
            std::uint16_t x{};
            std::uint16_t y{};
            if (!ParseDecimal(item.substr(0, colon), x) ||
                !ParseDecimal(item.substr(colon + 1), y) || x > 1000 || y > 1000) {
                return false;
            }
            const ColorAttachmentProvenancePoint point{x, y};
            bool duplicate{};
            for (std::size_t index = 0; index < result.point_count; ++index) {
                duplicate |= result.points[index] == point;
            }
            if (!duplicate) {
                result.points[result.point_count++] = point;
            }
            if (comma == std::string_view::npos) {
                break;
            }
            begin = comma + 1;
            if (begin == text->size()) {
                return false;
            }
        }
        return result.point_count != 0;
    }

    [[nodiscard]] static bool ParseSelectors(const std::optional<std::string_view> text,
                                             ColorAttachmentProvenancePolicy& result) noexcept {
        if (!text || text->empty()) {
            return false;
        }
        std::size_t begin{};
        std::size_t entry_count{};
        while (begin < text->size()) {
            if (++entry_count > MaxSelectors) {
                return false;
            }
            const std::size_t comma = text->find(',', begin);
            const std::size_t end = comma == std::string_view::npos ? text->size() : comma;
            const auto item = text->substr(begin, end - begin);
            if (item.size() != 16) {
                return false;
            }
            std::uint64_t selector{};
            const auto [parsed_end, error] =
                std::from_chars(item.data(), item.data() + item.size(), selector, 16);
            if (error != std::errc{} || parsed_end != item.data() + item.size()) {
                return false;
            }
            bool duplicate{};
            for (std::size_t index = 0; index < result.selector_count; ++index) {
                duplicate |= result.selectors[index] == selector;
            }
            if (!duplicate) {
                if (result.selector_count == MaxSelectors) {
                    return false;
                }
                result.selectors[result.selector_count++] = selector;
            }
            if (comma == std::string_view::npos) {
                break;
            }
            begin = comma + 1;
            if (begin == text->size()) {
                return false;
            }
        }
        return result.selector_count != 0;
    }
};

enum class ColorAttachmentProvenanceEnd {
    Window,
    Capacity,
    Teardown,
};

struct ColorAttachmentProvenanceSummary {
    ColorAttachmentProvenanceEnd reason{};
    std::uint64_t elapsed_ms{};
    std::uint32_t matched_commands{};
    std::uint32_t completed_commands{};
};

struct ColorAttachmentProvenanceCommandToken {
    std::uint64_t generation{};
    std::uint64_t selector{};
    std::uint32_t ordinal{};
    auto operator<=>(const ColorAttachmentProvenanceCommandToken&) const = default;
};

struct ColorAttachmentProvenanceBegin {
    std::optional<ColorAttachmentProvenanceCommandToken> capture;
    std::optional<ColorAttachmentProvenanceSummary> end;
};

// Serial pre/post command receipt. A command consumes capacity only after its exact
// token is finished, so a pre-sample can never close the receipt without its post-sample.
class ColorAttachmentProvenanceReceipt {
public:
    explicit ColorAttachmentProvenanceReceipt(ColorAttachmentProvenancePolicy policy_)
        : policy{policy_} {}

    [[nodiscard]] ColorAttachmentProvenanceBegin Begin(const std::uint64_t elapsed_ms,
                                                       const std::uint64_t selector) noexcept {
        if (!Enabled() || active.has_value() || elapsed_ms < policy.start_ms) {
            return {};
        }
        if (elapsed_ms >= policy.start_ms + policy.duration_ms) {
            return {.end = FinishReceipt(ColorAttachmentProvenanceEnd::Window, elapsed_ms)};
        }
        if (!policy.Selects(selector)) {
            return {};
        }
        ColorAttachmentProvenanceCommandToken token{
            .generation = ++generation,
            .selector = selector,
            .ordinal = matched_commands,
        };
        ++matched_commands;
        active = token;
        return {.capture = token};
    }

    [[nodiscard]] std::optional<ColorAttachmentProvenanceSummary> Finish(
        const ColorAttachmentProvenanceCommandToken token,
        const std::uint64_t elapsed_ms) noexcept {
        if (!Enabled() || !active || *active != token) {
            return std::nullopt;
        }
        active.reset();
        ++completed_commands;
        if (completed_commands == policy.command_capacity) {
            return FinishReceipt(ColorAttachmentProvenanceEnd::Capacity, elapsed_ms);
        }
        if (elapsed_ms >= policy.start_ms + policy.duration_ms) {
            return FinishReceipt(ColorAttachmentProvenanceEnd::Window, elapsed_ms);
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<ColorAttachmentProvenanceSummary> Close(
        const std::uint64_t elapsed_ms) noexcept {
        if (!Enabled()) {
            return std::nullopt;
        }
        return FinishReceipt(ColorAttachmentProvenanceEnd::Teardown, elapsed_ms);
    }

    [[nodiscard]] bool Enabled() const noexcept {
        return policy.enabled && policy.valid && !closed;
    }

private:
    [[nodiscard]] ColorAttachmentProvenanceSummary FinishReceipt(
        const ColorAttachmentProvenanceEnd reason, const std::uint64_t elapsed_ms) noexcept {
        closed = true;
        active.reset();
        return {reason, elapsed_ms, matched_commands, completed_commands};
    }

    ColorAttachmentProvenancePolicy policy;
    std::optional<ColorAttachmentProvenanceCommandToken> active;
    std::uint64_t generation{};
    std::uint32_t matched_commands{};
    std::uint32_t completed_commands{};
    bool closed{};
};

struct ColorAttachmentProvenanceTarget {
    std::uint64_t image_uid{};
    std::uint32_t format{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t levels{};
    std::uint32_t layers{};
    std::uint32_t mip{};
    std::uint32_t layer{};
    std::uint32_t bits_per_texel{};
    std::uint32_t samples{1};
    bool compressed{};
    bool depth{};
    bool is_3d{};
};

struct ColorAttachmentProvenanceSlotPair {
    std::uint64_t pre_offset{};
    std::uint64_t post_offset{};
};

class ColorAttachmentProvenanceSlotAllocator {
public:
    static constexpr std::uint64_t SlotStride = 48;
    static constexpr std::uint64_t PairBytes = SlotStride * 2;

    explicit ColorAttachmentProvenanceSlotAllocator(
        const std::uint64_t byte_capacity_, const std::uint64_t slot_stride_ = SlotStride)
        : byte_capacity{byte_capacity_}, slot_stride{std::max(slot_stride_, SlotStride)} {}

    [[nodiscard]] std::optional<ColorAttachmentProvenanceSlotPair> ReservePair() noexcept {
        const auto pair_bytes = slot_stride * 2;
        if (used_bytes > byte_capacity || pair_bytes > byte_capacity - used_bytes) {
            return std::nullopt;
        }
        const ColorAttachmentProvenanceSlotPair result{used_bytes, used_bytes + slot_stride};
        used_bytes += pair_bytes;
        return result;
    }

    [[nodiscard]] std::uint64_t UsedBytes() const noexcept {
        return used_bytes;
    }

private:
    std::uint64_t byte_capacity{};
    std::uint64_t slot_stride{SlotStride};
    std::uint64_t used_bytes{};
};

enum class ColorAttachmentProvenanceRejection {
    None,
    UndefinedFormat,
    Multisampled,
    Compressed,
    Depth,
    ThreeDimensional,
    InvalidTexelSize,
    InvalidMip,
    InvalidLayer,
    InvalidExtent,
    InvalidPoint,
    Capacity,
};

struct ColorAttachmentProvenanceSamplePlan {
    std::uint32_t x{};
    std::uint32_t y{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t mip{};
    std::uint32_t layer{};
    std::uint32_t payload_bytes{};
    std::uint64_t pre_offset{};
    std::uint64_t post_offset{};
};

struct ColorAttachmentProvenanceSampleResult {
    std::optional<ColorAttachmentProvenanceSamplePlan> plan;
    ColorAttachmentProvenanceRejection rejection{};
};

[[nodiscard]] inline ColorAttachmentProvenanceSampleResult PlanColorAttachmentProvenanceSample(
    const ColorAttachmentProvenanceTarget& target, const ColorAttachmentProvenancePoint point,
    ColorAttachmentProvenanceSlotAllocator& slots) noexcept {
    const auto reject = [](const ColorAttachmentProvenanceRejection reason) {
        return ColorAttachmentProvenanceSampleResult{.rejection = reason};
    };
    if (target.format == 0) {
        return reject(ColorAttachmentProvenanceRejection::UndefinedFormat);
    }
    if (target.samples != 1) {
        return reject(ColorAttachmentProvenanceRejection::Multisampled);
    }
    if (target.compressed) {
        return reject(ColorAttachmentProvenanceRejection::Compressed);
    }
    if (target.depth) {
        return reject(ColorAttachmentProvenanceRejection::Depth);
    }
    if (target.is_3d) {
        return reject(ColorAttachmentProvenanceRejection::ThreeDimensional);
    }
    if (target.bits_per_texel == 0 || target.bits_per_texel > 128 ||
        target.bits_per_texel % 8 != 0) {
        return reject(ColorAttachmentProvenanceRejection::InvalidTexelSize);
    }
    if (target.levels == 0 || target.mip >= target.levels || target.mip >= 32) {
        return reject(ColorAttachmentProvenanceRejection::InvalidMip);
    }
    if (target.layers == 0 || target.layer >= target.layers) {
        return reject(ColorAttachmentProvenanceRejection::InvalidLayer);
    }
    if (target.width == 0 || target.height == 0) {
        return reject(ColorAttachmentProvenanceRejection::InvalidExtent);
    }
    if (point.x_permille > 1000 || point.y_permille > 1000) {
        return reject(ColorAttachmentProvenanceRejection::InvalidPoint);
    }
    const auto pair = slots.ReservePair();
    if (!pair) {
        return reject(ColorAttachmentProvenanceRejection::Capacity);
    }

    const std::uint32_t width = std::max(target.width >> target.mip, 1u);
    const std::uint32_t height = std::max(target.height >> target.mip, 1u);
    const auto scale = [](const std::uint16_t permille, const std::uint32_t extent) {
        const auto coordinate = static_cast<std::uint64_t>(permille) * extent / 1000;
        return static_cast<std::uint32_t>(
            std::min<std::uint64_t>(coordinate, static_cast<std::uint64_t>(extent - 1)));
    };
    return {
        .plan =
            ColorAttachmentProvenanceSamplePlan{
                .x = scale(point.x_permille, width),
                .y = scale(point.y_permille, height),
                .width = width,
                .height = height,
                .mip = target.mip,
                .layer = target.layer,
                .payload_bytes = target.bits_per_texel / 8,
                .pre_offset = pair->pre_offset,
                .post_offset = pair->post_offset,
            },
    };
}

[[nodiscard]] inline std::vector<std::size_t> UniqueColorAttachmentProvenanceTargets(
    const std::span<const ColorAttachmentProvenanceTarget> targets) {
    std::vector<std::size_t> result;
    result.reserve(targets.size());
    for (std::size_t candidate = 0; candidate < targets.size(); ++candidate) {
        const auto& target = targets[candidate];
        bool duplicate{};
        for (const std::size_t accepted : result) {
            const auto& prior = targets[accepted];
            if (prior.image_uid == target.image_uid && prior.mip == target.mip &&
                prior.layer == target.layer) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            result.push_back(candidate);
        }
    }
    return result;
}

} // namespace VideoCore
