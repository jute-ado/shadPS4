// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/types.h"

namespace Vulkan {

enum class FinalGuestSurfaceImageType : u8 {
    Color1D,
    Color2D,
    Color3D,
    Other,
};

enum class FinalGuestSurfaceAspect : u8 {
    Color,
    Depth,
    Stencil,
    Other,
};

enum class FinalGuestSurfaceFormat : u8 {
    Unsupported,
    Rgba8,
    Bgra8,
    A2R10G10B10,
    A2B10G10R10,
    Rgba16,
    Rgba16Float,
    Block8,
    Block16,
};

enum class FinalGuestSurfaceComparison : u8 {
    ExactVisible,
    LocalizedVisualReturn,
    SampledLinearVisualReturn,
};

enum class FinalGuestSurfaceStage : u8 {
    GuestPreFsr,
    PostPp,
    PpInputShadow,
    PpSampledInput,
};

enum class FinalGuestSurfaceStatus : u8 {
    Complete,
    Unsupported,
    CapacityLoss,
    BusyLoss,
    InvalidationLoss,
    GapLoss,
    Pending,
    AlreadyConsumed,
};

struct FinalGuestSurfaceFormatBlock {
    u32 width{};
    u32 height{};
    u32 bytes{};
};

[[nodiscard]] constexpr FinalGuestSurfaceFormatBlock DescribeFinalGuestSurfaceFormat(
    FinalGuestSurfaceFormat format) noexcept {
    switch (format) {
    case FinalGuestSurfaceFormat::Rgba8:
    case FinalGuestSurfaceFormat::Bgra8:
    case FinalGuestSurfaceFormat::A2R10G10B10:
    case FinalGuestSurfaceFormat::A2B10G10R10:
        return {1, 1, 4};
    case FinalGuestSurfaceFormat::Rgba16:
    case FinalGuestSurfaceFormat::Rgba16Float:
        return {1, 1, 8};
    case FinalGuestSurfaceFormat::Block8:
        return {4, 4, 8};
    case FinalGuestSurfaceFormat::Block16:
        return {4, 4, 16};
    default:
        return {};
    }
}

struct FinalGuestSurfaceDescriptor {
    u32 width{};
    u32 height{};
    u32 depth{};
    u32 mip_level{};
    u32 mip_levels{};
    u32 base_array_layer{};
    u32 array_layers{};
    u32 samples{};
    FinalGuestSurfaceImageType type{FinalGuestSurfaceImageType::Other};
    FinalGuestSurfaceAspect aspect{FinalGuestSurfaceAspect::Other};
    FinalGuestSurfaceFormat format{FinalGuestSurfaceFormat::Unsupported};
    FinalGuestSurfaceComparison comparison{FinalGuestSurfaceComparison::ExactVisible};
    FinalGuestSurfaceStage stage{FinalGuestSurfaceStage::GuestPreFsr};
    u32 logical_width{};
    u32 logical_height{};
    s32 logical_offset_x{};
    s32 logical_offset_y{};
    bool logical_full_fit{};
    bool logical_top_left{};
    bool logical_no_y_flip{};
    u32 comparison_gamma_bits{};
};

struct FinalGuestSurfaceLoss {
    u32 unsupported_type{};
    u32 unsupported_samples{};
    u32 unsupported_mip{};
    u32 unsupported_layer{};
    u32 unsupported_aspect{};
    u32 unsupported_format{};
    u32 invalid_extent{};
    u32 logical_mapping{};
    u32 tile_capacity{};
    u32 byte_capacity{};
    u32 ordinal_capacity{};
    u32 busy{};
    u32 invalidation{};
    u32 gap{};
    u32 history{};
    u32 tile_detail{};

    [[nodiscard]] constexpr bool Any() const noexcept {
        return unsupported_type || unsupported_samples || unsupported_mip || unsupported_layer ||
               unsupported_aspect || unsupported_format || invalid_extent || tile_capacity ||
               logical_mapping || byte_capacity || ordinal_capacity || busy || invalidation ||
               gap || history || tile_detail;
    }

    bool operator==(const FinalGuestSurfaceLoss&) const = default;
};

struct FinalGuestSurfaceTile {
    u32 x{};
    u32 y{};
    u32 width{};
    u32 height{};
    u32 buffer_offset{};
    u32 byte_size{};

    auto operator<=>(const FinalGuestSurfaceTile&) const = default;
};

struct FinalGuestSurfacePairedRegion {
    u32 logical_ordinal{};
    u32 buffer_offset{};
    u32 byte_size{};
    u32 width{};
    u32 height{};

    bool operator==(const FinalGuestSurfacePairedRegion&) const = default;
};

struct FinalGuestSurfaceTileLimits {
    u32 max_tiles{4096};
    u32 max_bytes{16u << 20};
};

struct FinalGuestSurfaceTilePlan {
    static constexpr u32 MaxTiles = 4096;
    static constexpr u32 WindowExtent = 32;
    static constexpr u32 WindowStride = 16;

    u32 surface_width{};
    u32 surface_height{};
    u32 logical_width{};
    u32 logical_height{};
    u32 row_bytes{};
    u32 bytes_per_pixel{};
    u32 window_width{};
    u32 window_height{};
    u32 logical_columns{};
    u32 logical_rows{};
    u32 tile_count{};
    u32 sample_bytes{};
    u32 copy_region_count{};
    u32 scale_numerator{};
    u32 scale_denominator{};
    FinalGuestSurfaceComparison comparison{FinalGuestSurfaceComparison::ExactVisible};
    FinalGuestSurfaceStage stage{FinalGuestSurfaceStage::GuestPreFsr};
    u32 comparison_gamma_bits{};
    u32 paired_sampled_offset{};
    u32 paired_sampled_bytes{};
    u32 paired_sampled_row_bytes{};
    FinalGuestSurfaceFormat paired_sampled_format{FinalGuestSurfaceFormat::Unsupported};
    u32 paired_backing_offset{};
    u32 paired_backing_bytes{};
    u32 paired_backing_region_count{};
    std::array<FinalGuestSurfacePairedRegion, 32> paired_backing_regions{};
    FinalGuestSurfaceFormat paired_backing_format{FinalGuestSurfaceFormat::Unsupported};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::Complete};
    FinalGuestSurfaceLoss loss{};

    [[nodiscard]] constexpr FinalGuestSurfaceTile TileAt(u32 ordinal_index) const noexcept {
        if (ordinal_index >= tile_count || logical_columns == 0) {
            return {};
        }
        const u32 column = ordinal_index % logical_columns;
        const u32 row = ordinal_index / logical_columns;
        const u32 logical_x = column * WindowStride;
        const u32 logical_y = row * WindowStride;
        const u32 x = static_cast<u32>(static_cast<u64>(logical_x) * surface_width / logical_width);
        const u32 y =
            static_cast<u32>(static_cast<u64>(logical_y) * surface_height / logical_height);
        const u32 end_x = static_cast<u32>(static_cast<u64>(logical_x + window_width) *
                                           surface_width / logical_width);
        const u32 end_y = static_cast<u32>(static_cast<u64>(logical_y + window_height) *
                                           surface_height / logical_height);
        return {
            .x = x,
            .y = y,
            .width = end_x - x,
            .height = end_y - y,
            .buffer_offset = y * row_bytes + x * bytes_per_pixel,
            .byte_size = (end_x - x) * (end_y - y) * bytes_per_pixel,
        };
    }

    bool operator==(const FinalGuestSurfaceTilePlan&) const = default;
};

[[nodiscard]] inline FinalGuestSurfaceTilePlan PlanFinalGuestSurfaceTiles(
    const FinalGuestSurfaceDescriptor& desc, FinalGuestSurfaceTileLimits limits = {}) noexcept {
    FinalGuestSurfaceTilePlan rejected{};
    const auto unsupported = [&](u32 FinalGuestSurfaceLoss::* member) {
        auto result = rejected;
        result.status = FinalGuestSurfaceStatus::Unsupported;
        result.loss.*member = 1;
        return result;
    };
    if (desc.type != FinalGuestSurfaceImageType::Color1D &&
        desc.type != FinalGuestSurfaceImageType::Color2D) {
        return unsupported(&FinalGuestSurfaceLoss::unsupported_type);
    }
    if (desc.samples != 1) {
        return unsupported(&FinalGuestSurfaceLoss::unsupported_samples);
    }
    if (desc.mip_level != 0 || desc.mip_levels != 1) {
        return unsupported(&FinalGuestSurfaceLoss::unsupported_mip);
    }
    if (desc.base_array_layer != 0 || desc.array_layers != 1) {
        return unsupported(&FinalGuestSurfaceLoss::unsupported_layer);
    }
    if (desc.aspect != FinalGuestSurfaceAspect::Color) {
        return unsupported(&FinalGuestSurfaceLoss::unsupported_aspect);
    }
    if (desc.format == FinalGuestSurfaceFormat::Block8 ||
        desc.format == FinalGuestSurfaceFormat::Block16) {
        return unsupported(&FinalGuestSurfaceLoss::unsupported_format);
    }
    const auto block = DescribeFinalGuestSurfaceFormat(desc.format);
    if (block.width == 0 || block.height == 0 || block.bytes == 0) {
        return unsupported(&FinalGuestSurfaceLoss::unsupported_format);
    }
    if (desc.comparison == FinalGuestSurfaceComparison::LocalizedVisualReturn &&
        desc.format != FinalGuestSurfaceFormat::Rgba8 &&
        desc.format != FinalGuestSurfaceFormat::Bgra8) {
        return unsupported(&FinalGuestSurfaceLoss::unsupported_format);
    }
    if (desc.comparison == FinalGuestSurfaceComparison::SampledLinearVisualReturn &&
        (desc.format != FinalGuestSurfaceFormat::Rgba16Float ||
         !std::isfinite(std::bit_cast<float>(desc.comparison_gamma_bits)) ||
         std::bit_cast<float>(desc.comparison_gamma_bits) < 0.1f ||
         std::bit_cast<float>(desc.comparison_gamma_bits) > 2.0f)) {
        return unsupported(&FinalGuestSurfaceLoss::unsupported_format);
    }
    if (desc.width == 0 || desc.height == 0 || desc.depth != 1 ||
        (desc.type == FinalGuestSurfaceImageType::Color1D && desc.height != 1)) {
        return unsupported(&FinalGuestSurfaceLoss::invalid_extent);
    }
    if (desc.logical_width == 0 || desc.logical_height == 0 || desc.logical_offset_x != 0 ||
        desc.logical_offset_y != 0 || !desc.logical_full_fit || !desc.logical_top_left ||
        !desc.logical_no_y_flip ||
        static_cast<u64>(desc.width) * desc.logical_height !=
            static_cast<u64>(desc.height) * desc.logical_width) {
        return unsupported(&FinalGuestSurfaceLoss::logical_mapping);
    }

    const u32 window_width = std::min(desc.logical_width, FinalGuestSurfaceTilePlan::WindowExtent);
    const u32 window_height =
        std::min(desc.logical_height, FinalGuestSurfaceTilePlan::WindowExtent);
    const u32 columns =
        desc.logical_width <= window_width
            ? 1
            : (desc.logical_width - window_width) / FinalGuestSurfaceTilePlan::WindowStride + 1;
    const u32 rows =
        desc.type == FinalGuestSurfaceImageType::Color1D || desc.logical_height <= window_height
            ? 1
            : (desc.logical_height - window_height) / FinalGuestSurfaceTilePlan::WindowStride + 1;
    const u64 tile_count = static_cast<u64>(columns) * rows;
    if (tile_count > limits.max_tiles || tile_count > FinalGuestSurfaceTilePlan::MaxTiles) {
        FinalGuestSurfaceTilePlan result{};
        result.status = FinalGuestSurfaceStatus::CapacityLoss;
        result.loss.tile_capacity = 1;
        return result;
    }
    const auto maps_exactly = [](u32 logical_coordinate, u32 guest_extent,
                                 u32 logical_extent) noexcept {
        return static_cast<u64>(logical_coordinate) * guest_extent % logical_extent == 0;
    };
    for (u32 column = 0; column < columns; ++column) {
        const u32 start = column * FinalGuestSurfaceTilePlan::WindowStride;
        if (!maps_exactly(start, desc.width, desc.logical_width) ||
            !maps_exactly(start + window_width, desc.width, desc.logical_width)) {
            return unsupported(&FinalGuestSurfaceLoss::logical_mapping);
        }
    }
    for (u32 row = 0; row < rows; ++row) {
        const u32 start = row * FinalGuestSurfaceTilePlan::WindowStride;
        if (!maps_exactly(start, desc.height, desc.logical_height) ||
            !maps_exactly(start + window_height, desc.height, desc.logical_height)) {
            return unsupported(&FinalGuestSurfaceLoss::logical_mapping);
        }
    }
    const u64 row_bytes = static_cast<u64>(desc.width) * block.bytes;
    const u64 sample_bytes = row_bytes * desc.height;
    if (row_bytes > std::numeric_limits<u32>::max() || sample_bytes > limits.max_bytes ||
        sample_bytes > std::numeric_limits<u32>::max()) {
        FinalGuestSurfaceTilePlan result{};
        result.status = FinalGuestSurfaceStatus::CapacityLoss;
        result.loss.byte_capacity = 1;
        return result;
    }
    const u32 scale_divisor = std::gcd(desc.width, desc.logical_width);
    return {
        .surface_width = desc.width,
        .surface_height = desc.height,
        .logical_width = desc.logical_width,
        .logical_height = desc.logical_height,
        .row_bytes = static_cast<u32>(row_bytes),
        .bytes_per_pixel = block.bytes,
        .window_width = window_width,
        .window_height = window_height,
        .logical_columns = columns,
        .logical_rows = rows,
        .tile_count = static_cast<u32>(tile_count),
        .sample_bytes = static_cast<u32>(sample_bytes),
        .copy_region_count = 1,
        .scale_numerator = desc.width / scale_divisor,
        .scale_denominator = desc.logical_width / scale_divisor,
        .comparison = desc.comparison,
        .stage = desc.stage,
        .comparison_gamma_bits = desc.comparison_gamma_bits,
    };
}

struct FinalGuestSurfaceCaptureWindow {
    static constexpr u32 MaxFrameCount = 2048;

    u64 frame_start{3300};
    u32 frame_count{1700};

    [[nodiscard]] static constexpr FinalGuestSurfaceCaptureWindow Defaults() noexcept {
        return {};
    }

    [[nodiscard]] constexpr bool Contains(u64 sequence) const noexcept {
        return sequence >= frame_start && sequence - frame_start < frame_count;
    }

    [[nodiscard]] constexpr bool IsFinal(u64 sequence) const noexcept {
        return Contains(sequence) &&
               (sequence == std::numeric_limits<u64>::max() || !Contains(sequence + 1));
    }
};

[[nodiscard]] constexpr bool ShouldCaptureFinalGuestSurface(bool enabled,
                                                            FinalGuestSurfaceCaptureWindow window,
                                                            u64 sequence) noexcept {
    return enabled && window.Contains(sequence);
}

class FinalGuestSurfaceReadbackSlotPool {
public:
    static constexpr u32 MaxSlots = 8;

    struct Token {
        u32 slot{MaxSlots};
        u32 generation{};

        auto operator<=>(const Token&) const = default;
        [[nodiscard]] explicit constexpr operator bool() const noexcept {
            return slot < MaxSlots && generation != 0;
        }
    };

    [[nodiscard]] std::optional<Token> TryAcquire() noexcept {
        for (u32 slot = 0; slot < MaxSlots; ++slot) {
            auto& entry = entries[slot];
            if (entry.acquired) {
                continue;
            }
            ++entry.generation;
            if (entry.generation == 0) {
                ++entry.generation;
            }
            entry.acquired = true;
            return Token{slot, entry.generation};
        }
        return std::nullopt;
    }

    [[nodiscard]] bool ReleaseAfterCpuConsume(Token token) noexcept {
        if (!token) {
            return false;
        }
        auto& entry = entries[token.slot];
        if (!entry.acquired || entry.generation != token.generation) {
            return false;
        }
        entry.acquired = false;
        return true;
    }

private:
    struct Entry {
        u32 generation{};
        bool acquired{};
    };
    std::array<Entry, MaxSlots> entries{};
};

class FinalGuestSurfaceReadbackCompletion {
public:
    template <typename Invalidate>
    [[nodiscard]] FinalGuestSurfaceStatus TryConsume(bool gpu_complete, bool coherent,
                                                     Invalidate&& invalidate) noexcept {
        if (consumed) {
            return FinalGuestSurfaceStatus::AlreadyConsumed;
        }
        if (!gpu_complete) {
            return FinalGuestSurfaceStatus::Pending;
        }
        consumed = true;
        if (!coherent && !invalidate()) {
            return FinalGuestSurfaceStatus::InvalidationLoss;
        }
        return FinalGuestSurfaceStatus::Complete;
    }

private:
    bool consumed{};
};

class FinalGuestSurfaceBackingGenerationProvider {
public:
    [[nodiscard]] u64 Assign(u64& storage) noexcept {
        if (storage != 0) {
            return storage;
        }
        if (next == 0) {
            return 0;
        }
        storage = next++;
        return storage;
    }

private:
    u64 next{1};
};

struct FinalGuestSurfaceTransport {
    // Equality intentionally covers only the image UID, backing generation, exact format, and
    // extent. It does not claim a write serial or producer-path identity.
    u64 surface_identity{};
    u64 backing_generation{};
    FinalGuestSurfaceFormat format{FinalGuestSurfaceFormat::Unsupported};
    u32 width{};
    u32 height{};

    auto operator<=>(const FinalGuestSurfaceTransport&) const = default;
};

class FinalGuestSurfacePostPpTransportTracker {
public:
    [[nodiscard]] FinalGuestSurfaceTransport Observe(FinalGuestSurfaceFormat format, u32 width,
                                                     u32 height, bool hdr) noexcept {
        const Key current{format, width, height, hdr};
        if (!has_key || current != key) {
            key = current;
            has_key = true;
            if (generation != std::numeric_limits<u64>::max()) {
                ++generation;
            }
        }
        return {
            .surface_identity = 1,
            .backing_generation = generation,
            .format = format,
            .width = width,
            .height = height,
        };
    }

private:
    struct Key {
        FinalGuestSurfaceFormat format{FinalGuestSurfaceFormat::Unsupported};
        u32 width{};
        u32 height{};
        bool hdr{};

        bool operator==(const Key&) const = default;
    };

    Key key{};
    u64 generation{};
    bool has_key{};
};

struct FinalGuestSurfaceLagConfig {
    u64 cadence_us{100'000};
    u64 tolerance_us{25'000};

    [[nodiscard]] static constexpr FinalGuestSurfaceLagConfig Defaults() noexcept {
        return {};
    }
};

struct FinalGuestSurfaceWatchOrdinals {
    static constexpr u32 MaxOrdinals = 32;

    std::array<u32, MaxOrdinals> ordinals{};
    u32 count{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::Complete};
    u32 loss{};

    [[nodiscard]] constexpr bool Contains(u32 ordinal) const noexcept {
        return std::binary_search(ordinals.begin(), ordinals.begin() + count, ordinal);
    }
};

[[nodiscard]] inline FinalGuestSurfaceWatchOrdinals ParseFinalGuestSurfaceWatchOrdinals(
    std::string_view text) noexcept {
    FinalGuestSurfaceWatchOrdinals selector{};
    if (text.empty()) {
        return selector;
    }
    while (!text.empty()) {
        const size_t comma = text.find(',');
        const std::string_view token = text.substr(0, comma);
        u32 ordinal{};
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), ordinal);
        if (token.empty() || parsed.ec != std::errc{} ||
            parsed.ptr != token.data() + token.size() || ordinal == 0 ||
            ordinal > FinalGuestSurfaceTilePlan::MaxTiles) {
            selector = {};
            selector.status = FinalGuestSurfaceStatus::Unsupported;
            selector.loss = 1;
            return selector;
        }
        if (selector.count == selector.ordinals.size()) {
            selector = {};
            selector.status = FinalGuestSurfaceStatus::CapacityLoss;
            selector.loss = 1;
            return selector;
        }
        selector.ordinals[selector.count++] = ordinal;
        if (comma == std::string_view::npos) {
            break;
        }
        if (comma + 1 == text.size()) {
            selector = {};
            selector.status = FinalGuestSurfaceStatus::Unsupported;
            selector.loss = 1;
            return selector;
        }
        text.remove_prefix(comma + 1);
    }
    std::sort(selector.ordinals.begin(), selector.ordinals.begin() + selector.count);
    if (std::adjacent_find(selector.ordinals.begin(), selector.ordinals.begin() + selector.count) !=
        selector.ordinals.begin() + selector.count) {
        selector = {};
        selector.status = FinalGuestSurfaceStatus::Unsupported;
        selector.loss = 1;
    }
    return selector;
}

[[nodiscard]] constexpr FinalGuestSurfaceWatchOrdinals ValidateFinalGuestSurfaceWatchOrdinals(
    FinalGuestSurfaceWatchOrdinals selector, u32 actual_window_count) noexcept {
    if (selector.status != FinalGuestSurfaceStatus::Complete) {
        return selector;
    }
    for (u32 index = 0; index < selector.count; ++index) {
        if (selector.ordinals[index] > actual_window_count) {
            selector.status = FinalGuestSurfaceStatus::Unsupported;
            selector.loss = 1;
            return selector;
        }
    }
    return selector;
}

inline constexpr u32 FinalGuestSurfaceMaxScreenshotRequests = 1000;

struct FinalGuestSurfaceContentConfig {
    FinalGuestSurfaceCaptureWindow window{FinalGuestSurfaceCaptureWindow::Defaults()};
    FinalGuestSurfaceLagConfig lag{FinalGuestSurfaceLagConfig::Defaults()};
    FinalGuestSurfaceWatchOrdinals watch_ordinals{};
    FinalGuestSurfaceStage stage{FinalGuestSurfaceStage::GuestPreFsr};
    u32 expected_calibrations{};
    bool calibrated_triplets{};
};

template <typename ReadValue>
[[nodiscard]] std::optional<FinalGuestSurfaceContentConfig> ResolveFinalGuestSurfaceContentConfig(
    ReadValue&& read_value) {
    const auto enabled = read_value("SHADPS4_FINAL_GUEST_SURFACE_CONTENT");
    if (!enabled || *enabled != "1") {
        return std::nullopt;
    }
    const auto parse = [&](const char* name, u64 fallback, u64 maximum) {
        const auto value = read_value(name);
        if (!value) {
            return fallback;
        }
        u64 parsed{};
        const auto result = std::from_chars(value->data(), value->data() + value->size(), parsed);
        if (result.ec != std::errc{} || result.ptr != value->data() + value->size()) {
            return fallback;
        }
        return std::min(parsed, maximum);
    };

    FinalGuestSurfaceContentConfig config{};
    if (const auto stage = read_value("SHADPS4_FINAL_GUEST_SURFACE_STAGE")) {
        if (*stage == "post_pp") {
            config.stage = FinalGuestSurfaceStage::PostPp;
        } else if (*stage == "pp_input_shadow") {
            config.stage = FinalGuestSurfaceStage::PpInputShadow;
        } else if (*stage == "pp_sampled_input") {
            config.stage = FinalGuestSurfaceStage::PpSampledInput;
        } else if (*stage != "guest_pre_fsr") {
            return std::nullopt;
        }
    }
    if (const auto calibrated = read_value("SHADPS4_FINAL_GUEST_SURFACE_CALIBRATED_TRIPLETS")) {
        if (*calibrated != "1" || (config.stage != FinalGuestSurfaceStage::PostPp &&
                                   config.stage != FinalGuestSurfaceStage::PpInputShadow &&
                                   config.stage != FinalGuestSurfaceStage::PpSampledInput)) {
            return std::nullopt;
        }
        config.calibrated_triplets = true;
        const auto expected = read_value("SHADPS4_FINAL_GUEST_SURFACE_EXPECTED_CALIBRATIONS");
        if (!expected) {
            return std::nullopt;
        }
        u32 parsed{};
        const auto result =
            std::from_chars(expected->data(), expected->data() + expected->size(), parsed);
        if (result.ec != std::errc{} || result.ptr != expected->data() + expected->size() ||
            parsed == 0 || parsed > FinalGuestSurfaceMaxScreenshotRequests) {
            return std::nullopt;
        }
        config.expected_calibrations = parsed;
    }
    if ((config.stage == FinalGuestSurfaceStage::PpInputShadow ||
         config.stage == FinalGuestSurfaceStage::PpSampledInput) &&
        !config.calibrated_triplets) {
        return std::nullopt;
    }
    config.window.frame_start = parse("SHADPS4_FINAL_GUEST_SURFACE_FRAME_START",
                                      config.window.frame_start, std::numeric_limits<u64>::max());
    config.window.frame_count =
        static_cast<u32>(parse("SHADPS4_FINAL_GUEST_SURFACE_FRAME_COUNT", config.window.frame_count,
                               FinalGuestSurfaceCaptureWindow::MaxFrameCount));
    if (config.window.frame_count == 0) {
        config.window.frame_count = FinalGuestSurfaceCaptureWindow::Defaults().frame_count;
    }
    config.lag.cadence_us =
        parse("SHADPS4_FINAL_GUEST_SURFACE_LAG_CADENCE_US", config.lag.cadence_us, 1'000'000);
    if (config.lag.cadence_us == 0) {
        config.lag.cadence_us = FinalGuestSurfaceLagConfig::Defaults().cadence_us;
    }
    config.lag.tolerance_us = parse("SHADPS4_FINAL_GUEST_SURFACE_LAG_TOLERANCE_US",
                                    config.lag.tolerance_us, (config.lag.cadence_us - 1) / 2);
    if (const auto watch = read_value("SHADPS4_FINAL_GUEST_SURFACE_WATCH_ORDINALS")) {
        config.watch_ordinals = ParseFinalGuestSurfaceWatchOrdinals(*watch);
    }
    return config;
}

[[nodiscard]] constexpr bool ShouldObserveFinalGuestSurfaceAtPresent(
    FinalGuestSurfaceStage stage, bool is_reusing_frame, bool stamp_valid,
    bool in_capture_window) noexcept {
    return (stage == FinalGuestSurfaceStage::PostPp ||
            stage == FinalGuestSurfaceStage::PpInputShadow ||
            stage == FinalGuestSurfaceStage::PpSampledInput) &&
           !is_reusing_frame && stamp_valid && in_capture_window;
}

[[nodiscard]] constexpr bool IsPresentFinalGuestSurfaceStage(
    FinalGuestSurfaceStage stage) noexcept {
    return stage == FinalGuestSurfaceStage::PostPp ||
           stage == FinalGuestSurfaceStage::PpInputShadow ||
           stage == FinalGuestSurfaceStage::PpSampledInput;
}

struct FinalGuestSurfaceLogPolicyConfig {
    bool verbose_frame_reports{true};
    bool calibrated_triplet_reports{};
    bool stage_content_coverage{true};
    bool calibrated_coverage{};
};

[[nodiscard]] constexpr FinalGuestSurfaceLogPolicyConfig FinalGuestSurfaceLogPolicy(
    FinalGuestSurfaceStage stage) noexcept {
    if (stage == FinalGuestSurfaceStage::PpInputShadow ||
        stage == FinalGuestSurfaceStage::PpSampledInput) {
        return {
            .verbose_frame_reports = false,
            .calibrated_triplet_reports = true,
            .stage_content_coverage = true,
            .calibrated_coverage = true,
        };
    }
    return {};
}

struct FinalGuestSurfaceFrameDiagnosticStamp {
    u64 surface_sequence{};
    u64 surface_process_time_us{};
    u32 guest_width{};
    u32 guest_height{};
    bool valid{};

    constexpr void Assign(bool enabled, u64 sequence, u64 process_time_us, u32 width,
                          u32 height) noexcept {
        Clear();
        if (enabled) {
            surface_sequence = sequence;
            surface_process_time_us = process_time_us;
            guest_width = width;
            guest_height = height;
            valid = true;
        }
    }

    constexpr void Clear() noexcept {
        *this = {};
    }
};

struct FinalGuestSurfacePresentationMapping {
    u32 guest_width{};
    u32 guest_height{};
    u32 swapchain_width{};
    u32 swapchain_height{};
    s32 output_x{};
    s32 output_y{};
    u32 output_width{};
    u32 output_height{};
    bool top_left{};
    bool no_y_flip{};

    bool operator==(const FinalGuestSurfacePresentationMapping&) const = default;

    [[nodiscard]] constexpr std::array<u32, 2> ExactScale() const noexcept {
        if (guest_width == 0 || guest_height == 0 || output_width == 0 || output_height == 0 ||
            output_x != 0 || output_y != 0 || output_width != swapchain_width ||
            output_height != swapchain_height || !top_left || !no_y_flip ||
            static_cast<u64>(guest_width) * output_height !=
                static_cast<u64>(guest_height) * output_width) {
            return {};
        }
        const u32 divisor = std::gcd(guest_width, output_width);
        return {guest_width / divisor, output_width / divisor};
    }

    [[nodiscard]] constexpr bool IsIdentity() const noexcept {
        return ExactScale() == std::array<u32, 2>{1, 1};
    }
};

struct FinalGuestSurfaceCalibrationReport {
    u32 request_ordinal{};
    u64 surface_sequence{};
    u64 surface_process_time_us{};
    FinalGuestSurfacePresentationMapping mapping{};
    u32 overflow_loss{};
    u32 mapping_ordinal{};
    u32 mapping_loss{};
    u32 scale_numerator{};
    u32 scale_denominator{};
    bool emit{};
    bool fallback_time{};
    bool identity_mapping{};
    bool exact_scaled_mapping{};
    bool emit_mapping{};
    bool overflow_marker{};
};

class FinalGuestSurfaceScreenshotCalibration {
public:
    static constexpr u32 MaxRequests = FinalGuestSurfaceMaxScreenshotRequests;
    static constexpr u32 MaxMappingOrdinals = 16;

    explicit constexpr FinalGuestSurfaceScreenshotCalibration(bool enabled_) : enabled{enabled_} {}

    [[nodiscard]] constexpr FinalGuestSurfaceCalibrationReport Observe(
        const FinalGuestSurfaceFrameDiagnosticStamp& stamp,
        FinalGuestSurfacePresentationMapping mapping, u64 fallback_process_time_us) noexcept {
        if (!enabled) {
            return {};
        }
        if (observed_requests == MaxRequests) {
            if (overflow_emitted) {
                return {};
            }
            overflow_emitted = true;
            return {
                .mapping = mapping,
                .overflow_loss = 1,
                .emit = true,
                .overflow_marker = true,
            };
        }
        ++observed_requests;
        if (stamp.valid) {
            mapping.guest_width = stamp.guest_width;
            mapping.guest_height = stamp.guest_height;
        }
        const auto [mapping_ordinal, emit_mapping, mapping_loss] = MappingFor(mapping);
        const auto scale = mapping.ExactScale();
        return {
            .request_ordinal = observed_requests,
            .surface_sequence = stamp.valid ? stamp.surface_sequence : 0,
            .surface_process_time_us =
                stamp.valid ? stamp.surface_process_time_us : fallback_process_time_us,
            .mapping = mapping,
            .mapping_ordinal = mapping_ordinal,
            .mapping_loss = mapping_loss,
            .scale_numerator = scale[0],
            .scale_denominator = scale[1],
            .emit = true,
            .fallback_time = !stamp.valid,
            .identity_mapping = stamp.valid && mapping.IsIdentity(),
            .exact_scaled_mapping = stamp.valid && scale[0] != 0,
            .emit_mapping = emit_mapping,
        };
    }

    [[nodiscard]] constexpr u32 ObservedRequests() const noexcept {
        return observed_requests;
    }

private:
    struct MappingResult {
        u32 ordinal{};
        bool emit{};
        u32 loss{};
    };

    [[nodiscard]] constexpr MappingResult MappingFor(
        const FinalGuestSurfacePresentationMapping& mapping) noexcept {
        u32 ordinal{};
        for (u32 index = 0; index < mapping_count; ++index) {
            if (mappings[index] == mapping) {
                ordinal = index + 1;
                break;
            }
        }
        if (ordinal == 0) {
            if (mapping_count == mappings.size()) {
                return {.loss = 1};
            }
            mappings[mapping_count++] = mapping;
            ordinal = mapping_count;
        }
        const bool changed = ordinal != last_mapping_ordinal;
        last_mapping_ordinal = ordinal;
        return {.ordinal = ordinal, .emit = changed};
    }

    bool enabled{};
    bool overflow_emitted{};
    u32 observed_requests{};
    std::array<FinalGuestSurfacePresentationMapping, MaxMappingOrdinals> mappings{};
    u32 mapping_count{};
    u32 last_mapping_ordinal{};
};

[[nodiscard]] constexpr u32 FinalGuestSurfaceAutomationCalibrationCount(u32 /*notifying_count*/,
                                                                        u32 silent_count) noexcept {
    return silent_count;
}

struct FinalGuestSurfaceReport {
    static constexpr u32 MaxTileDetails = FinalGuestSurfaceWatchOrdinals::MaxOrdinals;

    u64 sequence{};
    u64 process_time_us{};
    u64 a_sequence{};
    u64 a_process_time_us{};
    u64 b_sequence{};
    u64 b_process_time_us{};
    u64 c_sequence{};
    u64 c_process_time_us{};
    u32 surface_ordinal{};
    u32 tile_count{};
    u32 changed_tiles{};
    u32 aba_tiles{};
    u32 unselected_aba_tiles{};
    std::array<u32, MaxTileDetails> aba_tile_ordinals{};
    u32 aba_tile_ordinal_count{};
    u32 selector_count{};
    u32 selector_loss{};
    FinalGuestSurfaceStatus selector_status{FinalGuestSurfaceStatus::Complete};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::Complete};
    FinalGuestSurfaceLoss loss{};
    FinalGuestSurfaceStage stage{FinalGuestSurfaceStage::GuestPreFsr};
    bool stable_transport{};
    bool exact_aba{};
    bool localized_aba{};
    bool whole_sample_aba{};
};

[[nodiscard]] inline std::string FormatFinalGuestSurfaceTileOrdinals(
    const FinalGuestSurfaceReport& report) {
    std::string tile_ordinals;
    for (u32 index = 0; index < report.aba_tile_ordinal_count; ++index) {
        if (!tile_ordinals.empty()) {
            tile_ordinals += ',';
        }
        tile_ordinals += std::to_string(report.aba_tile_ordinals[index]);
    }
    return tile_ordinals;
}

[[nodiscard]] constexpr u32 FinalGuestSurfaceLossMask(const FinalGuestSurfaceLoss& loss,
                                                      u32 selector_loss) noexcept {
    return (static_cast<u32>(loss.unsupported_type != 0) << 0) |
           (static_cast<u32>(loss.unsupported_samples != 0) << 1) |
           (static_cast<u32>(loss.unsupported_mip != 0) << 2) |
           (static_cast<u32>(loss.unsupported_layer != 0) << 3) |
           (static_cast<u32>(loss.unsupported_aspect != 0) << 4) |
           (static_cast<u32>(loss.unsupported_format != 0) << 5) |
           (static_cast<u32>(loss.invalid_extent != 0) << 6) |
           (static_cast<u32>(loss.logical_mapping != 0) << 7) |
           (static_cast<u32>(loss.tile_capacity != 0) << 8) |
           (static_cast<u32>(loss.byte_capacity != 0) << 9) |
           (static_cast<u32>(loss.ordinal_capacity != 0) << 10) |
           (static_cast<u32>(loss.busy != 0) << 11) |
           (static_cast<u32>(loss.invalidation != 0) << 12) |
           (static_cast<u32>(loss.gap != 0) << 13) | (static_cast<u32>(loss.history != 0) << 14) |
           (static_cast<u32>(loss.tile_detail != 0) << 15) |
           (static_cast<u32>(selector_loss != 0) << 16);
}

[[nodiscard]] inline std::string FormatFinalGuestSurfaceCompactReport(
    const FinalGuestSurfaceReport& report) {
    return "FGSC s=" + std::to_string(report.sequence) +
           " t=" + std::to_string(report.process_time_us) +
           " o=" + std::to_string(report.surface_ordinal) +
           " w=" + std::to_string(report.tile_count) +
           " d=" + std::to_string(report.changed_tiles) + " a=" + std::to_string(report.aba_tiles) +
           " u=" + std::to_string(report.unselected_aba_tiles) +
           " q=" + FormatFinalGuestSurfaceTileOrdinals(report) +
           " v=" + std::to_string(report.stable_transport) +
           " ws=" + std::to_string(report.whole_sample_aba) +
           " p=" + std::to_string(static_cast<u32>(report.stage)) +
           " la=" + std::to_string(report.localized_aba) +
           " st=" + std::to_string(static_cast<u32>(report.status)) +
           " lm=" + std::to_string(FinalGuestSurfaceLossMask(report.loss, report.selector_loss)) +
           " sel=" + std::to_string(report.selector_count) + '/' +
           std::to_string(static_cast<u32>(report.selector_status)) +
           " abc=" + std::to_string(report.a_sequence) + '/' + std::to_string(report.b_sequence) +
           '/' + std::to_string(report.c_sequence);
}

[[nodiscard]] inline std::string FormatFinalGuestSurfaceCompactMapping(
    const FinalGuestSurfaceCalibrationReport& report) {
    return "FGSCM m=" + std::to_string(report.mapping_ordinal) +
           " g=" + std::to_string(report.mapping.guest_width) + 'x' +
           std::to_string(report.mapping.guest_height) +
           " s=" + std::to_string(report.mapping.swapchain_width) + 'x' +
           std::to_string(report.mapping.swapchain_height) +
           " r=" + std::to_string(report.mapping.output_x) + ',' +
           std::to_string(report.mapping.output_y) + ',' +
           std::to_string(report.mapping.output_width) + ',' +
           std::to_string(report.mapping.output_height) +
           " tl=" + std::to_string(report.mapping.top_left) +
           " yf=" + std::to_string(!report.mapping.no_y_flip) +
           " sc=" + std::to_string(report.scale_numerator) + '/' +
           std::to_string(report.scale_denominator) +
           " ex=" + std::to_string(report.exact_scaled_mapping) +
           " ml=" + std::to_string(report.mapping_loss);
}

[[nodiscard]] inline std::string FormatFinalGuestSurfaceCompactCalibration(
    const FinalGuestSurfaceCalibrationReport& report) {
    return "FGSCC q=" + std::to_string(report.request_ordinal) +
           " s=" + std::to_string(report.surface_sequence) +
           " t=" + std::to_string(report.surface_process_time_us) +
           " m=" + std::to_string(report.mapping_ordinal) +
           " f=" + std::to_string(report.fallback_time) +
           " ov=" + std::to_string(report.overflow_loss) +
           " ml=" + std::to_string(report.mapping_loss);
}

[[nodiscard]] inline std::string FormatFinalGuestSurfaceReport(
    const FinalGuestSurfaceReport& report) {
    const std::string tile_ordinals = FormatFinalGuestSurfaceTileOrdinals(report);
    return "sequence=" + std::to_string(report.sequence) +
           " process_time_us=" + std::to_string(report.process_time_us) +
           " surface_ordinal=" + std::to_string(report.surface_ordinal) +
           " tiles=" + std::to_string(report.tile_count) +
           " changed_tiles=" + std::to_string(report.changed_tiles) +
           " aba_tiles=" + std::to_string(report.aba_tiles) +
           " unselected_aba_tiles=" + std::to_string(report.unselected_aba_tiles) +
           " aba_tile_ordinals=" + tile_ordinals +
           " selector_count=" + std::to_string(report.selector_count) +
           " selector_status=" + std::to_string(static_cast<u32>(report.selector_status)) +
           " selector_loss=" + std::to_string(report.selector_loss) +
           " status=" + std::to_string(static_cast<u32>(report.status)) +
           " stable=" + std::to_string(report.stable_transport) +
           " exact_aba=" + std::to_string(report.exact_aba) +
           " whole_sample_aba=" + std::to_string(report.whole_sample_aba) +
           " gap_loss=" + std::to_string(report.loss.gap) +
           " history_loss=" + std::to_string(report.loss.history) +
           " ordinal_loss=" + std::to_string(report.loss.ordinal_capacity) +
           " tile_detail_loss=" + std::to_string(report.loss.tile_detail);
}

struct FinalGuestSurfaceCalibratedStamp {
    u32 request_ordinal{};
    u64 sequence{};
    u64 process_time_us{};
    bool valid{};
};

struct FinalGuestSurfaceCalibratedLoss {
    u32 gap{};
    u32 history{};
    u32 duplicate{};
    u32 overflow{};
    u32 invalid{};
    u32 transport{};
    u32 time{};
    u32 selector{};

    [[nodiscard]] constexpr bool Any() const noexcept {
        return gap || history || duplicate || overflow || invalid || transport || time || selector;
    }

    bool operator==(const FinalGuestSurfaceCalibratedLoss&) const = default;
};

struct FinalGuestSurfaceCalibratedReport {
    u32 request_ordinal{};
    u64 a_sequence{};
    u64 a_process_time_us{};
    u64 b_sequence{};
    u64 b_process_time_us{};
    u64 c_sequence{};
    u64 c_process_time_us{};
    std::array<u32, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> matched_ordinals{};
    std::array<u32, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> pre_or_at_sample_ordinals{};
    std::array<u32, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> post_sample_ordinals{};
    std::array<u32, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> ambiguous_boundary_ordinals{};
    std::array<u32, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> backing_aba_ordinals{};
    std::array<u32, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> backing_stable_ordinals{};
    std::array<u32, FinalGuestSurfaceWatchOrdinals::MaxOrdinals> backing_ambiguous_ordinals{};
    u32 matched_ordinal_count{};
    u32 pre_or_at_sample_ordinal_count{};
    u32 post_sample_ordinal_count{};
    u32 ambiguous_boundary_ordinal_count{};
    u32 backing_aba_ordinal_count{};
    u32 backing_stable_ordinal_count{};
    u32 backing_ambiguous_ordinal_count{};
    u32 selector_count{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::Complete};
    FinalGuestSurfaceCalibratedLoss loss{};
    bool stable_transport{};
    bool exact_aba{};
};

struct FinalGuestSurfaceCalibratedCoverage {
    u32 calibrations{};
    u32 outside{};
    u32 eligible{};
    u32 emitted{};
    u32 complete{};
    FinalGuestSurfaceCalibratedLoss loss{};

    bool operator==(const FinalGuestSurfaceCalibratedCoverage&) const = default;
};

[[nodiscard]] constexpr u32 FinalGuestSurfaceCalibratedLossMask(
    const FinalGuestSurfaceCalibratedLoss& loss) noexcept {
    return (static_cast<u32>(loss.gap != 0) << 0) | (static_cast<u32>(loss.history != 0) << 1) |
           (static_cast<u32>(loss.duplicate != 0) << 2) |
           (static_cast<u32>(loss.overflow != 0) << 3) |
           (static_cast<u32>(loss.invalid != 0) << 4) |
           (static_cast<u32>(loss.transport != 0) << 5) | (static_cast<u32>(loss.time != 0) << 6) |
           (static_cast<u32>(loss.selector != 0) << 7);
}

[[nodiscard]] inline std::string FormatFinalGuestSurfaceCalibratedReport(
    const FinalGuestSurfaceCalibratedReport& report) {
    const auto format_ordinals = [](const auto& values, u32 count) {
        std::string ordinals;
        for (u32 index = 0; index < count; ++index) {
            if (!ordinals.empty()) {
                ordinals += ',';
            }
            ordinals += std::to_string(values[index]);
        }
        return ordinals;
    };
    const auto ordinals = format_ordinals(report.matched_ordinals, report.matched_ordinal_count);
    const auto pre =
        format_ordinals(report.pre_or_at_sample_ordinals, report.pre_or_at_sample_ordinal_count);
    const auto post =
        format_ordinals(report.post_sample_ordinals, report.post_sample_ordinal_count);
    const auto ambiguous = format_ordinals(report.ambiguous_boundary_ordinals,
                                           report.ambiguous_boundary_ordinal_count);
    const auto backing_aba =
        format_ordinals(report.backing_aba_ordinals, report.backing_aba_ordinal_count);
    const auto backing_stable =
        format_ordinals(report.backing_stable_ordinals, report.backing_stable_ordinal_count);
    const auto backing_ambiguous =
        format_ordinals(report.backing_ambiguous_ordinals, report.backing_ambiguous_ordinal_count);
    return "FGSCT q=" + std::to_string(report.request_ordinal) +
           " abc=" + std::to_string(report.a_sequence) + '/' + std::to_string(report.b_sequence) +
           '/' + std::to_string(report.c_sequence) +
           " t=" + std::to_string(report.a_process_time_us) + '/' +
           std::to_string(report.b_process_time_us) + '/' +
           std::to_string(report.c_process_time_us) + " r=" + ordinals + " pre=" + pre +
           " post=" + post + " amb=" + ambiguous + " ba=" + backing_aba + " bs=" + backing_stable +
           " bx=" + backing_ambiguous + " n=" + std::to_string(report.matched_ordinal_count) + '/' +
           std::to_string(report.selector_count) + " ex=" + std::to_string(report.exact_aba) +
           " v=" + std::to_string(report.stable_transport) +
           " st=" + std::to_string(static_cast<u32>(report.status)) +
           " lm=" + std::to_string(FinalGuestSurfaceCalibratedLossMask(report.loss));
}

[[nodiscard]] inline std::string FormatFinalGuestSurfaceCalibratedCoverage(
    const FinalGuestSurfaceCalibratedCoverage& coverage) {
    return "FGSCTC c=" + std::to_string(coverage.calibrations) +
           " o=" + std::to_string(coverage.outside) + " e=" + std::to_string(coverage.eligible) +
           '/' + std::to_string(coverage.emitted) + '/' + std::to_string(coverage.complete) +
           " lm=" + std::to_string(FinalGuestSurfaceCalibratedLossMask(coverage.loss));
}

class FinalGuestSurfaceReducer {
public:
    static constexpr u32 MaxHistory = 32;
    static constexpr u32 MaxSurfaceOrdinals = 16;

    explicit FinalGuestSurfaceReducer(FinalGuestSurfaceLagConfig config_,
                                      FinalGuestSurfaceWatchOrdinals selector_ = {})
        : config{config_}, selector{selector_} {}

    [[nodiscard]] FinalGuestSurfaceReport Observe(u64 sequence, u64 process_time_us,
                                                  FinalGuestSurfaceTransport transport,
                                                  const FinalGuestSurfaceTilePlan& plan,
                                                  std::span<const std::byte> bytes) {
        FinalGuestSurfaceReport report{
            .sequence = sequence,
            .process_time_us = process_time_us,
            .surface_ordinal = OrdinalFor(transport.surface_identity),
            .tile_count = plan.tile_count,
            .selector_count = selector.count,
            .selector_loss = selector.loss,
            .selector_status = selector.status,
            .status = plan.status,
            .loss = plan.loss,
            .stage = plan.stage,
        };
        const bool contiguous = !has_last || sequence == last_sequence + 1;
        const bool time_ordered = !has_last || process_time_us > last_process_time_us;
        last_sequence = sequence;
        last_process_time_us = process_time_us;
        has_last = true;
        if (!contiguous || !time_ordered) {
            ++report.loss.gap;
            report.status = FinalGuestSurfaceStatus::GapLoss;
            history.clear();
            history_evicted = false;
        }
        if (report.surface_ordinal == 0) {
            report.status = FinalGuestSurfaceStatus::CapacityLoss;
            report.loss.ordinal_capacity = 1;
            history.clear();
            history_evicted = false;
            return report;
        }
        if (plan.status != FinalGuestSurfaceStatus::Complete || plan.loss.Any() ||
            bytes.size() != plan.sample_bytes) {
            if (!report.loss.Any()) {
                ++report.loss.invalidation;
                report.status = FinalGuestSurfaceStatus::InvalidationLoss;
            }
            history.clear();
            history_evicted = false;
            return report;
        }
        const auto runtime_selector =
            ValidateFinalGuestSurfaceWatchOrdinals(selector, plan.tile_count);
        report.selector_count = runtime_selector.count;
        report.selector_status = runtime_selector.status;
        report.selector_loss = runtime_selector.loss;
        if (runtime_selector.status != FinalGuestSurfaceStatus::Complete ||
            runtime_selector.loss != 0) {
            report.status = FinalGuestSurfaceStatus::Unsupported;
            history.clear();
            history_evicted = false;
            return report;
        }

        if (!history.empty()) {
            report.changed_tiles = ChangedTiles(history.back(), transport.format, plan, bytes);
        }
        const auto b = FindClosest(process_time_us, config.cadence_us);
        const auto a =
            config.cadence_us <= std::numeric_limits<u64>::max() / 2
                ? FindClosest(process_time_us, config.cadence_us * 2, b ? b->index : history.size())
                : std::nullopt;
        if (a && b) {
            const auto& baseline = history[a->index];
            const auto& departure = history[b->index];
            report.a_sequence = baseline.sequence;
            report.a_process_time_us = baseline.process_time_us;
            report.b_sequence = departure.sequence;
            report.b_process_time_us = departure.process_time_us;
            report.c_sequence = sequence;
            report.c_process_time_us = process_time_us;
            bool stable = baseline.transport == transport;
            for (u32 i = a->index + 1; i < history.size(); ++i) {
                stable &= history[i].transport == transport;
            }
            report.stable_transport = stable;
            PopulateAbaDetails(baseline, departure, transport.format, plan, bytes, report);
        } else if (history_evicted && process_time_us >= config.cadence_us * 2) {
            ++report.loss.history;
        }
        Push(sequence, process_time_us, transport, plan, bytes);
        return report;
    }

    [[nodiscard]] size_t RetainedContentObservationCount() const noexcept {
        return history.size();
    }

    [[nodiscard]] std::optional<FinalGuestSurfaceCalibratedReport> EvaluateCalibratedTriplet(
        FinalGuestSurfaceCalibratedStamp a_stamp, FinalGuestSurfaceCalibratedStamp b_stamp,
        FinalGuestSurfaceCalibratedStamp c_stamp, bool finish) const noexcept {
        FinalGuestSurfaceCalibratedReport report{
            .request_ordinal = c_stamp.request_ordinal,
            .a_sequence = a_stamp.sequence,
            .a_process_time_us = a_stamp.process_time_us,
            .b_sequence = b_stamp.sequence,
            .b_process_time_us = b_stamp.process_time_us,
            .c_sequence = c_stamp.sequence,
            .c_process_time_us = c_stamp.process_time_us,
            .selector_count = selector.count,
        };
        const auto fail = [&](FinalGuestSurfaceStatus status,
                              u32 FinalGuestSurfaceCalibratedLoss::* loss) {
            report.status = status;
            ++(report.loss.*loss);
            return std::optional{report};
        };
        if (!a_stamp.valid || !b_stamp.valid || !c_stamp.valid) {
            return fail(FinalGuestSurfaceStatus::InvalidationLoss,
                        &FinalGuestSurfaceCalibratedLoss::invalid);
        }
        if (a_stamp.sequence >= b_stamp.sequence || b_stamp.sequence >= c_stamp.sequence) {
            return fail(FinalGuestSurfaceStatus::GapLoss, &FinalGuestSurfaceCalibratedLoss::gap);
        }
        if (a_stamp.process_time_us >= b_stamp.process_time_us ||
            b_stamp.process_time_us >= c_stamp.process_time_us) {
            return fail(FinalGuestSurfaceStatus::InvalidationLoss,
                        &FinalGuestSurfaceCalibratedLoss::time);
        }
        const auto runtime_selector =
            history.empty()
                ? selector
                : ValidateFinalGuestSurfaceWatchOrdinals(selector, history.back().plan.tile_count);
        if (runtime_selector.status != FinalGuestSurfaceStatus::Complete ||
            runtime_selector.loss != 0 || runtime_selector.count == 0) {
            return fail(FinalGuestSurfaceStatus::Unsupported,
                        &FinalGuestSurfaceCalibratedLoss::selector);
        }
        if (!history.empty() && a_stamp.sequence < history.front().sequence) {
            return history_evicted ? fail(FinalGuestSurfaceStatus::CapacityLoss,
                                          &FinalGuestSurfaceCalibratedLoss::history)
                                   : fail(FinalGuestSurfaceStatus::GapLoss,
                                          &FinalGuestSurfaceCalibratedLoss::gap);
        }
        std::array<const Observation*, 3> endpoints{FindObservation(a_stamp.sequence),
                                                    FindObservation(b_stamp.sequence),
                                                    FindObservation(c_stamp.sequence)};
        if (std::ranges::find(endpoints, nullptr) != endpoints.end()) {
            return finish ? fail(FinalGuestSurfaceStatus::GapLoss,
                                 &FinalGuestSurfaceCalibratedLoss::gap)
                          : std::nullopt;
        }
        if (endpoints[0]->process_time_us != a_stamp.process_time_us ||
            endpoints[1]->process_time_us != b_stamp.process_time_us ||
            endpoints[2]->process_time_us != c_stamp.process_time_us) {
            return fail(FinalGuestSurfaceStatus::InvalidationLoss,
                        &FinalGuestSurfaceCalibratedLoss::time);
        }
        const auto& plan = endpoints[2]->plan;
        const auto format = endpoints[2]->transport.format;
        for (const auto& observation : history) {
            if (observation.sequence < a_stamp.sequence ||
                observation.sequence > c_stamp.sequence) {
                continue;
            }
            if (observation.transport != endpoints[2]->transport || observation.plan != plan) {
                return fail(FinalGuestSurfaceStatus::Unsupported,
                            &FinalGuestSurfaceCalibratedLoss::transport);
            }
        }
        if (plan.paired_backing_format != FinalGuestSurfaceFormat::Unsupported) {
            if (plan.paired_backing_region_count != runtime_selector.count) {
                return fail(FinalGuestSurfaceStatus::Unsupported,
                            &FinalGuestSurfaceCalibratedLoss::selector);
            }
            for (u32 selected = 0; selected < runtime_selector.count; ++selected) {
                if (plan.paired_backing_regions[selected].logical_ordinal !=
                    runtime_selector.ordinals[selected]) {
                    return fail(FinalGuestSurfaceStatus::Unsupported,
                                &FinalGuestSurfaceCalibratedLoss::selector);
                }
            }
        }
        report.stable_transport = true;
        for (u32 selected = 0; selected < runtime_selector.count; ++selected) {
            const u32 index = runtime_selector.ordinals[selected] - 1;
            const bool returned =
                EqualTile(*endpoints[0], index, format, plan, endpoints[2]->bytes);
            const bool departed =
                !EqualTile(*endpoints[0], index, format, plan, endpoints[1]->bytes);
            report.exact_aba |= returned && departed;
            if ((plan.comparison == FinalGuestSurfaceComparison::LocalizedVisualReturn ||
                 plan.comparison == FinalGuestSurfaceComparison::SampledLinearVisualReturn) &&
                IsLocalizedVisualReturn(*endpoints[0], *endpoints[1], format, plan,
                                        endpoints[2]->bytes, index)) {
                report.matched_ordinals[report.matched_ordinal_count++] = index + 1;
                if (plan.paired_sampled_format != FinalGuestSurfaceFormat::Unsupported) {
                    const bool raw_returned =
                        EqualPairedSampleTile(*endpoints[0], index, plan, endpoints[2]->bytes);
                    const bool raw_departed =
                        !EqualPairedSampleTile(*endpoints[0], index, plan, endpoints[1]->bytes);
                    if (raw_returned && raw_departed) {
                        report.pre_or_at_sample_ordinals[report.pre_or_at_sample_ordinal_count++] =
                            index + 1;
                    } else if (raw_returned) {
                        report.post_sample_ordinals[report.post_sample_ordinal_count++] = index + 1;
                    } else {
                        report.ambiguous_boundary_ordinals
                            [report.ambiguous_boundary_ordinal_count++] = index + 1;
                    }
                }
                if (plan.paired_backing_format != FinalGuestSurfaceFormat::Unsupported) {
                    const bool backing_returned = EqualPairedBackingRegion(
                        *endpoints[0], index + 1, plan, endpoints[2]->bytes);
                    const bool backing_departed = !EqualPairedBackingRegion(
                        *endpoints[0], index + 1, plan, endpoints[1]->bytes);
                    if (backing_returned && backing_departed) {
                        report.backing_aba_ordinals[report.backing_aba_ordinal_count++] = index + 1;
                    } else if (backing_returned) {
                        report.backing_stable_ordinals[report.backing_stable_ordinal_count++] =
                            index + 1;
                    } else {
                        report
                            .backing_ambiguous_ordinals[report.backing_ambiguous_ordinal_count++] =
                            index + 1;
                    }
                }
            }
        }
        return report;
    }

private:
    struct Observation {
        u64 sequence{};
        u64 process_time_us{};
        FinalGuestSurfaceTransport transport{};
        FinalGuestSurfaceTilePlan plan{};
        std::vector<std::byte> bytes{};
    };

    struct Closest {
        u32 index{};
        u64 distance{};
    };

    [[nodiscard]] const Observation* FindObservation(u64 sequence) const noexcept {
        const auto observation = std::ranges::find(history, sequence, &Observation::sequence);
        return observation != history.end() ? &*observation : nullptr;
    }

    [[nodiscard]] u32 OrdinalFor(u64 identity) noexcept {
        for (u32 i = 0; i < identity_count; ++i) {
            if (identities[i] == identity) {
                return i + 1;
            }
        }
        if (identity_count == identities.size()) {
            return 0;
        }
        identities[identity_count++] = identity;
        return identity_count;
    }

    [[nodiscard]] std::optional<Closest> FindClosest(u64 current_time, u64 lag,
                                                     u32 end = MaxHistory) const noexcept {
        if (current_time < lag) {
            return std::nullopt;
        }
        const u64 target = current_time - lag;
        end = std::min<u32>(end, static_cast<u32>(history.size()));
        std::optional<Closest> closest;
        for (u32 i = 0; i < end; ++i) {
            const u64 observed = history[i].process_time_us;
            const u64 distance = observed > target ? observed - target : target - observed;
            if (distance <= config.tolerance_us && (!closest || distance < closest->distance)) {
                closest = Closest{i, distance};
            }
        }
        return closest;
    }

    [[nodiscard]] static bool SameLayout(const Observation& observation,
                                         const FinalGuestSurfaceTilePlan& plan) noexcept {
        return observation.plan == plan;
    }

    [[nodiscard]] static bool EqualContent(const Observation& observation,
                                           FinalGuestSurfaceFormat format,
                                           const FinalGuestSurfaceTilePlan& plan,
                                           std::span<const std::byte> bytes) noexcept {
        if (!SameLayout(observation, plan) || observation.transport.format != format ||
            observation.bytes.size() != bytes.size()) {
            return false;
        }
        const u64 output_bytes = static_cast<u64>(plan.row_bytes) * plan.surface_height;
        if (output_bytes > bytes.size()) {
            return false;
        }
        return EqualVisibleBytes(
            format,
            std::span<const std::byte>{observation.bytes}.first(static_cast<size_t>(output_bytes)),
            bytes.first(static_cast<size_t>(output_bytes)));
    }

    [[nodiscard]] static bool EqualVisibleBytes(FinalGuestSurfaceFormat format,
                                                std::span<const std::byte> left,
                                                std::span<const std::byte> right) noexcept {
        if (left.size() != right.size()) {
            return false;
        }
        u32 texel_bytes{};
        u32 visible_bytes{};
        bool mask_packed_alpha{};
        switch (format) {
        case FinalGuestSurfaceFormat::Rgba8:
        case FinalGuestSurfaceFormat::Bgra8:
            texel_bytes = 4;
            visible_bytes = 3;
            break;
        case FinalGuestSurfaceFormat::A2R10G10B10:
        case FinalGuestSurfaceFormat::A2B10G10R10:
            texel_bytes = 4;
            visible_bytes = 4;
            mask_packed_alpha = true;
            break;
        case FinalGuestSurfaceFormat::Rgba16:
        case FinalGuestSurfaceFormat::Rgba16Float:
            texel_bytes = 8;
            visible_bytes = 6;
            break;
        case FinalGuestSurfaceFormat::Block8:
        case FinalGuestSurfaceFormat::Block16:
            return std::equal(left.begin(), left.end(), right.begin());
        default:
            return false;
        }
        if (left.size() % texel_bytes != 0) {
            return false;
        }
        for (size_t offset = 0; offset < left.size(); offset += texel_bytes) {
            for (u32 byte = 0; byte < visible_bytes; ++byte) {
                if (mask_packed_alpha && byte == 3) {
                    if ((std::to_integer<u8>(left[offset + byte]) & 0x3f) !=
                        (std::to_integer<u8>(right[offset + byte]) & 0x3f)) {
                        return false;
                    }
                } else if (left[offset + byte] != right[offset + byte]) {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] static bool EqualTile(const Observation& observation, u32 index,
                                        FinalGuestSurfaceFormat format,
                                        const FinalGuestSurfaceTilePlan& plan,
                                        std::span<const std::byte> bytes) noexcept {
        if (!SameLayout(observation, plan) || observation.transport.format != format) {
            return false;
        }
        const auto tile = plan.TileAt(index);
        const u32 row_visible_bytes = tile.width * plan.bytes_per_pixel;
        if (tile.width == 0 || tile.height == 0 ||
            tile.buffer_offset + static_cast<u64>(tile.height - 1) * plan.row_bytes +
                    row_visible_bytes >
                bytes.size() ||
            observation.bytes.size() != bytes.size()) {
            return false;
        }
        for (u32 row = 0; row < tile.height; ++row) {
            const size_t offset = tile.buffer_offset + static_cast<size_t>(row) * plan.row_bytes;
            if (!EqualVisibleBytes(format,
                                   std::span<const std::byte>{observation.bytes}.subspan(
                                       offset, row_visible_bytes),
                                   bytes.subspan(offset, row_visible_bytes))) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] static bool EqualPairedSampleTile(const Observation& observation, u32 index,
                                                    const FinalGuestSurfaceTilePlan& plan,
                                                    std::span<const std::byte> bytes) noexcept {
        if (!SameLayout(observation, plan) ||
            plan.paired_sampled_format == FinalGuestSurfaceFormat::Unsupported ||
            plan.paired_sampled_row_bytes == 0 || observation.bytes.size() != bytes.size()) {
            return false;
        }
        const auto block = DescribeFinalGuestSurfaceFormat(plan.paired_sampled_format);
        const auto tile = plan.TileAt(index);
        if (block.width != 1 || block.height != 1 || block.bytes == 0 || tile.width == 0 ||
            tile.height == 0) {
            return false;
        }
        const u32 row_visible_bytes = tile.width * block.bytes;
        const u64 start = static_cast<u64>(plan.paired_sampled_offset) +
                          static_cast<u64>(tile.y) * plan.paired_sampled_row_bytes +
                          static_cast<u64>(tile.x) * block.bytes;
        const u64 end = start + static_cast<u64>(tile.height - 1) * plan.paired_sampled_row_bytes +
                        row_visible_bytes;
        if (end > bytes.size() ||
            end > static_cast<u64>(plan.paired_sampled_offset) + plan.paired_sampled_bytes) {
            return false;
        }
        for (u32 row = 0; row < tile.height; ++row) {
            const size_t offset = static_cast<size_t>(start) +
                                  static_cast<size_t>(row) * plan.paired_sampled_row_bytes;
            if (!EqualVisibleBytes(plan.paired_sampled_format,
                                   std::span<const std::byte>{observation.bytes}.subspan(
                                       offset, row_visible_bytes),
                                   bytes.subspan(offset, row_visible_bytes))) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] static bool EqualPairedBackingRegion(const Observation& observation,
                                                       u32 logical_ordinal,
                                                       const FinalGuestSurfaceTilePlan& plan,
                                                       std::span<const std::byte> bytes) noexcept {
        if (!SameLayout(observation, plan) ||
            (plan.paired_backing_format != FinalGuestSurfaceFormat::Rgba8 &&
             plan.paired_backing_format != FinalGuestSurfaceFormat::Bgra8) ||
            observation.bytes.size() != bytes.size()) {
            return false;
        }
        const auto region = std::ranges::find(plan.paired_backing_regions, logical_ordinal,
                                              &FinalGuestSurfacePairedRegion::logical_ordinal);
        if (region == plan.paired_backing_regions.begin() + plan.paired_backing_region_count ||
            region == plan.paired_backing_regions.end() || region->byte_size == 0 ||
            region->byte_size != static_cast<u64>(region->width) * region->height * 4) {
            return false;
        }
        const u64 start = static_cast<u64>(plan.paired_backing_offset) + region->buffer_offset;
        const u64 end = start + region->byte_size;
        if (end > bytes.size() ||
            end > static_cast<u64>(plan.paired_backing_offset) + plan.paired_backing_bytes) {
            return false;
        }
        return EqualVisibleBytes(plan.paired_backing_format,
                                 std::span<const std::byte>{observation.bytes}.subspan(
                                     static_cast<size_t>(start), region->byte_size),
                                 bytes.subspan(static_cast<size_t>(start), region->byte_size));
    }

    [[nodiscard]] static u32 ChangedTiles(const Observation& previous,
                                          FinalGuestSurfaceFormat format,
                                          const FinalGuestSurfaceTilePlan& plan,
                                          std::span<const std::byte> bytes) noexcept {
        if (!SameLayout(previous, plan) || previous.transport.format != format) {
            return plan.tile_count;
        }
        u32 changed{};
        for (u32 i = 0; i < plan.tile_count; ++i) {
            changed += !EqualTile(previous, i, format, plan, bytes);
        }
        return changed;
    }

    [[nodiscard]] static u16 ReadU16(std::span<const std::byte> bytes) noexcept {
        return static_cast<u16>(std::to_integer<u8>(bytes[0])) |
               static_cast<u16>(std::to_integer<u8>(bytes[1]) << 8);
    }

    [[nodiscard]] static float HalfToFloat(u16 value) noexcept {
        const u32 sign = static_cast<u32>(value & 0x8000u) << 16;
        u32 exponent = (value >> 10) & 0x1fu;
        u32 mantissa = value & 0x03ffu;
        u32 bits{};
        if (exponent == 0) {
            if (mantissa == 0) {
                bits = sign;
            } else {
                s32 adjusted_exponent = 1;
                while ((mantissa & 0x0400u) == 0) {
                    mantissa <<= 1;
                    --adjusted_exponent;
                }
                mantissa &= 0x03ffu;
                bits = sign | (static_cast<u32>(adjusted_exponent + 112) << 23) | (mantissa << 13);
            }
        } else if (exponent == 0x1fu) {
            bits = sign | 0x7f80'0000u | (mantissa << 13);
        } else {
            bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
        }
        return std::bit_cast<float>(bits);
    }

    [[nodiscard]] static std::optional<float> EncodePpSample(float linear, float gamma) noexcept {
        if (!std::isfinite(linear) || !std::isfinite(gamma) || gamma < 0.1f || gamma > 2.0f) {
            return std::nullopt;
        }
        constexpr float cutoff = 0.0031308f;
        constexpr float a = 1.055f;
        constexpr float b = 0.055f;
        constexpr float d = 12.92f;
        const float encoded =
            linear < cutoff ? d * linear / gamma : a * std::pow(linear, 1.0f / (3.4f - gamma)) - b;
        if (!std::isfinite(encoded)) {
            return std::nullopt;
        }
        return std::clamp(encoded, 0.0f, 1.0f);
    }

    [[nodiscard]] static std::optional<u32> ChangedVisualPixels(
        FinalGuestSurfaceFormat format, const FinalGuestSurfaceTilePlan& plan,
        std::span<const std::byte> left, std::span<const std::byte> right, u32 index) noexcept {
        if (format != FinalGuestSurfaceFormat::Rgba8 && format != FinalGuestSurfaceFormat::Bgra8 &&
            format != FinalGuestSurfaceFormat::Rgba16Float) {
            return std::nullopt;
        }
        const auto tile = plan.TileAt(index);
        const u32 texel_bytes = format == FinalGuestSurfaceFormat::Rgba16Float ? 8u : 4u;
        const u32 row_bytes = tile.width * plan.bytes_per_pixel;
        if (tile.width == 0 || tile.height == 0 || plan.bytes_per_pixel != texel_bytes ||
            left.size() != right.size() ||
            tile.buffer_offset + static_cast<u64>(tile.height - 1) * plan.row_bytes + row_bytes >
                left.size()) {
            return std::nullopt;
        }
        u32 changed{};
        for (u32 row = 0; row < tile.height; ++row) {
            const size_t row_offset =
                tile.buffer_offset + static_cast<size_t>(row) * plan.row_bytes;
            for (u32 column = 0; column < tile.width; ++column) {
                const size_t offset = row_offset + static_cast<size_t>(column) * texel_bytes;
                if (format == FinalGuestSurfaceFormat::Rgba16Float) {
                    const float gamma = std::bit_cast<float>(plan.comparison_gamma_bits);
                    float difference{};
                    for (u32 channel = 0; channel < 3; ++channel) {
                        const auto first = EncodePpSample(
                            HalfToFloat(ReadU16(left.subspan(offset + channel * 2, 2))), gamma);
                        const auto second = EncodePpSample(
                            HalfToFloat(ReadU16(right.subspan(offset + channel * 2, 2))), gamma);
                        if (!first || !second) {
                            return std::nullopt;
                        }
                        difference += std::abs(*first - *second);
                    }
                    changed += difference >= 48.0f / 255.0f;
                } else {
                    u32 difference{};
                    for (u32 channel = 0; channel < 3; ++channel) {
                        const u8 first = std::to_integer<u8>(left[offset + channel]);
                        const u8 second = std::to_integer<u8>(right[offset + channel]);
                        difference += first > second ? first - second : second - first;
                    }
                    changed += difference >= 48;
                }
            }
        }
        return changed;
    }

    [[nodiscard]] static bool IsLocalizedVisualReturn(const Observation& baseline,
                                                      const Observation& departure,
                                                      FinalGuestSurfaceFormat format,
                                                      const FinalGuestSurfaceTilePlan& plan,
                                                      std::span<const std::byte> bytes,
                                                      u32 index) noexcept {
        if (!SameLayout(baseline, plan) || !SameLayout(departure, plan) ||
            baseline.transport.format != format || departure.transport.format != format) {
            return false;
        }
        const auto baseline_to_departure =
            ChangedVisualPixels(format, plan, baseline.bytes, departure.bytes, index);
        const auto departure_to_current =
            ChangedVisualPixels(format, plan, departure.bytes, bytes, index);
        const auto baseline_to_current =
            ChangedVisualPixels(format, plan, baseline.bytes, bytes, index);
        if (!baseline_to_departure || !departure_to_current || !baseline_to_current) {
            return false;
        }
        const auto tile = plan.TileAt(index);
        const u64 pixels = static_cast<u64>(tile.width) * tile.height;
        return static_cast<u64>(*baseline_to_departure) * 4 >= pixels &&
               static_cast<u64>(*departure_to_current) * 4 >= pixels &&
               static_cast<u64>(*baseline_to_current) * 100 <= pixels;
    }

    void PopulateAbaDetails(const Observation& baseline, const Observation& departure,
                            FinalGuestSurfaceFormat format, const FinalGuestSurfaceTilePlan& plan,
                            std::span<const std::byte> bytes,
                            FinalGuestSurfaceReport& report) const noexcept {
        if (!SameLayout(baseline, plan) || !SameLayout(departure, plan) ||
            baseline.transport.format != format || departure.transport.format != format) {
            return;
        }
        const bool all_returned = EqualContent(baseline, format, plan, bytes);
        const bool any_departed = !EqualContent(baseline, format, departure.plan, departure.bytes);
        bool any_exact_tile{};
        for (u32 i = 0; i < plan.tile_count; ++i) {
            const bool returned = EqualTile(baseline, i, format, plan, bytes);
            const bool departed = !EqualTile(baseline, i, format, departure.plan, departure.bytes);
            any_exact_tile |= returned && departed;
            const bool localized =
                plan.comparison == FinalGuestSurfaceComparison::LocalizedVisualReturn ||
                plan.comparison == FinalGuestSurfaceComparison::SampledLinearVisualReturn;
            const bool matched =
                localized ? IsLocalizedVisualReturn(baseline, departure, format, plan, bytes, i)
                          : returned && departed;
            if (!matched) {
                continue;
            }
            ++report.aba_tiles;
            if (!selector.Contains(i + 1)) {
                ++report.unselected_aba_tiles;
            } else if (report.aba_tile_ordinal_count < report.aba_tile_ordinals.size()) {
                report.aba_tile_ordinals[report.aba_tile_ordinal_count++] = i + 1;
            } else {
                ++report.loss.tile_detail;
            }
        }
        report.exact_aba = any_exact_tile;
        report.localized_aba =
            (plan.comparison == FinalGuestSurfaceComparison::LocalizedVisualReturn ||
             plan.comparison == FinalGuestSurfaceComparison::SampledLinearVisualReturn) &&
            report.aba_tiles != 0;
        report.whole_sample_aba = all_returned && any_departed;
    }

    void Push(u64 sequence, u64 process_time_us, FinalGuestSurfaceTransport transport,
              const FinalGuestSurfaceTilePlan& plan, std::span<const std::byte> bytes) {
        if (history.size() == MaxHistory) {
            history.pop_front();
            history_evicted = true;
        }
        history.push_back({
            .sequence = sequence,
            .process_time_us = process_time_us,
            .transport = transport,
            .plan = plan,
            .bytes = std::vector<std::byte>{bytes.begin(), bytes.end()},
        });
    }

    FinalGuestSurfaceLagConfig config{};
    FinalGuestSurfaceWatchOrdinals selector{};
    std::deque<Observation> history{};
    std::array<u64, MaxSurfaceOrdinals> identities{};
    u32 identity_count{};
    u64 last_sequence{};
    u64 last_process_time_us{};
    bool has_last{};
    bool history_evicted{};
};

class FinalGuestSurfaceCalibratedTriplets {
public:
    static constexpr u32 MaxPendingReports = 16;

    explicit FinalGuestSurfaceCalibratedTriplets(bool enabled_,
                                                 FinalGuestSurfaceWatchOrdinals selector_,
                                                 FinalGuestSurfaceCaptureWindow window_,
                                                 u32 expected_calibrations_)
        : enabled{enabled_}, selector{selector_}, window{window_},
          expected_calibrations{expected_calibrations_} {
        if (!enabled) {
            return;
        }
        if (expected_calibrations == 0 ||
            expected_calibrations > FinalGuestSurfaceMaxScreenshotRequests) {
            poisoned = true;
            coverage.loss.overflow = 1;
            return;
        }
        calibrations.resize(expected_calibrations + 1);
        classified.resize(expected_calibrations + 1);
        eligible.resize(expected_calibrations + 1);
    }

    void ObserveCalibration(FinalGuestSurfaceCalibratedStamp stamp,
                            const FinalGuestSurfaceReducer& reducer) noexcept {
        if (!enabled || finished || poisoned) {
            return;
        }
        if (selector.status != FinalGuestSurfaceStatus::Complete || selector.loss != 0 ||
            selector.count == 0) {
            Poison(&FinalGuestSurfaceCalibratedLoss::selector);
            return;
        }
        if (stamp.request_ordinal == 0 || stamp.request_ordinal > expected_calibrations ||
            stamp.request_ordinal > FinalGuestSurfaceMaxScreenshotRequests) {
            Poison(&FinalGuestSurfaceCalibratedLoss::overflow);
            return;
        }
        if (calibrations[stamp.request_ordinal].has_value()) {
            Poison(&FinalGuestSurfaceCalibratedLoss::duplicate);
            return;
        }
        calibrations[stamp.request_ordinal] = stamp;
        ++coverage.calibrations;
        Reconcile(reducer);
    }

    void Reconcile(const FinalGuestSurfaceReducer& reducer) noexcept {
        if (!enabled || finished || poisoned) {
            return;
        }
        for (u32 request = 1; request <= expected_calibrations; ++request) {
            if (classified[request] || !calibrations[request]) {
                continue;
            }
            if (request < 3) {
                classified[request] = true;
                ++coverage.outside;
                continue;
            }
            if (!calibrations[request - 2] || !calibrations[request - 1]) {
                continue;
            }
            const auto& a = *calibrations[request - 2];
            const auto& b = *calibrations[request - 1];
            const auto& c = *calibrations[request];
            if (!window.Contains(a.sequence) || !window.Contains(b.sequence) ||
                !window.Contains(c.sequence)) {
                classified[request] = true;
                ++coverage.outside;
                continue;
            }
            if (!eligible[request]) {
                eligible[request] = true;
                ++coverage.eligible;
            }
            if (auto report = reducer.EvaluateCalibratedTriplet(a, b, c, false)) {
                Complete(request, std::move(*report));
            }
        }
    }

    void Finish(const FinalGuestSurfaceReducer& reducer) noexcept {
        if (!enabled || finished) {
            return;
        }
        if (!poisoned) {
            Reconcile(reducer);
            for (u32 request = 1; request <= expected_calibrations; ++request) {
                if (classified[request]) {
                    continue;
                }
                if (!calibrations[request]) {
                    ++coverage.loss.gap;
                    continue;
                }
                if (request < 3) {
                    classified[request] = true;
                    ++coverage.outside;
                    continue;
                }
                if (!calibrations[request - 2] || !calibrations[request - 1]) {
                    ++coverage.loss.gap;
                    classified[request] = true;
                    continue;
                }
                const auto& a = *calibrations[request - 2];
                const auto& b = *calibrations[request - 1];
                const auto& c = *calibrations[request];
                if (!window.Contains(a.sequence) || !window.Contains(b.sequence) ||
                    !window.Contains(c.sequence)) {
                    classified[request] = true;
                    ++coverage.outside;
                    continue;
                }
                if (!eligible[request]) {
                    eligible[request] = true;
                    ++coverage.eligible;
                }
                if (auto report = reducer.EvaluateCalibratedTriplet(a, b, c, true)) {
                    Complete(request, std::move(*report));
                }
            }
        }
        finished = true;
    }

    [[nodiscard]] std::vector<FinalGuestSurfaceCalibratedReport> TakeReports() {
        std::vector<FinalGuestSurfaceCalibratedReport> result;
        result.reserve(reports.size());
        while (!reports.empty()) {
            result.push_back(std::move(reports.front()));
            reports.pop_front();
        }
        return result;
    }

    [[nodiscard]] constexpr const FinalGuestSurfaceCalibratedCoverage& GetCoverage()
        const noexcept {
        return coverage;
    }

    [[nodiscard]] constexpr bool CoverageReady() const noexcept {
        return enabled && finished;
    }

    [[nodiscard]] static constexpr size_t RetainedContentObservationCount() noexcept {
        return 0;
    }

    [[nodiscard]] size_t RetainedCalibrationCapacity() const noexcept {
        return calibrations.size();
    }

private:
    using LossMember = u32 FinalGuestSurfaceCalibratedLoss::*;

    void Accumulate(const FinalGuestSurfaceCalibratedLoss& loss) noexcept {
        coverage.loss.gap += loss.gap;
        coverage.loss.history += loss.history;
        coverage.loss.duplicate += loss.duplicate;
        coverage.loss.overflow += loss.overflow;
        coverage.loss.invalid += loss.invalid;
        coverage.loss.transport += loss.transport;
        coverage.loss.time += loss.time;
        coverage.loss.selector += loss.selector;
    }

    void Poison(LossMember member) noexcept {
        ++(coverage.loss.*member);
        reports.clear();
        poisoned = true;
    }

    void Complete(u32 request, FinalGuestSurfaceCalibratedReport report) noexcept {
        classified[request] = true;
        if (reports.size() == MaxPendingReports) {
            Poison(&FinalGuestSurfaceCalibratedLoss::overflow);
            return;
        }
        ++coverage.emitted;
        coverage.complete +=
            report.status == FinalGuestSurfaceStatus::Complete && !report.loss.Any();
        Accumulate(report.loss);
        reports.push_back(std::move(report));
    }

    bool enabled{};
    bool finished{};
    bool poisoned{};
    FinalGuestSurfaceWatchOrdinals selector{};
    FinalGuestSurfaceCaptureWindow window{};
    u32 expected_calibrations{};
    std::vector<std::optional<FinalGuestSurfaceCalibratedStamp>> calibrations{};
    std::vector<bool> classified{};
    std::vector<bool> eligible{};
    std::deque<FinalGuestSurfaceCalibratedReport> reports{};
    FinalGuestSurfaceCalibratedCoverage coverage{};
};

} // namespace Vulkan
