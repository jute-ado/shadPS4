// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <deque>
#include <limits>
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
    Block8,
    Block16,
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
};

struct FinalGuestSurfaceLoss {
    u32 unsupported_type{};
    u32 unsupported_samples{};
    u32 unsupported_mip{};
    u32 unsupported_layer{};
    u32 unsupported_aspect{};
    u32 unsupported_format{};
    u32 invalid_extent{};
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
               byte_capacity || ordinal_capacity || busy || invalidation || gap || history ||
               tile_detail;
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
    u32 row_bytes{};
    u32 bytes_per_pixel{};
    u32 window_width{};
    u32 window_height{};
    u32 logical_columns{};
    u32 logical_rows{};
    u32 tile_count{};
    u32 sample_bytes{};
    u32 copy_region_count{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::Complete};
    FinalGuestSurfaceLoss loss{};

    [[nodiscard]] constexpr FinalGuestSurfaceTile TileAt(u32 ordinal_index) const noexcept {
        if (ordinal_index >= tile_count || logical_columns == 0) {
            return {};
        }
        const u32 column = ordinal_index % logical_columns;
        const u32 row = ordinal_index / logical_columns;
        const u32 x = column * WindowStride;
        const u32 y = row * WindowStride;
        return {
            .x = x,
            .y = y,
            .width = window_width,
            .height = window_height,
            .buffer_offset = y * row_bytes + x * bytes_per_pixel,
            .byte_size = window_width * window_height * bytes_per_pixel,
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
    if (desc.width == 0 || desc.height == 0 || desc.depth != 1 ||
        (desc.type == FinalGuestSurfaceImageType::Color1D && desc.height != 1)) {
        return unsupported(&FinalGuestSurfaceLoss::invalid_extent);
    }

    const u32 window_width = std::min(desc.width, FinalGuestSurfaceTilePlan::WindowExtent);
    const u32 window_height = std::min(desc.height, FinalGuestSurfaceTilePlan::WindowExtent);
    const u32 columns =
        desc.width <= window_width
            ? 1
            : (desc.width - window_width) / FinalGuestSurfaceTilePlan::WindowStride + 1;
    const u32 rows =
        desc.type == FinalGuestSurfaceImageType::Color1D || desc.height <= window_height
            ? 1
            : (desc.height - window_height) / FinalGuestSurfaceTilePlan::WindowStride + 1;
    const u64 tile_count = static_cast<u64>(columns) * rows;
    if (tile_count > limits.max_tiles || tile_count > FinalGuestSurfaceTilePlan::MaxTiles) {
        FinalGuestSurfaceTilePlan result{};
        result.status = FinalGuestSurfaceStatus::CapacityLoss;
        result.loss.tile_capacity = 1;
        return result;
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
    return {
        .surface_width = desc.width,
        .surface_height = desc.height,
        .row_bytes = static_cast<u32>(row_bytes),
        .bytes_per_pixel = block.bytes,
        .window_width = window_width,
        .window_height = window_height,
        .logical_columns = columns,
        .logical_rows = rows,
        .tile_count = static_cast<u32>(tile_count),
        .sample_bytes = static_cast<u32>(sample_bytes),
        .copy_region_count = 1,
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

struct FinalGuestSurfaceContentConfig {
    FinalGuestSurfaceCaptureWindow window{FinalGuestSurfaceCaptureWindow::Defaults()};
    FinalGuestSurfaceLagConfig lag{FinalGuestSurfaceLagConfig::Defaults()};
    FinalGuestSurfaceWatchOrdinals watch_ordinals{};
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

    [[nodiscard]] constexpr bool IsIdentity() const noexcept {
        return guest_width != 0 && guest_height != 0 && guest_width == swapchain_width &&
               guest_height == swapchain_height && output_x == 0 && output_y == 0 &&
               output_width == guest_width && output_height == guest_height && top_left &&
               no_y_flip;
    }
};

struct FinalGuestSurfaceCalibrationReport {
    u32 request_ordinal{};
    u64 surface_sequence{};
    u64 surface_process_time_us{};
    FinalGuestSurfacePresentationMapping mapping{};
    u32 overflow_loss{};
    bool emit{};
    bool fallback_time{};
    bool identity_mapping{};
    bool overflow_marker{};
};

class FinalGuestSurfaceScreenshotCalibration {
public:
    static constexpr u32 MaxRequests = 1000;

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
        return {
            .request_ordinal = observed_requests,
            .surface_sequence = stamp.valid ? stamp.surface_sequence : 0,
            .surface_process_time_us =
                stamp.valid ? stamp.surface_process_time_us : fallback_process_time_us,
            .mapping = mapping,
            .emit = true,
            .fallback_time = !stamp.valid,
            .identity_mapping = stamp.valid && mapping.IsIdentity(),
        };
    }

    [[nodiscard]] constexpr u32 ObservedRequests() const noexcept {
        return observed_requests;
    }

private:
    bool enabled{};
    bool overflow_emitted{};
    u32 observed_requests{};
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
    bool stable_transport{};
    bool exact_aba{};
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
        return EqualVisibleBytes(format, observation.bytes, bytes);
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
        for (u32 i = 0; i < plan.tile_count; ++i) {
            const bool returned = EqualTile(baseline, i, format, plan, bytes);
            const bool departed = !EqualTile(baseline, i, format, departure.plan, departure.bytes);
            if (!returned || !departed) {
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
        report.exact_aba = report.aba_tiles != 0;
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

} // namespace Vulkan
