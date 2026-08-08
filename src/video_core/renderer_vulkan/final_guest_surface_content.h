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
    u32 max_tiles{16 * 9};
    u32 max_bytes{16u << 20};
};

struct FinalGuestSurfaceTilePlan {
    static constexpr u32 MaxTiles = 16 * 9;

    std::array<FinalGuestSurfaceTile, MaxTiles> tiles{};
    u32 tile_count{};
    u32 sample_bytes{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::Complete};
    FinalGuestSurfaceLoss loss{};
};

namespace Detail {

[[nodiscard]] constexpr u32 AlignDown(u32 value, u32 alignment) noexcept {
    return value / alignment * alignment;
}

[[nodiscard]] constexpr std::optional<u32> AlignUpChecked(u32 value, u32 alignment) noexcept {
    if (alignment == 0 || value > std::numeric_limits<u32>::max() - (alignment - 1)) {
        return std::nullopt;
    }
    return (value + alignment - 1) / alignment * alignment;
}

} // namespace Detail

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

    FinalGuestSurfaceTilePlan candidate{};
    const u32 block_columns = (desc.width + block.width - 1) / block.width;
    const u32 block_rows = (desc.height + block.height - 1) / block.height;
    const u32 columns = std::min(16u, block_columns);
    const u32 rows =
        desc.type == FinalGuestSurfaceImageType::Color1D ? 1 : std::min(9u, block_rows);
    for (u32 row = 0; row < rows; ++row) {
        for (u32 column = 0; column < columns; ++column) {
            const u32 start_block_x =
                static_cast<u32>(static_cast<u64>(column) * block_columns / columns);
            const u32 end_block_x =
                static_cast<u32>(static_cast<u64>(column + 1) * block_columns / columns);
            const u32 start_block_y = static_cast<u32>(static_cast<u64>(row) * block_rows / rows);
            const u32 end_block_y = static_cast<u32>(static_cast<u64>(row + 1) * block_rows / rows);
            const u32 x = start_block_x * block.width;
            const u32 y = start_block_y * block.height;
            const u32 end_x = std::min(desc.width, end_block_x * block.width);
            const u32 end_y = std::min(desc.height, end_block_y * block.height);
            const u32 width = end_x - x;
            const u32 height = end_y - y;

            if (candidate.tile_count >= limits.max_tiles ||
                candidate.tile_count >= candidate.tiles.size()) {
                FinalGuestSurfaceTilePlan result{};
                result.status = FinalGuestSurfaceStatus::CapacityLoss;
                result.loss.tile_capacity = 1;
                return result;
            }
            const u64 blocks_x = (static_cast<u64>(width) + block.width - 1) / block.width;
            const u64 blocks_y = (static_cast<u64>(height) + block.height - 1) / block.height;
            const u64 byte_size = blocks_x * blocks_y * block.bytes;
            const u32 buffer_alignment = std::max(4u, block.bytes);
            const auto aligned_offset =
                Detail::AlignUpChecked(candidate.sample_bytes, buffer_alignment);
            if (!aligned_offset || byte_size > std::numeric_limits<u32>::max() ||
                *aligned_offset > limits.max_bytes ||
                byte_size > limits.max_bytes - *aligned_offset) {
                FinalGuestSurfaceTilePlan result{};
                result.status = FinalGuestSurfaceStatus::CapacityLoss;
                result.loss.byte_capacity = 1;
                return result;
            }
            candidate.tiles[candidate.tile_count++] = {
                .x = x,
                .y = y,
                .width = width,
                .height = height,
                .buffer_offset = *aligned_offset,
                .byte_size = static_cast<u32>(byte_size),
            };
            candidate.sample_bytes = static_cast<u32>(*aligned_offset + byte_size);
        }
    }
    return candidate;
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

struct FinalGuestSurfaceContentConfig {
    FinalGuestSurfaceCaptureWindow window{FinalGuestSurfaceCaptureWindow::Defaults()};
    FinalGuestSurfaceLagConfig lag{FinalGuestSurfaceLagConfig::Defaults()};
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
    return config;
}

struct FinalGuestSurfaceReport {
    static constexpr u32 MaxTileDetails = FinalGuestSurfaceTilePlan::MaxTiles;

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
    std::array<u32, MaxTileDetails> aba_tile_ordinals{};
    u32 aba_tile_ordinal_count{};
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
           " aba_tile_ordinals=" + tile_ordinals +
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

    explicit FinalGuestSurfaceReducer(FinalGuestSurfaceLagConfig config_) : config{config_} {}

    [[nodiscard]] FinalGuestSurfaceReport Observe(u64 sequence, u64 process_time_us,
                                                  FinalGuestSurfaceTransport transport,
                                                  const FinalGuestSurfaceTilePlan& plan,
                                                  std::span<const std::byte> bytes) {
        FinalGuestSurfaceReport report{
            .sequence = sequence,
            .process_time_us = process_time_us,
            .surface_ordinal = OrdinalFor(transport.surface_identity),
            .tile_count = plan.tile_count,
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
        if (observation.plan.tile_count != plan.tile_count ||
            observation.plan.sample_bytes != plan.sample_bytes) {
            return false;
        }
        return std::equal(observation.plan.tiles.begin(),
                          observation.plan.tiles.begin() + plan.tile_count, plan.tiles.begin());
    }

    [[nodiscard]] static bool EqualContent(const Observation& observation,
                                           FinalGuestSurfaceFormat format,
                                           const FinalGuestSurfaceTilePlan& plan,
                                           std::span<const std::byte> bytes) noexcept {
        if (!SameLayout(observation, plan) || observation.transport.format != format ||
            observation.bytes.size() != bytes.size()) {
            return false;
        }
        for (u32 i = 0; i < plan.tile_count; ++i) {
            if (!EqualTile(observation, i, format, plan, bytes)) {
                return false;
            }
        }
        return true;
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
        const auto& tile = plan.tiles[index];
        return EqualVisibleBytes(format,
                                 std::span<const std::byte>{observation.bytes}.subspan(
                                     tile.buffer_offset, tile.byte_size),
                                 bytes.subspan(tile.buffer_offset, tile.byte_size));
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

    static void PopulateAbaDetails(const Observation& baseline, const Observation& departure,
                                   FinalGuestSurfaceFormat format,
                                   const FinalGuestSurfaceTilePlan& plan,
                                   std::span<const std::byte> bytes,
                                   FinalGuestSurfaceReport& report) noexcept {
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
            if (report.aba_tile_ordinal_count < report.aba_tile_ordinals.size()) {
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
    std::deque<Observation> history{};
    std::array<u64, MaxSurfaceOrdinals> identities{};
    u32 identity_count{};
    u64 last_sequence{};
    u64 last_process_time_us{};
    bool has_last{};
    bool history_evicted{};
};

} // namespace Vulkan
