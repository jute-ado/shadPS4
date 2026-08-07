// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>

#include "common/types.h"

namespace VideoCore {

enum class DrawTopologyKind : u32 {
    Direct,
    DirectIndexed,
    Indirect,
    IndirectIndexed,
    Count,
};

enum class DrawTopologyResult : u32 {
    Submitted,
    FilterFastClear,
    FilterFmaskDecompress,
    FilterResolve,
    FilterPrimitiveNone,
    FilterDepthStencilCopy,
    MissingPipeline,
    BindingFailed,
    Count,
};

enum class OcclusionEventKind : u32 {
    Control,
    Dump,
    Reset,
    Count,
};

enum class DrawTopologyPacket : u32 {
    DrawIndex2,
    DrawIndexOffset2,
    DrawIndexAuto,
    DrawIndirect,
    DrawIndirectMulti,
    DrawIndexIndirect,
    DrawIndexIndirectMulti,
    DrawIndexIndirectCountMulti,
    SetBase,
    IndexBufferSize,
    SetPredication,
    Count,
};

struct FrameDrawTopologySnapshot {
    bool should_report{};
    u64 sequence{};
    u64 direct{};
    u64 direct_indexed{};
    u64 indirect{};
    u64 indirect_indexed{};
    u64 submitted{};
    u64 filtered{};
    u64 filter_fast_clear{};
    u64 filter_fmask_decompress{};
    u64 filter_resolve{};
    u64 filter_primitive_none{};
    u64 filter_depth_stencil_copy{};
    u64 missing_pipeline{};
    u64 binding_failed{};
    u64 occlusion_control{};
    u64 occlusion_dump{};
    u64 occlusion_reset{};
    u64 set_base{};
    u64 index_buffer_size{};
    u64 set_predication{};
    u64 packet_hash{};
};

class FrameDrawTopologyDiagnostic {
public:
    explicit FrameDrawTopologyDiagnostic(u64 report_limit_) : report_limit{report_limit_} {}

    static constexpr u64 EmptyPacketHash() noexcept {
        return 1469598103934665603ULL;
    }

    void ObservePacket(DrawTopologyPacket packet) noexcept {
        packet_counts[static_cast<size_t>(packet)].fetch_add(1, std::memory_order_relaxed);
        u64 hash = packet_hash.load(std::memory_order_relaxed);
        const u64 value = static_cast<u64>(packet) + 1;
        while (!packet_hash.compare_exchange_weak(hash, (hash ^ value) * 1099511628211ULL,
                                                  std::memory_order_relaxed)) {
        }
    }

    void ObserveDraw(DrawTopologyKind kind) noexcept {
        draw_counts[static_cast<size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
    }

    void ObserveDrawResult(DrawTopologyResult result) noexcept {
        result_counts[static_cast<size_t>(result)].fetch_add(1, std::memory_order_relaxed);
    }

    void ObserveOcclusion(OcclusionEventKind kind) noexcept {
        occlusion_counts[static_cast<size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
    }

    FrameDrawTopologySnapshot TakeSnapshot() noexcept {
        const u64 sequence = frame_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
        const u64 filter_fast_clear = Take(result_counts, DrawTopologyResult::FilterFastClear);
        const u64 filter_fmask_decompress =
            Take(result_counts, DrawTopologyResult::FilterFmaskDecompress);
        const u64 filter_resolve = Take(result_counts, DrawTopologyResult::FilterResolve);
        const u64 filter_primitive_none =
            Take(result_counts, DrawTopologyResult::FilterPrimitiveNone);
        const u64 filter_depth_stencil_copy =
            Take(result_counts, DrawTopologyResult::FilterDepthStencilCopy);
        return {
            .should_report = sequence <= report_limit,
            .sequence = sequence,
            .direct = Take(draw_counts, DrawTopologyKind::Direct),
            .direct_indexed = Take(draw_counts, DrawTopologyKind::DirectIndexed),
            .indirect = Take(draw_counts, DrawTopologyKind::Indirect),
            .indirect_indexed = Take(draw_counts, DrawTopologyKind::IndirectIndexed),
            .submitted = Take(result_counts, DrawTopologyResult::Submitted),
            .filtered = filter_fast_clear + filter_fmask_decompress + filter_resolve +
                        filter_primitive_none + filter_depth_stencil_copy,
            .filter_fast_clear = filter_fast_clear,
            .filter_fmask_decompress = filter_fmask_decompress,
            .filter_resolve = filter_resolve,
            .filter_primitive_none = filter_primitive_none,
            .filter_depth_stencil_copy = filter_depth_stencil_copy,
            .missing_pipeline = Take(result_counts, DrawTopologyResult::MissingPipeline),
            .binding_failed = Take(result_counts, DrawTopologyResult::BindingFailed),
            .occlusion_control = Take(occlusion_counts, OcclusionEventKind::Control),
            .occlusion_dump = Take(occlusion_counts, OcclusionEventKind::Dump),
            .occlusion_reset = Take(occlusion_counts, OcclusionEventKind::Reset),
            .set_base = Take(packet_counts, DrawTopologyPacket::SetBase),
            .index_buffer_size = Take(packet_counts, DrawTopologyPacket::IndexBufferSize),
            .set_predication = Take(packet_counts, DrawTopologyPacket::SetPredication),
            .packet_hash = packet_hash.exchange(EmptyPacketHash(), std::memory_order_relaxed),
        };
    }

private:
    template <typename Enum, size_t Size>
    static u64 Take(std::array<std::atomic<u64>, Size>& counts, Enum index) noexcept {
        return counts[static_cast<size_t>(index)].exchange(0, std::memory_order_relaxed);
    }

    u64 report_limit;
    std::atomic<u64> frame_sequence{};
    std::array<std::atomic<u64>, static_cast<size_t>(DrawTopologyKind::Count)> draw_counts{};
    std::array<std::atomic<u64>, static_cast<size_t>(DrawTopologyResult::Count)> result_counts{};
    std::array<std::atomic<u64>, static_cast<size_t>(OcclusionEventKind::Count)> occlusion_counts{};
    std::array<std::atomic<u64>, static_cast<size_t>(DrawTopologyPacket::Count)> packet_counts{};
    std::atomic<u64> packet_hash{EmptyPacketHash()};
};

} // namespace VideoCore
