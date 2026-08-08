// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <vector>

#include "video_core/amdgpu/input_assembly_device_integrity.h"

namespace AmdGpu {

enum class InputAssemblyHostUsage : u8 {
    Stream,
    DeviceLocal,
};

struct BoundInputAssemblySource {
    InputAssemblyBufferToken token{};
    InputAssemblyHostUsage usage{};
    u64 host_offset{};
    u64 host_size{};
    u64 write_serial{};
    InputAssemblyAuthority authority{InputAssemblyAuthority::Unknown};
};

struct NormalizedInputAssemblyRange {
    InputAssemblySemanticOrdinal semantic{};
    InputAssemblyBufferToken source{};
    InputAssemblyHostUsage usage{};
    u64 source_offset{};
    u64 source_size{};
    u64 size{};
    u64 write_serial{};
    InputAssemblyAuthority authority{InputAssemblyAuthority::Unknown};
};

[[nodiscard]] inline std::optional<NormalizedInputAssemblyRange> NormalizeVertexInputRange(
    const BoundInputAssemblySource& source, u64 merged_guest_base, u64 binding_guest_base,
    u64 binding_size, InputAssemblySemanticOrdinal semantic) noexcept {
    if (!source.token || binding_size == 0 || binding_guest_base < merged_guest_base) {
        return std::nullopt;
    }
    const u64 relative = binding_guest_base - merged_guest_base;
    if (relative > std::numeric_limits<u64>::max() - source.host_offset) {
        return std::nullopt;
    }
    const u64 offset = source.host_offset + relative;
    if (offset > source.host_size || binding_size > source.host_size - offset) {
        return std::nullopt;
    }
    return NormalizedInputAssemblyRange{
        .semantic = semantic,
        .source = source.token,
        .usage = source.usage,
        .source_offset = offset,
        .source_size = source.host_size,
        .size = binding_size,
        .write_serial = source.write_serial,
        .authority = source.authority,
    };
}

[[nodiscard]] inline std::optional<NormalizedInputAssemblyRange> NormalizeIndexInputRange(
    const BoundInputAssemblySource& source, u64 index_offset, u64 index_stride, u64 index_count,
    InputAssemblySemanticOrdinal semantic) noexcept {
    if (!source.token || index_stride == 0 || index_count == 0 ||
        index_offset > std::numeric_limits<u64>::max() / index_stride ||
        index_count > std::numeric_limits<u64>::max() / index_stride) {
        return std::nullopt;
    }
    const u64 relative = index_offset * index_stride;
    const u64 size = index_count * index_stride;
    if (relative > std::numeric_limits<u64>::max() - source.host_offset) {
        return std::nullopt;
    }
    const u64 offset = source.host_offset + relative;
    if (offset > source.host_size || size > source.host_size - offset) {
        return std::nullopt;
    }
    return NormalizedInputAssemblyRange{
        .semantic = semantic,
        .source = source.token,
        .usage = source.usage,
        .source_offset = offset,
        .source_size = source.host_size,
        .size = size,
        .write_serial = source.write_serial,
        .authority = source.authority,
    };
}

struct InputAssemblyReadbackSample {
    InputAssemblyBufferToken source{};
    u32 capture_draw{};
    u64 source_offset{};
    u64 write_serial{};
    u32 destination_offset{};
    u32 size{};
    InputAssemblyHostUsage usage{};
    InputAssemblyAuthority authority{InputAssemblyAuthority::Unknown};
};

struct InputAssemblyReadbackReference {
    u32 sample_index{};
    u32 destination_offset{};
    u32 size{};
};

struct InputAssemblyReadbackSemantic {
    static constexpr u32 MaxReferences = 3;

    InputAssemblySemanticOrdinal semantic{};
    std::array<InputAssemblyReadbackReference, MaxReferences> references{};
    u32 reference_count{};
};

struct InputAssemblyReadbackLoss {
    u32 invalid_range{};
    u32 source_conflict{};
    u32 sample_capacity{};
    u32 byte_capacity{};
    u32 semantic_capacity{};

    [[nodiscard]] constexpr bool Any() const noexcept {
        return invalid_range != 0 || source_conflict != 0 || sample_capacity != 0 ||
               byte_capacity != 0 || semantic_capacity != 0;
    }
};

struct InputAssemblyReadbackFramePlan {
    std::vector<InputAssemblyReadbackSample> samples{};
    std::vector<InputAssemblyReadbackSemantic> semantics{};
    u64 sequence{};
    u32 sample_bytes{};
    u32 sample_count{};
    u32 semantic_count{};
    InputAssemblyReadbackLoss loss{};
    bool complete{true};
};

struct InputAssemblyCaptureDecision {
    std::array<InputAssemblyReadbackReference, 3> references{};
    std::array<u32, 3> new_copy_indices{};
    u32 reference_count{};
    u32 new_copy_count{};
    bool accepted{};
};

class InputAssemblyDeviceReadbackPlanner {
public:
    static constexpr u32 SampleBytes = 32;
    static constexpr u32 MaxPhysicalSampleBytes = SampleBytes + 4;
    static constexpr u32 MaxSamplesPerFrame = 2048;
    static constexpr u32 MaxSemanticsPerFrame = 8192;
    static constexpr u32 MaxSampleBytesPerFrame = MaxSamplesPerFrame * MaxPhysicalSampleBytes;

    InputAssemblyDeviceReadbackPlanner() {
        plan.samples.reserve(MaxSamplesPerFrame);
        plan.semantics.reserve(MaxSemanticsPerFrame);
    }

    void BeginFrame(u64 sequence) {
        plan.samples.clear();
        plan.semantics.clear();
        plan.sequence = sequence;
        plan.sample_bytes = 0;
        plan.sample_count = 0;
        plan.semantic_count = 0;
        plan.loss = {};
        plan.complete = true;
        frame_active = true;
    }

    [[nodiscard]] InputAssemblyCaptureDecision Plan(
        const NormalizedInputAssemblyRange& range) noexcept {
        InputAssemblyCaptureDecision decision{};
        if (!frame_active || !range.source || range.size == 0 ||
            range.source_offset > range.source_size ||
            range.size > range.source_size - range.source_offset) {
            Reject(plan.loss.invalid_range);
            return decision;
        }
        if (plan.semantic_count >= MaxSemanticsPerFrame) {
            Reject(plan.loss.semantic_capacity);
            return decision;
        }

        for (const auto& existing : plan.samples) {
            if (existing.source == range.source &&
                (existing.usage != range.usage || existing.authority != range.authority)) {
                Reject(plan.loss.source_conflict);
                return decision;
            }
        }

        struct LogicalSample {
            u64 logical_offset{};
            u32 logical_size{};
            u64 physical_offset{};
            u32 physical_size{};
            std::optional<u32> existing_index{};
        };
        std::array<LogicalSample, 3> logical_samples{};
        u32 logical_count{};
        const auto add_logical_sample = [&](u64 offset, u32 size) {
            for (u32 i = 0; i < logical_count; ++i) {
                if (logical_samples[i].logical_offset == offset &&
                    logical_samples[i].logical_size == size) {
                    return true;
                }
            }
            const u64 physical_offset = offset & ~u64{3};
            const u64 prefix = offset - physical_offset;
            const u64 required = prefix + size;
            if (required > std::numeric_limits<u32>::max() - 3) {
                return false;
            }
            const u32 physical_size = static_cast<u32>((required + 3) & ~u64{3});
            if (physical_offset > range.source_size ||
                physical_size > range.source_size - physical_offset) {
                return false;
            }
            logical_samples[logical_count++] = {
                .logical_offset = offset,
                .logical_size = size,
                .physical_offset = physical_offset,
                .physical_size = physical_size,
            };
            return true;
        };

        const u32 copy_size = static_cast<u32>(std::min<u64>(range.size, SampleBytes));
        bool valid_samples = add_logical_sample(range.source_offset, copy_size);
        if (range.size > SampleBytes) {
            const u64 middle = range.source_offset + (range.size - SampleBytes) / 2;
            valid_samples &= add_logical_sample(middle, SampleBytes);
            valid_samples &=
                add_logical_sample(range.source_offset + range.size - SampleBytes, SampleBytes);
        }
        if (!valid_samples) {
            Reject(plan.loss.invalid_range);
            return decision;
        }

        u32 new_samples{};
        u32 new_bytes{};
        for (u32 logical = 0; logical < logical_count; ++logical) {
            for (u32 sample = 0; sample < plan.sample_count; ++sample) {
                const auto& existing = plan.samples[sample];
                if (existing.source == range.source &&
                    existing.capture_draw == range.semantic.draw &&
                    existing.source_offset == logical_samples[logical].physical_offset &&
                    existing.size == logical_samples[logical].physical_size &&
                    existing.write_serial == range.write_serial) {
                    logical_samples[logical].existing_index = sample;
                    break;
                }
            }
            if (logical_samples[logical].existing_index.has_value()) {
                continue;
            }
            bool duplicates_new_sample{};
            for (u32 previous = 0; previous < logical; ++previous) {
                duplicates_new_sample |= !logical_samples[previous].existing_index.has_value() &&
                                         logical_samples[previous].physical_offset ==
                                             logical_samples[logical].physical_offset &&
                                         logical_samples[previous].physical_size ==
                                             logical_samples[logical].physical_size;
            }
            if (!duplicates_new_sample) {
                ++new_samples;
                new_bytes += logical_samples[logical].physical_size;
            }
        }
        if (new_samples > MaxSamplesPerFrame - plan.sample_count) {
            Reject(plan.loss.sample_capacity);
            return decision;
        }
        if (new_bytes > MaxSampleBytesPerFrame - plan.sample_bytes) {
            Reject(plan.loss.byte_capacity);
            return decision;
        }

        InputAssemblyReadbackSemantic semantic{.semantic = range.semantic};
        std::array<u32, 3> resolved_indices{};
        for (u32 logical = 0; logical < logical_count; ++logical) {
            std::optional<u32> resolved{};
            if (logical_samples[logical].existing_index.has_value()) {
                resolved = *logical_samples[logical].existing_index;
            } else {
                for (u32 previous = 0; previous < logical; ++previous) {
                    if (logical_samples[previous].physical_offset ==
                            logical_samples[logical].physical_offset &&
                        logical_samples[previous].physical_size ==
                            logical_samples[logical].physical_size) {
                        resolved = resolved_indices[previous];
                        break;
                    }
                }
                if (!resolved.has_value()) {
                    resolved = plan.sample_count++;
                    plan.samples.push_back({
                        .source = range.source,
                        .capture_draw = range.semantic.draw,
                        .source_offset = logical_samples[logical].physical_offset,
                        .write_serial = range.write_serial,
                        .destination_offset = plan.sample_bytes,
                        .size = logical_samples[logical].physical_size,
                        .usage = range.usage,
                        .authority = range.authority,
                    });
                    plan.sample_bytes += logical_samples[logical].physical_size;
                    decision.new_copy_indices[decision.new_copy_count] = *resolved;
                    ++decision.new_copy_count;
                }
            }
            const u32 sample_index = *resolved;
            resolved_indices[logical] = sample_index;
            const auto& sample = plan.samples[sample_index];
            const InputAssemblyReadbackReference reference{
                .sample_index = sample_index,
                .destination_offset = static_cast<u32>(sample.destination_offset +
                                                       logical_samples[logical].logical_offset -
                                                       logical_samples[logical].physical_offset),
                .size = logical_samples[logical].logical_size,
            };
            semantic.references[semantic.reference_count++] = reference;
            decision.references[decision.reference_count++] = reference;
        }
        plan.semantics.push_back(semantic);
        ++plan.semantic_count;
        decision.accepted = true;
        return decision;
    }

    [[nodiscard]] const InputAssemblyReadbackFramePlan& EndFrame() noexcept {
        frame_active = false;
        plan.complete &= !plan.loss.Any();
        return plan;
    }

    [[nodiscard]] const InputAssemblyReadbackFramePlan& CurrentFrame() const noexcept {
        return plan;
    }

private:
    void Reject(u32& counter) noexcept {
        ++counter;
        plan.complete = false;
    }

    InputAssemblyReadbackFramePlan plan{};
    bool frame_active{};
};

enum class InputAssemblyCopyAccess : u32 {
    HostWrite = 1u << 0,
    TransferWrite = 1u << 1,
    ShaderWrite = 1u << 2,
    MemoryWrite = 1u << 3,
    TransferRead = 1u << 4,
    VertexRead = 1u << 5,
    IndexRead = 1u << 6,
};

struct InputAssemblyCopyBarrierPlan {
    u32 pre_source_access{};
    u32 post_source_access{};
    bool copy_reads_source{};

    [[nodiscard]] constexpr bool PreSourceHas(InputAssemblyCopyAccess access) const noexcept {
        return (pre_source_access & static_cast<u32>(access)) != 0;
    }

    [[nodiscard]] constexpr bool PostSourceHas(InputAssemblyCopyAccess access) const noexcept {
        return (post_source_access & static_cast<u32>(access)) != 0;
    }
};

[[nodiscard]] constexpr InputAssemblyCopyBarrierPlan MakeInputAssemblyCopyBarrierPlan(
    InputAssemblyHostUsage usage, bool vertex, bool index) noexcept {
    u32 pre = static_cast<u32>(InputAssemblyCopyAccess::TransferWrite) |
              static_cast<u32>(InputAssemblyCopyAccess::MemoryWrite);
    if (usage == InputAssemblyHostUsage::Stream) {
        pre |= static_cast<u32>(InputAssemblyCopyAccess::HostWrite);
    } else {
        pre |= static_cast<u32>(InputAssemblyCopyAccess::ShaderWrite);
    }
    u32 post{};
    if (vertex) {
        post |= static_cast<u32>(InputAssemblyCopyAccess::VertexRead);
    }
    if (index) {
        post |= static_cast<u32>(InputAssemblyCopyAccess::IndexRead);
    }
    return {.pre_source_access = pre, .post_source_access = post, .copy_reads_source = true};
}

class InputAssemblyReadbackCompletion {
public:
    explicit InputAssemblyReadbackCompletion(u64 completion_tick) noexcept
        : completion_tick{completion_tick} {}

    [[nodiscard]] bool TryClaimInvalidation(u64 completed_tick) noexcept {
        if (claimed || completed_tick < completion_tick) {
            return false;
        }
        claimed = true;
        return true;
    }

private:
    u64 completion_tick{};
    bool claimed{};
};

} // namespace AmdGpu
