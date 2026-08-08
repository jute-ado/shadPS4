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

struct InputAssemblyCaptureWindow {
    u64 frame_start{};
    u64 frame_count{};
    u32 draw_start{};
    u32 draw_count{};

    [[nodiscard]] static constexpr InputAssemblyCaptureWindow Defaults() noexcept {
        return {.frame_start = 4000, .frame_count = 1000, .draw_start = 0, .draw_count = 1152};
    }

    [[nodiscard]] constexpr bool ContainsFrame(u64 frame) const noexcept {
        return frame >= frame_start && frame - frame_start < frame_count;
    }

    [[nodiscard]] constexpr bool ContainsDraw(u32 draw) const noexcept {
        return draw >= draw_start && draw - draw_start < draw_count;
    }
};

struct BoundInputAssemblySource {
    InputAssemblyBufferToken token{};
    InputAssemblyHostUsage usage{};
    u64 host_offset{};
    u64 host_size{};
    u64 write_serial{};
    InputAssemblyAuthority authority{InputAssemblyAuthority::Unknown};
};

struct InputAssemblyPostObtainState {
    bool needs_input_barrier{};
    InputAssemblyAuthority authority{InputAssemblyAuthority::CpuAuthoritative};
};

[[nodiscard]] constexpr InputAssemblyPostObtainState ResolveInputAssemblyPostObtainState(
    bool gpu_modified_after_obtain) noexcept {
    return {
        .needs_input_barrier = gpu_modified_after_obtain,
        .authority = gpu_modified_after_obtain ? InputAssemblyAuthority::GpuAuthoritative
                                               : InputAssemblyAuthority::CpuAuthoritative,
    };
}

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
    static constexpr u32 MaxSamplesPerFrame = 8192;
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
        if (plan.semantic_count >= MaxSemanticsPerFrame) {
            Reject(plan.loss.semantic_capacity);
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

class InputAssemblyReadbackSlotPool {
public:
    static constexpr u32 MaxSlots = 8;

    struct Token {
        u32 slot{MaxSlots};
        u32 generation{};

        auto operator<=>(const Token&) const = default;

        [[nodiscard]] constexpr explicit operator bool() const noexcept {
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
            return Token{.slot = slot, .generation = entry.generation};
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

struct InputAssemblyImmediateSnapshot {
    u64 sequence{};
    u64 process_time_us{};
    InputAssemblySemanticOrdinal semantic{};
    InputAssemblyBufferToken source{};
    u64 write_serial{};
    InputAssemblyAuthority authority{InputAssemblyAuthority::Unknown};
    std::span<const std::byte> bytes{};
    bool complete{};
};

struct InputAssemblyLagConfig {
    bool enabled{true};
    u64 cadence_us{100'000};
    u64 tolerance_us{25'000};

    [[nodiscard]] static constexpr InputAssemblyLagConfig Defaults() noexcept {
        return {};
    }

    [[nodiscard]] constexpr bool HasDisjointTargets() const noexcept {
        return cadence_us != 0 && tolerance_us <= (cadence_us - 1) / 2;
    }

    [[nodiscard]] constexpr InputAssemblyLagConfig Normalized() const noexcept {
        auto normalized = *this;
        if (cadence_us == 0) {
            normalized.enabled = false;
            normalized.tolerance_us = 0;
            return normalized;
        }
        normalized.tolerance_us = std::min(tolerance_us, (cadence_us - 1) / 2);
        return normalized;
    }

    [[nodiscard]] constexpr u64 HistoryHorizonUs() const noexcept {
        if (cadence_us > (std::numeric_limits<u64>::max() - tolerance_us) / 3) {
            return std::numeric_limits<u64>::max();
        }
        return cadence_us * 3 + tolerance_us;
    }
};

struct InputAssemblyImmediateChange {
    bool first_observation{};
    bool changed{};
    bool exact_aba_return{};
    bool stable_transport_aba{};
    bool lag_exact_aba_return{};
    bool lag_stable_transport_aba{};
    bool lag_episode_return{};
    bool lag_stable_transport_episode_return{};
    bool lag_source_changed{};
    bool lag_write_serial_changed{};
    bool lag_unavailable{};
    bool lag_ambiguous{};
    bool lag_history_loss{};
    bool time_gap{};
    bool disabled{};
    u64 lag_baseline_sequence{};
    u64 lag_baseline_process_time_us{};
    u64 lag_departure_sequence{};
    u64 lag_departure_process_time_us{};
    u64 lag_return_sequence{};
    u64 lag_return_process_time_us{};
    u64 lag_episode_baseline_sequence{};
    u64 lag_episode_baseline_process_time_us{};
    u64 lag_episode_departure_sequence{};
    u64 lag_episode_departure_process_time_us{};
    u64 lag_episode_return_sequence{};
    u64 lag_episode_return_process_time_us{};
    bool source_changed{};
    bool write_serial_changed{};
    bool authority_ambiguous{};
    bool sequence_gap{};
    bool incomplete{};
    bool baseline_reset{};
    bool capacity_exceeded{};
};

struct InputAssemblyLagEvent {
    InputAssemblySemanticOrdinal semantic{};
    u64 baseline_sequence{};
    u64 baseline_process_time_us{};
    u64 departure_sequence{};
    u64 departure_process_time_us{};
    u64 return_sequence{};
    u64 return_process_time_us{};
    bool stable_transport{};
};

class InputAssemblyLagEventDetails {
public:
    static constexpr u32 MaxEvents = 8;

    [[nodiscard]] bool Append(InputAssemblySemanticOrdinal semantic,
                              const InputAssemblyImmediateChange& change) noexcept {
        if (!change.lag_exact_aba_return && !change.lag_episode_return) {
            return true;
        }
        if (count == events.size()) {
            ++lost_events;
            return false;
        }
        const bool exact = change.lag_exact_aba_return;
        events[count++] = {
            .semantic = semantic,
            .baseline_sequence =
                exact ? change.lag_baseline_sequence : change.lag_episode_baseline_sequence,
            .baseline_process_time_us = exact ? change.lag_baseline_process_time_us
                                              : change.lag_episode_baseline_process_time_us,
            .departure_sequence =
                exact ? change.lag_departure_sequence : change.lag_episode_departure_sequence,
            .departure_process_time_us = exact ? change.lag_departure_process_time_us
                                               : change.lag_episode_departure_process_time_us,
            .return_sequence =
                exact ? change.lag_return_sequence : change.lag_episode_return_sequence,
            .return_process_time_us = exact ? change.lag_return_process_time_us
                                            : change.lag_episode_return_process_time_us,
            .stable_transport = exact ? change.lag_stable_transport_aba
                                      : change.lag_stable_transport_episode_return,
        };
        return true;
    }

    [[nodiscard]] std::span<const InputAssemblyLagEvent> Events() const noexcept {
        return {events.data(), count};
    }

    [[nodiscard]] u32 LostEvents() const noexcept {
        return lost_events;
    }

private:
    std::array<InputAssemblyLagEvent, MaxEvents> events{};
    u32 count{};
    u32 lost_events{};
};

class InputAssemblyImmediateReadbackReducer {
public:
    static constexpr u32 MaxEntries = InputAssemblyDeviceIntegrityPlanner::MaxHistoryIdentities;
    static constexpr u32 MaxSnapshotBytes = InputAssemblyDeviceIntegrityReducer::MaxSnapshotBytes;
    static constexpr u32 MaxLagHistoryObservations = 32;

    explicit InputAssemblyImmediateReadbackReducer(
        InputAssemblyLagConfig config_ = InputAssemblyLagConfig::Defaults())
        : config{config_.Normalized()},
          entries{config.enabled && config.cadence_us != 0 ? std::make_unique<Entry[]>(MaxEntries)
                                                           : nullptr} {}

    [[nodiscard]] InputAssemblyImmediateChange Observe(
        const InputAssemblyImmediateSnapshot& snapshot) noexcept {
        if (entries == nullptr) {
            return {.disabled = true};
        }
        Entry* entry{};
        for (u32 i = 0; i < count; ++i) {
            if (entries[i].semantic == snapshot.semantic) {
                entry = &entries[i];
                break;
            }
        }
        if (!snapshot.complete) {
            if (entry != nullptr) {
                ClearBaseline(*entry);
                ClearLagHistory(*entry);
                entry->observed = true;
            }
            return {.lag_ambiguous = true, .incomplete = true};
        }
        if (snapshot.bytes.empty() || snapshot.bytes.size() > MaxSnapshotBytes) {
            if (entry != nullptr) {
                ClearBaseline(*entry);
                ClearLagHistory(*entry);
                entry->observed = true;
            }
            return {.lag_ambiguous = true, .capacity_exceeded = true};
        }
        if (entry == nullptr) {
            if (count >= MaxEntries) {
                return {.capacity_exceeded = true};
            }
            entry = &entries[count++];
            entry->semantic = snapshot.semantic;
        }

        if (snapshot.authority == InputAssemblyAuthority::Unknown) {
            ClearBaseline(*entry);
            ClearLagHistory(*entry);
            entry->observed = true;
            entry->authority = snapshot.authority;
            return {.lag_ambiguous = true, .authority_ambiguous = true};
        }
        if (!entry->has_baseline) {
            const bool prior_observation = entry->observed;
            SetBaseline(*entry, snapshot);
            PushLagObservation(*entry, snapshot);
            return prior_observation
                       ? InputAssemblyImmediateChange{.lag_ambiguous = true, .baseline_reset = true}
                       : InputAssemblyImmediateChange{.first_observation = true};
        }
        if (entry->authority != snapshot.authority) {
            SetBaseline(*entry, snapshot);
            ClearLagHistory(*entry);
            PushLagObservation(*entry, snapshot);
            return {.lag_ambiguous = true, .authority_ambiguous = true};
        }
        const bool contiguous = entry->last_sequence != std::numeric_limits<u64>::max() &&
                                snapshot.sequence == entry->last_sequence + 1;
        if (!contiguous) {
            SetBaseline(*entry, snapshot);
            ClearLagHistory(*entry);
            PushLagObservation(*entry, snapshot);
            return {.lag_ambiguous = true, .sequence_gap = true};
        }
        if ((snapshot.process_time_us != 0 || entry->last_process_time_us != 0) &&
            snapshot.process_time_us <= entry->last_process_time_us) {
            SetBaseline(*entry, snapshot);
            ClearLagHistory(*entry);
            PushLagObservation(*entry, snapshot);
            return {.lag_ambiguous = true, .time_gap = true};
        }

        const auto previous =
            std::span<const std::byte>{entry->previous.data(), entry->previous_size};
        const bool changed = !std::ranges::equal(previous, snapshot.bytes);
        const auto previous_previous = std::span<const std::byte>{entry->previous_previous.data(),
                                                                  entry->previous_previous_size};
        const bool exact_aba =
            changed && entry->has_previous_previous &&
            entry->previous_previous_sequence <= std::numeric_limits<u64>::max() - 2 &&
            entry->previous_previous_sequence + 2 == snapshot.sequence &&
            std::ranges::equal(previous_previous, snapshot.bytes);
        const bool stable_transport_aba =
            exact_aba && entry->source == snapshot.source &&
            entry->previous_previous_source == snapshot.source &&
            entry->write_serial == snapshot.write_serial &&
            entry->previous_previous_write_serial == snapshot.write_serial;
        const bool source_changed = entry->source != snapshot.source;
        const bool write_serial_changed = entry->write_serial != snapshot.write_serial;
        auto lag_change = CompareLagHistory(*entry, snapshot);

        entry->previous_previous_size = entry->previous_size;
        std::copy_n(entry->previous.begin(), entry->previous_size,
                    entry->previous_previous.begin());
        entry->previous_previous_sequence = entry->last_sequence;
        entry->previous_previous_source = entry->source;
        entry->previous_previous_write_serial = entry->write_serial;
        entry->has_previous_previous = true;
        entry->previous_size = static_cast<u32>(snapshot.bytes.size());
        std::ranges::copy(snapshot.bytes, entry->previous.begin());
        entry->source = snapshot.source;
        entry->write_serial = snapshot.write_serial;
        entry->last_sequence = snapshot.sequence;
        entry->last_process_time_us = snapshot.process_time_us;
        PushLagObservation(*entry, snapshot);
        lag_change.changed = changed;
        lag_change.exact_aba_return = exact_aba;
        lag_change.stable_transport_aba = stable_transport_aba;
        lag_change.source_changed = source_changed;
        lag_change.write_serial_changed = write_serial_changed;
        return lag_change;
    }

    void Reset() noexcept {
        for (u32 i = 0; i < count; ++i) {
            entries[i] = {};
        }
        count = 0;
    }

    [[nodiscard]] u32 EntryCount() const noexcept {
        return count;
    }

    [[nodiscard]] u32 RetainedObservationCount() const noexcept {
        u32 retained{};
        for (u32 i = 0; i < count; ++i) {
            retained += entries[i].lag_history_count;
        }
        return retained;
    }

private:
    struct LagObservation {
        InputAssemblyBufferToken source{};
        u64 write_serial{};
        u64 sequence{};
        u64 process_time_us{};
        std::array<std::byte, MaxSnapshotBytes> bytes{};
        u32 size{};
    };

    struct Entry {
        InputAssemblySemanticOrdinal semantic{};
        InputAssemblyBufferToken source{};
        InputAssemblyBufferToken previous_previous_source{};
        u64 write_serial{};
        u64 previous_previous_write_serial{};
        u64 previous_previous_sequence{};
        u64 last_sequence{};
        u64 last_process_time_us{};
        InputAssemblyAuthority authority{InputAssemblyAuthority::Unknown};
        std::array<std::byte, MaxSnapshotBytes> previous_previous{};
        std::array<std::byte, MaxSnapshotBytes> previous{};
        u32 previous_previous_size{};
        u32 previous_size{};
        bool observed{};
        bool has_baseline{};
        bool has_previous_previous{};
        std::array<LagObservation, MaxLagHistoryObservations> lag_history{};
        u32 lag_history_count{};
        bool lag_history_evicted{};
    };

    static void ClearBaseline(Entry& entry) noexcept {
        entry.previous_previous_size = 0;
        entry.previous_size = 0;
        entry.has_baseline = false;
        entry.has_previous_previous = false;
    }

    static void ClearLagHistory(Entry& entry) noexcept {
        entry.lag_history_count = 0;
        entry.lag_history_evicted = false;
    }

    static void SetBaseline(Entry& entry, const InputAssemblyImmediateSnapshot& snapshot) noexcept {
        entry.source = snapshot.source;
        entry.write_serial = snapshot.write_serial;
        entry.authority = snapshot.authority;
        entry.previous_size = static_cast<u32>(snapshot.bytes.size());
        std::ranges::copy(snapshot.bytes, entry.previous.begin());
        entry.previous_previous_size = 0;
        entry.last_sequence = snapshot.sequence;
        entry.last_process_time_us = snapshot.process_time_us;
        entry.observed = true;
        entry.has_baseline = true;
        entry.has_previous_previous = false;
    }

    static void PushLagObservation(Entry& entry,
                                   const InputAssemblyImmediateSnapshot& snapshot) noexcept {
        if (entry.lag_history_count == MaxLagHistoryObservations) {
            std::move(entry.lag_history.begin() + 1, entry.lag_history.end(),
                      entry.lag_history.begin());
            --entry.lag_history_count;
            entry.lag_history_evicted = true;
        }
        auto& observation = entry.lag_history[entry.lag_history_count++];
        observation.source = snapshot.source;
        observation.write_serial = snapshot.write_serial;
        observation.sequence = snapshot.sequence;
        observation.process_time_us = snapshot.process_time_us;
        observation.size = static_cast<u32>(snapshot.bytes.size());
        std::ranges::copy(snapshot.bytes, observation.bytes.begin());
    }

    [[nodiscard]] const LagObservation* FindClosest(const Entry& entry, u64 target_time_us,
                                                    u32 before_index) const noexcept {
        const LagObservation* closest{};
        u64 closest_distance = std::numeric_limits<u64>::max();
        for (u32 i = 0; i < before_index; ++i) {
            const auto& observation = entry.lag_history[i];
            const u64 distance = observation.process_time_us > target_time_us
                                     ? observation.process_time_us - target_time_us
                                     : target_time_us - observation.process_time_us;
            if (distance <= config.tolerance_us && distance < closest_distance) {
                closest = &observation;
                closest_distance = distance;
            }
        }
        return closest;
    }

    [[nodiscard]] static bool EqualBytes(const LagObservation& observation,
                                         std::span<const std::byte> bytes) noexcept {
        return observation.size == bytes.size() &&
               std::ranges::equal(
                   std::span<const std::byte>{observation.bytes.data(), observation.size}, bytes);
    }

    [[nodiscard]] InputAssemblyImmediateChange CompareLagHistory(
        const Entry& entry, const InputAssemblyImmediateSnapshot& snapshot) const noexcept {
        InputAssemblyImmediateChange result{};
        const u64 cadence = config.cadence_us;
        if (cadence <= std::numeric_limits<u64>::max() / 2 &&
            snapshot.process_time_us >= cadence * 2) {
            const u64 prior_target = snapshot.process_time_us - cadence;
            const u64 prior_previous_target = snapshot.process_time_us - cadence * 2;
            const auto* prior = FindClosest(entry, prior_target, entry.lag_history_count);
            u32 prior_index = entry.lag_history_count;
            if (prior != nullptr) {
                prior_index = static_cast<u32>(prior - entry.lag_history.data());
            }
            const auto* prior_previous = FindClosest(entry, prior_previous_target, prior_index);
            if (prior == nullptr || prior_previous == nullptr) {
                result.lag_unavailable = true;
                if (entry.lag_history_evicted && entry.lag_history_count != 0 &&
                    entry.lag_history.front().process_time_us > prior_previous_target) {
                    result.lag_history_loss = true;
                }
            } else {
                const bool returned = EqualBytes(*prior_previous, snapshot.bytes);
                const bool departed = !EqualBytes(*prior, snapshot.bytes);
                result.lag_exact_aba_return = returned && departed;
                result.lag_source_changed =
                    prior_previous->source != snapshot.source || prior->source != snapshot.source;
                result.lag_write_serial_changed =
                    prior_previous->write_serial != snapshot.write_serial ||
                    prior->write_serial != snapshot.write_serial;
                result.lag_stable_transport_aba = result.lag_exact_aba_return &&
                                                  !result.lag_source_changed &&
                                                  !result.lag_write_serial_changed;
                if (result.lag_exact_aba_return) {
                    result.lag_baseline_sequence = prior_previous->sequence;
                    result.lag_baseline_process_time_us = prior_previous->process_time_us;
                    result.lag_departure_sequence = prior->sequence;
                    result.lag_departure_process_time_us = prior->process_time_us;
                    result.lag_return_sequence = snapshot.sequence;
                    result.lag_return_process_time_us = snapshot.process_time_us;
                }
            }
        }

        const u64 horizon = config.HistoryHorizonUs();
        u32 oldest = 0;
        while (oldest < entry.lag_history_count &&
               snapshot.process_time_us - entry.lag_history[oldest].process_time_us > horizon) {
            ++oldest;
        }
        std::optional<u32> departure;
        std::optional<u32> baseline;
        for (u32 i = entry.lag_history_count; i-- > oldest;) {
            if (!departure.has_value()) {
                if (!EqualBytes(entry.lag_history[i], snapshot.bytes)) {
                    departure = i;
                }
                continue;
            }
            if (EqualBytes(entry.lag_history[i], snapshot.bytes)) {
                baseline = i;
                break;
            }
        }
        if (baseline.has_value() && departure.has_value() &&
            *baseline + 1 < entry.lag_history_count) {
            const auto& baseline_observation = entry.lag_history[*baseline];
            const auto& departure_observation = entry.lag_history[*baseline + 1];
            result.lag_episode_return = true;
            bool episode_source_changed{};
            bool episode_serial_changed{};
            for (u32 i = *baseline; i < entry.lag_history_count; ++i) {
                episode_source_changed |= entry.lag_history[i].source != snapshot.source;
                episode_serial_changed |=
                    entry.lag_history[i].write_serial != snapshot.write_serial;
            }
            result.lag_source_changed |= episode_source_changed;
            result.lag_write_serial_changed |= episode_serial_changed;
            result.lag_stable_transport_episode_return =
                !episode_source_changed && !episode_serial_changed;
            result.lag_episode_baseline_sequence = baseline_observation.sequence;
            result.lag_episode_baseline_process_time_us = baseline_observation.process_time_us;
            result.lag_episode_departure_sequence = departure_observation.sequence;
            result.lag_episode_departure_process_time_us = departure_observation.process_time_us;
            result.lag_episode_return_sequence = snapshot.sequence;
            result.lag_episode_return_process_time_us = snapshot.process_time_us;
            if (!result.lag_exact_aba_return) {
                result.lag_baseline_sequence = baseline_observation.sequence;
                result.lag_baseline_process_time_us = baseline_observation.process_time_us;
                result.lag_departure_sequence = departure_observation.sequence;
                result.lag_departure_process_time_us = departure_observation.process_time_us;
                result.lag_return_sequence = snapshot.sequence;
                result.lag_return_process_time_us = snapshot.process_time_us;
            }
        } else if (departure.has_value() && entry.lag_history_evicted && oldest == 0) {
            result.lag_history_loss = true;
        }
        return result;
    }

    InputAssemblyLagConfig config{};
    std::unique_ptr<Entry[]> entries;
    u32 count{};
};

} // namespace AmdGpu
