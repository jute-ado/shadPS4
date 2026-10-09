// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <utility>

#include "common/types.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {

enum class SamplerLodRelation : u8 {
    Idle = 0,
    Exact = 1,
    NonMipRawRange = 2,
    InvertedRange = 3,
    Mismatch = 4,
};

enum class SamplerCompareRelation : u8 {
    Idle,
    DescriptorExact,
    RegularExact,
    DepthCompareExact,
    DepthCompareMissing,
    RegularUnexpectedCompare,
    MixedUse,
    MissingOperationUse,
    Invalid,
    CreateInfoMismatch,
};

enum class SamplerReductionRelation : u8 {
    Idle,
    Exact,
    Unsupported,
    FeatureDisabled,
    Missing,
    Mismatch,
};

enum class SamplerPNextOrder : u8 {
    None,
    CustomOnly,
    ReductionOnly,
    CustomThenReduction,
    ReductionThenCustom,
    Invalid,
};

enum class SamplerRealizationDisposition : u8 {
    Idle,
    Exact,
    LodMismatch,
    CompareMismatch,
    ReductionMismatch,
    InvalidTransition,
};

struct SamplerRealizationFact {
    SamplerLodRelation lod_relation{SamplerLodRelation::Idle};
    SamplerCompareRelation compare_relation{SamplerCompareRelation::Idle};
    bool compare_enabled{};
    bool compare_create_info_exact{};
    SamplerReductionRelation reduction_relation{SamplerReductionRelation::Idle};
    SamplerPNextOrder pnext_order{SamplerPNextOrder::None};
    bool malformed{};
    bool observed{};
    SamplerRealizationDisposition disposition{SamplerRealizationDisposition::Idle};

    auto operator<=>(const SamplerRealizationFact&) const = default;
};

struct SamplerRealizationDescriptor {
    AmdGpu::Sampler effective_sampler{};
    const vk::SamplerCreateInfo* actual_create_info{};
    u32 pair_operation_use_mask{};
    bool pair_operation_use_known{true};
    bool sampler_filter_minmax_supported{};
    bool sampler_filter_minmax_enabled{};
};

struct SamplerRealizationObservation {
    AmdGpu::Sampler effective_sampler{};
    const vk::SamplerCreateInfo* actual_create_info{};
    SamplerRealizationFact fact{};
};

[[nodiscard]] SamplerRealizationObservation
ObserveSamplerRealizationOnce(const SamplerRealizationDescriptor& descriptor) noexcept;

[[nodiscard]] SamplerRealizationFact ResolveSamplerRealizationForPair(
    const SamplerRealizationFact& creation, const AmdGpu::Sampler& effective_sampler,
    u32 pair_operation_use_mask) noexcept;

enum class SamplerCachePath : u8 {
    Idle,
    Inserted,
    Hit,
};

enum class SamplerCacheDisposition : u8 {
    Idle,
    Exact,
    InvalidTransition,
};

struct SamplerCacheEntryObservation {
    vk::Sampler sampler{};
    SamplerRealizationFact creation{};
    bool provenance_current{};
};

struct SamplerCacheLookupResult {
    bool found{};
    SamplerCacheEntryObservation entry{};
};

struct SamplerCacheFact {
    SamplerCachePath path{SamplerCachePath::Idle};
    SamplerRealizationFact creation{};
    bool returned_handle_exact{};
    bool provenance_current{};
    bool observed{};
    SamplerCacheDisposition disposition{SamplerCacheDisposition::Idle};

    auto operator<=>(const SamplerCacheFact&) const = default;
};

struct SamplerCacheObservation {
    vk::Sampler sampler{};
    vk::Sampler returned_sampler{};
    SamplerCacheFact fact{};
};

[[nodiscard]] constexpr SamplerCacheDisposition
ClassifySamplerCacheFact(const SamplerCacheFact& fact) noexcept {
    if (!fact.observed || (fact.path != SamplerCachePath::Inserted &&
                           fact.path != SamplerCachePath::Hit) ||
        !fact.creation.observed ||
        fact.creation.disposition != SamplerRealizationDisposition::Exact ||
        !fact.returned_handle_exact || !fact.provenance_current) {
        return SamplerCacheDisposition::InvalidTransition;
    }
    return SamplerCacheDisposition::Exact;
}

template <typename Lookup, typename Create, typename Touch>
[[nodiscard]] SamplerCacheObservation ObserveSamplerCacheOnce(Lookup&& lookup, Create&& create,
                                                               Touch&& touch) {
    const SamplerCacheLookupResult lookup_result = std::invoke(std::forward<Lookup>(lookup));
    SamplerCacheEntryObservation entry{};
    const SamplerCachePath path = lookup_result.found ? SamplerCachePath::Hit
                                                       : SamplerCachePath::Inserted;
    if (lookup_result.found) {
        entry = lookup_result.entry;
        std::invoke(std::forward<Touch>(touch));
    } else {
        entry = std::invoke(std::forward<Create>(create));
    }

    SamplerCacheObservation observation{
        .sampler = entry.sampler,
        .returned_sampler = entry.sampler,
        .fact = {
            .path = path,
            .creation = entry.creation,
            .returned_handle_exact = true,
            .provenance_current = entry.provenance_current,
            .observed = true,
        },
    };
    observation.fact.disposition = ClassifySamplerCacheFact(observation.fact);
    return observation;
}

template <typename Lookup, typename Create, typename Touch>
[[nodiscard]] vk::Sampler ObserveSamplerCacheOnce(Lookup&& lookup, Create&& create, Touch&& touch,
                                                   std::nullptr_t) {
    return ObserveSamplerCacheOnce(std::forward<Lookup>(lookup), std::forward<Create>(create),
                                   std::forward<Touch>(touch))
        .returned_sampler;
}

} // namespace VideoCore
