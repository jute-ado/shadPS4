// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/texture_cache/sampler_observation.h"

namespace VideoCore {
namespace {

struct PNextObservation {
    SamplerPNextOrder order{SamplerPNextOrder::None};
    const vk::SamplerReductionModeCreateInfo* reduction{};
    bool custom{};
    bool malformed{};
};

[[nodiscard]] PNextObservation ObservePNext(const void* first) noexcept {
    PNextObservation result{};
    if (first == nullptr) {
        return result;
    }

    const VkBaseInStructure* nodes[2]{};
    u32 count = 0;
    auto* node = static_cast<const VkBaseInStructure*>(first);
    while (node != nullptr) {
        if (count == 2 || node == nodes[0] || node == nodes[1]) {
            result.order = SamplerPNextOrder::Invalid;
            result.malformed = true;
            return result;
        }
        nodes[count++] = node;
        switch (node->sType) {
        case VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT:
            if (result.custom) {
                result.order = SamplerPNextOrder::Invalid;
                result.malformed = true;
                return result;
            }
            result.custom = true;
            break;
        case VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO:
            if (result.reduction != nullptr) {
                result.order = SamplerPNextOrder::Invalid;
                result.malformed = true;
                return result;
            }
            result.reduction =
                reinterpret_cast<const vk::SamplerReductionModeCreateInfo*>(node);
            break;
        default:
            result.order = SamplerPNextOrder::Invalid;
            result.malformed = true;
            return result;
        }
        node = node->pNext;
    }

    if (count == 1) {
        result.order = result.custom ? SamplerPNextOrder::CustomOnly
                                     : SamplerPNextOrder::ReductionOnly;
    } else if (nodes[0]->sType == VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT) {
        result.order = SamplerPNextOrder::CustomThenReduction;
    } else {
        result.order = SamplerPNextOrder::ReductionThenCustom;
    }
    return result;
}

[[nodiscard]] SamplerLodRelation ClassifyLod(const AmdGpu::Sampler& sampler,
                                             const vk::SamplerCreateInfo& create_info) noexcept {
    const float raw_min = sampler.MinLod();
    const float raw_max = sampler.MaxLod();
    if (sampler.mip_filter == AmdGpu::MipFilter::None) {
        if (create_info.minLod == 0.0f && create_info.maxLod == 0.0f) {
            return SamplerLodRelation::Exact;
        }
        if (create_info.minLod == raw_min && create_info.maxLod == raw_max) {
            return SamplerLodRelation::NonMipRawRange;
        }
        return SamplerLodRelation::Mismatch;
    }
    if (raw_min > raw_max) {
        if (create_info.minLod == raw_min && create_info.maxLod == raw_min) {
            return SamplerLodRelation::Exact;
        }
        if (create_info.minLod == raw_min && create_info.maxLod == raw_max) {
            return SamplerLodRelation::InvertedRange;
        }
        return SamplerLodRelation::Mismatch;
    }
    return create_info.minLod == raw_min && create_info.maxLod == raw_max
               ? SamplerLodRelation::Exact
               : SamplerLodRelation::Mismatch;
}

[[nodiscard]] SamplerCompareRelation ClassifyPairCompare(bool compare_enabled,
                                                         u32 operation_use_mask) noexcept {
    constexpr u32 RegularSample = 1u << 0;
    constexpr u32 DepthCompare = 1u << 1;
    constexpr u32 Gather = 1u << 2;
    constexpr u32 Known = RegularSample | DepthCompare | Gather;
    if ((operation_use_mask & ~Known) != 0) {
        return SamplerCompareRelation::Invalid;
    }
    const bool regular = (operation_use_mask & RegularSample) != 0;
    const bool depth = (operation_use_mask & DepthCompare) != 0;
    if (regular && depth) {
        return SamplerCompareRelation::MixedUse;
    }
    if (!regular && !depth) {
        return SamplerCompareRelation::MissingOperationUse;
    }
    if (depth) {
        return compare_enabled ? SamplerCompareRelation::DepthCompareExact
                               : SamplerCompareRelation::DepthCompareMissing;
    }
    return compare_enabled ? SamplerCompareRelation::RegularUnexpectedCompare
                           : SamplerCompareRelation::RegularExact;
}

[[nodiscard]] SamplerCompareRelation
ClassifyCompare(const AmdGpu::Sampler& sampler, const vk::SamplerCreateInfo& create_info,
                u32 operation_use_mask, bool operation_use_known) noexcept {
    const bool expected_enable = sampler.depth_compare_func != AmdGpu::DepthCompare::Never;
    if (static_cast<bool>(create_info.compareEnable) != expected_enable ||
        create_info.compareOp != Vulkan::LiverpoolToVK::DepthCompare(sampler.depth_compare_func)) {
        return SamplerCompareRelation::CreateInfoMismatch;
    }
    return operation_use_known
               ? ClassifyPairCompare(expected_enable, operation_use_mask)
               : SamplerCompareRelation::DescriptorExact;
}

[[nodiscard]] SamplerReductionRelation
ClassifyReduction(const AmdGpu::Sampler& sampler, const PNextObservation& chain,
                  bool supported, bool enabled) noexcept {
    const auto mode = sampler.filter_mode.Value();
    if (mode == AmdGpu::FilterMode::Blend) {
        return chain.reduction == nullptr ? SamplerReductionRelation::Exact
                                          : SamplerReductionRelation::Mismatch;
    }
    if (!supported) {
        return SamplerReductionRelation::Unsupported;
    }
    if (!enabled) {
        return SamplerReductionRelation::FeatureDisabled;
    }
    if (chain.reduction == nullptr) {
        return SamplerReductionRelation::Missing;
    }
    const vk::SamplerReductionMode expected =
        mode == AmdGpu::FilterMode::Min ? vk::SamplerReductionMode::eMin
                                        : vk::SamplerReductionMode::eMax;
    return chain.reduction->reductionMode == expected ? SamplerReductionRelation::Exact
                                                       : SamplerReductionRelation::Mismatch;
}

} // namespace

SamplerRealizationObservation
ObserveSamplerRealizationOnce(const SamplerRealizationDescriptor& descriptor) noexcept {
    SamplerRealizationObservation observation{
        .effective_sampler = descriptor.effective_sampler,
        .actual_create_info = descriptor.actual_create_info,
    };
    auto& fact = observation.fact;
    if (descriptor.actual_create_info == nullptr) {
        fact.disposition = SamplerRealizationDisposition::InvalidTransition;
        return observation;
    }

    const auto chain = ObservePNext(descriptor.actual_create_info->pNext);
    fact.pnext_order = chain.order;
    fact.malformed = chain.malformed;
    fact.observed = true;
    if (chain.malformed) {
        fact.disposition = SamplerRealizationDisposition::InvalidTransition;
        return observation;
    }

    const bool custom_expected =
        descriptor.effective_sampler.border_color_type == AmdGpu::BorderColor::Custom;
    if (chain.custom != custom_expected) {
        fact.disposition = SamplerRealizationDisposition::InvalidTransition;
        return observation;
    }

    fact.lod_relation =
        ClassifyLod(descriptor.effective_sampler, *descriptor.actual_create_info);
    fact.compare_enabled = static_cast<bool>(descriptor.actual_create_info->compareEnable);
    fact.compare_create_info_exact =
        fact.compare_enabled ==
            (descriptor.effective_sampler.depth_compare_func != AmdGpu::DepthCompare::Never) &&
        descriptor.actual_create_info->compareOp == Vulkan::LiverpoolToVK::DepthCompare(
                                                        descriptor.effective_sampler
                                                            .depth_compare_func);
    fact.compare_relation = ClassifyCompare(descriptor.effective_sampler,
                                            *descriptor.actual_create_info,
                                            descriptor.pair_operation_use_mask,
                                            descriptor.pair_operation_use_known);
    fact.reduction_relation = ClassifyReduction(
        descriptor.effective_sampler, chain, descriptor.sampler_filter_minmax_supported,
        descriptor.sampler_filter_minmax_enabled);

    if (fact.compare_relation == SamplerCompareRelation::MixedUse ||
        fact.compare_relation == SamplerCompareRelation::MissingOperationUse ||
        fact.compare_relation == SamplerCompareRelation::Invalid) {
        fact.disposition = SamplerRealizationDisposition::InvalidTransition;
    } else if (fact.lod_relation != SamplerLodRelation::Exact) {
        fact.disposition = SamplerRealizationDisposition::LodMismatch;
    } else if (fact.compare_relation != SamplerCompareRelation::DescriptorExact &&
               fact.compare_relation != SamplerCompareRelation::RegularExact &&
               fact.compare_relation != SamplerCompareRelation::DepthCompareExact) {
        fact.disposition = SamplerRealizationDisposition::CompareMismatch;
    } else if (fact.reduction_relation != SamplerReductionRelation::Exact) {
        fact.disposition = SamplerRealizationDisposition::ReductionMismatch;
    } else {
        fact.disposition = SamplerRealizationDisposition::Exact;
    }
    return observation;
}

SamplerRealizationFact ResolveSamplerRealizationForPair(
    const SamplerRealizationFact& creation, const AmdGpu::Sampler& effective_sampler,
    u32 pair_operation_use_mask) noexcept {
    SamplerRealizationFact resolved = creation;
    const bool expected_compare_enabled =
        effective_sampler.depth_compare_func != AmdGpu::DepthCompare::Never;
    if (!creation.observed || creation.malformed ||
        creation.compare_relation != SamplerCompareRelation::DescriptorExact ||
        !creation.compare_create_info_exact ||
        creation.compare_enabled != expected_compare_enabled) {
        resolved.compare_relation = SamplerCompareRelation::Invalid;
        resolved.disposition = SamplerRealizationDisposition::InvalidTransition;
        return resolved;
    }

    resolved.compare_relation =
        ClassifyPairCompare(creation.compare_enabled, pair_operation_use_mask);
    if (resolved.compare_relation == SamplerCompareRelation::MixedUse ||
        resolved.compare_relation == SamplerCompareRelation::MissingOperationUse ||
        resolved.compare_relation == SamplerCompareRelation::Invalid) {
        resolved.disposition = SamplerRealizationDisposition::InvalidTransition;
    } else if (resolved.lod_relation != SamplerLodRelation::Exact) {
        resolved.disposition = SamplerRealizationDisposition::LodMismatch;
    } else if (resolved.compare_relation != SamplerCompareRelation::RegularExact &&
               resolved.compare_relation != SamplerCompareRelation::DepthCompareExact) {
        resolved.disposition = SamplerRealizationDisposition::CompareMismatch;
    } else if (resolved.reduction_relation != SamplerReductionRelation::Exact) {
        resolved.disposition = SamplerRealizationDisposition::ReductionMismatch;
    } else {
        resolved.disposition = SamplerRealizationDisposition::Exact;
    }
    return resolved;
}

} // namespace VideoCore
