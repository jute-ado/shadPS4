// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <span>

#include "common/types.h"
#include "video_core/renderer_vulkan/final_guest_surface_content.h"

namespace Vulkan::HostPasses {

struct PpUpstreamSampleViewFingerprint {
    u32 type{};
    u32 format{};
    u32 base_mip{};
    u32 mip_count{};
    u32 base_layer{};
    u32 layer_count{};
    std::array<u32, 4> mapping{};
    u32 min_lod{};
    bool storage{};

    auto operator<=>(const PpUpstreamSampleViewFingerprint&) const = default;
};

struct PpUpstreamSampleSamplerFingerprint {
    u64 raw0{};
    u64 raw1{};

    auto operator<=>(const PpUpstreamSampleSamplerFingerprint&) const = default;
};

struct PpUpstreamSampleRoute {
    static constexpr u32 MaxSamplers = 16;

    u32 logical_stage{};
    u32 image_resource_index{};
    u32 array_element{};
    PpUpstreamSampleViewFingerprint view{};
    std::array<PpUpstreamSampleSamplerFingerprint, MaxSamplers> samplers{};
    u32 sampler_count{};
    bool valid{true};

    auto operator<=>(const PpUpstreamSampleRoute&) const = default;
};

struct PpUpstreamSampleRouteObservation {
    u32 input_index{};
    PpUpstreamSampleRoute route{};
};

struct PpUpstreamSampleRouteSet {
    static constexpr u32 MaxRoutes = 16;

    std::array<PpUpstreamSampleRoute, MaxRoutes> routes{};
    u32 selected_input_index{};
    u32 route_count{};
    FinalGuestSurfaceStatus status{FinalGuestSurfaceStatus::AlreadyConsumed};
    FinalGuestSurfaceLoss loss{};
    bool retains_image{};
    bool retains_view{};
    bool retains_sampler{};
    bool logs_private_words{};

    bool operator==(const PpUpstreamSampleRouteSet&) const = default;
};

struct PpUpstreamSampleRouteDescriptor {
    bool enabled{};
    u32 selected_input_index{};
    u32 input_count{};
    std::span<const PpUpstreamSampleRouteObservation> observations{};
};

[[nodiscard]] inline PpUpstreamSampleRouteSet PlanPpUpstreamSampleRoutes(
    const PpUpstreamSampleRouteDescriptor& descriptor) noexcept {
    PpUpstreamSampleRouteSet result{.selected_input_index = descriptor.selected_input_index};
    if (!descriptor.enabled) {
        return result;
    }
    const auto reject = [&](FinalGuestSurfaceStatus status,
                            u32 FinalGuestSurfaceLoss::* member) {
        PpUpstreamSampleRouteSet rejected{
            .selected_input_index = descriptor.selected_input_index,
            .status = status,
        };
        rejected.loss.*member = 1;
        return rejected;
    };
    if (descriptor.input_count == 0 || descriptor.selected_input_index >= descriptor.input_count) {
        return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                      &FinalGuestSurfaceLoss::invalidation);
    }
    for (const auto& observation : descriptor.observations) {
        if (observation.input_index != descriptor.selected_input_index) {
            continue;
        }
        const auto& route = observation.route;
        if (!route.valid || route.sampler_count > route.MaxSamplers) {
            return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                          &FinalGuestSurfaceLoss::invalidation);
        }
        const auto same_key = [&](const PpUpstreamSampleRoute& candidate) {
            return candidate.logical_stage == route.logical_stage &&
                   candidate.image_resource_index == route.image_resource_index &&
                   candidate.array_element == route.array_element;
        };
        const auto existing = std::find_if(result.routes.begin(),
                                           result.routes.begin() + result.route_count, same_key);
        if (existing != result.routes.begin() + result.route_count) {
            if (*existing != route) {
                return reject(FinalGuestSurfaceStatus::InvalidationLoss,
                              &FinalGuestSurfaceLoss::invalidation);
            }
            continue;
        }
        if (result.route_count == result.MaxRoutes) {
            return reject(FinalGuestSurfaceStatus::CapacityLoss,
                          &FinalGuestSurfaceLoss::tile_capacity);
        }
        result.routes[result.route_count++] = route;
    }
    if (result.route_count == 0) {
        return reject(FinalGuestSurfaceStatus::GapLoss, &FinalGuestSurfaceLoss::gap);
    }
    std::sort(result.routes.begin(), result.routes.begin() + result.route_count,
              [](const PpUpstreamSampleRoute& left, const PpUpstreamSampleRoute& right) {
                  if (left.logical_stage != right.logical_stage) {
                      return left.logical_stage < right.logical_stage;
                  }
                  if (left.image_resource_index != right.image_resource_index) {
                      return left.image_resource_index < right.image_resource_index;
                  }
                  return left.array_element < right.array_element;
              });
    result.status = FinalGuestSurfaceStatus::Complete;
    return result;
}

} // namespace Vulkan::HostPasses
