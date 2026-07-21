// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vulkan/vulkan.hpp>

namespace Vulkan {

inline vk::PhysicalDeviceFeatures BuildCoreDeviceFeatures(
    const vk::PhysicalDeviceFeatures& supported) {
    vk::PhysicalDeviceFeatures enabled{};
    enabled.robustBufferAccess = supported.robustBufferAccess;
    enabled.imageCubeArray = supported.imageCubeArray;
    enabled.independentBlend = supported.independentBlend;
    enabled.geometryShader = supported.geometryShader;
    enabled.tessellationShader = supported.tessellationShader;
    enabled.sampleRateShading = supported.sampleRateShading;
    enabled.dualSrcBlend = supported.dualSrcBlend;
    enabled.logicOp = supported.logicOp;
    enabled.multiDrawIndirect = supported.multiDrawIndirect;
    enabled.drawIndirectFirstInstance = supported.drawIndirectFirstInstance;
    enabled.depthClamp = supported.depthClamp;
    enabled.depthBiasClamp = supported.depthBiasClamp;
    enabled.fillModeNonSolid = supported.fillModeNonSolid;
    enabled.depthBounds = supported.depthBounds;
    enabled.wideLines = supported.wideLines;
    enabled.multiViewport = supported.multiViewport;
    enabled.samplerAnisotropy = supported.samplerAnisotropy;
    enabled.vertexPipelineStoresAndAtomics = supported.vertexPipelineStoresAndAtomics;
    enabled.fragmentStoresAndAtomics = supported.fragmentStoresAndAtomics;
    enabled.shaderImageGatherExtended = supported.shaderImageGatherExtended;
    enabled.shaderStorageImageExtendedFormats = supported.shaderStorageImageExtendedFormats;
    enabled.shaderStorageImageMultisample = supported.shaderStorageImageMultisample;
    enabled.shaderClipDistance = supported.shaderClipDistance;
    enabled.shaderFloat64 = supported.shaderFloat64;
    enabled.shaderInt64 = supported.shaderInt64;
    enabled.shaderInt16 = supported.shaderInt16;
    return enabled;
}

} // namespace Vulkan
