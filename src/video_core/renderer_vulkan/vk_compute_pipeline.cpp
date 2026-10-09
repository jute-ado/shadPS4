// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/small_vector.hpp>

#include "common/io_file.h"
#include "common/path_util.h"
#include "shader_recompiler/info.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_compute_dispatch_policy.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

ComputePipeline::ComputePipeline(const Instance& instance, Scheduler& scheduler,
                                 DescriptorHeap& desc_heap, const Shader::Profile& profile,
                                 vk::PipelineCache pipeline_cache, ComputePipelineKey compute_key_,
                                 const Shader::Info& info_, vk::ShaderModule module,
                                 SerializationSupport& sdata, bool preloading /*=false*/)
    : Pipeline{instance, scheduler, desc_heap, profile, pipeline_cache, true},
      compute_key{compute_key_} {
    auto& info = stages[int(Shader::LogicalStage::Compute)];
    info = &info_;
    const auto debug_str = GetDebugString();

    const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_ci = {
        .requiredSubgroupSize = 64,
    };
    const vk::PipelineShaderStageCreateInfo shader_ci = {
        .pNext = instance.IsSubgroupSize64Supported() ? &subgroup_size_ci : nullptr,
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = module,
        .pName = "main",
    };

    u32 binding{};
    boost::container::small_vector<vk::DescriptorSetLayoutBinding, 32> bindings;
    for (const auto& buffer : info->buffers) {
        // During deserialization, we don't have access to the UD to fetch sharp data. To address
        // this properly we need to track shaprs or portion of them in `sdata`, but since we're
        // interested only in "is storage" flag (which is not even effective atm), we can take a
        // shortcut there.
        const auto sharp = preloading ? AmdGpu::Buffer{} : buffer.GetSharp(*info);
        bindings.push_back({
            .binding = binding++,
            .descriptorType = buffer.IsStorage(sharp) ? vk::DescriptorType::eStorageBuffer
                                                      : vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
    }
    for (const auto& image : info->images) {
        const u32 num_bindings = image.NumBindings(*info);
        bindings.push_back({
            .binding = binding,
            .descriptorType = image.is_written ? vk::DescriptorType::eStorageImage
                                               : vk::DescriptorType::eSampledImage,
            .descriptorCount = num_bindings,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
        binding += num_bindings;
    }
    for (const auto& sampler : info->samplers) {
        bindings.push_back({
            .binding = binding++,
            .descriptorType = vk::DescriptorType::eSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
    }

    const vk::PushConstantRange push_constants = {
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(Shader::PushData),
    };

    uses_push_descriptors = binding < instance.MaxPushDescriptors();
    const auto flags = uses_push_descriptors
                           ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
                           : vk::DescriptorSetLayoutCreateFlagBits{};
    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = flags,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
    const auto device = instance.GetDevice();
    auto [descriptor_set_result, descriptor_set] =
        device.createDescriptorSetLayoutUnique(desc_layout_ci);
    ASSERT_MSG(descriptor_set_result == vk::Result::eSuccess,
               "Failed to create compute descriptor set layout: {}",
               vk::to_string(descriptor_set_result));
    desc_layout = std::move(descriptor_set);

    const vk::DescriptorSetLayout set_layout = *desc_layout;
    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1U,
        .pPushConstantRanges = &push_constants,
    };
    auto [layout_result, layout] = instance.GetDevice().createPipelineLayoutUnique(layout_info);
    ASSERT_MSG(layout_result == vk::Result::eSuccess,
               "Failed to create compute pipeline layout: {}", vk::to_string(layout_result));
    pipeline_layout = std::move(layout);
    SetObjectName(device, *pipeline_layout, "Compute PipelineLayout {}", debug_str);

    vk::PipelineCreateFlags pipeline_flags = ComputeDispatchNeedsBasePipeline(info->pgm_hash)
                                                 ? vk::PipelineCreateFlagBits::eDispatchBase
                                                 : vk::PipelineCreateFlags{};
    const bool capture_executable = false;
    if (capture_executable) {
        pipeline_flags |= vk::PipelineCreateFlagBits::eCaptureInternalRepresentationsKHR |
                          vk::PipelineCreateFlagBits::eCaptureStatisticsKHR;
    }
    const vk::ComputePipelineCreateInfo compute_pipeline_ci = {
        .flags = pipeline_flags,
        .stage = shader_ci,
        .layout = *pipeline_layout,
    };
    auto [pipeline_result, pipe] =
        instance.GetDevice().createComputePipelineUnique(pipeline_cache, compute_pipeline_ci);
    ASSERT_MSG(pipeline_result == vk::Result::eSuccess, "Failed to create compute pipeline: {}",
               vk::to_string(pipeline_result));
    pipeline = std::move(pipe);
    SetObjectName(device, *pipeline, "Compute Pipeline {}", debug_str);

    // TEMP Shadow diagnostic: query the NVIDIA compiler's register/spill statistics for the
    // pathological compute shader. Remove with the executable-properties extension after use.
    if (capture_executable) {
        const VkPipelineInfoKHR pipeline_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR,
            .pNext = nullptr,
            .pipeline = static_cast<VkPipeline>(*pipeline),
        };
        u32 executable_count{};
        const auto raw_device = static_cast<VkDevice>(device);
        auto& dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER;
        if (dispatcher.vkGetPipelineExecutablePropertiesKHR(
                raw_device, &pipeline_info, &executable_count, nullptr) == VK_SUCCESS) {
            std::vector<VkPipelineExecutablePropertiesKHR> executables(
                executable_count, VkPipelineExecutablePropertiesKHR{
                                      .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
            dispatcher.vkGetPipelineExecutablePropertiesKHR(raw_device, &pipeline_info,
                                                              &executable_count,
                                                              executables.data());
            for (u32 executable_index = 0; executable_index < executable_count;
                 ++executable_index) {
                const auto& executable = executables[executable_index];
                LOG_INFO(Render_Vulkan,
                         "TEMP Shadow 87 executable {} name='{}' description='{}' subgroup={}",
                         executable_index, executable.name, executable.description,
                         executable.subgroupSize);
                const VkPipelineExecutableInfoKHR executable_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
                    .pNext = nullptr,
                    .pipeline = static_cast<VkPipeline>(*pipeline),
                    .executableIndex = executable_index,
                };
                u32 statistic_count{};
                dispatcher.vkGetPipelineExecutableStatisticsKHR(
                    raw_device, &executable_info, &statistic_count, nullptr);
                std::vector<VkPipelineExecutableStatisticKHR> statistics(
                    statistic_count, VkPipelineExecutableStatisticKHR{
                                         .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
                if (dispatcher.vkGetPipelineExecutableStatisticsKHR(
                        raw_device, &executable_info, &statistic_count,
                        statistics.data()) != VK_SUCCESS) {
                    continue;
                }
                for (const auto& statistic : statistics) {
                    switch (statistic.format) {
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                        LOG_INFO(Render_Vulkan, "TEMP Shadow 87 stat {} ({})={}", statistic.name,
                                 statistic.description, statistic.value.b32);
                        break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                        LOG_INFO(Render_Vulkan, "TEMP Shadow 87 stat {} ({})={}", statistic.name,
                                 statistic.description, statistic.value.i64);
                        break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                        LOG_INFO(Render_Vulkan, "TEMP Shadow 87 stat {} ({})={}", statistic.name,
                                 statistic.description, statistic.value.u64);
                        break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
                        LOG_INFO(Render_Vulkan, "TEMP Shadow 87 stat {} ({})={}", statistic.name,
                                 statistic.description, statistic.value.f64);
                        break;
                    default:
                        break;
                    }
                }

                u32 representation_count{};
                dispatcher.vkGetPipelineExecutableInternalRepresentationsKHR(
                    raw_device, &executable_info, &representation_count, nullptr);
                std::vector<VkPipelineExecutableInternalRepresentationKHR> representations(
                    representation_count);
                for (auto& representation : representations) {
                    representation.sType =
                        VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
                }
                if (dispatcher.vkGetPipelineExecutableInternalRepresentationsKHR(
                        raw_device, &executable_info, &representation_count,
                        representations.data()) != VK_SUCCESS) {
                    continue;
                }
                std::vector<std::vector<u8>> representation_data(representation_count);
                for (u32 index = 0; index < representation_count; ++index) {
                    representation_data[index].resize(representations[index].dataSize);
                    representations[index].pData = representation_data[index].data();
                }
                if (dispatcher.vkGetPipelineExecutableInternalRepresentationsKHR(
                        raw_device, &executable_info, &representation_count,
                        representations.data()) != VK_SUCCESS) {
                    continue;
                }
                using namespace Common::FS;
                const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
                std::filesystem::create_directories(dump_dir);
                for (u32 index = 0; index < representation_count; ++index) {
                    const auto& representation = representations[index];
                    LOG_INFO(Render_Vulkan,
                             "TEMP Shadow 87 representation {} name='{}' description='{}' "
                             "text={} bytes={}",
                             index, representation.name, representation.description,
                             representation.isText, representation.dataSize);
                    const auto extension = representation.isText ? "txt" : "bin";
                    const auto path = dump_dir / fmt::format("cs_0x{:016x}_native_{}.{}",
                                                              info->pgm_hash, index, extension);
                    const IOFile file{path, FileAccessMode::Create};
                    file.WriteSpan(std::span<const u8>{representation_data[index].data(),
                                                       representation.dataSize});
                }
            }
        }
    }
}

ComputePipeline::~ComputePipeline() = default;

} // namespace Vulkan
