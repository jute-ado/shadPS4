// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "shader_recompiler/buffer_access_range.h"
#include "shader_recompiler/ir/type.h"
#include "video_core/amdgpu/resource.h"

#include <boost/container/static_vector.hpp>

namespace Shader {

static constexpr u32 NUM_USER_DATA_REGS = 16;
static constexpr u32 NUM_IMAGES = 64;
// Dynamic descriptor tables expand one shader image resource into multiple Vulkan bindings.
// Keep room for a full 64-entry table alongside the ordinary per-shader image budget.
static constexpr u32 NUM_IMAGE_BINDINGS = NUM_IMAGES + 64;
static constexpr u32 NUM_BUFFERS = 40;
static constexpr u32 NUM_SAMPLERS = 16;
static constexpr u32 NUM_FMASKS = 8;

enum class BufferType : u32 {
    Guest,
    Flatbuf,
    BdaPagetable,
    FaultBuffer,
    GdsBuffer,
    SharedMemory,
    ClipPlanes,
};

struct Info;

struct BufferResource {
    u32 sharp_idx;
    IR::Type used_types;
    AmdGpu::Buffer inline_cbuf;
    BufferType buffer_type;
    u8 instance_attrib{};
    bool is_written{};
    bool is_formatted{};
    BufferAccessRange access_range{};

    bool IsSpecial() const noexcept {
        return buffer_type != BufferType::Guest;
    }

    u64 GetBindingSize(const AmdGpu::Buffer buffer) const noexcept {
        const u64 declared_size = buffer.GetSize();
        return access_range.Fit(declared_size, BufferAddressLayout::From(buffer));
    }

    bool IsStorage([[maybe_unused]] const AmdGpu::Buffer buffer) const noexcept {
        // When using uniform buffers, a size is required at compilation time, so we need to
        // either compile a lot of shader specializations to handle each size or just force it to
        // the maximum possible size always. However, for some vendors the shader-supplied size is
        // used for bounds checking uniform buffer accesses, so the latter would effectively turn
        // off buffer robustness behavior. Instead, force storage buffers which are bounds checked
        // using the actual buffer size. We are assuming the performance hit from this is
        // acceptable.
        return true; // buffer.GetSize() > profile.max_ubo_size || is_written;
    }
    constexpr AmdGpu::Buffer GetSharp(const auto& info) const noexcept {
        AmdGpu::Buffer buffer{};
        if (inline_cbuf) {
            buffer = inline_cbuf;
            if (inline_cbuf.base_address != 1) {
                buffer.base_address += info.pgm_base; // address fixup
            }
        } else {
            buffer = info.template ReadUdSharp<AmdGpu::Buffer>(sharp_idx);
        }
        if (!buffer.Valid()) {
            LOG_DEBUG(Render, "Encountered invalid buffer sharp");
            return AmdGpu::Buffer::Null();
        }
        return buffer;
    }
};
using BufferResourceList = boost::container::static_vector<BufferResource, NUM_BUFFERS>;

enum class MipStorageFallbackMode : u32 { None, DynamicIndex, ConstantIndex };

struct ImageResource {
    u32 sharp_idx;
    bool is_depth{};
    bool is_atomic{};
    bool is_array{};
    bool is_written{};
    bool uses_integer_coordinates{};
    bool queries_dimensions{};
    bool is_r128{};
    bool is_dynamic_table{};
    u32 dynamic_table_count{};
    u32 dynamic_table_offset_dw{};
    u32 dynamic_table_stride_dw{};
    AmdGpu::Image dynamic_table_template{};
    MipStorageFallbackMode mip_fallback_mode{};
    u32 constant_mip_index{};

    constexpr AmdGpu::Image ValidateSharp(AmdGpu::Image image) const noexcept {
        if (!image.Valid()) {
            LOG_DEBUG(Render_Vulkan, "Encountered invalid image sharp");
            image = AmdGpu::Image::Null(is_depth);
        } else if (is_depth) {
            const auto data_fmt = image.GetDataFmt();
            if (data_fmt != AmdGpu::DataFormat::Format16 &&
                data_fmt != AmdGpu::DataFormat::Format32) {
                LOG_DEBUG(Render_Vulkan,
                          "Encountered non-depth image used with depth instruction!");
                image = AmdGpu::Image::Null(true);
            }
        }
        return image;
    }

    constexpr AmdGpu::Image GetSharp(const auto& info, const u32 table_index = 0) const noexcept {
        AmdGpu::Image image{};
        if (is_dynamic_table) {
            // Shader compilation and specialization may outlive the guest table mapping. Use the
            // descriptor captured while tracking the shader; runtime binding refreshes each table
            // entry through GetSharpIfMapped instead of dereferencing this pointer again.
            image = dynamic_table_template;
        } else if (!is_r128) {
            image = info.template ReadUdSharp<AmdGpu::Image>(sharp_idx);
        } else {
            const auto raw = info.template ReadUdSharp<u128>(sharp_idx);
            std::memcpy(&image, &raw, sizeof(raw));
            image.pitch = image.width;
        }
        return ValidateSharp(image);
    }

    template <typename ReadMappedMemory>
    AmdGpu::Image GetSharpIfMapped(const auto& info, const u32 table_index,
                                   ReadMappedMemory&& read_mapped_memory) const noexcept {
        if (!is_dynamic_table) {
            return GetSharp(info, table_index);
        }
        if (table_index >= dynamic_table_count) {
            return AmdGpu::Image::Null(is_depth);
        }

        const VAddr table_base = info.template ReadUdSharp<VAddr>(sharp_idx);
        const u64 descriptor_size = is_r128 ? sizeof(u128) : sizeof(AmdGpu::Image);
        const u64 descriptor_offset_dw =
            u64{dynamic_table_offset_dw} + u64{table_index} * dynamic_table_stride_dw;
        if (descriptor_offset_dw > std::numeric_limits<u64>::max() / sizeof(u32)) {
            return AmdGpu::Image::Null(is_depth);
        }
        const u64 descriptor_offset = descriptor_offset_dw * sizeof(u32);
        if (table_base == 0 || table_base > std::numeric_limits<VAddr>::max() - descriptor_offset) {
            return AmdGpu::Image::Null(is_depth);
        }

        AmdGpu::Image image{};
        const VAddr descriptor_address = table_base + descriptor_offset;
        if (!std::forward<ReadMappedMemory>(read_mapped_memory)(descriptor_address, &image,
                                                                descriptor_size)) {
            return AmdGpu::Image::Null(is_depth);
        }
        if (is_r128) {
            image.pitch = image.width;
        }
        return ValidateSharp(image);
    }

    u32 NumBindings(const auto& info) const {
        if (is_dynamic_table) {
            return dynamic_table_count;
        }
        const AmdGpu::Image tsharp = GetSharp(info);
        return (mip_fallback_mode == MipStorageFallbackMode::DynamicIndex)
                   ? (tsharp.last_level - tsharp.base_level + 1)
                   : 1;
    }
};
using ImageResourceList = boost::container::static_vector<ImageResource, NUM_IMAGES>;

struct SamplerResource {
    u32 sharp_idx;
    AmdGpu::Sampler inline_sampler;
    u32 is_inline_sampler : 1;
    u32 associated_image : 4;
    u32 disable_aniso : 1;

    constexpr AmdGpu::Sampler GetSharp(const auto& info) const noexcept {
        return is_inline_sampler ? inline_sampler
                                 : info.template ReadUdSharp<AmdGpu::Sampler>(sharp_idx);
    }
};
using SamplerResourceList = boost::container::static_vector<SamplerResource, NUM_SAMPLERS>;

struct FMaskResource {
    u32 sharp_idx;

    constexpr AmdGpu::Image GetSharp(const auto& info) const noexcept {
        return info.template ReadUdSharp<AmdGpu::Image>(sharp_idx);
    }
};
using FMaskResourceList = boost::container::static_vector<FMaskResource, NUM_FMASKS>;

struct PushData {
    static constexpr u32 XOffsetIndex = 0;
    static constexpr u32 YOffsetIndex = 1;
    static constexpr u32 XScaleIndex = 2;
    static constexpr u32 YScaleIndex = 3;
    static constexpr u32 UdRegsIndex = 4;
    static constexpr u32 BufOffsetIndex = UdRegsIndex + NUM_USER_DATA_REGS / 4;
    static constexpr u32 HostInvScaleIndex = BufOffsetIndex + 3;

    float xoffset;
    float yoffset;
    float xscale;
    float yscale;
    std::array<u32, NUM_USER_DATA_REGS> ud_regs;
    std::array<u8, NUM_BUFFERS> buf_offsets;
    float host_inv_scale{1.0f};

    void AddOffset(u32 binding, u32 offset) {
        ASSERT(offset < 256 && binding < buf_offsets.size());
        buf_offsets[binding] = offset;
    }
};
static_assert(sizeof(PushData) <= 128,
              "PushData size is greater than minimum size guaranteed by Vulkan spec");

} // namespace Shader
