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

[[nodiscard]] inline bool IsSupportedDynamicImageModel(const AmdGpu::Image& model,
                                                       bool is_array) noexcept {
    if (!model.Valid() || AmdGpu::IsFmask(model.GetDataFmt())) {
        return false;
    }
    const auto view_type = model.GetViewType(is_array);
    return view_type != AmdGpu::ImageType::Color2DMsaa &&
           view_type != AmdGpu::ImageType::Color2DMsaaArray;
}

struct ImageResource {
    u32 sharp_idx;
    // A non-zero count describes one sampled-image descriptor-array binding backed by the
    // flattened SRT snapshot. It is a host-supported window, not the guest allocation length.
    u32 dynamic_descriptor_count{};
    u32 dynamic_descriptor_stride_dw{};
    bool is_depth{};
    bool is_atomic{};
    bool is_array{};
    bool is_written{};
    bool uses_integer_coordinates{};
    bool queries_dimensions{};
    bool is_r128{};
    bool requires_dynamic_oob_guard{};
    MipStorageFallbackMode mip_fallback_mode{};
    u32 constant_mip_index{};

    [[nodiscard]] constexpr bool HasValidDynamicRange(const auto& info) const noexcept {
        if (dynamic_descriptor_count == 0) {
            return true;
        }
        if (dynamic_descriptor_count > NUM_IMAGES || dynamic_descriptor_stride_dw == 0) {
            return false;
        }
        const u32 descriptor_dw = is_r128 ? 4U : 8U;
        if (dynamic_descriptor_stride_dw < descriptor_dw) {
            return false;
        }
        const u64 end_dw = u64{sharp_idx} +
                           u64{dynamic_descriptor_count - 1U} * dynamic_descriptor_stride_dw +
                           descriptor_dw;
        return end_dw <= info.flattened_ud_buf.size();
    }

    constexpr AmdGpu::Image ReadSharpUnchecked(const auto& info,
                                               u32 descriptor_index = 0) const noexcept {
        const u32 descriptor_offset =
            dynamic_descriptor_count != 0 ? descriptor_index * dynamic_descriptor_stride_dw : 0;
        const u32 descriptor_sharp_idx = sharp_idx + descriptor_offset;
        AmdGpu::Image image{};
        if (!is_r128) {
            image = info.template ReadUdSharp<AmdGpu::Image>(descriptor_sharp_idx);
        } else {
            const auto raw = info.template ReadUdSharp<u128>(descriptor_sharp_idx);
            std::memcpy(&image, &raw, sizeof(raw));
            image.pitch = image.width;
        }
        return image;
    }

    constexpr AmdGpu::Image GetSharp(const auto& info, u32 descriptor_index = 0) const noexcept {
        if (dynamic_descriptor_count != 0 &&
            (!HasValidDynamicRange(info) || descriptor_index >= dynamic_descriptor_count)) {
            return AmdGpu::Image::Null(is_depth);
        }
        AmdGpu::Image image = ReadSharpUnchecked(info, descriptor_index);
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

    bool IsTableEntryCompatible(const auto& info, u32 descriptor_index) const noexcept {
        if (dynamic_descriptor_count == 0 || !HasValidDynamicRange(info) ||
            descriptor_index >= dynamic_descriptor_count) {
            return false;
        }
        const auto model = ReadSharpUnchecked(info, 0);
        const auto candidate = ReadSharpUnchecked(info, descriptor_index);
        return IsSupportedDynamicImageModel(model, is_array) &&
               IsSupportedDynamicImageModel(candidate, is_array) &&
               model.GetViewType(is_array) == candidate.GetViewType(is_array) &&
               model.IsCube() == candidate.IsCube() &&
               AmdGpu::GetNumberClass(model.GetNumberFmt()) ==
                   AmdGpu::GetNumberClass(candidate.GetNumberFmt()) &&
               model.GetNumberConversion() == candidate.GetNumberConversion() &&
               (model.GetNumberFmt() == AmdGpu::NumberFormat::Srgb) ==
                   (candidate.GetNumberFmt() == AmdGpu::NumberFormat::Srgb);
    }

    bool HasSupportedDynamicModel(const auto& info) const noexcept {
        return dynamic_descriptor_count == 0 ||
               (HasValidDynamicRange(info) &&
                IsSupportedDynamicImageModel(ReadSharpUnchecked(info, 0), is_array));
    }

    u32 NumBindings(const auto& info) const {
        if (dynamic_descriptor_count != 0) {
            return HasValidDynamicRange(info) ? dynamic_descriptor_count : 1U;
        }
        const AmdGpu::Image tsharp = GetSharp(info);
        return (mip_fallback_mode == MipStorageFallbackMode::DynamicIndex)
                   ? (tsharp.last_level - tsharp.base_level + 1)
                   : 1;
    }
};

[[nodiscard]] constexpr bool CanAliasImageResources(const ImageResource& lhs,
                                                    const ImageResource& rhs) noexcept {
    return lhs.sharp_idx == rhs.sharp_idx &&
           lhs.dynamic_descriptor_count == rhs.dynamic_descriptor_count &&
           lhs.dynamic_descriptor_stride_dw == rhs.dynamic_descriptor_stride_dw &&
           lhs.is_depth == rhs.is_depth && lhs.is_array == rhs.is_array &&
           lhs.is_r128 == rhs.is_r128 &&
           lhs.requires_dynamic_oob_guard == rhs.requires_dynamic_oob_guard &&
           lhs.mip_fallback_mode == rhs.mip_fallback_mode &&
           lhs.constant_mip_index == rhs.constant_mip_index;
}

using ImageResourceList = boost::container::static_vector<ImageResource, NUM_IMAGES>;

struct SamplerResource {
    u32 sharp_idx;
    AmdGpu::Sampler inline_sampler;
    u32 is_inline_sampler : 1;
    u32 associated_image : 6;
    u32 disable_aniso : 1;

    constexpr AmdGpu::Sampler GetSharp(const auto& info) const noexcept {
        return is_inline_sampler ? inline_sampler
                                 : info.template ReadUdSharp<AmdGpu::Sampler>(sharp_idx);
    }
};

[[nodiscard]] constexpr bool CanAliasSamplerResources(const SamplerResource& lhs,
                                                      const SamplerResource& rhs) noexcept {
    return lhs.sharp_idx == rhs.sharp_idx && lhs.is_inline_sampler == rhs.is_inline_sampler &&
           lhs.inline_sampler == rhs.inline_sampler && lhs.disable_aniso == rhs.disable_aniso &&
           (!lhs.disable_aniso || lhs.associated_image == rhs.associated_image);
}

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
