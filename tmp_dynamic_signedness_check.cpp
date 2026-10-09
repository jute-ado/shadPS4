#include <cstring>
#include <vector>

#include "shader_recompiler/resource.h"

struct FlatInfo {
    std::vector<u32> flattened_ud_buf;

    template <typename T>
    T ReadUdSharp(u32 index) const noexcept {
        T value{};
        std::memcpy(&value, flattened_ud_buf.data() + index, sizeof(value));
        return value;
    }
};

static AmdGpu::Image MakeImage(VAddr address, AmdGpu::NumberFormat format) {
    auto image = AmdGpu::Image::Null(false);
    image.base_address = address >> 8;
    image.type = static_cast<u64>(AmdGpu::ImageType::Color2D);
    image.num_format = static_cast<u64>(format);
    return image;
}

int main() {
    FlatInfo info{.flattened_ud_buf = std::vector<u32>(56)};
    const auto unsigned_image = MakeImage(0x10000, AmdGpu::NumberFormat::Uint);
    const auto signed_image = MakeImage(0x20000, AmdGpu::NumberFormat::Sint);
    std::memcpy(info.flattened_ud_buf.data() + 4, &unsigned_image, sizeof(unsigned_image));
    std::memcpy(info.flattened_ud_buf.data() + 12, &signed_image, sizeof(signed_image));
    const Shader::ImageResource signedness_resource{
        .sharp_idx = 4,
        .dynamic_descriptor_count = 2,
        .dynamic_descriptor_stride_dw = 8,
    };
    if (signedness_resource.IsTableEntryCompatible(info, 1)) {
        return 1;
    }

    auto array_image = MakeImage(0x30000, AmdGpu::NumberFormat::Unorm);
    array_image.type = static_cast<u64>(AmdGpu::ImageType::Color2DArray);
    auto cube_image = MakeImage(0x40000, AmdGpu::NumberFormat::Unorm);
    cube_image.type = static_cast<u64>(AmdGpu::ImageType::Cube);
    auto converted_image = MakeImage(0x50000, AmdGpu::NumberFormat::Uscaled);
    converted_image.type = static_cast<u64>(AmdGpu::ImageType::Color2DArray);
    auto srgb_image = MakeImage(0x60000, AmdGpu::NumberFormat::Srgb);
    srgb_image.type = static_cast<u64>(AmdGpu::ImageType::Color2DArray);
    std::memcpy(info.flattened_ud_buf.data() + 20, &array_image, sizeof(array_image));
    std::memcpy(info.flattened_ud_buf.data() + 28, &cube_image, sizeof(cube_image));
    std::memcpy(info.flattened_ud_buf.data() + 36, &converted_image, sizeof(converted_image));
    std::memcpy(info.flattened_ud_buf.data() + 44, &srgb_image, sizeof(srgb_image));
    const Shader::ImageResource semantics_resource{
        .sharp_idx = 20,
        .dynamic_descriptor_count = 4,
        .dynamic_descriptor_stride_dw = 8,
        .is_array = true,
    };
    return semantics_resource.IsTableEntryCompatible(info, 1) ||
                   semantics_resource.IsTableEntryCompatible(info, 2) ||
                   semantics_resource.IsTableEntryCompatible(info, 3)
               ? 1
               : 0;
}
