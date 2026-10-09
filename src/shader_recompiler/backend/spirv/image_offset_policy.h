// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Shader::Backend::SPIRV {

enum class ImageOffsetInstruction {
    Sample,
    Gather,
};

constexpr bool CanUseRuntimeImageOffset(ImageOffsetInstruction instruction,
                                        bool supports_maintenance8,
                                        bool supports_image_gather_extended) {
    return supports_image_gather_extended &&
           (instruction == ImageOffsetInstruction::Gather || supports_maintenance8);
}

constexpr bool NeedsImageGatherExtendedCapability(bool has_image_gather,
                                                  bool uses_dynamic_sample_offset) {
    return has_image_gather || uses_dynamic_sample_offset;
}

} // namespace Shader::Backend::SPIRV
