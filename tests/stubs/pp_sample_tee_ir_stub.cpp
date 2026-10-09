// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>

#include "common/logging/log.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"

namespace Shader::IR {

void IREmitter::PpUpstreamSampleTee(const Value&, u32, const U32&, const U32&) {}

} // namespace Shader::IR

namespace Common {

std::string GetCurrentThreadName() {
    return "shadPS4::PpSampleTeeTest";
}

namespace Log {

std::unordered_map<std::string_view, std::shared_ptr<spdlog::logger>> ALL_LOGGERS{};

} // namespace Log
} // namespace Common

void assert_fail_impl() {
    std::abort();
}

namespace Vulkan::LiverpoolToVK {

vk::CompareOp DepthCompare(AmdGpu::DepthCompare comp) {
    switch (comp) {
    case AmdGpu::DepthCompare::Never:
        return vk::CompareOp::eNever;
    case AmdGpu::DepthCompare::Less:
        return vk::CompareOp::eLess;
    case AmdGpu::DepthCompare::Equal:
        return vk::CompareOp::eEqual;
    case AmdGpu::DepthCompare::LessEqual:
        return vk::CompareOp::eLessOrEqual;
    case AmdGpu::DepthCompare::Greater:
        return vk::CompareOp::eGreater;
    case AmdGpu::DepthCompare::NotEqual:
        return vk::CompareOp::eNotEqual;
    case AmdGpu::DepthCompare::GreaterEqual:
        return vk::CompareOp::eGreaterOrEqual;
    case AmdGpu::DepthCompare::Always:
        return vk::CompareOp::eAlways;
    }
    std::abort();
}

} // namespace Vulkan::LiverpoolToVK

[[noreturn]] void unreachable_impl() {
    std::abort();
}
