// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/vk_memory_diagnostics.h"

namespace {

std::string ReadText(const char* path) {
    std::ifstream source{path, std::ios::binary};
    EXPECT_TRUE(source.is_open());
    std::string text{std::istreambuf_iterator<char>{source}, std::istreambuf_iterator<char>{}};
    std::erase(text, '\r');
    return text;
}

std::string FunctionBody(const std::string& source, const std::string_view signature) {
    const auto signature_begin = source.find(signature);
    EXPECT_NE(signature_begin, std::string::npos);
    if (signature_begin == std::string::npos) {
        return {};
    }
    const auto body_begin = source.find('{', signature_begin);
    EXPECT_NE(body_begin, std::string::npos);
    if (body_begin == std::string::npos) {
        return {};
    }

    unsigned depth{};
    for (size_t index = body_begin; index < source.size(); ++index) {
        if (source[index] == '{') {
            ++depth;
        } else if (source[index] == '}' && --depth == 0) {
            return source.substr(body_begin, index - body_begin + 1);
        }
    }
    ADD_FAILURE() << "Unterminated function body for " << signature;
    return {};
}

struct FakeScreenshotReadback {
    bool coherent{};
    bool invalidation_succeeds{true};
    std::vector<std::string> events;

    bool Consume() {
        if (coherent) {
            events.emplace_back("coherent-fast-path");
        } else {
            events.emplace_back("invalidate");
            if (!invalidation_succeeds) {
                return false;
            }
        }
        events.emplace_back("convert");
        events.emplace_back("publish");
        return true;
    }
};

TEST(VulkanMemoryDiagnostics, RequiresExactOptInValue) {
    for (const auto value : {"", "0", "true", "yes", "01", "+1", " 1", "1 ", "1\n", "-1"}) {
        EXPECT_FALSE(Vulkan::ParseVulkanMemoryDiagnostics(value)) << value;
    }
    EXPECT_TRUE(Vulkan::ParseVulkanMemoryDiagnostics("1"));
}

TEST(VulkanMemoryDiagnosticsWiring, InstanceOwnsOptInBudgetSnapshot) {
    const auto source = ReadText(SHADPS4_VK_INSTANCE_SOURCE_PATH);
    const auto header = ReadText(SHADPS4_VK_INSTANCE_HEADER_PATH);
    EXPECT_NE(header.find("LogDeviceMemoryBudget"), std::string::npos);
    EXPECT_NE(source.find("VulkanMemoryDiagnosticsEnabled()"), std::string::npos);
    EXPECT_NE(source.find("vmaGetHeapBudgets"), std::string::npos);
    EXPECT_NE(source.find("Vulkan memory budget event="), std::string::npos);
}

TEST(VulkanMemoryDiagnosticsWiring, FailureBoundariesRequestBudgetSnapshot) {
    const auto scheduler = ReadText(SHADPS4_SCHEDULER_SOURCE_PATH);
    const auto semaphore = ReadText(SHADPS4_MASTER_SEMAPHORE_SOURCE_PATH);
    EXPECT_NE(scheduler.find("LogDeviceMemoryBudget"), std::string::npos);
    EXPECT_NE(semaphore.find("LogDeviceMemoryBudget"), std::string::npos);
}

TEST(VulkanMemoryDiagnosticsWiring, AllocationAndGcBoundariesRequestBudgetSnapshot) {
    const auto buffer = ReadText(SHADPS4_BUFFER_SOURCE_PATH);
    const auto image = ReadText(SHADPS4_IMAGE_SOURCE_PATH);
    const auto tile = ReadText(SHADPS4_TILE_MANAGER_SOURCE_PATH);
    const auto presenter = ReadText(SHADPS4_PRESENTER_SOURCE_PATH);
    const auto buffer_cache = ReadText(SHADPS4_BUFFER_CACHE_SOURCE_PATH);
    const auto texture_cache = ReadText(SHADPS4_TEXTURE_CACHE_SOURCE_PATH);
    for (const auto* source : {&buffer, &image, &tile, &presenter, &buffer_cache, &texture_cache}) {
        EXPECT_NE(source->find("LogDeviceMemoryBudget"), std::string::npos);
    }
}

TEST(ScreenshotReadback, NonCoherentReadbackInvalidatesBeforeCpuPublication) {
    FakeScreenshotReadback readback{.coherent = false};

    ASSERT_TRUE(readback.Consume());
    EXPECT_EQ(readback.events,
              (std::vector<std::string>{"invalidate", "convert", "publish"}));
}

TEST(ScreenshotReadback, InvalidationFailureSuppressesCpuPublication) {
    FakeScreenshotReadback readback{.coherent = false, .invalidation_succeeds = false};

    EXPECT_FALSE(readback.Consume());
    EXPECT_EQ(readback.events, (std::vector<std::string>{"invalidate"}));
}

TEST(ScreenshotReadback, CoherentReadbackUsesTheFastPath) {
    FakeScreenshotReadback readback{.coherent = true};

    ASSERT_TRUE(readback.Consume());
    EXPECT_EQ(readback.events,
              (std::vector<std::string>{"coherent-fast-path", "convert", "publish"}));
}

TEST(ScreenshotReadbackWiring, InvalidatesMappedRangeBeforeConversion) {
    const auto source = ReadText(SHADPS4_PRESENTER_SOURCE_PATH);
    const auto save = FunctionBody(source, "static void SavePendingScreenshots(");
    ASSERT_FALSE(save.empty());

    const auto coherent = save.find("readback.buffer.is_coherent");
    const auto invalidate = save.find("readback.buffer.InvalidateMappedRange");
    const auto conversion = save.find("ConvertReadbackToRgba8(readback, rgba)");
    const auto failure = save.find("continue", invalidate);
    ASSERT_NE(coherent, std::string::npos);
    ASSERT_NE(invalidate, std::string::npos);
    ASSERT_NE(conversion, std::string::npos);
    ASSERT_NE(failure, std::string::npos);
    EXPECT_LT(coherent, invalidate);
    EXPECT_LT(invalidate, conversion);
    EXPECT_LT(invalidate, failure);
}

} // namespace
