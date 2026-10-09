// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <optional>
#include <vector>

#include "core/memory.h"
#include "video_core/renderer_vulkan/compute_buffer_binding_guard.h"

namespace {

struct MappingInterval {
    VAddr begin;
    VAddr end;
    u64 token;
};

class RecordedMappingTokens {
public:
    void Map(VAddr begin, u64 size, u64 token) {
        mappings.push_back({begin, begin + size, token});
    }

    void Remap(VAddr begin, u64 size, u64 token) {
        const VAddr end = begin + size;
        std::erase_if(mappings, [&](const MappingInterval& mapping) {
            return mapping.begin < end && begin < mapping.end;
        });
        Map(begin, size, token);
    }

    [[nodiscard]] u64 GetMappingToken(VAddr address, u64 size) const {
        if (size == 0 || address > std::numeric_limits<VAddr>::max() - size) {
            return 0;
        }
        const VAddr end = address + size;
        const auto mapping = std::ranges::find_if(mappings, [&](const MappingInterval& item) {
            return item.begin <= address && item.end >= end;
        });
        return mapping == mappings.end() ? 0 : mapping->token;
    }

private:
    std::vector<MappingInterval> mappings;
};

} // namespace

TEST(GpuMappingAdmission, OnlyGpuVisibleMappingsRequireACompletedSubmissionBoundary) {
    using Core::MemoryProt;

    EXPECT_FALSE(Core::RequiresSubmissionBoundaryBeforeMapping(MemoryProt::NoAccess));
    EXPECT_FALSE(Core::RequiresSubmissionBoundaryBeforeMapping(MemoryProt::CpuReadWrite));
    EXPECT_TRUE(Core::RequiresSubmissionBoundaryBeforeMapping(MemoryProt::GpuRead));
    EXPECT_TRUE(Core::RequiresSubmissionBoundaryBeforeMapping(MemoryProt::GpuWrite));
    EXPECT_TRUE(Core::RequiresSubmissionBoundaryBeforeMapping(MemoryProt::CpuReadWrite |
                                                              MemoryProt::GpuReadWrite));
}

TEST(ComputeBufferBindingGuard, AcceptsFullyMappedStableRange) {
    RecordedMappingTokens mappings;
    mappings.Map(0x10000, 0x4000, 7);

    const auto selected = Vulkan::SelectComputeBufferBinding(
        0x11000, 0x2000,
        [&](VAddr address, u64 size) { return mappings.GetMappingToken(address, size); });

    ASSERT_TRUE(selected.has_value());
    EXPECT_TRUE(Vulkan::IsComputeBufferBindingCurrent(
        *selected,
        [&](VAddr address, u64 size) { return mappings.GetMappingToken(address, size); }));
}

TEST(ComputeBufferBindingGuard, ClampsOversizedDescriptorBeforeSelectingMappedRange) {
    RecordedMappingTokens mappings;
    constexpr VAddr address = 0x11000;
    constexpr u64 declared_size = 0xfffffffc;
    constexpr u64 mapped_extent = 0x3000;
    mappings.Map(0x10000, 0x5000, 7);

    u32 clamp_calls = 0;
    const auto range = Vulkan::ClampComputeBufferBindingRange(
        address, declared_size, [&](VAddr clamped_address, u64 clamped_declared_size) {
            ++clamp_calls;
            EXPECT_EQ(clamped_address, address);
            EXPECT_EQ(clamped_declared_size, declared_size);
            return mapped_extent;
        });
    const auto selected = Vulkan::SelectComputeBufferBinding(
        range,
        [&](VAddr selected_address, u64 selected_size) {
            EXPECT_EQ(selected_address, address);
            EXPECT_EQ(selected_size, mapped_extent);
            return mappings.GetMappingToken(selected_address, selected_size);
        });

    EXPECT_EQ(clamp_calls, 1);
    ASSERT_TRUE(selected.has_value());
    EXPECT_EQ(selected->address, address);
    EXPECT_EQ(selected->size, mapped_extent);

    mappings.Remap(0x10000, 0x5000, 8);
    EXPECT_FALSE(Vulkan::IsComputeBufferBindingCurrent(
        *selected,
        [&](VAddr selected_address, u64 selected_size) {
            return mappings.GetMappingToken(selected_address, selected_size);
        }));
}

TEST(ComputeBufferBindingGuard, RejectsUnmappedAndPartiallyMappedRanges) {
    RecordedMappingTokens mappings;
    mappings.Map(0x10000, 0x2000, 7);
    mappings.Map(0x30000, 0x1000, 8);
    mappings.Map(0x31000, 0x1000, 9);

    EXPECT_FALSE(Vulkan::SelectComputeBufferBinding(
                     0x20000, 0x1000,
                     [&](VAddr address, u64 size) {
                         return mappings.GetMappingToken(address, size);
                     })
                     .has_value());
    EXPECT_FALSE(Vulkan::SelectComputeBufferBinding(
                     0x11000, 0x2000,
                     [&](VAddr address, u64 size) {
                         return mappings.GetMappingToken(address, size);
                     })
                     .has_value());
    EXPECT_FALSE(Vulkan::SelectComputeBufferBinding(
                     0x30000, 0x2000,
                     [&](VAddr address, u64 size) {
                         return mappings.GetMappingToken(address, size);
                     })
                     .has_value());
}

TEST(ComputeBufferBindingGuard, RejectsGenerationChangeBetweenSelectionAndDispatch) {
    RecordedMappingTokens mappings;
    mappings.Map(0x10000, 0x4000, 7);
    const auto selected = Vulkan::SelectComputeBufferBinding(
        0x11000, 0x2000,
        [&](VAddr address, u64 size) { return mappings.GetMappingToken(address, size); });
    ASSERT_TRUE(selected.has_value());

    mappings.Remap(0x10000, 0x4000, 8);

    const std::array selections{*selected};
    EXPECT_FALSE(Vulkan::AreComputeBufferBindingsCurrent(
        selections,
        [&](VAddr address, u64 size) { return mappings.GetMappingToken(address, size); }));
}

TEST(ComputeBufferBindingGuard, ReselectionAcceptsCurrentGenerationForSameShaderPermutation) {
    RecordedMappingTokens mappings;
    mappings.Map(0x10000, 0x4000, 7);
    const auto old_selection = Vulkan::SelectComputeBufferBinding(
        0x11000, 0x2000,
        [&](VAddr address, u64 size) { return mappings.GetMappingToken(address, size); });
    ASSERT_TRUE(old_selection.has_value());

    mappings.Remap(0x10000, 0x4000, 8);
    const auto new_selection = Vulkan::SelectComputeBufferBinding(
        0x11000, 0x2000,
        [&](VAddr address, u64 size) { return mappings.GetMappingToken(address, size); });

    ASSERT_TRUE(new_selection.has_value());
    EXPECT_FALSE(Vulkan::IsComputeBufferBindingCurrent(
        *old_selection,
        [&](VAddr address, u64 size) { return mappings.GetMappingToken(address, size); }));
    EXPECT_TRUE(Vulkan::IsComputeBufferBindingCurrent(
        *new_selection,
        [&](VAddr address, u64 size) { return mappings.GetMappingToken(address, size); }));
    EXPECT_NE(old_selection->mapping_token, new_selection->mapping_token);
}

TEST(ComputeBufferBindingGuard, UnrelatedMappingChangesDoNotInvalidateSelection) {
    RecordedMappingTokens mappings;
    mappings.Map(0x10000, 0x4000, 7);
    mappings.Map(0x20000, 0x4000, 11);
    const auto selected = Vulkan::SelectComputeBufferBinding(
        0x11000, 0x2000,
        [&](VAddr address, u64 size) { return mappings.GetMappingToken(address, size); });
    ASSERT_TRUE(selected.has_value());

    mappings.Remap(0x20000, 0x4000, 12);

    const std::array selections{*selected};
    EXPECT_TRUE(Vulkan::AreComputeBufferBindingsCurrent(
        selections,
        [&](VAddr address, u64 size) { return mappings.GetMappingToken(address, size); }));
}
