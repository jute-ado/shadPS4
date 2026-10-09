// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <fstream>
#include <future>
#include <limits>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "core/memory.h"

TEST(GpuMappingAdmission, OnlyGpuVisibleMappingsRequireACompletedSubmissionBoundary) {
    using Core::MemoryProt;

    EXPECT_FALSE(Core::RequiresSubmissionBoundaryBeforeMapping(MemoryProt::NoAccess));
    EXPECT_FALSE(Core::RequiresSubmissionBoundaryBeforeMapping(MemoryProt::CpuReadWrite));
    EXPECT_TRUE(Core::RequiresSubmissionBoundaryBeforeMapping(MemoryProt::GpuRead));
    EXPECT_TRUE(Core::RequiresSubmissionBoundaryBeforeMapping(MemoryProt::GpuWrite));
    EXPECT_TRUE(Core::RequiresSubmissionBoundaryBeforeMapping(MemoryProt::CpuReadWrite |
                                                              MemoryProt::GpuReadWrite));
}

TEST(GpuMappingAdmission, OnlyGpuVisibleUnmapsRequireACompletedSubmissionBoundary) {
    using Core::MemoryProt;

    EXPECT_FALSE(Core::RequiresSubmissionBoundaryBeforeUnmapping(MemoryProt::NoAccess));
    EXPECT_FALSE(Core::RequiresSubmissionBoundaryBeforeUnmapping(MemoryProt::CpuReadWrite));
    EXPECT_TRUE(Core::RequiresSubmissionBoundaryBeforeUnmapping(MemoryProt::GpuRead));
    EXPECT_TRUE(Core::RequiresSubmissionBoundaryBeforeUnmapping(MemoryProt::GpuWrite));
    EXPECT_TRUE(Core::RequiresSubmissionBoundaryBeforeUnmapping(MemoryProt::CpuReadWrite |
                                                                MemoryProt::GpuReadWrite));
}

TEST(GpuMappingAdmission, UnmappingWaitsWhenAnyAreaInTheRangeIsGpuVisible) {
    using Core::MemoryProt;
    const auto query = [](VAddr address, MemoryProt& prot, VAddr& end) {
        if (address < 0x2000) {
            prot = MemoryProt::CpuReadWrite;
            end = 0x2000;
        } else {
            prot = MemoryProt::CpuReadWrite | MemoryProt::GpuReadWrite;
            end = 0x3000;
        }
        return true;
    };

    EXPECT_TRUE(Core::RequiresSubmissionBoundaryBeforeUnmapping(0x1000, 0x2000, query));
}

TEST(GpuMappingAdmission, CpuOnlyAndInvalidUnmapRangesDoNotWait) {
    using Core::MemoryProt;
    const auto cpu_only = [](VAddr, MemoryProt& prot, VAddr& end) {
        prot = MemoryProt::CpuReadWrite;
        end = 0x3000;
        return true;
    };
    const auto missing = [](VAddr, MemoryProt&, VAddr&) { return false; };
    const auto stalled = [](VAddr address, MemoryProt& prot, VAddr& end) {
        prot = MemoryProt::GpuReadWrite;
        end = address;
        return true;
    };

    EXPECT_FALSE(Core::RequiresSubmissionBoundaryBeforeUnmapping(0x1000, 0x2000, cpu_only));
    EXPECT_FALSE(Core::RequiresSubmissionBoundaryBeforeUnmapping(0x1000, 0, cpu_only));
    EXPECT_FALSE(Core::RequiresSubmissionBoundaryBeforeUnmapping(
        std::numeric_limits<VAddr>::max() - 0x1000, 0x2000, cpu_only));
    EXPECT_FALSE(Core::RequiresSubmissionBoundaryBeforeUnmapping(0x1000, 0x1000, missing));
    EXPECT_FALSE(Core::RequiresSubmissionBoundaryBeforeUnmapping(0x1000, 0x1000, stalled));
}

TEST(GpuMappingAdmission, GpuVisibleBackingRemainsLiveUntilTheBoundaryCompletes) {
    using namespace std::chrono_literals;

    std::promise<void> release_boundary;
    const auto release = release_boundary.get_future().share();
    std::promise<void> wait_started;
    auto started = wait_started.get_future();
    std::atomic<bool> has_backing{true};
    std::atomic<u64> mapping_token{7};

    const auto admission_owner = std::make_shared<u32>(0);
    auto unmap = std::async(std::launch::async, [&] {
        return Core::ChangeGpuVisibleAddressSpaceAfterSubmissionBoundary(
            true,
            [&] {
                wait_started.set_value();
                release.wait();
                return Common::UniqueFunction<void>{[admission_owner] {}};
            },
            [&] {
                EXPECT_EQ(admission_owner.use_count(), 2);
                has_backing = false;
                ++mapping_token;
                return 0;
            });
    });

    const auto started_status = started.wait_for(1s);
    EXPECT_EQ(started_status, std::future_status::ready);
    EXPECT_TRUE(has_backing.load());
    EXPECT_EQ(mapping_token.load(), 7);

    // Always release the worker, even if a heavily loaded test host missed the readiness deadline.
    release_boundary.set_value();

    ASSERT_EQ(unmap.wait_for(1s), std::future_status::ready);
    EXPECT_EQ(unmap.get(), 0);
    EXPECT_FALSE(has_backing.load());
    EXPECT_EQ(mapping_token.load(), 8);
    EXPECT_EQ(admission_owner.use_count(), 1);
}

TEST(GpuMappingAdmission, CpuOnlyUnmapSkipsTheBoundary) {
    u32 wait_count{};
    u32 unmap_count{};

    const auto result = Core::ChangeGpuVisibleAddressSpaceAfterSubmissionBoundary(
        false,
        [&] {
            ++wait_count;
            return Common::UniqueFunction<void>{};
        },
        [&] {
            ++unmap_count;
            return 17;
        });

    EXPECT_EQ(result, 17);
    EXPECT_EQ(wait_count, 0);
    EXPECT_EQ(unmap_count, 1);
}

TEST(GpuMappingAdmission, KernelAppliesTheBoundaryBeforeRevokingTheMapping) {
    std::ifstream input{SHADPS4_KERNEL_MEMORY_SOURCE_PATH, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    const std::string source = contents.str();

    const auto function_begin = source.find("s32 PS4_SYSV_ABI sceKernelMunmap(");
    const auto function_end = source.find("s32 PS4_SYSV_ABI posix_munmap(", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);

    const auto admission = body.find("memory->UnmapMemory(");
    const auto wait = body.find("AcquireGpuSubmissionBoundary", admission);
    const auto function_end_call = body.find(";", wait);
    ASSERT_NE(wait, std::string::npos);
    ASSERT_NE(function_end_call, std::string::npos);
    EXPECT_LT(admission, wait);
    EXPECT_EQ(body.find("QueryProtection"), std::string::npos);
}
