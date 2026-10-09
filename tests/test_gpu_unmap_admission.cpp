// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <fstream>
#include <future>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "core/libraries/gnmdriver/submission_gate.h"
#include "core/memory.h"

using namespace std::chrono_literals;

namespace {

struct RecordedMapping {
    std::atomic<bool> has_backing{true};
    std::atomic<u64> token{7};
    std::atomic<u32> revoke_count{};

    s32 Revoke() {
        has_backing = false;
        ++token;
        ++revoke_count;
        return 0;
    }
};

TEST(GpuUnmapAdmission, OnlyGpuVisibleProtectionRequiresTheSubmissionBoundary) {
    using Core::MemoryProt;

    EXPECT_FALSE(Core::RequiresSubmissionBoundaryBeforeUnmapping(MemoryProt::NoAccess));
    EXPECT_FALSE(Core::RequiresSubmissionBoundaryBeforeUnmapping(MemoryProt::CpuReadWrite));
    EXPECT_TRUE(Core::RequiresSubmissionBoundaryBeforeUnmapping(MemoryProt::GpuRead));
    EXPECT_TRUE(Core::RequiresSubmissionBoundaryBeforeUnmapping(MemoryProt::GpuWrite));
    EXPECT_TRUE(Core::RequiresSubmissionBoundaryBeforeUnmapping(MemoryProt::CpuReadWrite |
                                                                MemoryProt::GpuReadWrite));
}

TEST(GpuUnmapAdmission, GpuVisibleUnmapPreservesMappingUntilSubmittedBoundaryCompletes) {
    Libraries::GnmDriver::SubmissionGate gate;
    auto complete_boundary = gate.BeginBoundary();
    RecordedMapping mapping;
    std::promise<void> wait_started;
    auto wait_started_future = wait_started.get_future();

    auto unmap = std::async(std::launch::async, [&] {
        return Core::UnmapAfterSubmissionBoundary(
            Core::RequiresSubmissionBoundaryBeforeUnmapping(Core::MemoryProt::GpuReadWrite),
            [&] {
                wait_started.set_value();
                gate.WaitForBoundary();
            },
            [&] { return mapping.Revoke(); });
    });

    const auto wait_started_status = wait_started_future.wait_for(1s);
    const auto pending_status = unmap.wait_for(20ms);
    EXPECT_EQ(wait_started_status, std::future_status::ready);
    EXPECT_EQ(pending_status, std::future_status::timeout);
    EXPECT_TRUE(mapping.has_backing.load());
    EXPECT_EQ(mapping.token.load(), 7);
    EXPECT_EQ(mapping.revoke_count.load(), 0);

    complete_boundary();

    ASSERT_EQ(unmap.wait_for(1s), std::future_status::ready);
    EXPECT_EQ(unmap.get(), 0);
    EXPECT_FALSE(mapping.has_backing.load());
    EXPECT_EQ(mapping.token.load(), 8);
    EXPECT_EQ(mapping.revoke_count.load(), 1);
}

TEST(GpuUnmapAdmission, CpuOnlyUnmapDoesNotWaitForSubmittedBoundary) {
    Libraries::GnmDriver::SubmissionGate gate;
    auto complete_boundary = gate.BeginBoundary();
    RecordedMapping mapping;

    auto unmap = std::async(std::launch::async, [&] {
        return Core::UnmapAfterSubmissionBoundary(
            Core::RequiresSubmissionBoundaryBeforeUnmapping(Core::MemoryProt::CpuReadWrite),
            [&] { gate.WaitForBoundary(); }, [&] { return mapping.Revoke(); });
    });

    const auto unmap_status = unmap.wait_for(1s);
    const bool boundary_was_pending = !gate.IsBoundaryOpen();
    complete_boundary();

    ASSERT_EQ(unmap_status, std::future_status::ready);
    EXPECT_TRUE(boundary_was_pending);
    EXPECT_EQ(unmap.get(), 0);
    EXPECT_FALSE(mapping.has_backing.load());
    EXPECT_EQ(mapping.token.load(), 8);
    EXPECT_EQ(mapping.revoke_count.load(), 1);
}

TEST(GpuUnmapAdmission, KernelWaitsBeforeCallingTheMemoryManagerUnmapPath) {
    std::ifstream input{SHADPS4_KERNEL_MEMORY_SOURCE_PATH, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    const std::string source = contents.str();

    const auto function_begin = source.find("s32 PS4_SYSV_ABI sceKernelMunmap(");
    const auto function_end = source.find("s32 PS4_SYSV_ABI posix_munmap(", function_begin);
    ASSERT_NE(function_begin, std::string::npos);
    ASSERT_NE(function_end, std::string::npos);
    const auto body = source.substr(function_begin, function_end - function_begin);

    const auto policy = body.find("RequiresSubmissionBoundaryBeforeUnmapping(");
    const auto admission = body.find("UnmapAfterSubmissionBoundary(");
    const auto wait = body.find("WaitForSubmissionBoundary", admission);
    const auto revoke = body.find("memory->UnmapMemory(", wait);
    ASSERT_NE(policy, std::string::npos);
    ASSERT_NE(admission, std::string::npos);
    ASSERT_NE(wait, std::string::npos);
    ASSERT_NE(revoke, std::string::npos);
    EXPECT_LT(policy, admission);
    EXPECT_LT(admission, wait);
    EXPECT_LT(wait, revoke);
    EXPECT_EQ(body.find("scoped_lock"), std::string::npos);
    EXPECT_EQ(body.find("unique_lock"), std::string::npos);
}

} // namespace
