// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

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

TEST(GpuMappingAdmission, CpuOnlyUnmappingDoesNotWait) {
    using Core::MemoryProt;
    const auto query = [](VAddr, MemoryProt& prot, VAddr& end) {
        prot = MemoryProt::CpuReadWrite;
        end = 0x3000;
        return true;
    };

    EXPECT_FALSE(Core::RequiresSubmissionBoundaryBeforeUnmapping(0x1000, 0x2000, query));
}

TEST(GpuMappingAdmission, InvalidOrNonProgressingRangesDoNotRequestABoundary) {
    using Core::MemoryProt;
    const auto missing = [](VAddr, MemoryProt&, VAddr&) { return false; };
    const auto stalled = [](VAddr address, MemoryProt& prot, VAddr& end) {
        prot = MemoryProt::GpuReadWrite;
        end = address;
        return true;
    };

    EXPECT_FALSE(Core::RequiresSubmissionBoundaryBeforeUnmapping(0x1000, 0x1000, missing));
    EXPECT_FALSE(Core::RequiresSubmissionBoundaryBeforeUnmapping(0x1000, 0x1000, stalled));
}
