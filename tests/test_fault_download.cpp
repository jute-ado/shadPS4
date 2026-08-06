// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/buffer_cache/dma_publication_gate.h"
#include "video_core/buffer_cache/fault_download.h"

namespace VideoCore {
namespace {

TEST(FaultDownload, ReservesSlotZeroForTheReportedCount) {
    const auto count = BoundFaultDownloadCount(FaultDownloadSlotCount - 1, FaultDownloadSlotCount);

    EXPECT_EQ(count.address_count, FaultDownloadSlotCount - 1);
    EXPECT_FALSE(count.overflowed);
}

TEST(FaultDownload, CapsUntrustedCountAtTheDownloadedAddressSlots) {
    const auto count = BoundFaultDownloadCount(FaultDownloadSlotCount + 37, FaultDownloadSlotCount);

    EXPECT_EQ(count.address_count, FaultDownloadSlotCount - 1);
    EXPECT_TRUE(count.overflowed);
}

TEST(FaultDownload, ReportsOverflowWhenThereAreNoAddressSlots) {
    const auto count = BoundFaultDownloadCount(1, 1);

    EXPECT_EQ(count.address_count, 0);
    EXPECT_TRUE(count.overflowed);
}

TEST(FaultDownload, KeepsAnEmptyDownloadClean) {
    const auto count = BoundFaultDownloadCount(0, FaultDownloadSlotCount);

    EXPECT_EQ(count.address_count, 0);
    EXPECT_FALSE(count.overflowed);
}

TEST(FaultDownload, ClassifiesCleanFaultedInvalidAndOverflowEpochs) {
    const auto clean = ClassifyDmaFaultEpoch(BoundFaultDownloadCount(0, 1024), 0);
    EXPECT_EQ(clean.GetStatus(), DmaFaultEpoch::Status::Clean);

    const auto faulted = ClassifyDmaFaultEpoch(BoundFaultDownloadCount(2, 1024), 0);
    EXPECT_EQ(faulted.GetStatus(), DmaFaultEpoch::Status::Faulted);
    EXPECT_EQ(faulted.FaultCount(), 2);

    const auto invalid = ClassifyDmaFaultEpoch(BoundFaultDownloadCount(2, 1024), 1);
    EXPECT_EQ(invalid.GetStatus(), DmaFaultEpoch::Status::Invalid);

    const auto overflow = ClassifyDmaFaultEpoch(BoundFaultDownloadCount(1061, 1024), 1);
    EXPECT_EQ(overflow.GetStatus(), DmaFaultEpoch::Status::Overflow);
}

TEST(FaultDownload, MissingOrResetCompletionFailsClosed) {
    DmaFaultEpochCompletion completion;

    EXPECT_FALSE(completion.IsComplete());
    EXPECT_EQ(completion.ValueOrInvalid().GetStatus(), DmaFaultEpoch::Status::Invalid);

    completion.Complete(DmaFaultEpoch::Clean());
    EXPECT_TRUE(completion.IsComplete());
    EXPECT_EQ(completion.ValueOrInvalid().GetStatus(), DmaFaultEpoch::Status::Clean);

    completion.Reset();
    EXPECT_FALSE(completion.IsComplete());
    EXPECT_EQ(completion.ValueOrInvalid().GetStatus(), DmaFaultEpoch::Status::Invalid);
}

} // namespace
} // namespace VideoCore
