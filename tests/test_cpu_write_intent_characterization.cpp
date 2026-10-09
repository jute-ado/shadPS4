// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>
#include <gtest/gtest.h>

#include "video_core/buffer_cache/cpu_page_write_snapshot.h"

namespace {

using Page = std::array<u8, VideoCore::TRACKER_BYTES_PER_PAGE>;
constexpr VAddr PageAddress = 0x4000;
constexpr size_t PointerOffset = 0x70;
constexpr size_t GpuNeighborOffset = 0x90;
constexpr u64 BeforePointer = 0x0000000200004000;
constexpr u64 GpuPayload = 0x3f80000042cda6e0;
constexpr u64 GpuNeighbor = 0x0123456789abcdef;

void Store(Page& page, size_t offset, u64 value) {
    std::memcpy(page.data() + offset, &value, sizeof(value));
}

u64 Load(const Page& page, size_t offset) {
    u64 value;
    std::memcpy(&value, page.data() + offset, sizeof(value));
    return value;
}

struct Capture {
    VideoCore::CpuPageWriteTracker tracker;
    Page cpu{};
    Page composed{};

    explicit Capture(size_t first_write_offset = 0x20, size_t first_write_size = 8) {
        Store(cpu, PointerOffset, BeforePointer);
        EXPECT_TRUE(tracker.Capture(PageAddress, cpu, first_write_offset, first_write_size));
        Store(composed, PointerOffset, GpuPayload);
        Store(composed, GpuNeighborOffset, GpuNeighbor);
    }

    void Reconcile() {
        ASSERT_TRUE(tracker.ApplyPendingWrites(PageAddress, cpu, PageAddress, composed));
        EXPECT_EQ(Load(composed, GpuNeighborOffset), GpuNeighbor);
    }
};

// Intentionally RED characterizations: the tracker sees a first page fault but receives no
// notification for later stores after that page is writable. They must not be claimed GREEN by
// merely preferring all CPU words, which would break the untouched-GPU-neighbour controls.
TEST(CpuWriteIntentCharacterization, LaterSameValuePointerStoreRequiresUnrecordedIntent) {
    Capture capture;
    Store(capture.cpu, PointerOffset, BeforePointer); // A real same-value 64-bit CPU store.
    capture.Reconcile();
    EXPECT_EQ(Load(capture.composed, PointerOffset), BeforePointer);
}

TEST(CpuWriteIntentCharacterization, LaterPointerStoreWithUnchangedHighHalfIsOneCpuWrite) {
    Capture capture;
    constexpr u64 NewPointer = 0x0000000200008000;
    Store(capture.cpu, PointerOffset, NewPointer); // Low word changes, high word remains 2.
    capture.Reconcile();
    EXPECT_EQ(Load(capture.composed, PointerOffset), NewPointer);
}

TEST(CpuWriteIntentCharacterization, ExplicitPointerWritePreservesBothHalvesAndGpuNeighbor) {
    Capture capture{PointerOffset, sizeof(u64)};
    Store(capture.cpu, PointerOffset, BeforePointer);
    capture.Reconcile();
    EXPECT_EQ(Load(capture.composed, PointerOffset), BeforePointer);
}

TEST(CpuWriteIntentCharacterization, BothChangedPointerWordsAreRecoveredFromBeforeImage) {
    Capture capture;
    constexpr u64 NewPointer = 0x0000000300008000;
    Store(capture.cpu, PointerOffset, NewPointer);
    capture.Reconcile();
    EXPECT_EQ(Load(capture.composed, PointerOffset), NewPointer);
}

TEST(CpuWriteIntentCharacterization, UntouchedGpuPayloadMustNotBeReplacedByCpuBacking) {
    Capture capture;
    capture.Reconcile();
    EXPECT_EQ(Load(capture.composed, PointerOffset), GpuPayload);
}

} // namespace
