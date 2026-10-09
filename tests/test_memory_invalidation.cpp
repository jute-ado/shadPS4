// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include "video_core/buffer_cache/buffer_registration_index.h"
#include "video_core/buffer_cache/cpu_page_write_snapshot.h"
#include "video_core/buffer_cache/memory_invalidation.h"
#include "video_core/buffer_cache/region_manager.h"

namespace {
using VideoCore::CpuPageUploadRange;
using VideoCore::CpuPageWriteTracker;
using VideoCore::InvalidateTrackedRegion;
using VideoCore::MemoryInvalidationReason;
using VideoCore::PageManager;
using VideoCore::RegionManager;
using VideoCore::TRACKER_BYTES_PER_PAGE;
using VideoCore::Type;

struct WatchChange {
    bool watch;
    VAddr address;
};
std::vector<WatchChange> watch_changes;

class TrackedPage {
public:
    explicit TrackedPage(VAddr address) : region{&watcher, address} {
        region.ChangeRegionState<Type::CPU, false>(address, TRACKER_BYTES_PER_PAGE);
        region.GetRegionBits<Type::GPU>().SetRange(0, 1);
        watch_changes.clear();
    }
    PageManager watcher{nullptr};
    RegionManager region;
};

std::string ReadSource(const char* path) {
    std::ifstream input{path};
    std::ostringstream text;
    text << input.rdbuf();
    return text.str();
}
} // namespace

namespace VideoCore {
// Only the OS watcher sink is replaced. Region state, invalidation decision,
// before-images, forced-word tracking and their locks are production classes.
struct PageManager::Impl {};
PageManager::PageManager(Vulkan::Rasterizer*) : impl{std::make_unique<Impl>()} {}
PageManager::~PageManager() = default;
template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(VAddr address, RegionBits&) const {
    static_assert(!is_read);
    watch_changes.push_back({track, address});
}
template void PageManager::UpdatePageWatchersForRegion<true, false>(VAddr, RegionBits&) const;
template void PageManager::UpdatePageWatchersForRegion<false, false>(VAddr, RegionBits&) const;
} // namespace VideoCore

TEST(MemoryInvalidation, MappingChangeNeverReadsReservedGpuDirtyPage) {
    // Stale registration/GPU ownership is intentionally present. This is not an
    // unregistered-page shortcut. The OS must reject any attempted before-image read.
    ASSERT_EXIT(([] {
#ifdef _WIN32
                    void* reserved =
                        VirtualAlloc(nullptr, TRACKER_BYTES_PER_PAGE, MEM_RESERVE, PAGE_NOACCESS);
#else
                    void* reserved = mmap(nullptr, TRACKER_BYTES_PER_PAGE, PROT_NONE,
                                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                    if (reserved == MAP_FAILED) {
                        std::exit(2);
                    }
#endif
                    if (reserved == nullptr) {
                        std::exit(2);
                    }
                    const VAddr address = reinterpret_cast<VAddr>(reserved);
                    VideoCore::BufferRegistrationIndex<unsigned> registration;
                    registration.Add(address, TRACKER_BYTES_PER_PAGE, 1);
                    if (!registration.Intersects(address, TRACKER_BYTES_PER_PAGE)) {
                        std::exit(2);
                    }
                    TrackedPage tracked{address};
                    CpuPageWriteTracker snapshots;
                    int captures = 0;
                    const bool flush = InvalidateTrackedRegion(
                        tracked.region, 0, TRACKER_BYTES_PER_PAGE, false,
                        MemoryInvalidationReason::MappingChange,
                        [&](VAddr page, size_t offset, size_t size) {
                            ++captures;
                            const auto bytes = std::span<const u8, TRACKER_BYTES_PER_PAGE>{
                                reinterpret_cast<const u8*>(page), TRACKER_BYTES_PER_PAGE};
                            return snapshots.Capture(page, bytes, offset, size);
                        });
                    const bool correct =
                        !flush && captures == 0 &&
                        tracked.region.IsRegionModified<Type::CPU>(0, TRACKER_BYTES_PER_PAGE) &&
                        tracked.region.IsRegionModified<Type::GPU>(0, TRACKER_BYTES_PER_PAGE);
#ifdef _WIN32
                    VirtualFree(reserved, 0, MEM_RELEASE);
#else
                    munmap(reserved, TRACKER_BYTES_PER_PAGE);
#endif
                    std::exit(correct ? 0 : 3);
                }()),
                ::testing::ExitedWithCode(0), "");
}

TEST(MemoryInvalidation, CpuWritesStillPreserveFirstBeforeImageAndForcedWords) {
    constexpr VAddr Address = 0x400000;
    TrackedPage tracked{Address};
    CpuPageWriteTracker snapshots;
    std::array<u8, TRACKER_BYTES_PER_PAGE> page{};
    int captures = 0;
    const auto preserve = [&](VAddr address, size_t offset, size_t size) {
        ++captures;
        return snapshots.Capture(address, page, offset, size);
    };
    EXPECT_FALSE(InvalidateTrackedRegion(tracked.region, 0x100, 1, false,
                                         MemoryInvalidationReason::CpuWrite, preserve));
    page[0x108] = 9;
    EXPECT_FALSE(InvalidateTrackedRegion(tracked.region, 0x200, 1, false,
                                         MemoryInvalidationReason::CpuWrite, preserve));
    std::vector<CpuPageUploadRange> uploads;
    ASSERT_TRUE(snapshots.Consume(Address, page, 0, page.size(),
                                  [&](auto range) { uploads.push_back(range); }));
    EXPECT_EQ(captures, 2);
    EXPECT_EQ(uploads, (std::vector{CpuPageUploadRange{0x100, 4}, CpuPageUploadRange{0x108, 4},
                                    CpuPageUploadRange{0x200, 4}}));
    EXPECT_TRUE(tracked.region.IsRegionModified<Type::GPU>(0, TRACKER_BYTES_PER_PAGE));
}

TEST(MemoryInvalidation, MappingChangeKeepsConfiguredReadbackAndReleasesRegionLock) {
    constexpr VAddr Address = 0x400000;
    for (const auto reason :
         {MemoryInvalidationReason::CpuWrite, MemoryInvalidationReason::MappingChange}) {
        TrackedPage tracked{Address};
        int captures = 0;
        EXPECT_TRUE(InvalidateTrackedRegion(tracked.region, 0, TRACKER_BYTES_PER_PAGE, true, reason,
                                            [&](VAddr, size_t, size_t) {
                                                ++captures;
                                                return true;
                                            }));
        EXPECT_EQ(captures, 0);
        EXPECT_FALSE(tracked.region.IsRegionModified<Type::CPU>(0, TRACKER_BYTES_PER_PAGE));
        EXPECT_TRUE(tracked.region.IsRegionModified<Type::GPU>(0, TRACKER_BYTES_PER_PAGE));
        const bool unlocked = tracked.region.lock.try_lock();
        EXPECT_TRUE(unlocked); // The caller performs the existing readback outside this lock.
        if (unlocked) {
            tracked.region.lock.unlock();
        }
    }
}

TEST(MemoryInvalidation, CpuWriteCaptureRefusalStillUsesExistingReadbackFallback) {
    TrackedPage tracked{0x400000};
    int captures = 0;
    EXPECT_TRUE(InvalidateTrackedRegion(tracked.region, 0, TRACKER_BYTES_PER_PAGE, false,
                                        MemoryInvalidationReason::CpuWrite,
                                        [&](VAddr, size_t, size_t) {
                                            ++captures;
                                            return false;
                                        }));
    EXPECT_EQ(captures, 1);
    EXPECT_FALSE(tracked.region.IsRegionModified<Type::CPU>(0, TRACKER_BYTES_PER_PAGE));
    EXPECT_TRUE(tracked.region.IsRegionModified<Type::GPU>(0, TRACKER_BYTES_PER_PAGE));
}

TEST(MemoryInvalidation, NonGpuDirtyMappingDoesNotReadOrFlush) {
    TrackedPage tracked{0x400000};
    tracked.region.GetRegionBits<Type::GPU>().Clear();
    int captures = 0;
    EXPECT_FALSE(InvalidateTrackedRegion(tracked.region, 0, TRACKER_BYTES_PER_PAGE, true,
                                         MemoryInvalidationReason::MappingChange,
                                         [&](VAddr, size_t, size_t) {
                                             ++captures;
                                             return false;
                                         }));
    EXPECT_EQ(captures, 0);
    EXPECT_TRUE(tracked.region.IsRegionModified<Type::CPU>(0, TRACKER_BYTES_PER_PAGE));
    EXPECT_FALSE(tracked.region.IsRegionModified<Type::GPU>(0, TRACKER_BYTES_PER_PAGE));
}

TEST(MemoryInvalidation, WholePageUnmapRetainsExistingDisabledReadbackUploadPolicy) {
    constexpr VAddr Address = 0x400000;
    std::array<u8, TRACKER_BYTES_PER_PAGE> backing{};
    backing.fill(0x11);
    CpuPageWriteTracker old_policy;
    // The old unmap called CPU preservation with a whole-page write. That forced
    // every word into the next upload, even when the physical allocation survived.
    ASSERT_TRUE(old_policy.Capture(Address, backing, 0x100, 1));
    ASSERT_TRUE(old_policy.Capture(Address, backing, 0, backing.size()));
    std::vector<CpuPageUploadRange> old_uploads;
    ASSERT_TRUE(old_policy.Consume(Address, backing, 0, backing.size(),
                                   [&](auto range) { old_uploads.push_back(range); }));
    CpuPageWriteTracker new_policy;
    ASSERT_TRUE(new_policy.Capture(Address, backing, 0x100, 1));
    new_policy.Discard(Address, backing.size());
    std::vector<CpuPageUploadRange> new_uploads;
    if (!new_policy.Consume(Address, backing, 0, backing.size(),
                            [&](auto range) { new_uploads.push_back(range); })) {
        // Actual SynchronizeBuffer fallback when no before-image remains.
        new_uploads.push_back({0, backing.size()});
    }
    EXPECT_EQ(old_uploads, new_uploads);
    EXPECT_EQ(new_uploads, (std::vector{CpuPageUploadRange{0, backing.size()}}));
}

TEST(MemoryInvalidation, NewBackingAtSameVaDoesNotInheritOldWritesOrOtherVaSnapshots) {
    constexpr VAddr Address = 0x400000;
    std::array<u8, TRACKER_BYTES_PER_PAGE> old_backing{};
    old_backing.fill(0x11);
    std::array<u8, TRACKER_BYTES_PER_PAGE> replacement{};
    replacement.fill(0x88);
    CpuPageWriteTracker snapshots;
    ASSERT_TRUE(snapshots.Capture(Address, old_backing, 0x100, 1));
    ASSERT_TRUE(snapshots.Capture(Address + TRACKER_BYTES_PER_PAGE, old_backing, 0x200, 1));
    // Production MappingChange discards this VA range before the registration query.
    snapshots.Discard(Address, TRACKER_BYTES_PER_PAGE);
    EXPECT_FALSE(snapshots.Consume(Address, replacement, 0, replacement.size(), [](auto) {}));
    ASSERT_TRUE(snapshots.Capture(Address, replacement, 0x300, 1));
    std::vector<CpuPageUploadRange> new_writes;
    ASSERT_TRUE(snapshots.Consume(Address, replacement, 0, replacement.size(),
                                  [&](auto range) { new_writes.push_back(range); }));
    EXPECT_EQ(new_writes, (std::vector{CpuPageUploadRange{0x300, 4}}));
    std::vector<CpuPageUploadRange> unaffected;
    ASSERT_TRUE(snapshots.Consume(Address + TRACKER_BYTES_PER_PAGE, old_backing, 0,
                                  old_backing.size(),
                                  [&](auto range) { unaffected.push_back(range); }));
    EXPECT_EQ(unaffected, (std::vector{CpuPageUploadRange{0x200, 4}}));
}

TEST(MemoryInvalidation, UnmapReasonAndSnapshotDiscardAreWiredWithoutChangingCpuFaultPath) {
    const auto header = ReadSource(SHADPS4_INVALIDATION_BUFFER_HEADER_PATH);
    const auto declaration = header.find("void InvalidateMemory(");
    ASSERT_NE(declaration, std::string::npos);
    EXPECT_NE(header.substr(declaration, header.find(';', declaration) - declaration)
                  .find("reason = MemoryInvalidationReason::CpuWrite"),
              std::string::npos);
    const auto source = ReadSource(SHADPS4_INVALIDATION_BUFFER_SOURCE_PATH);
    const auto begin = source.find("void BufferCache::InvalidateMemory(");
    const auto end = source.find("void BufferCache::ReadMemory(", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    const auto body = source.substr(begin, end - begin);
    const auto discard = body.find("cpu_page_write_tracker.Discard(device_addr, size)");
    const auto registered = body.find("IsRegionRegistered(device_addr, size)");
    EXPECT_NE(body.find("MemoryInvalidationReason::MappingChange"), std::string::npos);
    ASSERT_NE(discard, std::string::npos);
    ASSERT_NE(registered, std::string::npos);
    EXPECT_LT(discard, registered); // Old before-images must not survive cache unregister/remap.
    const auto rasterizer = ReadSource(SHADPS4_INVALIDATION_RASTERIZER_SOURCE_PATH);
    const auto cpu_begin = rasterizer.find("bool Rasterizer::InvalidateMemory(");
    const auto cpu_end = rasterizer.find("bool Rasterizer::ReadMemory(", cpu_begin);
    ASSERT_NE(cpu_begin, std::string::npos);
    ASSERT_NE(cpu_end, std::string::npos);
    EXPECT_NE(rasterizer.substr(cpu_begin, cpu_end - cpu_begin)
                  .find("buffer_cache.InvalidateMemory(addr, size)"),
              std::string::npos);
    const auto unmap_begin = rasterizer.find("void Rasterizer::UnmapMemory(");
    const auto unmap_end = rasterizer.find("void Rasterizer::UpdateDynamicState(", unmap_begin);
    ASSERT_NE(unmap_begin, std::string::npos);
    ASSERT_NE(unmap_end, std::string::npos);
    EXPECT_NE(rasterizer.substr(unmap_begin, unmap_end - unmap_begin)
                  .find("MemoryInvalidationReason::MappingChange"),
              std::string::npos);
}
