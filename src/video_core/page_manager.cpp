// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <utility>
#include <boost/container/small_vector.hpp>
#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/range_lock.h"
#include "common/signal_context.h"
#include "core/memory.h"
#include "core/signals.h"
#include "video_core/mapped_page_protection.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/srt_snapshot_epoch_coordinator.h"

#ifndef _WIN64
#include <sys/mman.h>
#include "common/adaptive_mutex.h"
#ifdef ENABLE_USERFAULTFD
#include <thread>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <poll.h>
#include <sys/ioctl.h>
#include "common/error.h"
#endif
#else
#include <windows.h>
#include "common/spin_lock.h"
#endif

#ifdef __linux__
#include "common/adaptive_mutex.h"
#else
#include "common/spin_lock.h"
#endif

namespace VideoCore {

constexpr size_t PM_PAGE_SIZE = 4_KB;
constexpr size_t PM_PAGE_BITS = 12;

struct PageManager::Impl {
    struct PageState {
        u8 num_write_watchers : 7;
        // At the moment only buffer cache can request read watchers.
        // And buffers cannot overlap, thus only 1 can exist per page.
        u8 num_read_watchers : 1;

        Core::MemoryPermission WritePerm() const noexcept {
            return num_write_watchers == 0 ? Core::MemoryPermission::Write
                                           : Core::MemoryPermission::None;
        }

        Core::MemoryPermission ReadPerm() const noexcept {
            return num_read_watchers == 0 ? Core::MemoryPermission::Read
                                          : Core::MemoryPermission::None;
        }

        Core::MemoryPermission Perms() const noexcept {
            return ReadPerm() | WritePerm();
        }

        template <s32 delta, bool is_read>
        u8 AddDelta() {
            if constexpr (is_read) {
                if constexpr (delta == 1) {
                    return ++num_read_watchers;
                } else if (delta == -1) {
                    ASSERT_MSG(num_read_watchers > 0, "Not enough watchers");
                    return --num_read_watchers;
                } else {
                    return num_read_watchers;
                }
            } else {
                if constexpr (delta == 1) {
                    return ++num_write_watchers;
                } else if (delta == -1) {
                    ASSERT_MSG(num_write_watchers > 0, "Not enough watchers");
                    return --num_write_watchers;
                } else {
                    return num_write_watchers;
                }
            }
        }
    };

    static constexpr size_t ADDRESS_BITS = 40;
    static constexpr size_t NUM_ADDRESS_PAGES = 1ULL << (40 - PM_PAGE_BITS);
    static constexpr size_t NUM_ADDRESS_LOCKS = NUM_ADDRESS_PAGES / PAGES_PER_LOCK;
    inline static Vulkan::Rasterizer* rasterizer;
#ifdef ENABLE_USERFAULTFD
    Impl(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
        uffd = syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
        ASSERT_MSG(uffd != -1, "{}", Common::GetLastErrorMsg());

        // Request uffdio features from kernel.
        uffdio_api api;
        api.api = UFFD_API;
        api.features = UFFD_FEATURE_THREAD_ID;
        const int ret = ioctl(uffd, UFFDIO_API, &api);
        ASSERT(ret == 0 && api.api == UFFD_API);

        // Create uffd handler thread
        ufd_thread = std::jthread([&](std::stop_token token) { UffdHandler(token); });
    }

    void RegisterRange(VAddr address, size_t size) {
        uffdio_register reg;
        reg.range.start = address;
        reg.range.len = size;
        reg.mode = UFFDIO_REGISTER_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_REGISTER, &reg);
        ASSERT_MSG(ret != -1, "Uffdio register failed");
    }

    void UnregisterRange(VAddr address, size_t size) {
        uffdio_range range;
        range.start = address;
        range.len = size;
        const int ret = ioctl(uffd, UFFDIO_UNREGISTER, &range);
        ASSERT_MSG(ret != -1, "Uffdio unregister failed");
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) {
        bool allow_write = True(perms & Core::MemoryPermission::Write);
        uffdio_writeprotect wp;
        wp.range.start = address;
        wp.range.len = size;
        wp.mode = allow_write ? 0 : UFFDIO_WRITEPROTECT_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_WRITEPROTECT, &wp);
        ASSERT_MSG(ret != -1, "Uffdio writeprotect failed with error: {}",
                   Common::GetLastErrorMsg());
    }

    void UffdHandler(std::stop_token token) {
        while (!token.stop_requested()) {
            pollfd pollfd;
            pollfd.fd = uffd;
            pollfd.events = POLLIN;

            // Block until the descriptor is ready for data reads.
            const int pollres = poll(&pollfd, 1, -1);
            switch (pollres) {
            case -1:
                perror("Poll userfaultfd");
                continue;
                break;
            case 0:
                continue;
            case 1:
                break;
            default:
                UNREACHABLE_MSG("Unexpected number of descriptors {} out of poll", pollres);
            }

            // We don't want an error condition to have occured.
            ASSERT_MSG(!(pollfd.revents & POLLERR), "POLLERR on userfaultfd");

            // We waited until there is data to read, we don't care about anything else.
            if (!(pollfd.revents & POLLIN)) {
                continue;
            }

            // Read message from kernel.
            uffd_msg msg;
            const int readret = read(uffd, &msg, sizeof(msg));
            ASSERT_MSG(readret != -1 || errno == EAGAIN, "Unexpected result of uffd read");
            if (errno == EAGAIN) {
                continue;
            }
            ASSERT_MSG(readret == sizeof(msg), "Unexpected short read, exiting");
            ASSERT(msg.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WP);

            // Notify rasterizer about the fault.
            const VAddr addr = msg.arg.pagefault.address;
            rasterizer->InvalidateMemory(addr, 1);
        }
    }

    std::jthread ufd_thread;
    int uffd;
#else
    Impl(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;

        // Should be called first.
        constexpr auto priority = std::numeric_limits<u32>::min();
        Core::Signals::Instance()->RegisterAccessViolationHandler(GuestFaultSignalHandler,
                                                                  priority);
    }

    void RegisterRange(VAddr address, size_t size) {
        // No-op
    }

    void UnregisterRange(VAddr address, size_t size) {
        // No-op
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) {
        RENDERER_TRACE;
        auto* memory = Core::Memory::Instance();
        auto& impl = memory->GetAddressSpace();
        ASSERT_MSG(perms != Core::MemoryPermission::Write,
                   "Attempted to protect region as write-only which is not a valid permission");
        impl.ProtectForTracking(address, size, perms);
    }

    static bool GuestFaultSignalHandler(void* context, void* fault_address) {
        const auto addr = reinterpret_cast<VAddr>(fault_address);
        if (Common::IsWriteError(context)) {
            return rasterizer->InvalidateMemory(addr, 8);
        } else {
            return rasterizer->ReadMemory(addr, 8);
        }
        return false;
    }
#endif

    void ProtectMapped(VAddr address, u64 size, Core::MemoryPermission perms) {
        (void)protectable_mappings.ForEachMappedSubrange(
            address, size, [this, perms](VAddr mapped_address, u64 mapped_size) {
                Protect(mapped_address, mapped_size, perms);
            });
    }

    void ReplayPageWatchers(VAddr address, u64 size) {
        const bool replayed = ReplayActivePageProtectionRuns(
            address, size, PM_PAGE_SIZE,
            [this](VAddr page_address) -> std::optional<Core::MemoryPermission> {
                const auto perms = cached_pages[page_address >> PM_PAGE_BITS].Perms();
                if (perms == Core::MemoryPermission::ReadWrite) {
                    return std::nullopt;
                }
                return perms;
            },
            [this](VAddr run_address, u64 run_size, Core::MemoryPermission perms) {
                Protect(run_address, run_size, perms);
            });
        ASSERT_MSG(replayed, "Invalid GPU mapping range {:#x} - {:#x}", address, address + size);
    }

    void OnMap(VAddr address, size_t size, GpuMappingKind kind) {
        ASSERT_MSG(Detail::IsValidAddressRange(address, size),
                   "Invalid GPU mapping range {:#x} size {:#x}", address, size);
        std::scoped_lock lifecycle_lock{mapping_lifecycle_mutex};

        if (kind == GpuMappingKind::HostMapped) {
            ASSERT_MSG(address % PM_PAGE_SIZE == 0 && size % PM_PAGE_SIZE == 0,
                       "Host GPU mapping range must be page aligned");
            ASSERT_MSG(Common::DivCeil(address + size, PM_PAGE_SIZE) <= NUM_ADDRESS_PAGES,
                       "Host GPU mapping exceeds tracked address space");
            constexpr u64 LockBytes = PAGES_PER_LOCK * PM_PAGE_SIZE;
            const bool chunked = ForEachAddressRangeChunk(
                address, size, LockBytes, [this](VAddr chunk_address, u64 chunk_size) {
                    const size_t lock_index = (chunk_address >> PM_PAGE_BITS) / PAGES_PER_LOCK;
                    std::scoped_lock page_lock{locks[lock_index]};
                    const bool published = protectable_mappings.PublishMappedRange(
                        chunk_address, chunk_size,
                        [this](VAddr mapped_address, u64 mapped_size) {
                            RegisterRange(mapped_address, mapped_size);
                        },
                        [this](VAddr mapped_address, u64 mapped_size) {
                            ReplayPageWatchers(mapped_address, mapped_size);
                        });
                    ASSERT(published);
                });
            ASSERT(chunked);
        }

        const bool published = kind == GpuMappingKind::HostMapped
                                   ? logical_mappings.PublishHostMappedRange(address, size)
                                   : logical_mappings.PublishSparsePlaceholderRange(address, size);
        ASSERT(published);
    }

    void OnUnmap(VAddr address, size_t size, GpuMappingKind kind) {
        ASSERT_MSG(Detail::IsValidAddressRange(address, size),
                   "Invalid GPU unmapping range {:#x} size {:#x}", address, size);
        std::scoped_lock lifecycle_lock{mapping_lifecycle_mutex};

        if (kind == GpuMappingKind::HostMapped) {
            ASSERT_MSG(address % PM_PAGE_SIZE == 0 && size % PM_PAGE_SIZE == 0,
                       "Host GPU unmapping range must be page aligned");
            ASSERT_MSG(Common::DivCeil(address + size, PM_PAGE_SIZE) <= NUM_ADDRESS_PAGES,
                       "Host GPU unmapping exceeds tracked address space");
            constexpr u64 LockBytes = PAGES_PER_LOCK * PM_PAGE_SIZE;
            const bool chunked = ForEachAddressRangeChunk(
                address, size, LockBytes, [this](VAddr chunk_address, u64 chunk_size) {
                    const size_t lock_index = (chunk_address >> PM_PAGE_BITS) / PAGES_PER_LOCK;
                    std::scoped_lock page_lock{locks[lock_index]};
                    const bool retired = protectable_mappings.RetireMappedRange(
                        chunk_address, chunk_size, [this](VAddr mapped_address, u64 mapped_size) {
                            UnregisterRange(mapped_address, mapped_size);
                        });
                    ASSERT(retired);
                });
            ASSERT(chunked);

            const bool logically_retired = logical_mappings.RetireHostMappedRange(address, size);
            ASSERT(logically_retired);
        } else {
            const bool logically_retired =
                logical_mappings.RetireSparsePlaceholderRange(address, size);
            ASSERT(logically_retired);
        }
    }

    template <bool track, bool is_read>
    void UpdatePageWatchers(VAddr addr, u64 size) {
        RENDERER_TRACE;

        size_t page = addr >> PM_PAGE_BITS;
        const u64 page_end = Common::DivCeil(addr + size, PM_PAGE_SIZE);

        // Acquire locks for the range of pages
        const auto lock_start = locks.begin() + (page / PAGES_PER_LOCK);
        const auto lock_end = locks.begin() + Common::DivCeil(page_end, PAGES_PER_LOCK);
        Common::RangeLockGuard lk(lock_start, lock_end);

        auto perms = cached_pages[page].Perms();
        u64 range_begin = 0;
        u64 range_bytes = 0;
        u64 potential_range_bytes = 0;

        const auto release_pending = [&] {
            if (range_bytes > 0) {
                RENDERER_TRACE;
                // Perform pending (un)protect action
                ProtectMapped(range_begin << PM_PAGE_BITS, range_bytes, perms);
                range_bytes = 0;
                potential_range_bytes = 0;
            }
        };

        // Iterate requested pages
        const u64 aligned_addr = page << PM_PAGE_BITS;
        const u64 aligned_end = page_end << PM_PAGE_BITS;
        if (!logical_mappings.Contains(aligned_addr, aligned_end - aligned_addr)) {
            LOG_WARNING(Render,
                        "Tracking memory region {:#x} - {:#x} which is not fully GPU mapped.",
                        aligned_addr, aligned_end);
        }

        for (; page != page_end; ++page) {
            PageState& state = cached_pages[page];

            // Apply the change to the page state
            const u8 new_count = state.AddDelta<track ? 1 : -1, is_read>();

            if (auto new_perms = state.Perms(); new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_bytes != 0) {
                // If the protection did not change, extend the potential range
                potential_range_bytes += PM_PAGE_SIZE;
            }

            // Only start a new range if the page must be (un)protected
            if ((new_count == 0 && !track) || (new_count == 1 && track)) {
                if (range_bytes == 0) {
                    // Start a new potential range
                    range_begin = page;
                    potential_range_bytes = PM_PAGE_SIZE;
                }
                // Extend current range up to potential range
                range_bytes = potential_range_bytes;
            }
        }

        // Add pending (un)protect action
        release_pending();
    }

    template <bool track, bool is_read>
    void UpdatePageWatchersForRegion(VAddr base_addr, RegionBits& mask) {
        RENDERER_TRACE;
        auto start_range = mask.FirstRange();
        auto end_range = mask.LastRange();

        if (start_range.second == end_range.second) {
            // if all pages are contiguous, use the regular UpdatePageWatchers
            const VAddr start_addr = base_addr + (start_range.first << PM_PAGE_BITS);
            const u64 size = (start_range.second - start_range.first) << PM_PAGE_BITS;
            return UpdatePageWatchers<track, is_read>(start_addr, size);
        }

        size_t base_page = (base_addr >> PM_PAGE_BITS);
        ASSERT(base_page % PAGES_PER_LOCK == 0);
        std::scoped_lock lk(locks[base_page / PAGES_PER_LOCK]);
        auto perms = cached_pages[base_page + start_range.first].Perms();
        u64 range_begin = 0;
        u64 range_bytes = 0;
        u64 potential_range_bytes = 0;

        const auto release_pending = [&] {
            if (range_bytes > 0) {
                RENDERER_TRACE;
                // Perform pending (un)protect action
                ProtectMapped(range_begin << PM_PAGE_BITS, range_bytes, perms);
                range_bytes = 0;
                potential_range_bytes = 0;
            }
        };

        // Iterate pages
        for (size_t page = start_range.first; page < end_range.second; ++page) {
            PageState& state = cached_pages[base_page + page];
            const bool update = mask.Get(page);

            // Apply the change to the page state
            const u8 new_count =
                update ? state.AddDelta<track ? 1 : -1, is_read>() : state.AddDelta<0, is_read>();

            if (auto new_perms = state.Perms(); new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_bytes != 0) {
                // If the protection did not change, extend the potential range
                potential_range_bytes += PM_PAGE_SIZE;
            }

            // If the page is not being updated, skip it
            if (!update) {
                continue;
            }

            // If the page must be (un)protected
            if ((new_count == 0 && !track) || (new_count == 1 && track)) {
                if (range_bytes == 0) {
                    // Start a new potential range
                    range_begin = base_page + page;
                    potential_range_bytes = PM_PAGE_SIZE;
                }
                // Extend current rango up to potential range
                range_bytes = potential_range_bytes;
            }
        }

        // Add pending (un)protect action
        release_pending();
    }

    [[nodiscard]] u64 BeginSnapshotEpoch() {
        return snapshot_coordinator.BeginEpoch();
    }

    [[nodiscard]] bool BeginSnapshotPage(u64 id, VAddr address, u64 size) {
        return snapshot_coordinator.BeginPageCopy(
            id, address, size, [this](VAddr watcher_address, u64 watcher_size) {
                UpdatePageWatchers<true, false>(watcher_address, watcher_size);
            });
    }

    [[nodiscard]] bool EndSnapshotPage(u64 id, VAddr address, u64 size, bool copied) {
        return snapshot_coordinator.EndPageCopy(id, address, size, copied);
    }

    [[nodiscard]] bool IsSnapshotEpochValid(u64 id) const {
        return snapshot_coordinator.IsEpochValid(id);
    }

    [[nodiscard]] bool SealSnapshotEpoch(u64 id) {
        return snapshot_coordinator.SealEpoch(id, [this](VAddr address, u64 size) {
            UpdatePageWatchers<false, false>(address, size);
        });
    }

    void InvalidateSnapshotRange(VAddr address, u64 size) {
        snapshot_coordinator.InvalidateRangeAndWait(
            address, size, [this](VAddr watcher_address, u64 watcher_size) {
                UpdatePageWatchers<false, false>(watcher_address, watcher_size);
            });
    }

    std::array<PageState, NUM_ADDRESS_PAGES> cached_pages{};
#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
    using LockType = Common::AdaptiveMutex;
#else
    using LockType = Common::SpinLock;
#endif
    std::array<LockType, NUM_ADDRESS_LOCKS> locks{};
    LogicalGpuMappingState logical_mappings;
    MappedPageProtectionState protectable_mappings;
    std::mutex mapping_lifecycle_mutex;
    SrtSnapshotEpochCoordinator snapshot_coordinator;
};

PageManager::PageManager(Vulkan::Rasterizer* rasterizer_)
    : impl{std::make_unique<Impl>(rasterizer_)} {}

PageManager::~PageManager() = default;

PageManager::SnapshotEpoch::SnapshotEpoch(SnapshotEpoch&& other) noexcept
    : manager{std::exchange(other.manager, nullptr)}, id{std::exchange(other.id, 0)},
      result{other.result} {}

PageManager::SnapshotEpoch& PageManager::SnapshotEpoch::operator=(SnapshotEpoch&& other) noexcept {
    if (this != &other) {
        (void)Seal();
        manager = std::exchange(other.manager, nullptr);
        id = std::exchange(other.id, 0);
        result = other.result;
    }
    return *this;
}

PageManager::SnapshotEpoch::~SnapshotEpoch() {
    (void)Seal();
}

bool PageManager::SnapshotEpoch::BeginPageCopy(VAddr address, u64 size) {
    return manager && manager->BeginSnapshotPage(id, address, size);
}

bool PageManager::SnapshotEpoch::EndPageCopy(VAddr address, u64 size, bool copied) {
    return manager && manager->EndSnapshotPage(id, address, size, copied);
}

bool PageManager::SnapshotEpoch::Valid() const {
    return manager ? manager->IsSnapshotEpochValid(id) : result;
}

bool PageManager::SnapshotEpoch::Seal() {
    if (manager) {
        result = manager->SealSnapshotEpoch(id);
        manager = nullptr;
        id = 0;
    }
    return result;
}

PageManager::SnapshotEpoch PageManager::BeginSnapshotEpoch() {
    return SnapshotEpoch{this, impl->BeginSnapshotEpoch()};
}

void PageManager::NotifyCpuWrite(VAddr address, u64 size) {
    impl->InvalidateSnapshotRange(address, size);
}

void PageManager::PrepareForGpuUnmap(VAddr address, u64 size) {
    impl->InvalidateSnapshotRange(address, size);
}

void PageManager::OnGpuMap(VAddr address, size_t size, GpuMappingKind kind) {
    impl->OnMap(address, size, kind);
}

void PageManager::OnGpuUnmap(VAddr address, size_t size, GpuMappingKind kind) {
    impl->OnUnmap(address, size, kind);
}

bool PageManager::IsGpuMapped(VAddr address, size_t size) const {
    return impl->logical_mappings.Contains(address, size);
}

u64 PageManager::GetGpuMappedExtent(VAddr address, u64 size) const {
    return impl->logical_mappings.MappedExtent(address, size);
}

bool PageManager::BeginSnapshotPage(u64 id, VAddr address, u64 size) {
    return id != 0 && impl->BeginSnapshotPage(id, address, size);
}

bool PageManager::EndSnapshotPage(u64 id, VAddr address, u64 size, bool copied) {
    return id != 0 && impl->EndSnapshotPage(id, address, size, copied);
}

bool PageManager::IsSnapshotEpochValid(u64 id) const {
    return id != 0 && impl->IsSnapshotEpochValid(id);
}

bool PageManager::SealSnapshotEpoch(u64 id) {
    return id != 0 && impl->SealSnapshotEpoch(id);
}

template <bool track>
void PageManager::UpdatePageWatchers(VAddr addr, u64 size) const {
    impl->UpdatePageWatchers<track, false>(addr, size);
}

template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(VAddr base_addr, RegionBits& mask) const {
    impl->UpdatePageWatchersForRegion<track, is_read>(base_addr, mask);
}

template void PageManager::UpdatePageWatchers<true>(VAddr addr, u64 size) const;
template void PageManager::UpdatePageWatchers<false>(VAddr addr, u64 size) const;
template void PageManager::UpdatePageWatchersForRegion<true, true>(VAddr base_addr,
                                                                   RegionBits& mask) const;
template void PageManager::UpdatePageWatchersForRegion<true, false>(VAddr base_addr,
                                                                    RegionBits& mask) const;
template void PageManager::UpdatePageWatchersForRegion<false, true>(VAddr base_addr,
                                                                    RegionBits& mask) const;
template void PageManager::UpdatePageWatchersForRegion<false, false>(VAddr base_addr,
                                                                     RegionBits& mask) const;

} // namespace VideoCore
