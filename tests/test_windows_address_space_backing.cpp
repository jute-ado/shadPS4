// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <concepts>
#include <memory>
#include <string_view>
#include <vector>
#include <windows.h>
#include <gtest/gtest.h>
#include "core/address_space.h"
#include "core/windows_address_space_backing.h"

namespace {

using Core::AddressSpaceBackingLease;
using Core::WindowsAddressSpaceBacking;
using Core::WindowsAddressSpaceBackingApi;

template <typename T>
concept ExposesMapping = requires(const T& value) { value.Mapping(); };
template <typename T>
concept ExposesHandle = requires(const T& value) { value.Handle(); };
template <typename T>
concept ExposesNativeHandle = requires(const T& value) { value.NativeHandle(); };

static_assert(!ExposesMapping<AddressSpaceBackingLease>);
static_assert(!ExposesHandle<AddressSpaceBackingLease>);
static_assert(!ExposesNativeHandle<AddressSpaceBackingLease>);
static_assert(std::same_as<decltype(std::declval<const AddressSpaceBackingLease&>().Base()), u8*>);
static_assert(std::same_as<decltype(std::declval<const AddressSpaceBackingLease&>().Size()), u64>);
static_assert(std::same_as<decltype(std::declval<const Core::AddressSpace&>().AcquireBackingLease()),
                           std::optional<AddressSpaceBackingLease>>);

class FakeBackingApi final : public WindowsAddressSpaceBackingApi {
public:
    enum class Failure {
        None,
        Create,
        Reserve,
        Map,
        WrongMapBase,
    };

    struct CreateArgs {
        void* file;
        u32 desired_access;
        u32 page_protection;
        u32 allocation_attributes;
        u64 maximum_size;
    } create_args{};

    struct ReserveArgs {
        void* process;
        u64 size;
        u32 allocation_type;
        u32 protection;
    } reserve_args{};

    struct MapArgs {
        void* mapping;
        void* process;
        u8* placeholder;
        u64 offset;
        u64 size;
        u32 allocation_type;
        u32 protection;
    } map_args{};

    explicit FakeBackingApi(Failure failure_ = Failure::None) : failure{failure_} {}

    void* CreateFileMapping(void* file, u32 desired_access, u32 page_protection,
                            u32 allocation_attributes, u64 maximum_size) override {
        events.emplace_back("create");
        create_args = {file, desired_access, page_protection, allocation_attributes, maximum_size};
        return failure == Failure::Create ? nullptr : mapping;
    }

    u8* ReservePlaceholder(void* process, u64 size, u32 allocation_type,
                           u32 protection) override {
        events.emplace_back("reserve");
        reserve_args = {process, size, allocation_type, protection};
        return failure == Failure::Reserve ? nullptr : placeholder;
    }

    u8* MapPlaceholder(void* mapping_, void* process, u8* placeholder_, u64 offset, u64 size,
                       u32 allocation_type, u32 protection) override {
        events.emplace_back("map");
        map_args = {mapping_, process, placeholder_, offset, size, allocation_type, protection};
        if (failure == Failure::Map) {
            return nullptr;
        }
        return failure == Failure::WrongMapBase ? wrong_base : placeholder;
    }

    bool UnmapPreservingPlaceholder(void*, u8* base, u32 flags) override {
        events.emplace_back("unmap");
        unmap_base = base;
        unmap_flags = flags;
        return cleanup_succeeds;
    }

    bool ReleasePlaceholder(void*, u8* base, u64 size, u32 flags) override {
        events.emplace_back("release");
        release_base = base;
        release_size = size;
        release_flags = flags;
        return cleanup_succeeds;
    }

    bool CloseMapping(void* mapping_) override {
        events.emplace_back("close");
        closed_mapping = mapping_;
        return cleanup_succeeds;
    }

    Failure failure;
    bool cleanup_succeeds{true};
    std::vector<std::string_view> events;
    void* const mapping{reinterpret_cast<void*>(0x1000)};
    u8* const placeholder{reinterpret_cast<u8*>(0x2000)};
    u8* const wrong_base{reinterpret_cast<u8*>(0x3000)};
    u8* unmap_base{};
    u32 unmap_flags{};
    u8* release_base{};
    u64 release_size{};
    u32 release_flags{};
    void* closed_mapping{};
};

constexpr u64 BackingSize = 0x210000000ULL;
void* const Process = reinterpret_cast<void*>(0x4000);

TEST(WindowsAddressSpaceBacking, UsesTheExactProductionAllocationSequenceAndFlags) {
    auto api = std::make_shared<FakeBackingApi>();
    auto backing = WindowsAddressSpaceBacking::Create(api, Process, BackingSize);

    ASSERT_NE(backing, nullptr);
    EXPECT_EQ(api->events, (std::vector<std::string_view>{"create", "reserve", "map"}));
    EXPECT_EQ(api->create_args.file, INVALID_HANDLE_VALUE);
    EXPECT_EQ(api->create_args.desired_access, FILE_MAP_ALL_ACCESS);
    EXPECT_EQ(api->create_args.page_protection, PAGE_EXECUTE_READWRITE);
    EXPECT_EQ(api->create_args.allocation_attributes, SEC_COMMIT);
    EXPECT_EQ(api->create_args.maximum_size, BackingSize);
    EXPECT_EQ(api->reserve_args.process, Process);
    EXPECT_EQ(api->reserve_args.size, BackingSize);
    EXPECT_EQ(api->reserve_args.allocation_type, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER);
    EXPECT_EQ(api->reserve_args.protection, PAGE_NOACCESS);
    EXPECT_EQ(api->map_args.mapping, api->mapping);
    EXPECT_EQ(api->map_args.process, Process);
    EXPECT_EQ(api->map_args.placeholder, api->placeholder);
    EXPECT_EQ(api->map_args.offset, 0);
    EXPECT_EQ(api->map_args.size, BackingSize);
    EXPECT_EQ(api->map_args.allocation_type, MEM_REPLACE_PLACEHOLDER);
    EXPECT_EQ(api->map_args.protection, PAGE_EXECUTE_READWRITE);

    const auto lease = backing->AcquireLease();
    EXPECT_TRUE(lease);
    EXPECT_EQ(lease.Base(), api->placeholder);
    EXPECT_EQ(lease.Size(), BackingSize);
}

TEST(WindowsAddressSpaceBacking, RollsBackEachFailedAcquisitionStage) {
    struct Case {
        FakeBackingApi::Failure failure;
        std::vector<std::string_view> events;
    };
    const Case cases[]{
        {FakeBackingApi::Failure::Create, {"create"}},
        {FakeBackingApi::Failure::Reserve, {"create", "reserve", "close"}},
        {FakeBackingApi::Failure::Map, {"create", "reserve", "map", "release", "close"}},
        {FakeBackingApi::Failure::WrongMapBase,
         {"create", "reserve", "map", "unmap", "release", "close"}},
    };

    for (const auto& test_case : cases) {
        auto api = std::make_shared<FakeBackingApi>(test_case.failure);
        EXPECT_EQ(WindowsAddressSpaceBacking::Create(api, Process, BackingSize), nullptr);
        EXPECT_EQ(api->events, test_case.events);
    }
}

TEST(WindowsAddressSpaceBacking, CleansUpInOrderAndContinuesAfterCleanupFailures) {
    auto api = std::make_shared<FakeBackingApi>();
    api->cleanup_succeeds = false;
    {
        auto backing = WindowsAddressSpaceBacking::Create(api, Process, BackingSize);
        ASSERT_NE(backing, nullptr);
    }

    EXPECT_EQ(api->events,
              (std::vector<std::string_view>{"create", "reserve", "map", "unmap", "release",
                                             "close"}));
    EXPECT_EQ(api->unmap_base, api->placeholder);
    EXPECT_EQ(api->unmap_flags, MEM_PRESERVE_PLACEHOLDER);
    EXPECT_EQ(api->release_base, api->placeholder);
    EXPECT_EQ(api->release_size, 0);
    EXPECT_EQ(api->release_flags, MEM_RELEASE);
    EXPECT_EQ(api->closed_mapping, api->mapping);
}

TEST(WindowsAddressSpaceBacking, MultipleLeasesDoNotAllocateAgain) {
    auto api = std::make_shared<FakeBackingApi>();
    auto backing = WindowsAddressSpaceBacking::Create(api, Process, BackingSize);
    ASSERT_NE(backing, nullptr);
    const auto allocation_events = api->events;

    const auto first = backing->AcquireLease();
    const auto second = backing->AcquireLease();

    EXPECT_TRUE(first);
    EXPECT_TRUE(second);
    EXPECT_EQ(api->events, allocation_events);
}

TEST(WindowsAddressSpaceBacking, LeaseRetainsBackingAfterOwnerRelease) {
    auto api = std::make_shared<FakeBackingApi>();
    auto backing = WindowsAddressSpaceBacking::Create(api, Process, BackingSize);
    ASSERT_NE(backing, nullptr);
    auto lease = backing->AcquireLease();

    backing.reset();
    EXPECT_EQ(api->events, (std::vector<std::string_view>{"create", "reserve", "map"}));
    EXPECT_EQ(lease.Base(), api->placeholder);
    EXPECT_EQ(lease.Size(), BackingSize);

    lease = {};
    EXPECT_EQ(api->events,
              (std::vector<std::string_view>{"create", "reserve", "map", "unmap", "release",
                                             "close"}));
}

} // namespace
