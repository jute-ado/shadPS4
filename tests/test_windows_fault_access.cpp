// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

#include <gtest/gtest.h>
#include <windows.h>

#include "video_core/windows_fault_access.h"

namespace {

std::atomic<bool> fault_entered;
std::atomic<bool> release_fault;
std::atomic<bool> delayed_fault_allowed;

LONG CALLBACK DelayedFaultHandler(EXCEPTION_POINTERS* exception) {
    if (exception->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    fault_entered.store(true, std::memory_order_release);
    while (!release_fault.load(std::memory_order_acquire)) {
        SwitchToThread();
    }

    const auto access = VideoCore::DecodeWindowsFaultAccess(
        exception->ExceptionRecord->ExceptionInformation[0]);
    const auto* address =
        reinterpret_cast<const void*>(exception->ExceptionRecord->ExceptionInformation[1]);
    delayed_fault_allowed.store(VideoCore::IsWindowsFaultAccessAllowed(address, access),
                                std::memory_order_release);
    return EXCEPTION_CONTINUE_EXECUTION;
}

} // namespace

TEST(WindowsFaultAccess, DecodesWindowsExceptionOperations) {
    EXPECT_EQ(VideoCore::DecodeWindowsFaultAccess(0), VideoCore::WindowsFaultAccess::Read);
    EXPECT_EQ(VideoCore::DecodeWindowsFaultAccess(1), VideoCore::WindowsFaultAccess::Write);
    EXPECT_EQ(VideoCore::DecodeWindowsFaultAccess(8), VideoCore::WindowsFaultAccess::Execute);
    EXPECT_EQ(VideoCore::DecodeWindowsFaultAccess(2), VideoCore::WindowsFaultAccess::Unknown);
}

TEST(WindowsFaultAccess, MatchesCommittedPageProtection) {
    SYSTEM_INFO system_info{};
    GetSystemInfo(&system_info);
    const auto page_size = static_cast<SIZE_T>(system_info.dwPageSize);
    auto* page = VirtualAlloc(nullptr, page_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    ASSERT_NE(page, nullptr);

    EXPECT_TRUE(VideoCore::IsWindowsFaultAccessAllowed(page, VideoCore::WindowsFaultAccess::Read));
    EXPECT_TRUE(VideoCore::IsWindowsFaultAccessAllowed(page, VideoCore::WindowsFaultAccess::Write));
    EXPECT_FALSE(
        VideoCore::IsWindowsFaultAccessAllowed(page, VideoCore::WindowsFaultAccess::Execute));

    DWORD old_protection{};
    ASSERT_TRUE(VirtualProtect(page, page_size, PAGE_READONLY, &old_protection));
    EXPECT_TRUE(VideoCore::IsWindowsFaultAccessAllowed(page, VideoCore::WindowsFaultAccess::Read));
    EXPECT_FALSE(VideoCore::IsWindowsFaultAccessAllowed(page, VideoCore::WindowsFaultAccess::Write));

    ASSERT_TRUE(VirtualProtect(page, page_size, PAGE_NOACCESS, &old_protection));
    EXPECT_FALSE(VideoCore::IsWindowsFaultAccessAllowed(page, VideoCore::WindowsFaultAccess::Read));
    EXPECT_FALSE(VideoCore::IsWindowsFaultAccessAllowed(page, VideoCore::WindowsFaultAccess::Write));

    ASSERT_TRUE(VirtualProtect(page, page_size, PAGE_EXECUTE_READWRITE, &old_protection));
    EXPECT_TRUE(VideoCore::IsWindowsFaultAccessAllowed(page, VideoCore::WindowsFaultAccess::Read));
    EXPECT_TRUE(VideoCore::IsWindowsFaultAccessAllowed(page, VideoCore::WindowsFaultAccess::Write));
    EXPECT_FALSE(
        VideoCore::IsWindowsFaultAccessAllowed(page, VideoCore::WindowsFaultAccess::Execute));

    ASSERT_TRUE(VirtualProtect(page, page_size, PAGE_READWRITE | PAGE_GUARD, &old_protection));
    EXPECT_FALSE(VideoCore::IsWindowsFaultAccessAllowed(page, VideoCore::WindowsFaultAccess::Read));
    EXPECT_FALSE(VideoCore::IsWindowsFaultAccessAllowed(page, VideoCore::WindowsFaultAccess::Write));
    EXPECT_TRUE(VirtualFree(page, 0, MEM_RELEASE));

    auto* reserved = VirtualAlloc(nullptr, page_size, MEM_RESERVE, PAGE_NOACCESS);
    ASSERT_NE(reserved, nullptr);
    EXPECT_FALSE(
        VideoCore::IsWindowsFaultAccessAllowed(reserved, VideoCore::WindowsFaultAccess::Read));
    EXPECT_FALSE(
        VideoCore::IsWindowsFaultAccessAllowed(reserved, VideoCore::WindowsFaultAccess::Write));
    EXPECT_TRUE(VirtualFree(reserved, 0, MEM_RELEASE));
}

TEST(WindowsFaultAccess, AcceptsDelayedWriteAfterAnotherThreadRestoresAccess) {
    SYSTEM_INFO system_info{};
    GetSystemInfo(&system_info);
    const auto page_size = static_cast<SIZE_T>(system_info.dwPageSize);
    auto* page = static_cast<volatile std::uint8_t*>(
        VirtualAlloc(nullptr, page_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    ASSERT_NE(page, nullptr);

    DWORD old_protection{};
    ASSERT_TRUE(VirtualProtect(const_cast<std::uint8_t*>(page), page_size, PAGE_NOACCESS,
                               &old_protection));
    auto* handler = AddVectoredExceptionHandler(1, DelayedFaultHandler);
    ASSERT_NE(handler, nullptr);
    fault_entered.store(false, std::memory_order_release);
    release_fault.store(false, std::memory_order_release);
    delayed_fault_allowed.store(false, std::memory_order_release);

    std::thread writer([&] { page[0] = 0x5a; });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!fault_entered.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }

    const bool entered = fault_entered.load(std::memory_order_acquire);
    EXPECT_TRUE(VirtualProtect(const_cast<std::uint8_t*>(page), page_size, PAGE_READWRITE,
                               &old_protection));
    release_fault.store(true, std::memory_order_release);
    writer.join();

    EXPECT_TRUE(entered);
    EXPECT_TRUE(delayed_fault_allowed.load(std::memory_order_acquire));
    EXPECT_EQ(page[0], 0x5a);
    EXPECT_TRUE(RemoveVectoredExceptionHandler(handler));
    EXPECT_TRUE(VirtualFree(const_cast<std::uint8_t*>(page), 0, MEM_RELEASE));
}
