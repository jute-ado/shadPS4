// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <fstream>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "core/gpu_owner_lifetime.h"

namespace {
using namespace std::chrono_literals;
using Lifetime = Core::GpuOwnerLifetime;

auto MakeLifetime() {
    return std::make_unique<Lifetime>();
}

auto Acquire(Lifetime& lifetime) {
    return lifetime.Acquire();
}

bool WaitUntilClosed(Lifetime& lifetime) {
    const auto deadline = std::chrono::steady_clock::now() + 1s;
    while (std::chrono::steady_clock::now() < deadline) {
        if (!Acquire(lifetime)) {
            return true;
        }
        std::this_thread::yield();
    }
    return false;
}

TEST(GpuOwnerLifetime, OpenLeaseDoesNotRequestGuestSubmissionProgress) {
    auto lifetime = MakeLifetime();
    std::mutex allocator;
    std::unique_lock allocator_lock{allocator};
    std::promise<void> old_gpu_work;
    auto old_completion = old_gpu_work.get_future();
    auto lease = Acquire(*lifetime);
    ASSERT_TRUE(lease);
    EXPECT_TRUE(allocator_lock.owns_lock());
    EXPECT_EQ(old_completion.wait_for(0ms), std::future_status::timeout);
    // This producer is allowed to mutate/cache-invalidate before the old guest work completes.
    // A whole-queue wait here would prevent the action which makes that work ready.
    old_gpu_work.set_value();
    EXPECT_EQ(old_completion.wait_for(0ms), std::future_status::ready);
}

TEST(GpuOwnerLifetime, ClosedGateRefusesEveryLaterOwnerUse) {
    auto lifetime = MakeLifetime();
    lifetime->CloseAndWait();
    EXPECT_FALSE(Acquire(*lifetime));
    EXPECT_FALSE(Acquire(*lifetime));
    lifetime->CloseAndWait();
}

TEST(GpuOwnerLifetime, AdmittedOperationFinishesBeforeOwnerStops) {
    auto lifetime = MakeLifetime();
    auto lease = Acquire(*lifetime);
    ASSERT_TRUE(lease);
    std::atomic<bool> owner_running{true};
    std::atomic<bool> returned_with_live_operation{false};
    std::atomic<bool> operation_finished{false};
    auto closer = std::async(std::launch::async, [&] {
        lifetime->CloseAndWait();
        returned_with_live_operation = !operation_finished.load();
        owner_running = false;
    });
    const bool closed = WaitUntilClosed(*lifetime);
    EXPECT_TRUE(closed); // Proves that CloseAndWait actually entered, not just an unscheduled task.
    EXPECT_EQ(closer.wait_for(20ms), std::future_status::timeout);
    EXPECT_TRUE(owner_running.load());
    operation_finished = true;
    lease.reset(); // Always release before joining, even on failed assertions.
    ASSERT_EQ(closer.wait_for(1s), std::future_status::ready);
    closer.get();
    EXPECT_FALSE(returned_with_live_operation.load());
    EXPECT_FALSE(owner_running.load());
}

TEST(GpuOwnerLifetime, MovedLeaseRetainsTheOperationUntilItsFinalOwnerReleases) {
    auto lifetime = MakeLifetime();
    auto original = Acquire(*lifetime);
    ASSERT_TRUE(original);
    auto moved = std::move(original);
    original.reset();
    auto closer = std::async(std::launch::async, [&] { lifetime->CloseAndWait(); });
    EXPECT_TRUE(WaitUntilClosed(*lifetime));
    EXPECT_EQ(closer.wait_for(20ms), std::future_status::timeout);
    moved.reset();
    ASSERT_EQ(closer.wait_for(1s), std::future_status::ready);
    closer.get();
}

TEST(GpuOwnerLifetime, ExceptionUnwindingReleasesTheAdmission) {
    auto lifetime = MakeLifetime();
    EXPECT_THROW(
        {
            auto lease = Acquire(*lifetime);
            ASSERT_TRUE(lease);
            throw std::runtime_error{"synthetic owner operation failure"};
        },
        std::runtime_error);
    lifetime->CloseAndWait();
    EXPECT_FALSE(Acquire(*lifetime));
}

std::string ReadSource(const char* file) {
    std::ifstream input{std::string{GPU_OWNER_SOURCE_ROOT} + "/" + file, std::ios::binary};
    EXPECT_TRUE(input.good());
    std::string source{std::istreambuf_iterator<char>{input}, {}};
    std::erase(source, '\r');
    return source;
}

std::string Function(const std::string& source, const std::string& signature) {
    const auto begin = source.find(signature);
    EXPECT_NE(begin, std::string::npos);
    if (begin == std::string::npos) {
        return {};
    }
    return source.substr(begin, source.find("\n}\n", begin) - begin);
}

TEST(GpuOwnerLifetimeWiring, MappingAdmissionDoesNotReleaseAllocatorOrConsultGpuWork) {
    const auto source = ReadSource("core/memory.cpp");
    const auto acquire = Function(source, "MemoryManager::AcquireGpuOwnerLease(");
    EXPECT_NE(acquire.find("gpu_owner_lifetime->Acquire()"), std::string::npos);
    for (const char* forbidden : {"unlock", "Retire", "Capture", "Finish", "Wait", "allocator"}) {
        EXPECT_EQ(acquire.find(forbidden), std::string::npos) << forbidden;
    }
    EXPECT_EQ(source.find("AcquireOptimisticMemoryRetirement"), std::string::npos);
    EXPECT_EQ(source.find("CaptureMemoryRetirementPlan"), std::string::npos);
    for (const char* operation :
         {"Free", "PoolCommit", "MapMemory", "MapFile", "PoolDecommit", "UnmapMemory"}) {
        const auto body = Function(source, "MemoryManager::" + std::string{operation} + "(");
        const auto lease = body.find("auto lifetime = AcquireGpuOwnerLease()");
        ASSERT_NE(lease, std::string::npos) << operation;
        EXPECT_LT(body.find("std::unique_lock lk{unmap_mutex}"), lease) << operation;
        EXPECT_EQ(body.find("lk.unlock()"), std::string::npos) << operation;
        EXPECT_EQ(body.find("AcquireMemoryRetirement"), std::string::npos) << operation;
    }
}

TEST(GpuOwnerLifetimeWiring, SparseMappingsAndExplicitInvalidationAlsoRetainOwnerLifetime) {
    const auto source = ReadSource("core/memory.cpp");
    for (const char* operation : {"SetPrtArea", "InvalidateMemory"}) {
        const auto body = Function(source, "MemoryManager::" + std::string{operation} + "(");
        const auto lease = body.find("auto lifetime = AcquireGpuOwnerLease()");
        ASSERT_NE(lease, std::string::npos) << operation;
        EXPECT_LT(lease, body.find("if (!lifetime)"));
        EXPECT_LT(body.find("if (!lifetime)"), body.find("rasterizer->"));
    }
}

TEST(GpuOwnerLifetimeWiring, OwnerInstallHasNoRetirementCallbackAndClosePrecedesStop) {
    const auto install =
        Function(ReadSource("core/libraries/gnmdriver/gnmdriver.cpp"), "void RegisterLib(");
    EXPECT_NE(install.find("std::make_shared<Core::GpuOwnerLifetime>()"), std::string::npos);
    EXPECT_EQ(install.find("MemoryRetirementProvider"), std::string::npos);
    EXPECT_EQ(install.find("AcquireSubmissionRetirement"), std::string::npos);
    EXPECT_EQ(install.find("WaitGpuCompletion"), std::string::npos);
    const auto close =
        Function(ReadSource("video_core/amdgpu/liverpool.cpp"), "Liverpool::~Liverpool()");
    EXPECT_LT(close.find("gpu_owner_lifetime->CloseAndWait()"),
              close.find("process_thread.request_stop()"));
    EXPECT_LT(close.find("process_thread.request_stop()"), close.find("process_thread.join()"));
}

TEST(GpuOwnerLifetimeWiring, ExactReleaseAndDeferredBackingIdentityRemainInProduction) {
    const auto source = ReadSource("core/memory.cpp");
    const auto free = Function(source, "MemoryManager::Free(");
    EXPECT_NE(free.find("PlanPhysicalMemoryRelease("), std::string::npos);
    EXPECT_NE(free.find("IntersectMemoryReleaseRanges("), std::string::npos);
    // GPU mappings are retired through the partial-range aware helper before the VMA
    // release plan mutates the address space.  Keep the ordering contract at the
    // MemoryManager boundary rather than requiring the helper's rasterizer call here.
    EXPECT_LT(free.find("UnmapGpuMappings"), free.find("UnmapMemoryImpl"));
    EXPECT_NE(source.find("CaptureDeferredBackingWrite(vma_map, virtual_identity"),
              std::string::npos);
    EXPECT_NE(source.find("PublishDeferredBackingWrite(mutex, vma_map, virtual_identity"),
              std::string::npos);
}

} // namespace
