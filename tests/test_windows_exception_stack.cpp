// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstddef>
#include <cstdint>
#include <latch>
#include <thread>

#include <gtest/gtest.h>

#include <windows.h>

#include "common/types.h"
#include "core/windows_exception_stack.h"

extern "C" void* PS4_SYSV_ABI _runOnAnotherStack(void* argument, void* function,
                                                   void* stack_top) asm("_runOnAnotherStack");

namespace {

struct CallbackObservation {
    bool outer_on_exception_stack{};
    bool nested_on_exception_stack{};
    bool outer_still_on_exception_stack{};
    bool outer_registered_with_windows{};
    bool nested_registered_with_windows{};
    std::intptr_t nested_result{};
    std::uintptr_t outer_stack_address{};
    std::uintptr_t nested_stack_address{};
};

bool IsInsideCurrentWindowsStack(const void* pointer) {
    ULONG_PTR low{};
    ULONG_PTR high{};
    GetCurrentThreadStackLimits(&low, &high);
    const auto value = reinterpret_cast<std::uintptr_t>(pointer);
    return value >= low && value < high;
}

std::intptr_t ObserveNestedStack(void* opaque) noexcept {
    auto& observation = *static_cast<CallbackObservation*>(opaque);
    std::byte stack_marker{};
    observation.nested_on_exception_stack =
        Core::IsOnPreparedWindowsExceptionStack(&stack_marker);
    observation.nested_registered_with_windows =
        IsInsideCurrentWindowsStack(&stack_marker);
    observation.nested_stack_address =
        reinterpret_cast<std::uintptr_t>(&stack_marker);
    return 41;
}

std::intptr_t ObserveOuterStack(void* opaque) noexcept {
    auto& observation = *static_cast<CallbackObservation*>(opaque);
    std::byte stack_marker{};
    observation.outer_on_exception_stack =
        Core::IsOnPreparedWindowsExceptionStack(&stack_marker);
    observation.outer_registered_with_windows =
        IsInsideCurrentWindowsStack(&stack_marker);
    observation.outer_stack_address =
        reinterpret_cast<std::uintptr_t>(&stack_marker);
    observation.nested_result =
        Core::RunOnWindowsExceptionStack(ObserveNestedStack, opaque);
    observation.outer_still_on_exception_stack =
        Core::IsOnPreparedWindowsExceptionStack(&stack_marker);
    return observation.nested_result + 1;
}

class GuardedGuestStack {
public:
    explicit GuardedGuestStack(std::size_t usable_size) {
        SYSTEM_INFO system_info{};
        GetSystemInfo(&system_info);
        guard_size = system_info.dwPageSize;
        const auto aligned_usable_size =
            (usable_size + guard_size - 1) / guard_size * guard_size;
        reservation_size = guard_size + aligned_usable_size;
        reservation = static_cast<std::byte*>(VirtualAlloc(
            nullptr, reservation_size, MEM_RESERVE, PAGE_NOACCESS));
        if (reservation == nullptr) {
            return;
        }
        usable_begin = static_cast<std::byte*>(VirtualAlloc(
            reservation + guard_size, aligned_usable_size, MEM_COMMIT,
            PAGE_READWRITE));
        if (usable_begin == nullptr) {
            VirtualFree(reservation, 0, MEM_RELEASE);
            reservation = nullptr;
            return;
        }
        usable_end = usable_begin + aligned_usable_size;
    }

    ~GuardedGuestStack() {
        if (reservation != nullptr) {
            VirtualFree(reservation, 0, MEM_RELEASE);
        }
    }

    [[nodiscard]] bool IsValid() const {
        return reservation != nullptr && usable_begin != nullptr;
    }

    [[nodiscard]] bool Contains(const void* pointer) const {
        const auto* value = static_cast<const std::byte*>(pointer);
        return value >= usable_begin && value < usable_end;
    }

    [[nodiscard]] void* Top() const {
        return usable_end;
    }

private:
    std::byte* reservation{};
    std::byte* usable_begin{};
    std::byte* usable_end{};
    std::size_t guard_size{};
    std::size_t reservation_size{};
};

struct GuestStackObservation {
    const GuardedGuestStack* guest_stack{};
    CallbackObservation callback{};
    bool guest_marker_on_guest_stack{};
    ULONG_PTR guest_low_before{};
    ULONG_PTR guest_high_before{};
    ULONG_PTR guest_low_after{};
    ULONG_PTR guest_high_after{};
    std::intptr_t result{};
};

void* PS4_SYSV_ABI InvokeFromGuestStack(void* opaque) {
    auto& observation = *static_cast<GuestStackObservation*>(opaque);
    std::byte guest_marker{};
    observation.guest_marker_on_guest_stack =
        observation.guest_stack->Contains(&guest_marker);
    GetCurrentThreadStackLimits(&observation.guest_low_before,
                                &observation.guest_high_before);
    observation.result = Core::RunOnWindowsExceptionStack(
        ObserveOuterStack, &observation.callback);
    GetCurrentThreadStackLimits(&observation.guest_low_after,
                                &observation.guest_high_after);
    return reinterpret_cast<void*>(observation.result);
}

class WindowsExceptionStackTest : public ::testing::Test {
protected:
    void SetUp() override {
        Core::CleanupWindowsExceptionStack();
    }

    void TearDown() override {
        Core::CleanupWindowsExceptionStack();
    }
};

} // namespace

TEST_F(WindowsExceptionStackTest, RunsHandlerAndNestedEntryOnPreparedStack) {
    ASSERT_TRUE(Core::PrepareWindowsExceptionStack());
    EXPECT_TRUE(Core::PrepareWindowsExceptionStack());
    std::byte caller_stack_marker{};
    CallbackObservation observation{};
    ULONG_PTR original_low{};
    ULONG_PTR original_high{};
    GetCurrentThreadStackLimits(&original_low, &original_high);

    EXPECT_FALSE(Core::IsOnPreparedWindowsExceptionStack(&caller_stack_marker));
    EXPECT_EQ(
        Core::RunOnWindowsExceptionStack(ObserveOuterStack, &observation),
        42);
    EXPECT_TRUE(observation.outer_on_exception_stack);
    EXPECT_TRUE(observation.nested_on_exception_stack);
    EXPECT_TRUE(observation.outer_still_on_exception_stack);
    EXPECT_TRUE(observation.outer_registered_with_windows);
    EXPECT_TRUE(observation.nested_registered_with_windows);
    EXPECT_EQ(observation.nested_result, 41);
    EXPECT_FALSE(Core::IsOnPreparedWindowsExceptionStack(&caller_stack_marker));
    ULONG_PTR restored_low{};
    ULONG_PTR restored_high{};
    GetCurrentThreadStackLimits(&restored_low, &restored_high);
    EXPECT_EQ(restored_low, original_low);
    EXPECT_EQ(restored_high, original_high);
}

TEST_F(WindowsExceptionStackTest, UnpreparedThreadKeepsOriginalHandlerPath) {
    std::byte caller_stack_marker{};
    CallbackObservation observation{};

    EXPECT_EQ(Core::RunOnWindowsExceptionStack(ObserveNestedStack, &observation),
              41);
    EXPECT_FALSE(observation.nested_on_exception_stack);
    EXPECT_TRUE(observation.nested_registered_with_windows);
    EXPECT_FALSE(Core::IsOnPreparedWindowsExceptionStack(&caller_stack_marker));
}

TEST_F(WindowsExceptionStackTest,
       SwitchesAwayFromGuestStackAndRestoresGuestAndNativeTebBounds) {
    ASSERT_TRUE(Core::PrepareWindowsExceptionStack());
    GuardedGuestStack guest_stack{64 * 1024};
    ASSERT_TRUE(guest_stack.IsValid());

    ULONG_PTR original_low{};
    ULONG_PTR original_high{};
    GetCurrentThreadStackLimits(&original_low, &original_high);
    GuestStackObservation observation{.guest_stack = &guest_stack};

    const auto result = _runOnAnotherStack(
        &observation, reinterpret_cast<void*>(InvokeFromGuestStack),
        guest_stack.Top());

    EXPECT_EQ(reinterpret_cast<std::intptr_t>(result), 42);
    EXPECT_EQ(observation.result, 42);
    EXPECT_TRUE(observation.guest_marker_on_guest_stack);
    EXPECT_EQ(observation.guest_low_before, 0u);
    EXPECT_EQ(observation.guest_high_before, 0u);
    EXPECT_EQ(observation.guest_low_after, 0u);
    EXPECT_EQ(observation.guest_high_after, 0u);
    EXPECT_TRUE(observation.callback.outer_on_exception_stack);
    EXPECT_TRUE(observation.callback.nested_on_exception_stack);
    EXPECT_FALSE(guest_stack.Contains(reinterpret_cast<const void*>(
        observation.callback.outer_stack_address)));
    EXPECT_FALSE(guest_stack.Contains(reinterpret_cast<const void*>(
        observation.callback.nested_stack_address)));
    ULONG_PTR restored_low{};
    ULONG_PTR restored_high{};
    GetCurrentThreadStackLimits(&restored_low, &restored_high);
    EXPECT_EQ(restored_low, original_low);
    EXPECT_EQ(restored_high, original_high);
}

TEST_F(WindowsExceptionStackTest, CleanupRestoresThreadAndAllowsReprepare) {
    const bool initially_a_fiber = IsThreadAFiber() != FALSE;
    CallbackObservation first{};
    ASSERT_TRUE(Core::PrepareWindowsExceptionStack());
    EXPECT_EQ(Core::RunOnWindowsExceptionStack(ObserveNestedStack, &first), 41);

    Core::CleanupWindowsExceptionStack();
    EXPECT_EQ(IsThreadAFiber() != FALSE, initially_a_fiber);

    CallbackObservation second{};
    ASSERT_TRUE(Core::PrepareWindowsExceptionStack());
    EXPECT_EQ(Core::RunOnWindowsExceptionStack(ObserveNestedStack, &second), 41);
    EXPECT_TRUE(second.nested_on_exception_stack);
}

TEST_F(WindowsExceptionStackTest, HostThreadsUseIndependentPreparedStacks) {
    struct ThreadObservation {
        bool prepared{};
        bool on_exception_stack{};
        bool registered_with_windows{};
        bool fiber_state_restored{};
        std::uintptr_t stack_address{};
    } first, second;
    std::latch observations_ready{2};
    std::latch permit_cleanup{1};

    const auto worker = [&](ThreadObservation& result) {
        const bool initially_a_fiber = IsThreadAFiber() != FALSE;
        CallbackObservation callback{};
        result.prepared = Core::PrepareWindowsExceptionStack();
        if (result.prepared) {
            Core::RunOnWindowsExceptionStack(ObserveNestedStack, &callback);
            result.on_exception_stack = callback.nested_on_exception_stack;
            result.registered_with_windows = callback.nested_registered_with_windows;
            result.stack_address = callback.nested_stack_address;
        }
        observations_ready.count_down();
        permit_cleanup.wait();
        Core::CleanupWindowsExceptionStack();
        result.fiber_state_restored =
            (IsThreadAFiber() != FALSE) == initially_a_fiber;
    };

    std::thread first_thread{worker, std::ref(first)};
    std::thread second_thread{worker, std::ref(second)};
    observations_ready.wait();
    EXPECT_TRUE(first.prepared);
    EXPECT_TRUE(second.prepared);
    EXPECT_TRUE(first.on_exception_stack);
    EXPECT_TRUE(second.on_exception_stack);
    EXPECT_TRUE(first.registered_with_windows);
    EXPECT_TRUE(second.registered_with_windows);
    EXPECT_NE(first.stack_address, second.stack_address);
    permit_cleanup.count_down();
    first_thread.join();
    second_thread.join();
    EXPECT_TRUE(first.fiber_state_restored);
    EXPECT_TRUE(second.fiber_state_restored);
}

TEST_F(WindowsExceptionStackTest, StackSwitchEntryHasWindowsUnwindMetadata) {
    DWORD64 image_base{};
    const auto* runtime_function = RtlLookupFunctionEntry(
        reinterpret_cast<DWORD64>(&Core::RunOnWindowsExceptionStack),
        &image_base, nullptr);

    EXPECT_NE(runtime_function, nullptr);
    EXPECT_NE(image_base, 0u);
}
