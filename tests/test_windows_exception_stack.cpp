// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstddef>
#include <cstdint>
#include <latch>
#include <thread>

#include <gtest/gtest.h>

#include <windows.h>
#include <xmmintrin.h>

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
    observation.nested_on_exception_stack = Core::IsOnPreparedWindowsExceptionStack(&stack_marker);
    observation.nested_registered_with_windows = IsInsideCurrentWindowsStack(&stack_marker);
    observation.nested_stack_address = reinterpret_cast<std::uintptr_t>(&stack_marker);
    return 41;
}

std::intptr_t ReturnContinueExecution(void*) noexcept {
    return EXCEPTION_CONTINUE_EXECUTION;
}

struct CleanupInsideHandlerObservation {
    bool before_cleanup{};
    bool after_cleanup{};
};

std::intptr_t AttemptCleanupInsideHandler(void* opaque) noexcept {
    auto& observation = *static_cast<CleanupInsideHandlerObservation*>(opaque);
    std::byte stack_marker{};
    observation.before_cleanup = Core::IsOnPreparedWindowsExceptionStack(&stack_marker);
    Core::CleanupWindowsExceptionStack();
    observation.after_cleanup = Core::IsOnPreparedWindowsExceptionStack(&stack_marker);
    return 7;
}

struct MxcsrObservation {
    unsigned int handler_before{};
    unsigned int handler_after{};
};

std::intptr_t ChangeHandlerRoundingMode(void* opaque) noexcept {
    auto& observation = *static_cast<MxcsrObservation*>(opaque);
    observation.handler_before = _mm_getcsr();
    _mm_setcsr((observation.handler_before & ~_MM_ROUND_MASK) | _MM_ROUND_DOWN);
    observation.handler_after = _mm_getcsr();
    return 9;
}

class ScopedMxcsrRestore {
public:
    ScopedMxcsrRestore() : original{_mm_getcsr()} {}
    ~ScopedMxcsrRestore() {
        _mm_setcsr(original);
    }

private:
    unsigned int original{};
};

struct RawTebStackBounds {
    std::uintptr_t base{};
    std::uintptr_t limit{};
};

RawTebStackBounds ReadRawTebStackBounds() {
    const auto* tib = reinterpret_cast<const NT_TIB*>(NtCurrentTeb());
    return {
        .base = reinterpret_cast<std::uintptr_t>(tib->StackBase),
        .limit = reinterpret_cast<std::uintptr_t>(tib->StackLimit),
    };
}

std::intptr_t ObserveOuterStack(void* opaque) noexcept {
    auto& observation = *static_cast<CallbackObservation*>(opaque);
    std::byte stack_marker{};
    observation.outer_on_exception_stack = Core::IsOnPreparedWindowsExceptionStack(&stack_marker);
    observation.outer_registered_with_windows = IsInsideCurrentWindowsStack(&stack_marker);
    observation.outer_stack_address = reinterpret_cast<std::uintptr_t>(&stack_marker);
    observation.nested_result = Core::RunOnWindowsExceptionStack(ObserveNestedStack, opaque);
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
        const auto aligned_usable_size = (usable_size + guard_size - 1) / guard_size * guard_size;
        const auto reservation_size = guard_size + aligned_usable_size;
        reservation = static_cast<std::byte*>(
            VirtualAlloc(nullptr, reservation_size, MEM_RESERVE, PAGE_NOACCESS));
        if (reservation == nullptr) {
            return;
        }
        usable_begin = static_cast<std::byte*>(VirtualAlloc(
            reservation + guard_size, aligned_usable_size, MEM_COMMIT, PAGE_READWRITE));
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
        const auto value = reinterpret_cast<std::uintptr_t>(pointer);
        return value >= reinterpret_cast<std::uintptr_t>(usable_begin) &&
               value < reinterpret_cast<std::uintptr_t>(usable_end);
    }

    [[nodiscard]] void* Top() const {
        return usable_end;
    }

private:
    std::byte* reservation{};
    std::byte* usable_begin{};
    std::byte* usable_end{};
    std::size_t guard_size{};
};

struct GuestStackObservation {
    const GuardedGuestStack* guest_stack{};
    CallbackObservation callback{};
    bool guest_marker_on_guest_stack{};
    RawTebStackBounds guest_bounds_before{};
    RawTebStackBounds guest_bounds_after{};
    std::intptr_t result{};
};

void* PS4_SYSV_ABI InvokeFromGuestStack(void* opaque) {
    auto& observation = *static_cast<GuestStackObservation*>(opaque);
    std::byte guest_marker{};
    observation.guest_marker_on_guest_stack = observation.guest_stack->Contains(&guest_marker);
    observation.guest_bounds_before = ReadRawTebStackBounds();
    observation.result = Core::RunOnWindowsExceptionStack(ObserveOuterStack, &observation.callback);
    observation.guest_bounds_after = ReadRawTebStackBounds();
    return reinterpret_cast<void*>(observation.result);
}

struct VehObservation {
    const GuardedGuestStack* guest_stack{};
    volatile std::byte* protected_page{};
    std::size_t page_size{};
    bool guest_marker_on_guest_stack{};
    bool handler_on_exception_stack{};
    bool handler_registered_with_windows{};
    bool resumed_after_fault{};
    RawTebStackBounds guest_bounds_before{};
    RawTebStackBounds guest_bounds_after{};
};

thread_local VehObservation* active_veh_observation{};

std::intptr_t RepairFaultOnExceptionStack(void* opaque) noexcept {
    auto& observation = *static_cast<VehObservation*>(opaque);
    std::byte stack_marker{};
    observation.handler_on_exception_stack = Core::IsOnPreparedWindowsExceptionStack(&stack_marker);
    observation.handler_registered_with_windows = IsInsideCurrentWindowsStack(&stack_marker);
    DWORD old_protection{};
    if (VirtualProtect(const_cast<std::byte*>(observation.protected_page), observation.page_size,
                       PAGE_READWRITE, &old_protection) == FALSE) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    return EXCEPTION_CONTINUE_EXECUTION;
}

LONG WINAPI TestVehHandler(EXCEPTION_POINTERS* exception) noexcept {
    auto* const observation = active_veh_observation;
    if (observation == nullptr ||
        exception->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION ||
        exception->ExceptionRecord->NumberParameters < 2 ||
        exception->ExceptionRecord->ExceptionInformation[1] !=
            reinterpret_cast<ULONG_PTR>(observation->protected_page)) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    return static_cast<LONG>(
        Core::RunOnWindowsExceptionStack(RepairFaultOnExceptionStack, observation));
}

void* PS4_SYSV_ABI FaultFromGuestStack(void* opaque) {
    auto& observation = *static_cast<VehObservation*>(opaque);
    std::byte guest_marker{};
    observation.guest_marker_on_guest_stack = observation.guest_stack->Contains(&guest_marker);
    observation.guest_bounds_before = ReadRawTebStackBounds();
    *observation.protected_page = std::byte{0x2a};
    observation.resumed_after_fault = true;
    observation.guest_bounds_after = ReadRawTebStackBounds();
    return nullptr;
}

struct FlsCleanupObservation {
    DWORD slot{FLS_OUT_OF_INDEXES};
    bool callback_invoked{};
    bool callback_used_deleting_stack{};
    std::intptr_t callback_result{};
};

std::intptr_t ObserveCleanupReentry(void* opaque) noexcept {
    auto& observation = *static_cast<FlsCleanupObservation*>(opaque);
    std::byte stack_marker{};
    observation.callback_used_deleting_stack =
        Core::IsOnPreparedWindowsExceptionStack(&stack_marker);
    return 73;
}

void NTAPI FlsCleanupCallback(void* opaque) noexcept {
    auto& observation = *static_cast<FlsCleanupObservation*>(opaque);
    observation.callback_invoked = true;
    observation.callback_result =
        Core::RunOnWindowsExceptionStack(ObserveCleanupReentry, &observation);
}

std::intptr_t InstallFlsCleanupCallback(void* opaque) noexcept {
    auto& observation = *static_cast<FlsCleanupObservation*>(opaque);
    observation.slot = FlsAlloc(FlsCleanupCallback);
    if (observation.slot == FLS_OUT_OF_INDEXES ||
        FlsSetValue(observation.slot, &observation) == FALSE) {
        return 0;
    }
    return 1;
}

struct GuestExitReleaseObservation {
    RawTebStackBounds bounds_before{};
    RawTebStackBounds bounds_after{};
    bool still_a_fiber{};
};

void* PS4_SYSV_ABI ReleaseForExitFromGuestStack(void* opaque) {
    auto& observation = *static_cast<GuestExitReleaseObservation*>(opaque);
    observation.bounds_before = ReadRawTebStackBounds();
    Core::ReleaseWindowsExceptionStackForThreadExit();
    observation.bounds_after = ReadRawTebStackBounds();
    observation.still_a_fiber = IsThreadAFiber() != FALSE;
    return nullptr;
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
    EXPECT_EQ(Core::RunOnWindowsExceptionStack(ObserveOuterStack, &observation), 42);
    EXPECT_TRUE(observation.outer_on_exception_stack);
    EXPECT_TRUE(observation.nested_on_exception_stack);
    EXPECT_TRUE(observation.outer_still_on_exception_stack);
    EXPECT_TRUE(observation.outer_registered_with_windows);
    EXPECT_TRUE(observation.nested_registered_with_windows);
    EXPECT_EQ(observation.nested_result, 41);
    EXPECT_FALSE(Core::IsOnPreparedWindowsExceptionStack(
        reinterpret_cast<const void*>(observation.nested_stack_address)));
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

    EXPECT_EQ(Core::RunOnWindowsExceptionStack(ObserveNestedStack, &observation), 41);
    EXPECT_FALSE(observation.nested_on_exception_stack);
    EXPECT_TRUE(observation.nested_registered_with_windows);
    EXPECT_FALSE(Core::IsOnPreparedWindowsExceptionStack(&caller_stack_marker));
}

TEST_F(WindowsExceptionStackTest, PreservesNegativeHandlerResult) {
    ASSERT_TRUE(Core::PrepareWindowsExceptionStack());
    EXPECT_EQ(Core::RunOnWindowsExceptionStack(ReturnContinueExecution, nullptr),
              EXCEPTION_CONTINUE_EXECUTION);
}

TEST_F(WindowsExceptionStackTest, CleanupInsideHandlerLeavesActiveStackIntact) {
    ASSERT_TRUE(Core::PrepareWindowsExceptionStack());
    CleanupInsideHandlerObservation observation{};

    EXPECT_EQ(Core::RunOnWindowsExceptionStack(AttemptCleanupInsideHandler, &observation), 7);
    EXPECT_TRUE(observation.before_cleanup);
    EXPECT_TRUE(observation.after_cleanup);
}

TEST_F(WindowsExceptionStackTest, PreservesCallerFloatingPointControlState) {
    ASSERT_TRUE(Core::PrepareWindowsExceptionStack());
    ScopedMxcsrRestore restore_mxcsr;
    const auto caller_mxcsr = (_mm_getcsr() & ~_MM_ROUND_MASK) | _MM_ROUND_UP;
    _mm_setcsr(caller_mxcsr);
    MxcsrObservation observation{};

    EXPECT_EQ(Core::RunOnWindowsExceptionStack(ChangeHandlerRoundingMode, &observation), 9);
    EXPECT_EQ(_mm_getcsr(), caller_mxcsr);
    EXPECT_EQ(observation.handler_after & _MM_ROUND_MASK, _MM_ROUND_DOWN);
}

TEST_F(WindowsExceptionStackTest, SwitchesAwayFromGuestStackAndRestoresGuestAndNativeTebBounds) {
    ASSERT_TRUE(Core::PrepareWindowsExceptionStack());
    GuardedGuestStack guest_stack{64 * 1024};
    ASSERT_TRUE(guest_stack.IsValid());

    ULONG_PTR original_low{};
    ULONG_PTR original_high{};
    GetCurrentThreadStackLimits(&original_low, &original_high);
    GuestStackObservation observation{.guest_stack = &guest_stack};

    const auto result = _runOnAnotherStack(
        &observation, reinterpret_cast<void*>(InvokeFromGuestStack), guest_stack.Top());

    EXPECT_EQ(reinterpret_cast<std::intptr_t>(result), 42);
    EXPECT_EQ(observation.result, 42);
    EXPECT_TRUE(observation.guest_marker_on_guest_stack);
    EXPECT_EQ(observation.guest_bounds_before.base, 0u);
    EXPECT_EQ(observation.guest_bounds_before.limit, 0u);
    EXPECT_EQ(observation.guest_bounds_after.base, 0u);
    EXPECT_EQ(observation.guest_bounds_after.limit, 0u);
    EXPECT_TRUE(observation.callback.outer_on_exception_stack);
    EXPECT_TRUE(observation.callback.nested_on_exception_stack);
    EXPECT_FALSE(guest_stack.Contains(
        reinterpret_cast<const void*>(observation.callback.outer_stack_address)));
    EXPECT_FALSE(guest_stack.Contains(
        reinterpret_cast<const void*>(observation.callback.nested_stack_address)));
    ULONG_PTR restored_low{};
    ULONG_PTR restored_high{};
    GetCurrentThreadStackLimits(&restored_low, &restored_high);
    EXPECT_EQ(restored_low, original_low);
    EXPECT_EQ(restored_high, original_high);
}

TEST_F(WindowsExceptionStackTest, RealVehResumesFaultingGuestStackThroughPreparedStack) {
    ASSERT_TRUE(Core::PrepareWindowsExceptionStack());
    GuardedGuestStack guest_stack{64 * 1024};
    ASSERT_TRUE(guest_stack.IsValid());
    SYSTEM_INFO system_info{};
    GetSystemInfo(&system_info);
    auto* const protected_page = static_cast<std::byte*>(
        VirtualAlloc(nullptr, system_info.dwPageSize, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS));
    ASSERT_NE(protected_page, nullptr);
    auto* const veh = AddVectoredExceptionHandler(1, TestVehHandler);
    ASSERT_NE(veh, nullptr);

    VehObservation observation{
        .guest_stack = &guest_stack,
        .protected_page = protected_page,
        .page_size = system_info.dwPageSize,
    };
    active_veh_observation = &observation;
    _runOnAnotherStack(&observation, reinterpret_cast<void*>(FaultFromGuestStack),
                       guest_stack.Top());
    active_veh_observation = nullptr;

    EXPECT_TRUE(RemoveVectoredExceptionHandler(veh));
    EXPECT_TRUE(VirtualFree(protected_page, 0, MEM_RELEASE));
    EXPECT_TRUE(observation.guest_marker_on_guest_stack);
    EXPECT_TRUE(observation.handler_on_exception_stack);
    EXPECT_TRUE(observation.handler_registered_with_windows);
    EXPECT_TRUE(observation.resumed_after_fault);
    EXPECT_EQ(observation.guest_bounds_before.base, 0u);
    EXPECT_EQ(observation.guest_bounds_before.limit, 0u);
    EXPECT_EQ(observation.guest_bounds_after.base, 0u);
    EXPECT_EQ(observation.guest_bounds_after.limit, 0u);
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

TEST_F(WindowsExceptionStackTest, CleanupDoesNotUndoBorrowedFiberConversion) {
    ASSERT_FALSE(IsThreadAFiber());
    ASSERT_NE(ConvertThreadToFiberEx(nullptr, FIBER_FLAG_FLOAT_SWITCH), nullptr);

    ASSERT_TRUE(Core::PrepareWindowsExceptionStack());
    Core::CleanupWindowsExceptionStack();
    EXPECT_TRUE(IsThreadAFiber());
    EXPECT_TRUE(ConvertFiberToThread());
}

TEST_F(WindowsExceptionStackTest, PrepareRecoversAfterBorrowedFiberOwnerConvertsBackToThread) {
    ASSERT_FALSE(IsThreadAFiber());
    ASSERT_NE(ConvertThreadToFiberEx(nullptr, FIBER_FLAG_FLOAT_SWITCH), nullptr);
    ASSERT_TRUE(Core::PrepareWindowsExceptionStack());
    ASSERT_TRUE(ConvertFiberToThread());

    ASSERT_TRUE(Core::PrepareWindowsExceptionStack());
    CallbackObservation observation{};
    EXPECT_EQ(Core::RunOnWindowsExceptionStack(ObserveNestedStack, &observation), 41);
    EXPECT_TRUE(observation.nested_on_exception_stack);
    Core::CleanupWindowsExceptionStack();
    EXPECT_FALSE(IsThreadAFiber());
}

TEST_F(WindowsExceptionStackTest, CleanupInvalidatesFiberBeforeFlsCallbacksCanReenter) {
    ASSERT_TRUE(Core::PrepareWindowsExceptionStack());
    FlsCleanupObservation observation{};
    ASSERT_EQ(Core::RunOnWindowsExceptionStack(InstallFlsCleanupCallback, &observation), 1);
    ASSERT_NE(observation.slot, FLS_OUT_OF_INDEXES);

    Core::CleanupWindowsExceptionStack();

    EXPECT_TRUE(observation.callback_invoked);
    EXPECT_EQ(observation.callback_result, 73);
    EXPECT_FALSE(observation.callback_used_deleting_stack);
    EXPECT_TRUE(FlsFree(observation.slot));
}

TEST_F(WindowsExceptionStackTest, ExitReleaseIsSafeFromManualGuestStack) {
    ASSERT_TRUE(Core::PrepareWindowsExceptionStack());
    GuardedGuestStack guest_stack{64 * 1024};
    ASSERT_TRUE(guest_stack.IsValid());
    GuestExitReleaseObservation observation{};

    _runOnAnotherStack(&observation, reinterpret_cast<void*>(ReleaseForExitFromGuestStack),
                       guest_stack.Top());

    EXPECT_EQ(observation.bounds_before.base, 0u);
    EXPECT_EQ(observation.bounds_before.limit, 0u);
    EXPECT_EQ(observation.bounds_after.base, 0u);
    EXPECT_EQ(observation.bounds_after.limit, 0u);
    EXPECT_TRUE(observation.still_a_fiber);
    ASSERT_TRUE(IsThreadAFiber());
    EXPECT_TRUE(ConvertFiberToThread());
}

TEST_F(WindowsExceptionStackTest, HostThreadsUseIndependentPreparedStacks) {
    struct ThreadObservation {
        bool prepared{};
        bool on_exception_stack{};
        bool registered_with_windows{};
        bool fiber_state_restored{};
        std::intptr_t callback_result{};
        std::uintptr_t stack_address{};
    } first, second;
    std::latch observations_ready{2};
    std::latch permit_cleanup{1};

    const auto worker = [&](ThreadObservation& result) {
        const bool initially_a_fiber = IsThreadAFiber() != FALSE;
        CallbackObservation callback{};
        result.prepared = Core::PrepareWindowsExceptionStack();
        if (result.prepared) {
            result.callback_result =
                Core::RunOnWindowsExceptionStack(ObserveNestedStack, &callback);
            result.on_exception_stack = callback.nested_on_exception_stack;
            result.registered_with_windows = callback.nested_registered_with_windows;
            result.stack_address = callback.nested_stack_address;
        }
        observations_ready.count_down();
        permit_cleanup.wait();
        Core::CleanupWindowsExceptionStack();
        result.fiber_state_restored = (IsThreadAFiber() != FALSE) == initially_a_fiber;
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
    EXPECT_EQ(first.callback_result, 41);
    EXPECT_EQ(second.callback_result, 41);
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
        reinterpret_cast<DWORD64>(&Core::RunOnWindowsExceptionStack), &image_base, nullptr);

    EXPECT_NE(runtime_function, nullptr);
    EXPECT_NE(image_base, 0u);
}
