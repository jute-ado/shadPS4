// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <fstream>
#include <future>
#include <iterator>
#include <semaphore>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "core/libraries/videoout/driver.h"

namespace {

using namespace std::chrono_literals;
using Libraries::VideoOut::VideoOutPort;

std::string ReadText(const char* path) {
    std::ifstream stream{path, std::ios::binary};
    return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
}

TEST(VideoOutLabelWait, ConcurrentGpuWorkInterruptsTheHardWaitWithoutPolling) {
    VideoOutPort port{};
    std::atomic<u64> work_generation{};
    std::atomic_bool label_ready{};
    std::atomic_uint predicate_calls{};
    std::binary_semaphore wait_entered{0};
    std::promise<bool> returned_for_label;
    auto returned = returned_for_label.get_future();

    std::jthread waiter{[&] {
        const u64 observed_generation = work_generation.load(std::memory_order_acquire);
        port.WaitVoLabel(
            [&] {
                if (predicate_calls.fetch_add(1, std::memory_order_relaxed) == 0) {
                    wait_entered.release();
                }
                return label_ready.load(std::memory_order_acquire);
            },
            [&] { return work_generation.load(std::memory_order_acquire) != observed_generation; });
        returned_for_label.set_value(label_ready.load(std::memory_order_acquire));
    }};

    wait_entered.acquire();
    EXPECT_EQ(returned.wait_for(20ms), std::future_status::timeout);

    work_generation.fetch_add(1, std::memory_order_release);
    port.InterruptVoWait();
    const auto status = returned.wait_for(1s);

    if (status != std::future_status::ready) {
        label_ready.store(true, std::memory_order_release);
    port.InterruptVoWait();
    }
    waiter.join();

    ASSERT_EQ(status, std::future_status::ready);
    EXPECT_FALSE(returned.get());
    EXPECT_LT(predicate_calls.load(std::memory_order_relaxed), 16u);
}

TEST(VideoOutLabelWait, ConcurrentGpuWorkArrivingBeforeTheWaitIsNotLost) {
    VideoOutPort port{};
    std::atomic<u64> work_generation{};
    const u64 observed_generation = work_generation.load(std::memory_order_acquire);
    std::atomic_uint predicate_calls{};

    work_generation.fetch_add(1, std::memory_order_release);
    port.WaitVoLabel([] { return false; },
                     [&] {
                         predicate_calls.fetch_add(1, std::memory_order_relaxed);
                         return work_generation.load(std::memory_order_acquire) !=
                                observed_generation;
                     });

    EXPECT_EQ(predicate_calls.load(std::memory_order_relaxed), 1u);
}

TEST(VideoOutLabelWait, HardWaitReturnsAfterABoundedIntervalWithoutAnyNotification) {
    VideoOutPort port{};
    std::atomic_uint predicate_calls{};
    const auto start = std::chrono::steady_clock::now();

    const bool completed = port.WaitVoLabelFor(
        [&] {
            predicate_calls.fetch_add(1, std::memory_order_relaxed);
            return false;
        },
        [] { return false; }, 5ms);

    const auto elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_FALSE(completed);
    EXPECT_GE(predicate_calls.load(std::memory_order_relaxed), 2u);
    EXPECT_GE(elapsed, 1ms);
    EXPECT_LT(elapsed, 1s);
}

TEST(VideoOutLabelWait, LiverpoolWakesTheWaitForIndependentlyRunnableWork) {
    const auto header = ReadText(SHADPS4_LIVERPOOL_HEADER_PATH);
    const auto source = ReadText(SHADPS4_LIVERPOOL_SOURCE_PATH);
    ASSERT_FALSE(header.empty());
    ASSERT_FALSE(source.empty());

    EXPECT_NE(header.find("concurrent_work_generation"), std::string::npos);
    EXPECT_NE(header.find("NotifyConcurrentWorkAvailable()"), std::string::npos);

    const auto send_command = header.find("void SendCommand(");
    const auto compute_registers = header.find("inline ComputeProgram& GetCsRegs()", send_command);
    ASSERT_NE(send_command, std::string::npos);
    ASSERT_NE(compute_registers, std::string::npos);
    const auto command_body = header.substr(send_command, compute_registers - send_command);
    const auto first_command_wake = command_body.find("NotifyConcurrentWorkAvailable()");
    ASSERT_NE(first_command_wake, std::string::npos);
    const auto second_command_wake =
        command_body.find("NotifyConcurrentWorkAvailable()", first_command_wake + 1);
    EXPECT_NE(second_command_wake, std::string::npos);

    const auto graphics_only_probe = source.find("bool Liverpool::HasOnlyGraphicsSubmissions()");
    const auto process = source.find("void Liverpool::Process(", graphics_only_probe);
    ASSERT_NE(graphics_only_probe, std::string::npos);
    ASSERT_NE(process, std::string::npos);
    const auto probe_body = source.substr(graphics_only_probe, process - graphics_only_probe);
    EXPECT_NE(probe_body.find("num_commands.load("), std::string::npos);

    const auto wake_helper = source.find("void Liverpool::NotifyConcurrentWorkAvailable()");
    ASSERT_NE(wake_helper, std::string::npos);
    ASSERT_LT(wake_helper, graphics_only_probe);
    const auto wake_body = source.substr(wake_helper, graphics_only_probe - wake_helper);
    EXPECT_NE(wake_body.find("concurrent_work_generation.fetch_add("), std::string::npos);
    EXPECT_NE(wake_body.find("vo_port->InterruptVoWait()"), std::string::npos);

    const auto compute_submit = source.find("void Liverpool::SubmitAsc(");
    const auto submit_done = source.find("void Liverpool::SubmitDone(", compute_submit);
    ASSERT_NE(compute_submit, std::string::npos);
    ASSERT_NE(submit_done, std::string::npos);
    const auto compute_body = source.substr(compute_submit, submit_done - compute_submit);
    EXPECT_NE(compute_body.find("NotifyConcurrentWorkAvailable()"), std::string::npos);

    const auto hard_wait = source.find("case PM4ItOpcode::WaitRegMem:");
    const auto next_packet = source.find("case PM4ItOpcode::IndirectBuffer:", hard_wait);
    ASSERT_NE(hard_wait, std::string::npos);
    ASSERT_NE(next_packet, std::string::npos);
    const auto wait_body = source.substr(hard_wait, next_packet - hard_wait);
    const auto generation = wait_body.find("concurrent_work_generation.load(");
    const auto idle_probe = wait_body.find("HasOnlyGraphicsSubmissions()", generation);
    const auto wait = wait_body.find("WaitVoLabel(", idle_probe);
    const auto bounded_wait = wait_body.find("WaitVoLabelFor(", idle_probe);
    ASSERT_NE(generation, std::string::npos);
    ASSERT_NE(idle_probe, std::string::npos);
    EXPECT_EQ(wait, std::string::npos);
    ASSERT_NE(bounded_wait, std::string::npos);
    EXPECT_LT(generation, idle_probe);
    EXPECT_LT(idle_probe, bounded_wait);
}

} // namespace
