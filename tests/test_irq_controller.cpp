// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <future>
#include <latch>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "core/platform.h"

namespace Common {
std::string GetCurrentThreadName() {
    return "shadPS4::IrqControllerTest";
}
} // namespace Common

namespace Common::Log {
std::unordered_map<std::string_view, std::shared_ptr<spdlog::logger>> ALL_LOGGERS;
} // namespace Common::Log

void assert_fail_impl() {
    std::abort();
}

using namespace std::chrono_literals;

TEST(IrqController, DeliversConcurrentSignalsForDistinctInterrupts) {
    constexpr u32 NumInterrupts = 16;
    constexpr u32 NumRounds = 64;

    for (u32 round = 0; round < NumRounds; ++round) {
        Platform::IrqController controller;
        std::array<std::atomic<u32>, NumInterrupts> deliveries{};
        std::atomic<u32> wrong_interrupts{};
        std::barrier start{NumInterrupts};
        std::barrier registered{NumInterrupts};
        std::vector<std::jthread> workers;
        workers.reserve(NumInterrupts);

        for (u32 index = 0; index < NumInterrupts; ++index) {
            workers.emplace_back([&, index] {
                const auto irq = static_cast<Platform::InterruptId>(index);
                void* const uid = &deliveries[index];

                start.arrive_and_wait();
                controller.Register(
                    irq,
                    [&](Platform::InterruptId delivered_irq) {
                        if (delivered_irq != irq) {
                            wrong_interrupts.fetch_add(1, std::memory_order_relaxed);
                        }
                        deliveries[index].fetch_add(1, std::memory_order_relaxed);
                    },
                    uid);
                registered.arrive_and_wait();
                controller.Signal(irq);
                controller.Unregister(irq, uid);
            });
        }

        workers.clear();
        for (const auto& count : deliveries) {
            EXPECT_EQ(count.load(std::memory_order_relaxed), 1u);
        }
        EXPECT_EQ(wrong_interrupts.load(std::memory_order_relaxed), 0u);
    }
}

TEST(IrqController, RegistrationChangesDoNotWaitForAnActiveHandler) {
    Platform::IrqController controller;
    std::latch handler_entered{1};
    std::latch release_handler{1};
    int registration{};

    controller.Register(
        Platform::InterruptId::GfxEop,
        [&](Platform::InterruptId irq) {
            EXPECT_EQ(irq, Platform::InterruptId::GfxEop);
            handler_entered.count_down();
            release_handler.wait();
        },
        &registration);

    auto signal =
        std::async(std::launch::async, [&] { controller.Signal(Platform::InterruptId::GfxEop); });
    handler_entered.wait();

    auto unregister = std::async(std::launch::async, [&] {
        controller.Unregister(Platform::InterruptId::GfxEop, &registration);
    });
    const auto unregister_status = unregister.wait_for(100ms);

    release_handler.count_down();
    signal.get();
    unregister.get();

    EXPECT_EQ(unregister_status, std::future_status::ready);
}

TEST(IrqController, PersistentHandlerCanUnregisterItself) {
    Platform::IrqController controller;
    constexpr auto irq = Platform::InterruptId::GfxEop;
    u32 deliveries = 0;
    void* const uid = &deliveries;

    controller.Register(
        irq,
        [&](Platform::InterruptId delivered_irq) {
            EXPECT_EQ(delivered_irq, irq);
            ++deliveries;
            controller.Unregister(irq, uid);
        },
        uid);

    controller.Signal(irq);
    controller.Signal(irq);

    EXPECT_EQ(deliveries, 1u);
}
