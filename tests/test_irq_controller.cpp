// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <barrier>
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
                    irq, [&](Platform::InterruptId delivered_irq) {
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
