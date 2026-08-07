// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <future>
#include <latch>

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

    auto signal = std::async(std::launch::async,
                             [&] { controller.Signal(Platform::InterruptId::GfxEop); });
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
