// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <sstream>

#include <gtest/gtest.h>
#include <spdlog/sinks/ostream_sink.h>

#include "common/logging/log.h"

namespace Common {
std::string GetCurrentThreadName() {
    return "shadPS4::LogOnceTest";
}
} // namespace Common

namespace Common::Log {
std::unordered_map<std::string_view, std::shared_ptr<spdlog::logger>> ALL_LOGGERS;
} // namespace Common::Log

namespace {

void LogPollingStub() {
    LOG_ERROR_ONCE(Lib_SystemGesture, "(STUBBED) polling call");
}

void LogAnotherPollingStub() {
    LOG_ERROR_ONCE(Lib_SystemGesture, "(STUBBED) another polling call");
}

TEST(LogOnce, EmitsOnlyTheFirstInvocationAtEachCallSite) {
    std::ostringstream output;
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(output);
    auto logger = std::make_shared<spdlog::logger>("log-once-test", sink);
    logger->set_pattern("%v");
    Common::Log::ALL_LOGGERS[Common::Log::Class::Lib_SystemGesture] = logger;

    for (int invocation = 0; invocation < 100; ++invocation) {
        LogPollingStub();
        LogAnotherPollingStub();
    }
    logger->flush();

    const std::string text = output.str();
    const auto first = text.find("(STUBBED) polling call");
    ASSERT_NE(first, std::string::npos);
    EXPECT_EQ(text.find("(STUBBED) polling call", first + 1), std::string::npos);
    const auto another_first = text.find("(STUBBED) another polling call");
    ASSERT_NE(another_first, std::string::npos);
    EXPECT_EQ(text.find("(STUBBED) another polling call", another_first + 1),
              std::string::npos);
}

} // namespace
