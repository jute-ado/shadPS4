// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "core/windows_exception_policy.h"

namespace {

std::string ReadSource(const char* path) {
    std::ifstream input{path, std::ios::binary};
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::string FunctionBody(const std::string& source, const std::string& signature) {
    const auto signature_pos = source.find(signature);
    if (signature_pos == std::string::npos) {
        return {};
    }
    const auto open = source.find('{', signature_pos + signature.size());
    if (open == std::string::npos) {
        return {};
    }
    u32 depth = 0;
    for (auto pos = open; pos < source.size(); ++pos) {
        if (source[pos] == '{') {
            ++depth;
        } else if (source[pos] == '}' && --depth == 0) {
            return source.substr(open, pos - open + 1);
        }
    }
    return {};
}

TEST(WindowsExceptionPolicy, CaughtMsvcCppExceptionDoesNotBeginEmulatorShutdown) {
    EXPECT_FALSE(Core::WindowsException::ShouldShutdownForUnclaimedException(
        Core::WindowsException::MsvcCppExceptionCode));
}

TEST(WindowsExceptionPolicy, UnclaimedAccessViolationStillBeginsEmulatorShutdown) {
    EXPECT_TRUE(Core::WindowsException::ShouldShutdownForUnclaimedException(0xc0000005));
}

TEST(WindowsExceptionPolicy, BreakpointRemainsOwnedByAssertPath) {
    EXPECT_FALSE(Core::WindowsException::ShouldShutdownForUnclaimedException(
        Core::WindowsException::BreakpointExceptionCode));
}

TEST(WindowsExceptionPolicy, StaticProtectionReportsOnlyItsOwnAccessViolations) {
    EXPECT_FALSE(Core::WindowsException::ShouldReportUnhandledException(
        0xc0000005, true, false));
    EXPECT_TRUE(Core::WindowsException::ShouldReportUnhandledException(
        0xc0000005, true, true));
}

TEST(WindowsExceptionPolicy, LegacyProtectionStillReportsUnhandledAccessViolations) {
    EXPECT_TRUE(Core::WindowsException::ShouldReportUnhandledException(
        0xc0000005, false, false));
}

TEST(WindowsExceptionPolicy, CppExceptionsRemainOwnedByTheLanguageRuntime) {
    EXPECT_FALSE(Core::WindowsException::ShouldReportUnhandledException(
        Core::WindowsException::MsvcCppExceptionCode, false, false));
    EXPECT_FALSE(Core::WindowsException::ShouldReportUnhandledException(
        Core::WindowsException::MsvcCppExceptionCode, true, true));
}

TEST(WindowsExceptionPolicy, QuickExitShutdownKeepsHandlersUntilThreadsAreTerminated) {
    const auto source = ReadSource(SHADPS4_EMULATOR_SOURCE_PATH);
    const auto shutdown = FunctionBody(source, "void Emulator::Shutdown()");

    ASSERT_FALSE(shutdown.empty());
    EXPECT_EQ(shutdown.find("RemoveHandlers("), std::string::npos);
}

TEST(WindowsExceptionPolicy, SignalDispatcherDestructionOwnsHandlerRemoval) {
    const auto source = ReadSource(SHADPS4_SIGNALS_SOURCE_PATH);
    const auto destructor = FunctionBody(source, "SignalDispatch::~SignalDispatch()");

    ASSERT_FALSE(destructor.empty());
    EXPECT_NE(destructor.find("RemoveHandlers("), std::string::npos);
}

TEST(WindowsExceptionPolicy, VectoredHandlerRunsBodyOnPreparedExceptionStack) {
    const auto source = ReadSource(SHADPS4_SIGNALS_SOURCE_PATH);
    const auto handler = FunctionBody(source, "static LONG WINAPI SignalHandler(");

    ASSERT_FALSE(handler.empty());
    EXPECT_NE(handler.find("RunOnWindowsExceptionStack("), std::string::npos);
    EXPECT_NE(handler.find("SignalHandlerBody"), std::string::npos);
}

TEST(WindowsExceptionPolicy, NativeGuestThreadsOwnPreparedExceptionStackLifetime) {
    const auto source = ReadSource(SHADPS4_THREAD_SOURCE_PATH);
    const auto initialize = FunctionBody(source, "void NativeThread::Initialize()");
    const auto exit = FunctionBody(source, "void NativeThread::Exit()");

    ASSERT_FALSE(initialize.empty());
    ASSERT_FALSE(exit.empty());
    EXPECT_NE(initialize.find("PrepareWindowsExceptionStack("), std::string::npos);
    const auto cleanup = exit.find("CleanupWindowsExceptionStack(");
    const auto terminate = exit.find("ExitThread(");
    ASSERT_NE(cleanup, std::string::npos);
    ASSERT_NE(terminate, std::string::npos);
    EXPECT_LT(cleanup, terminate);
}

TEST(WindowsExceptionPolicy, AssertPathRemovesHandlersImmediatelyBeforeBreakpointCrash) {
    const auto source = ReadSource(SHADPS4_ASSERT_SOURCE_PATH);
    const auto assert_fail = FunctionBody(source, "void assert_fail_impl()");

    ASSERT_FALSE(assert_fail.empty());
    const auto remove_handlers = assert_fail.find("RemoveHandlers(");
    const auto crash = assert_fail.find("Crash()");
    ASSERT_NE(remove_handlers, std::string::npos);
    ASSERT_NE(crash, std::string::npos);
    EXPECT_LT(remove_handlers, crash);
    EXPECT_EQ(assert_fail.find("Emulator>::Instance()->Shutdown()"), std::string::npos);
}

} // namespace
