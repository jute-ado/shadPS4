// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <fstream>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/download_readback_policy.h"
#include "video_core/buffer_cache/srt_cpu_snapshot_selection.h"

namespace VideoCore {
namespace {

TEST(SrtCpuSnapshotSelection, OnlyExactOneOptsIn) {
    EXPECT_TRUE(ParseSrtCpuSnapshotSelection("1"));
    for (std::string_view value : {"", "0", "true", "01", " 1", "1 ", "1\n", "2"}) {
        EXPECT_FALSE(ParseSrtCpuSnapshotSelection(value)) << value;
    }
}

TEST(SrtCpuSnapshotSelection, DefaultRetainsNormalCompositionWithoutInspectingCapture) {
    SrtCpuSnapshotSelection policy;
    std::array<u8, 4> output{8, 9, 10, 11};
    unsigned logs{};
    EXPECT_TRUE(
        policy.Complete(true, {}, std::numeric_limits<size_t>::max(), output, [&] { ++logs; }));
    EXPECT_EQ(output, (std::array<u8, 4>{8, 9, 10, 11}));
    EXPECT_EQ(logs, 0U);
}

TEST(SrtCpuSnapshotSelection, EnabledUsesOnlyRequestedSliceAndRetainsCanaries) {
    SrtCpuSnapshotSelection policy{true};
    const std::array<u8, 8> capture{1, 2, 3, 4, 5, 6, 7, 8};
    std::array<u8, 6> output{0xaa, 9, 9, 9, 9, 0xbb};
    unsigned logs{};
    EXPECT_TRUE(
        policy.Complete(true, capture, 2, std::span{output}.subspan(1, 4), [&] { ++logs; }));
    EXPECT_EQ(output, (std::array<u8, 6>{0xaa, 3, 4, 5, 6, 0xbb}));
    EXPECT_EQ(capture, (std::array<u8, 8>{1, 2, 3, 4, 5, 6, 7, 8}));
    EXPECT_EQ(logs, 1U);
}

TEST(SrtCpuSnapshotSelection, NormalFailureCannotBeTurnedIntoSuccessOrSelection) {
    SrtCpuSnapshotSelection policy{true};
    const std::array<u8, 4> capture{1, 2, 3, 4};
    std::array<u8, 4> output{8, 9, 10, 11};
    unsigned logs{};
    EXPECT_FALSE(policy.Complete(false, capture, 0, output, [&] { ++logs; }));
    EXPECT_EQ(output, (std::array<u8, 4>{8, 9, 10, 11}));
    EXPECT_EQ(logs, 0U);
}

TEST(SrtCpuSnapshotSelection, InvalidSliceNeverPartiallyCopiesOrLogs) {
    SrtCpuSnapshotSelection policy{true};
    const std::array<u8, 4> capture{1, 2, 3, 4};
    for (const size_t offset :
         {size_t{1}, size_t{4}, size_t{5}, std::numeric_limits<size_t>::max()}) {
        std::array<u8, 4> output{8, 9, 10, 11};
        unsigned logs{};
        EXPECT_FALSE(policy.Complete(true, capture, offset, output, [&] { ++logs; })) << offset;
        EXPECT_EQ(output, (std::array<u8, 4>{8, 9, 10, 11}));
        EXPECT_EQ(logs, 0U);
    }
}

TEST(SrtCpuSnapshotSelection, EmptyOutputIsNotAnActualSelection) {
    SrtCpuSnapshotSelection policy{true};
    const std::array<u8, 4> capture{1, 2, 3, 4};
    unsigned logs{};
    EXPECT_TRUE(policy.Complete(true, capture, capture.size(), {}, [&] { ++logs; }));
    EXPECT_EQ(logs, 0U);
    std::array<u8, 4> output{};
    EXPECT_TRUE(policy.Complete(true, capture, 0, output, [&] { ++logs; }));
    EXPECT_EQ(logs, 1U);
}

TEST(SrtCpuSnapshotSelection, KeepsFinishInvalidateCaptureOverlayPendingWriteOrdering) {
    SrtCpuSnapshotSelection policy{true};
    std::array<u8, 4> capture{};
    std::array<u8, 4> output{};
    unsigned cpu_reads{};
    std::vector<std::string_view> operations;
    EXPECT_TRUE(CompletePrivateSnapshotReadback([&] { operations.emplace_back("finish"); },
                                                [&] {
                                                    operations.emplace_back("invalidate");
                                                    return true;
                                                },
                                                [&] {
                                                    operations.emplace_back("capture");
                                                    ++cpu_reads;
                                                    capture = {1, 2, 3, 4};
                                                    output = capture;
                                                    return true;
                                                },
                                                [&] {
                                                    operations.emplace_back("gpu-overlay");
                                                    output.fill(9);
                                                    operations.emplace_back("pending-cpu-writes");
                                                    output[0] = 7;
                                                    return policy.Complete(
                                                        true, capture, 0, output, [&] {
                                                            operations.emplace_back("selected");
                                                        });
                                                }));
    EXPECT_EQ(operations,
              (std::vector<std::string_view>{"finish", "invalidate", "capture", "gpu-overlay",
                                             "pending-cpu-writes", "selected"}));
    EXPECT_EQ(cpu_reads, 1U);
    EXPECT_EQ(output, capture);
}

TEST(SrtCpuSnapshotSelection, RetainsEveryNormalReadbackFailureAndShortCircuit) {
    for (int fail_at = 0; fail_at < 4; ++fail_at) {
        SrtCpuSnapshotSelection policy{true};
        const std::array<u8, 4> capture{1, 2, 3, 4};
        std::array<u8, 4> output{8, 9, 10, 11};
        std::vector<int> calls;
        unsigned logs{};
        const auto step = [&](int index) {
            calls.push_back(index);
            return index != fail_at;
        };
        EXPECT_FALSE(CompletePrivateSnapshotReadback(
            [&] { calls.push_back(-1); }, [&] { return step(0); }, [&] { return step(1); },
            [&] {
                const bool completed = step(2) && step(3);
                return policy.Complete(completed, capture, 0, output, [&] { ++logs; });
            }));
        std::vector<int> expected{-1};
        for (int i = 0; i <= fail_at; ++i) {
            expected.push_back(i);
        }
        EXPECT_EQ(calls, expected);
        EXPECT_EQ(logs, 0U);
        EXPECT_EQ(output, (std::array<u8, 4>{8, 9, 10, 11}));
    }
}

TEST(SrtCpuSnapshotSelection, OneAttemptedLogOnlyAfterActualSuccessfulSelection) {
    SrtCpuSnapshotSelection policy{true};
    const std::array<u8, 4> capture{1, 2, 3, 4};
    std::array<u8, 4> output{};
    unsigned logs{};
    const auto log = [&] { ++logs; };
    EXPECT_FALSE(policy.Complete(false, capture, 0, output, log));
    EXPECT_TRUE(policy.Complete(true, capture, 0, output, log));
    output.fill(9);
    EXPECT_TRUE(policy.Complete(true, capture, 0, output, log));
    EXPECT_EQ(output, capture);
    EXPECT_EQ(logs, 1U);
}

TEST(SrtCpuSnapshotSelection, LoggingFailureDoesNotChangeSuccessfulSelectionOrRetryLogging) {
    SrtCpuSnapshotSelection policy{true};
    const std::array<u8, 4> capture{1, 2, 3, 4};
    std::array<u8, 4> output{};
    unsigned logs{};
    EXPECT_TRUE(policy.Complete(true, capture, 0, output, [&] {
        ++logs;
        throw std::runtime_error{"diagnostic sink"};
    }));
    EXPECT_EQ(output, capture);
    EXPECT_TRUE(policy.Complete(true, capture, 0, output, [&] { ++logs; }));
    EXPECT_EQ(logs, 1U);
}

TEST(SrtCpuSnapshotSelection, ConcurrentOwnedOutputsAttemptAtMostOneLog) {
    SrtCpuSnapshotSelection policy{true};
    const std::array<u8, 4> capture{1, 2, 3, 4};
    std::array<std::array<u8, 4>, 8> output{};
    std::atomic_uint logs{};
    std::atomic_uint successful{};
    std::vector<std::jthread> workers;
    for (auto& bytes : output) {
        workers.emplace_back([&, target = std::span{bytes}] {
            if (policy.Complete(true, capture, 0, target, [&] { ++logs; })) {
                ++successful;
            }
        });
    }
    workers.clear();
    EXPECT_EQ(successful.load(), output.size());
    EXPECT_EQ(logs.load(), 1U);
    for (const auto& bytes : output) {
        EXPECT_EQ(bytes, capture);
    }
}

TEST(SrtCpuSnapshotSelection, SourceUsesOnlySuccessfulPrivateReadbackComposition) {
    std::ifstream input{SHADPS4_BUFFER_CACHE_SOURCE_PATH, std::ios::binary};
    ASSERT_TRUE(input);
    std::ostringstream stream;
    stream << input.rdbuf();
    auto text = stream.str();
    text.erase(std::remove_if(text.begin(), text.end(),
                              [](unsigned char ch) {
                                  return ch == ' ' || ch == '\r' || ch == '\n' || ch == '\t';
                              }),
               text.end());
    const auto begin = text.find("boolBufferCache::DownloadBufferMemorySnapshot(");
    ASSERT_NE(begin, std::string::npos);
    const auto end = text.find("BufferCache::PreparedVertexBuffers", begin);
    ASSERT_NE(end, std::string::npos);
    const auto function = text.substr(begin, end - begin);
    const auto finish = function.find("scheduler.Finish()");
    const auto invalidate = function.find("download_buffer.InvalidateMappedRange", finish);
    const auto capture = function.find("memory->CopySparseMemory(cpu_capture_address", invalidate);
    const auto overlay = function.find("ApplyMemorySnapshotOverlays(", capture);
    const auto pending = function.find("cpu_page_write_tracker.ApplyPendingWrites(", overlay);
    const auto select = function.find("GetSrtCpuSnapshotSelection().Complete(", pending);
    ASSERT_NE(finish, std::string::npos);
    ASSERT_NE(invalidate, std::string::npos);
    ASSERT_NE(capture, std::string::npos);
    ASSERT_NE(overlay, std::string::npos);
    ASSERT_NE(pending, std::string::npos);
    ASSERT_NE(select, std::string::npos);
    EXPECT_NE(
        function.find("normal_completed,cpu_capture,device_addr-cpu_capture_address,destination",
                      select),
        std::string::npos);
    EXPECT_EQ(text.find("GetSrtCpuSnapshotSelection().Complete("), begin + select);
    EXPECT_EQ(text.find("GetSrtCpuSnapshotSelection().Complete(", begin + select + 1),
              std::string::npos);
}

} // namespace
} // namespace VideoCore
