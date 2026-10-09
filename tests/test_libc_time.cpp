// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <limits>

#include "core/libraries/libc_internal/libc_internal_time.h"

namespace Libraries::LibcInternal {
namespace {

static_assert(sizeof(OrbisTm) == 36);

TEST(LibcTime, GmtimeSUsesGuestArgumentOrderAndUnixEpoch) {
    const s64 seconds = 0;
    OrbisTm result{};

    ASSERT_EQ(internal_gmtime_s(&seconds, &result), 0);
    EXPECT_EQ(result.tm_sec, 0);
    EXPECT_EQ(result.tm_min, 0);
    EXPECT_EQ(result.tm_hour, 0);
    EXPECT_EQ(result.tm_mday, 1);
    EXPECT_EQ(result.tm_mon, 0);
    EXPECT_EQ(result.tm_year, 70);
    EXPECT_EQ(result.tm_wday, 4);
    EXPECT_EQ(result.tm_yday, 0);
    EXPECT_EQ(result.tm_isdst, 0);
}

TEST(LibcTime, GmtimeSHandlesGregorianLeapDays) {
    const s64 seconds = 951827696; // 2000-02-29 12:34:56 UTC
    OrbisTm result{};

    ASSERT_EQ(internal_gmtime_s(&seconds, &result), 0);
    EXPECT_EQ(result.tm_sec, 56);
    EXPECT_EQ(result.tm_min, 34);
    EXPECT_EQ(result.tm_hour, 12);
    EXPECT_EQ(result.tm_mday, 29);
    EXPECT_EQ(result.tm_mon, 1);
    EXPECT_EQ(result.tm_year, 100);
    EXPECT_EQ(result.tm_wday, 2);
    EXPECT_EQ(result.tm_yday, 59);
    EXPECT_EQ(result.tm_isdst, 0);
}

TEST(LibcTime, GmtimeSSupportsTimesBeforeTheUnixEpoch) {
    const s64 seconds = -1;
    OrbisTm result{};

    ASSERT_EQ(internal_gmtime_s(&seconds, &result), 0);
    EXPECT_EQ(result.tm_sec, 59);
    EXPECT_EQ(result.tm_min, 59);
    EXPECT_EQ(result.tm_hour, 23);
    EXPECT_EQ(result.tm_mday, 31);
    EXPECT_EQ(result.tm_mon, 11);
    EXPECT_EQ(result.tm_year, 69);
    EXPECT_EQ(result.tm_wday, 3);
    EXPECT_EQ(result.tm_yday, 364);
    EXPECT_EQ(result.tm_isdst, 0);
}

TEST(LibcTime, GmtimeSRejectsNullArguments) {
    const s64 seconds = 0;
    OrbisTm result{};

    EXPECT_EQ(internal_gmtime_s(nullptr, &result), 22);
    EXPECT_EQ(internal_gmtime_s(&seconds, nullptr), 22);
}

TEST(LibcTime, GmtimeSRejectsUnrepresentableGuestYearsWithoutChangingOutput) {
    const s64 seconds = std::numeric_limits<s64>::max();
    const OrbisTm sentinel{
        .tm_sec = 1,
        .tm_min = 2,
        .tm_hour = 3,
        .tm_mday = 4,
        .tm_mon = 5,
        .tm_year = 6,
        .tm_wday = 0,
        .tm_yday = 7,
        .tm_isdst = 8,
    };
    OrbisTm result = sentinel;

    EXPECT_EQ(internal_gmtime_s(&seconds, &result), 84);
    EXPECT_EQ(result.tm_sec, sentinel.tm_sec);
    EXPECT_EQ(result.tm_min, sentinel.tm_min);
    EXPECT_EQ(result.tm_hour, sentinel.tm_hour);
    EXPECT_EQ(result.tm_mday, sentinel.tm_mday);
    EXPECT_EQ(result.tm_mon, sentinel.tm_mon);
    EXPECT_EQ(result.tm_year, sentinel.tm_year);
    EXPECT_EQ(result.tm_wday, sentinel.tm_wday);
    EXPECT_EQ(result.tm_yday, sentinel.tm_yday);
    EXPECT_EQ(result.tm_isdst, sentinel.tm_isdst);
}

} // namespace
} // namespace Libraries::LibcInternal
