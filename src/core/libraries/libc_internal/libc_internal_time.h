// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>

#include "common/types.h"
#include "core/libraries/kernel/posix_error.h"

namespace Core::Loader {
class SymbolsResolver;
}

namespace Libraries::LibcInternal {

// The PS4 ABI uses the nine-field C tm layout, without host-specific timezone extensions.
struct OrbisTm {
    s32 tm_sec;
    s32 tm_min;
    s32 tm_hour;
    s32 tm_mday;
    s32 tm_mon;
    s32 tm_year;
    s32 tm_wday;
    s32 tm_yday;
    s32 tm_isdst;
};
static_assert(sizeof(OrbisTm) == 36);

namespace Detail {

struct CivilDate {
    s64 year;
    s32 month;
    s32 day;
};

[[nodiscard]] constexpr CivilDate CivilFromDays(s64 days_since_epoch) noexcept {
    const s64 z = days_since_epoch + 719468;
    const s64 era = (z >= 0 ? z : z - 146096) / 146097;
    const s64 day_of_era = z - era * 146097;
    const s64 year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36524 - day_of_era / 146096) /
        365;
    s64 year = year_of_era + era * 400;
    const s64 day_of_year =
        day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    const s64 month_prime = (5 * day_of_year + 2) / 153;
    const s32 day = static_cast<s32>(day_of_year - (153 * month_prime + 2) / 5 + 1);
    const s32 month = static_cast<s32>(month_prime + (month_prime < 10 ? 3 : -9));
    year += month <= 2;
    return {.year = year, .month = month, .day = day};
}

[[nodiscard]] constexpr s64 DaysFromCivil(s64 year, s32 month, s32 day) noexcept {
    year -= month <= 2;
    const s64 era = (year >= 0 ? year : year - 399) / 400;
    const s64 year_of_era = year - era * 400;
    const s64 month_prime = month + (month > 2 ? -3 : 9);
    const s64 day_of_year = (153 * month_prime + 2) / 5 + day - 1;
    const s64 day_of_era =
        year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    return era * 146097 + day_of_era - 719468;
}

} // namespace Detail

inline s32 PS4_SYSV_ABI internal_gmtime_s(const s64* timer, OrbisTm* result) noexcept {
    if (timer == nullptr || result == nullptr) {
        return POSIX_EINVAL;
    }

    constexpr s64 SecondsPerDay = 24 * 60 * 60;
    s64 days = *timer / SecondsPerDay;
    s64 seconds_in_day = *timer % SecondsPerDay;
    if (seconds_in_day < 0) {
        seconds_in_day += SecondsPerDay;
        --days;
    }

    const auto date = Detail::CivilFromDays(days);
    const s64 years_since_1900 = date.year - 1900;
    if (years_since_1900 < std::numeric_limits<s32>::min() ||
        years_since_1900 > std::numeric_limits<s32>::max()) {
        return POSIX_EOVERFLOW;
    }

    s64 weekday = (days + 4) % 7;
    if (weekday < 0) {
        weekday += 7;
    }

    OrbisTm converted{};
    converted.tm_sec = static_cast<s32>(seconds_in_day % 60);
    converted.tm_min = static_cast<s32>((seconds_in_day / 60) % 60);
    converted.tm_hour = static_cast<s32>(seconds_in_day / (60 * 60));
    converted.tm_mday = date.day;
    converted.tm_mon = date.month - 1;
    converted.tm_year = static_cast<s32>(years_since_1900);
    converted.tm_wday = static_cast<s32>(weekday);
    converted.tm_yday =
        static_cast<s32>(days - Detail::DaysFromCivil(date.year, 1, 1));
    converted.tm_isdst = 0;
    *result = converted;
    return 0;
}

void RegisterlibSceLibcInternalTime(Core::Loader::SymbolsResolver* sym);

} // namespace Libraries::LibcInternal
