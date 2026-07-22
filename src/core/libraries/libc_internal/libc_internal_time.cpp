// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/libraries/libs.h"
#include "libc_internal_time.h"

namespace Libraries::LibcInternal {

void RegisterlibSceLibcInternalTime(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("5bBacGLyLOs", "libSceLibcInternal", 1, "libSceLibcInternal", internal_gmtime_s);
}

} // namespace Libraries::LibcInternal
