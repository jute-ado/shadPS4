// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging/log.h"

namespace Common::Log {

std::unordered_map<std::string_view, std::shared_ptr<spdlog::logger>> ALL_LOGGERS{};

} // namespace Common::Log
