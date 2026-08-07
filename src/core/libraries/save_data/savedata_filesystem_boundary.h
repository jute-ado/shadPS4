// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <functional>
#include <utility>

#include "common/types.h"
#include "savedata_error.h"

namespace Libraries::SaveData {

template <typename Operation, typename FailureHandler>
Error RunFilesystemOperation(Operation&& operation, FailureHandler&& failure_handler) {
    try {
        std::invoke(std::forward<Operation>(operation));
        return Error::OK;
    } catch (const std::filesystem::filesystem_error& error) {
        std::invoke(std::forward<FailureHandler>(failure_handler), error);
        return Error::INTERNAL;
    }
}

} // namespace Libraries::SaveData
