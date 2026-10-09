// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <functional>
#include <type_traits>
#include <utility>

#include "common/types.h"
#include "savedata_error.h"

namespace Libraries::SaveData {

template <typename Operation, typename FailureHandler>
Error RunFilesystemOperation(Operation&& operation, FailureHandler&& failure_handler) {
    try {
        using OperationResult = std::invoke_result_t<Operation>;
        if constexpr (std::is_same_v<std::remove_cvref_t<OperationResult>, Error>) {
            return std::invoke(std::forward<Operation>(operation));
        } else {
            static_assert(std::is_void_v<OperationResult>);
            std::invoke(std::forward<Operation>(operation));
            return Error::OK;
        }
    } catch (const std::filesystem::filesystem_error& error) {
        using FailureResult =
            std::invoke_result_t<FailureHandler, const std::filesystem::filesystem_error&>;
        if constexpr (std::is_same_v<std::remove_cvref_t<FailureResult>, Error>) {
            return std::invoke(std::forward<FailureHandler>(failure_handler), error);
        } else {
            static_assert(std::is_void_v<FailureResult>);
            std::invoke(std::forward<FailureHandler>(failure_handler), error);
            return Error::INTERNAL;
        }
    }
}

} // namespace Libraries::SaveData
