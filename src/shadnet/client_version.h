// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>
#include <string_view>

namespace ShadNet {

struct ClientBuildMetadata {
    std::string_view version;
    bool is_release;
    std::string_view revision;
    std::string_view branch;
    std::string_view description;
    std::string_view remote_name;
    std::string_view remote_url;
};

inline std::string BuildNetworkClientVersion(const ClientBuildMetadata& metadata) {
    std::string result{"shadPS4 v"};
    result.append(metadata.version);
    return result;
}

} // namespace ShadNet
