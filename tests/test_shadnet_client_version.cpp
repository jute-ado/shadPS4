// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <string>

#include <gtest/gtest.h>

#include "shadnet/client_version.h"

TEST(ShadNetClientVersion, DevelopmentBuildDoesNotDiscloseForkMetadata) {
    const ShadNet::ClientBuildMetadata metadata{
        .version = "0.19.0",
        .is_release = false,
        .revision = "private-revision",
        .branch = "investigation/private-game-route",
        .description = "private-description-dirty",
        .remote_name = "personal-origin",
        .remote_url = "https://example.invalid/private-user/private-fork.git",
    };

    const std::string version = ShadNet::BuildNetworkClientVersion(metadata);

    EXPECT_EQ(version, "shadPS4 v0.19.0");
    for (const std::string_view private_value :
         std::array{metadata.revision, metadata.branch, metadata.description,
                    metadata.remote_name, metadata.remote_url}) {
        EXPECT_EQ(version.find(private_value), std::string::npos);
    }
}

TEST(ShadNetClientVersion, ReleaseBuildUsesTheSamePublicVersionShape) {
    const ShadNet::ClientBuildMetadata metadata{
        .version = "0.19.0",
        .is_release = true,
        .revision = "release-revision",
        .branch = "release-branch",
        .description = "release-description",
        .remote_name = "upstream",
        .remote_url = "https://example.invalid/upstream/repository.git",
    };

    EXPECT_EQ(ShadNet::BuildNetworkClientVersion(metadata), "shadPS4 v0.19.0");
}
