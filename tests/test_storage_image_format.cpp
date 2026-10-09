// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "shader_recompiler/backend/spirv/storage_image_format_policy.h"

TEST(StorageImageFormat, ReadOnlySampledImagesRemainFormatless) {
    EXPECT_FALSE(Shader::Backend::SPIRV::NeedsStorageImageFormat(false, false));
}

TEST(StorageImageFormat, WritableImagesUseTheirSpecializedFormat) {
    EXPECT_TRUE(Shader::Backend::SPIRV::NeedsStorageImageFormat(true, false));
}

TEST(StorageImageFormat, AtomicImagesUseTheirSpecializedFormat) {
    EXPECT_TRUE(Shader::Backend::SPIRV::NeedsStorageImageFormat(false, true));
}
