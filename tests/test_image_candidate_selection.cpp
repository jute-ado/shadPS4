// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include <gtest/gtest.h>

#include "video_core/texture_cache/image_candidate_selection.h"

namespace VideoCore {

TEST(ImageCandidateSelection, ContainingParentWinsOverExactStandaloneMip) {
    const std::array candidates{
        ImageCandidateMatch{.exact = true},
        ImageCandidateMatch{.parent_mip = 1, .parent_slice = 0},
    };

    const auto selection = SelectCanonicalImageCandidate(candidates);

    ASSERT_TRUE(selection);
    EXPECT_EQ(selection->index, 1u);
    EXPECT_EQ(selection->mip, 1);
    EXPECT_EQ(selection->slice, 0);
}

TEST(ImageCandidateSelection, ExactImageWinsWithoutContainingParent) {
    const std::array candidates{
        ImageCandidateMatch{},
        ImageCandidateMatch{.exact = true},
    };

    const auto selection = SelectCanonicalImageCandidate(candidates);

    ASSERT_TRUE(selection);
    EXPECT_EQ(selection->index, 1u);
    EXPECT_EQ(selection->mip, -1);
    EXPECT_EQ(selection->slice, -1);
}

} // namespace VideoCore
