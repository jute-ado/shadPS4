// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/amdgpu/draw_resource_fingerprint_diagnostic.h"

using AmdGpu::DrawResourceDescriptorKind;
using AmdGpu::DrawResourceFingerprintDiagnostic;

TEST(DrawResourceFingerprintDiagnostic, IdentifiesConsumedDescriptorChangesByDrawOrdinal) {
    DrawResourceFingerprintDiagnostic diagnostic{/*report_limit=*/4};
    std::array<u32, 4> buffer{0x1000, 16, 3, 4};
    std::array<u32, 8> image{0x2000, 2, 3, 4, 5, 6, 7, 8};
    std::array<u32, 4> vertex{0x3000, 32, 7, 8};

    const auto record_frame = [&] {
        diagnostic.BeginDraw();
        diagnostic.RecordDescriptor(DrawResourceDescriptorKind::Buffer, buffer.data(),
                                    sizeof(buffer));
        diagnostic.RecordDescriptor(DrawResourceDescriptorKind::Image, image.data(),
                                    sizeof(image));
        diagnostic.EndDraw();

        diagnostic.BeginDraw();
        diagnostic.RecordDescriptor(DrawResourceDescriptorKind::Vertex, vertex.data(),
                                    sizeof(vertex));
        diagnostic.EndDraw();
        return diagnostic.TakeSnapshot();
    };

    const auto baseline = record_frame();
    EXPECT_TRUE(baseline.should_report);
    EXPECT_FALSE(baseline.matches_previous_frame);
    EXPECT_EQ(baseline.draws, 2);
    EXPECT_EQ(baseline.descriptors, 3);
    EXPECT_EQ(baseline.bytes_hashed, sizeof(buffer) + sizeof(image) + sizeof(vertex));
    EXPECT_EQ(baseline.changed_draws, 2);
    ASSERT_EQ(baseline.reported_changed_draws, 2);
    EXPECT_EQ(baseline.first_changed_draw_ordinals[0], 0);
    EXPECT_EQ(baseline.first_changed_draw_ordinals[1], 1);

    const auto unchanged = record_frame();
    EXPECT_TRUE(unchanged.matches_previous_frame);
    EXPECT_EQ(unchanged.combined_hash, baseline.combined_hash);
    EXPECT_EQ(unchanged.changed_draws, 0);

    image[3] ^= 1;
    const auto changed = record_frame();
    EXPECT_FALSE(changed.matches_previous_frame);
    EXPECT_EQ(changed.changed_draws, 1);
    ASSERT_EQ(changed.reported_changed_draws, 1);
    EXPECT_EQ(changed.first_changed_draw_ordinals[0], 0);
}

TEST(DrawResourceFingerprintDiagnostic, BoundsDrawsDescriptorsAndBytes) {
    DrawResourceFingerprintDiagnostic diagnostic{/*report_limit=*/1};
    std::vector<u8> payload(DrawResourceFingerprintDiagnostic::MaxBytesPerFrame);

    for (u32 draw = 0; draw < DrawResourceFingerprintDiagnostic::MaxDrawsPerFrame + 2; ++draw) {
        diagnostic.BeginDraw();
        for (u32 descriptor = 0;
             descriptor < DrawResourceFingerprintDiagnostic::MaxDescriptorsPerDraw + 1;
             ++descriptor) {
            diagnostic.RecordDescriptor(DrawResourceDescriptorKind::Buffer, payload.data(),
                                        payload.size());
        }
        diagnostic.EndDraw();
    }

    const auto bounded = diagnostic.TakeSnapshot();
    EXPECT_EQ(bounded.draws, DrawResourceFingerprintDiagnostic::MaxDrawsPerFrame);
    EXPECT_EQ(bounded.bytes_hashed, DrawResourceFingerprintDiagnostic::MaxBytesPerFrame);
    EXPECT_GT(bounded.truncated_draws, 0);
    EXPECT_GT(bounded.truncated_descriptors, 0);
    EXPECT_GT(bounded.truncated_bytes, 0);
    EXPECT_FALSE(diagnostic.TakeSnapshot().should_report);
}

TEST(DrawResourceFingerprintDiagnostic, SeparatesDescriptorLocationFromShapeChanges) {
    DrawResourceFingerprintDiagnostic diagnostic{/*report_limit=*/3};
    std::array<u32, 4> descriptor{0x1000, 16, 3, 4};
    std::array<u32, 4> shape = descriptor;
    shape[0] = 0;

    const auto record = [&] {
        diagnostic.BeginDraw();
        diagnostic.RecordDescriptor(DrawResourceDescriptorKind::Buffer, descriptor.data(),
                                    sizeof(descriptor), shape.data(), sizeof(shape));
        diagnostic.EndDraw();
        return diagnostic.TakeSnapshot();
    };

    const auto baseline = record();
    EXPECT_EQ(baseline.changed_draws, 1);
    EXPECT_EQ(baseline.changed_shape_draws, 1);

    descriptor[0] = 0x2000;
    const auto location_only = record();
    EXPECT_EQ(location_only.changed_draws, 1);
    EXPECT_EQ(location_only.changed_shape_draws, 0);
    EXPECT_TRUE(location_only.shape_matches_previous_frame);

    shape[1] = 32;
    const auto shape_changed = record();
    EXPECT_EQ(shape_changed.changed_draws, 0);
    EXPECT_EQ(shape_changed.changed_shape_draws, 1);
    ASSERT_EQ(shape_changed.reported_changed_shape_draws, 1);
    EXPECT_EQ(shape_changed.first_changed_shape_draw_ordinals[0], 0);
}

TEST(DrawResourceFingerprintDiagnostic, ReportsLocationOnlyABAReturnsByDrawOrdinal) {
    DrawResourceFingerprintDiagnostic diagnostic{/*report_limit=*/3};
    std::array<u32, 4> descriptor{0x1000, 16, 3, 4};
    std::array<u32, 4> shape = descriptor;
    shape[0] = 0;
    u64 location = descriptor[0];

    const auto record = [&] {
        descriptor[0] = static_cast<u32>(location);
        diagnostic.BeginDraw();
        diagnostic.RecordDescriptor(DrawResourceDescriptorKind::Buffer, descriptor.data(),
                                    sizeof(descriptor), shape.data(), sizeof(shape), &location,
                                    sizeof(location));
        diagnostic.EndDraw();
        return diagnostic.TakeSnapshot();
    };

    const auto first = record();
    EXPECT_EQ(first.changed_location_draws, 1);
    EXPECT_EQ(first.location_aba_return_draws, 0);

    location = 0x2000;
    const auto middle = record();
    EXPECT_EQ(middle.changed_location_draws, 1);
    EXPECT_EQ(middle.location_aba_return_draws, 0);

    location = 0x1000;
    const auto returned = record();
    EXPECT_EQ(returned.changed_location_draws, 1);
    EXPECT_EQ(returned.location_aba_return_draws, 1);
    ASSERT_EQ(returned.reported_location_aba_return_draws, 1);
    EXPECT_EQ(returned.first_location_aba_return_draw_ordinals[0], 0);
}

TEST(DrawResourceFingerprintDiagnostic, ReportsHostIdentityABAReturnsByDrawOrdinal) {
    DrawResourceFingerprintDiagnostic diagnostic{/*report_limit=*/3};

    const auto record = [&](u64 image_uid) {
        diagnostic.BeginDraw();
        diagnostic.RecordHostIdentity(/*binding_ordinal=*/0, /*slot=*/7, image_uid,
                                      /*backing=*/11);
        diagnostic.EndDraw();
        return diagnostic.TakeSnapshot();
    };

    EXPECT_EQ(record(1).host_identity_aba_return_draws, 0);
    const auto middle = record(2);
    EXPECT_EQ(middle.changed_host_identity_draws, 1);
    EXPECT_EQ(middle.host_identity_aba_return_draws, 0);

    const auto returned = record(1);
    EXPECT_EQ(returned.changed_host_identity_draws, 1);
    EXPECT_EQ(returned.host_identity_aba_return_draws, 1);
    ASSERT_EQ(returned.reported_host_identity_aba_return_draws, 1);
    EXPECT_EQ(returned.first_host_identity_aba_return_draw_ordinals[0], 0);
}

TEST(DrawResourceFingerprintDiagnostic, ReportsHostBufferRangeABAReturnsByDrawOrdinal) {
    DrawResourceFingerprintDiagnostic diagnostic{/*report_limit=*/3};

    const auto record = [&](u64 offset) {
        diagnostic.BeginDraw();
        diagnostic.RecordHostBufferIdentity(/*binding_ordinal=*/0, /*role=*/1,
                                            /*object=*/7, /*backing=*/11, offset,
                                            /*size=*/256);
        diagnostic.EndDraw();
        return diagnostic.TakeSnapshot();
    };

    EXPECT_EQ(record(64).host_identity_aba_return_draws, 0);
    const auto middle = record(128);
    EXPECT_EQ(middle.changed_host_identity_draws, 1);
    EXPECT_EQ(middle.host_identity_aba_return_draws, 0);

    const auto returned = record(64);
    EXPECT_EQ(returned.changed_host_identity_draws, 1);
    EXPECT_EQ(returned.host_identity_aba_return_draws, 1);
    ASSERT_EQ(returned.reported_host_identity_aba_return_draws, 1);
    EXPECT_EQ(returned.first_host_identity_aba_return_draw_ordinals[0], 0);
}
