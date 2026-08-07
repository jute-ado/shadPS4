// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <mutex>
#include <span>
#include <unordered_map>

#include "common/types.h"

namespace Vulkan {

struct AttachmentSubresource {
    u32 base_level{};
    u32 levels{};
    u32 base_layer{};
    u32 layers{};
};

struct AttachmentTarget {
    u64 image_uid{};
    AttachmentSubresource subresource{};
};

struct AttachmentPublicationSnapshot {
    u32 scopes{};
    u32 draws_issued{};
    u32 barriers{};
    u32 destructive_writes{};
    u32 samples{};
    u32 attachment_samples{};
    u32 without_issued_producer{};
    u32 before_scope_end{};
    u32 without_covering_barrier{};
    u32 after_destructive_write{};
    u32 truncated_images{};
    u32 truncated_scope_targets{};
};

/**
 * Bounded diagnostic state for attachment publication into later shader samples.
 * Image identities are internal stable ordinals and are never emitted by this class.
 */
class AttachmentPublicationDiagnostic {
public:
    static constexpr u32 MaxTrackedImages = 4096;
    static constexpr u32 MaxScopeTargets = 9;

    AttachmentPublicationDiagnostic() {
        images.reserve(MaxTrackedImages);
    }

    void BeginScope(std::span<const AttachmentTarget> targets) {
        std::scoped_lock lock{mutex};
        EndScopeUnlocked();
        ++current.scopes;
        active_count = 0;
        for (const auto& target : targets) {
            if (target.image_uid == 0) {
                continue;
            }
            auto* state = FindOrCreate(target.image_uid);
            if (!state) {
                ++current.truncated_images;
                continue;
            }

            const auto active_it = std::find(active_images.begin(),
                                             active_images.begin() + active_count,
                                             target.image_uid);
            if (active_it != active_images.begin() + active_count) {
                state->produced = Union(state->produced, target.subresource);
                continue;
            }
            if (active_count >= MaxScopeTargets) {
                ++current.truncated_scope_targets;
                continue;
            }

            *state = ImageState{
                .produced = target.subresource,
                .scope_active = true,
            };
            active_images[active_count++] = target.image_uid;
        }
    }

    void RecordDrawIssued() {
        std::scoped_lock lock{mutex};
        ++current.draws_issued;
        for (u32 index = 0; index < active_count; ++index) {
            images.at(active_images[index]).draw_issued = true;
        }
    }

    void EndScope() {
        std::scoped_lock lock{mutex};
        EndScopeUnlocked();
    }

    void RecordBarrier(u64 image_uid, AttachmentSubresource subresource) {
        std::scoped_lock lock{mutex};
        ++current.barriers;
        const auto it = images.find(image_uid);
        if (it == images.end() || !it->second.scope_ended) {
            return;
        }
        auto& state = it->second;
        state.barrier = subresource;
        state.has_barrier = true;
    }

    void RecordDestructiveWrite(u64 image_uid, AttachmentSubresource subresource) {
        std::scoped_lock lock{mutex};
        ++current.destructive_writes;
        const auto it = images.find(image_uid);
        if (it == images.end() || !Overlaps(it->second.produced, subresource)) {
            return;
        }
        auto& state = it->second;
        state.destructive_write = subresource;
        state.has_destructive_write = true;
        state.has_barrier = false;
    }

    void RecordSample(u64 image_uid, AttachmentSubresource subresource) {
        std::scoped_lock lock{mutex};
        ++current.samples;
        const auto it = images.find(image_uid);
        if (it == images.end() || !Overlaps(it->second.produced, subresource)) {
            return;
        }

        ++current.attachment_samples;
        const auto& state = it->second;
        current.without_issued_producer += !state.draw_issued;
        current.before_scope_end += state.scope_active || !state.scope_ended;
        current.without_covering_barrier +=
            !state.has_barrier || !Covers(state.barrier, subresource);
        current.after_destructive_write +=
            state.has_destructive_write && Overlaps(state.destructive_write, subresource);
    }

    [[nodiscard]] AttachmentPublicationSnapshot TakeSnapshot() {
        std::scoped_lock lock{mutex};
        const auto snapshot = current;
        current = {};
        return snapshot;
    }

private:
    void EndScopeUnlocked() {
        for (u32 index = 0; index < active_count; ++index) {
            auto& state = images.at(active_images[index]);
            state.scope_active = false;
            state.scope_ended = true;
        }
        active_count = 0;
    }
    struct ImageState {
        AttachmentSubresource produced{};
        AttachmentSubresource barrier{};
        AttachmentSubresource destructive_write{};
        bool scope_active{};
        bool scope_ended{};
        bool draw_issued{};
        bool has_barrier{};
        bool has_destructive_write{};
    };

    [[nodiscard]] ImageState* FindOrCreate(u64 image_uid) {
        if (const auto it = images.find(image_uid); it != images.end()) {
            return &it->second;
        }
        if (images.size() >= MaxTrackedImages) {
            return nullptr;
        }
        return &images.emplace(image_uid, ImageState{}).first->second;
    }

    [[nodiscard]] static constexpr u64 End(u32 base, u32 count) {
        return static_cast<u64>(base) + count;
    }

    [[nodiscard]] static constexpr bool Covers(AttachmentSubresource outer,
                                               AttachmentSubresource inner) {
        return outer.base_level <= inner.base_level &&
               End(outer.base_level, outer.levels) >= End(inner.base_level, inner.levels) &&
               outer.base_layer <= inner.base_layer &&
               End(outer.base_layer, outer.layers) >= End(inner.base_layer, inner.layers);
    }

    [[nodiscard]] static constexpr bool Overlaps(AttachmentSubresource left,
                                                 AttachmentSubresource right) {
        return left.base_level < End(right.base_level, right.levels) &&
               right.base_level < End(left.base_level, left.levels) &&
               left.base_layer < End(right.base_layer, right.layers) &&
               right.base_layer < End(left.base_layer, left.layers);
    }

    [[nodiscard]] static constexpr AttachmentSubresource Union(AttachmentSubresource left,
                                                                AttachmentSubresource right) {
        const u32 base_level = std::min(left.base_level, right.base_level);
        const u32 base_layer = std::min(left.base_layer, right.base_layer);
        return {
            .base_level = base_level,
            .levels = static_cast<u32>(std::max(End(left.base_level, left.levels),
                                                End(right.base_level, right.levels)) -
                                       base_level),
            .base_layer = base_layer,
            .layers = static_cast<u32>(std::max(End(left.base_layer, left.layers),
                                                End(right.base_layer, right.layers)) -
                                       base_layer),
        };
    }

    std::unordered_map<u64, ImageState> images;
    std::array<u64, MaxScopeTargets> active_images{};
    u32 active_count{};
    AttachmentPublicationSnapshot current{};
    std::mutex mutex;
};

} // namespace Vulkan
