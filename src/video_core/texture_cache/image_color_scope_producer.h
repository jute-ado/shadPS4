// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>

#include "common/types.h"
#include "video_core/texture_cache/image_producer.h"
#include "video_core/texture_cache/types.h"

namespace VideoCore {

enum class ImageColorScopeDrawKind : u8 {
    Unknown,
    Direct,
    Indirect,
};

static constexpr u32 MaxImageColorScopeTerminalDraws = 8;

struct ImageColorScopePrivateLink {
    ImageId id{};
    u64 uid{};

    [[nodiscard]] constexpr bool Valid() const noexcept {
        return static_cast<bool>(id) && uid != 0;
    }

    auto operator<=>(const ImageColorScopePrivateLink&) const = default;
};

[[nodiscard]] constexpr bool ValidateImageColorScopePrivateLink(ImageColorScopePrivateLink link,
                                                                ImageId id, u64 uid) noexcept {
    return link.Valid() && link.id == id && link.uid == uid;
}

struct ImageColorScopeDrawSummary {
    ImageColorScopeDrawKind kind{ImageColorScopeDrawKind::Unknown};
    bool indexed{};
    u32 element_count{};
    u32 instance_count{};
    u32 sampled_images{};
    u32 storage_writes{};
    ImageProducerClass sampled_input_producer{ImageProducerClass::Unknown};
    bool sampled_input_fresh{};
    bool sampled_input_alias{};
    bool sampled_input_valid{};

    auto operator<=>(const ImageColorScopeDrawSummary&) const = default;
};

enum class ImageColorScopeAncestryTerminal : u8 {
    Unknown,
    NonColorProducer,
    Alias,
    InvalidProducer,
    InvalidScope,
    Overflow,
    ZeroDraws,
    MultipleDraws,
    ZeroInputs,
    MultipleInputs,
    StorageWrite,
    HistoryUnavailable,
    DepthCap,
    Count,
};

static constexpr u32 MaxImageColorScopeAncestryDepth = 8;

struct ImageColorScopeAncestryNode {
    ImageProducerClass producer{ImageProducerClass::Unknown};
    bool fresh{};
    bool alias{};
    bool producer_valid{};
    u32 draw_count{};
    ImageColorScopeDrawKind last_draw{ImageColorScopeDrawKind::Unknown};
    bool indexed{};
    u32 element_count{};
    u32 instance_count{};
    u32 sampled_images{};
    u32 storage_writes{};
    bool clear_at_begin{};
    bool scope_valid{};
    bool overflow{};

    auto operator<=>(const ImageColorScopeAncestryNode&) const = default;
};

struct ImageColorScopeAncestry {
    std::array<ImageColorScopeAncestryNode, MaxImageColorScopeAncestryDepth> nodes{};
    u32 depth{};
    ImageColorScopeAncestryTerminal terminal{ImageColorScopeAncestryTerminal::Unknown};
    bool truncated{};
    std::array<ImageColorScopeDrawSummary, MaxImageColorScopeTerminalDraws> terminal_draws{};
    u32 terminal_draw_count{};
    bool terminal_draws_truncated{};

    auto operator<=>(const ImageColorScopeAncestry&) const = default;
};

[[nodiscard]] constexpr bool IsImageColorScopeAncestryLoss(
    ImageColorScopeAncestryTerminal terminal) noexcept {
    switch (terminal) {
    case ImageColorScopeAncestryTerminal::InvalidProducer:
    case ImageColorScopeAncestryTerminal::InvalidScope:
    case ImageColorScopeAncestryTerminal::Overflow:
    case ImageColorScopeAncestryTerminal::HistoryUnavailable:
    case ImageColorScopeAncestryTerminal::DepthCap:
    case ImageColorScopeAncestryTerminal::Unknown:
        return true;
    default:
        return false;
    }
}

struct ImageColorScopeDrawDescriptor {
    ImageColorScopeDrawKind kind{ImageColorScopeDrawKind::Unknown};
    bool indexed{};
    u32 element_count{};
    u32 instance_count{};
    u32 sampled_bindings{};
    u32 sampled_images{};
    u32 storage_writes{};
    ImageProducerClass sampled_input_producer{ImageProducerClass::Unknown};
    bool sampled_input_fresh{};
    bool sampled_input_alias{};
    bool sampled_input_valid{};
    ImageColorScopePrivateLink sampled_input_image{};
    u32 sampled_input_scope_draw_count{};
    ImageColorScopeDrawKind sampled_input_scope_last_draw{ImageColorScopeDrawKind::Unknown};
    bool sampled_input_scope_indexed{};
    u32 sampled_input_scope_element_count{};
    u32 sampled_input_scope_instance_count{};
    u32 sampled_input_scope_sampled_images{};
    u32 sampled_input_scope_storage_writes{};
    bool sampled_input_scope_clear_at_begin{};
    bool sampled_input_scope_valid{};
    bool sampled_input_scope_overflow{};
    ImageProducerClass sampled_input_scope_input_producer{ImageProducerClass::Unknown};
    bool sampled_input_scope_input_fresh{};
    bool sampled_input_scope_input_alias{};
    bool sampled_input_scope_input_valid{};
    ImageColorScopeAncestry ancestry{};
};

struct ImageColorScopeProducerObservation {
    u32 draw_count{};
    ImageColorScopeDrawKind last_draw{ImageColorScopeDrawKind::Unknown};
    bool indexed{};
    u32 element_count{};
    u32 instance_count{};
    u32 sampled_bindings{};
    u32 sampled_images{};
    u32 storage_writes{};
    ImageProducerClass sampled_input_producer{ImageProducerClass::Unknown};
    bool sampled_input_fresh{};
    bool sampled_input_alias{};
    bool sampled_input_valid{};
    ImageColorScopePrivateLink sampled_input_image{};
    u32 sampled_input_scope_draw_count{};
    ImageColorScopeDrawKind sampled_input_scope_last_draw{ImageColorScopeDrawKind::Unknown};
    bool sampled_input_scope_indexed{};
    u32 sampled_input_scope_element_count{};
    u32 sampled_input_scope_instance_count{};
    u32 sampled_input_scope_sampled_images{};
    u32 sampled_input_scope_storage_writes{};
    bool sampled_input_scope_clear_at_begin{};
    bool sampled_input_scope_valid{};
    bool sampled_input_scope_overflow{};
    ImageProducerClass sampled_input_scope_input_producer{ImageProducerClass::Unknown};
    bool sampled_input_scope_input_fresh{};
    bool sampled_input_scope_input_alias{};
    bool sampled_input_scope_input_valid{};
    ImageColorScopeAncestry ancestry{};
    std::array<ImageColorScopeDrawSummary, MaxImageColorScopeTerminalDraws> draw_summaries{};
    u32 draw_summary_count{};
    bool draw_summaries_truncated{};
    bool clear_at_begin{};
    bool valid{};
    bool overflow{};

    auto operator<=>(const ImageColorScopeProducerObservation&) const = default;
};

[[nodiscard]] inline ImageColorScopeAncestry BuildImageColorScopeAncestry(
    ImageProducerObservation producer, bool producer_valid, bool alias,
    const ImageColorScopeProducerObservation* scope) noexcept {
    ImageColorScopeAncestry ancestry{};
    auto& node = ancestry.nodes[0];
    node.producer = producer.classification;
    node.fresh = producer.produced_since_last_observation;
    node.alias = alias;
    node.producer_valid = producer_valid;
    ancestry.depth = 1;
    if (!producer_valid) {
        ancestry.terminal = ImageColorScopeAncestryTerminal::InvalidProducer;
        return ancestry;
    }
    if (producer.classification == ImageProducerClass::Unknown) {
        ancestry.terminal = ImageColorScopeAncestryTerminal::InvalidProducer;
        return ancestry;
    }
    if (alias) {
        ancestry.terminal = ImageColorScopeAncestryTerminal::Alias;
        return ancestry;
    }
    if (producer.classification != ImageProducerClass::ColorAttachment) {
        ancestry.terminal = ImageColorScopeAncestryTerminal::NonColorProducer;
        return ancestry;
    }
    if (!scope) {
        ancestry.terminal = ImageColorScopeAncestryTerminal::HistoryUnavailable;
        return ancestry;
    }
    node.draw_count = scope->draw_count;
    node.last_draw = scope->last_draw;
    node.indexed = scope->indexed;
    node.element_count = scope->element_count;
    node.instance_count = scope->instance_count;
    node.sampled_images = scope->sampled_images;
    node.storage_writes = scope->storage_writes;
    node.clear_at_begin = scope->clear_at_begin;
    node.scope_valid = scope->valid;
    node.overflow = scope->overflow;
    if (scope->overflow) {
        ancestry.terminal = ImageColorScopeAncestryTerminal::Overflow;
        return ancestry;
    }
    if (!scope->valid) {
        ancestry.terminal = ImageColorScopeAncestryTerminal::InvalidScope;
        return ancestry;
    }
    if (scope->draw_count == 0) {
        ancestry.terminal = ImageColorScopeAncestryTerminal::ZeroDraws;
        return ancestry;
    }
    if (scope->draw_count != 1) {
        ancestry.terminal = ImageColorScopeAncestryTerminal::MultipleDraws;
        ancestry.terminal_draw_count = scope->draw_summary_count;
        ancestry.terminal_draws_truncated = scope->draw_summaries_truncated;
        for (u32 index = 0; index < ancestry.terminal_draw_count; ++index) {
            ancestry.terminal_draws[index] = scope->draw_summaries[index];
        }
        return ancestry;
    }
    if (scope->storage_writes != 0) {
        ancestry.terminal = ImageColorScopeAncestryTerminal::StorageWrite;
        return ancestry;
    }
    if (scope->sampled_images == 0) {
        ancestry.terminal = ImageColorScopeAncestryTerminal::ZeroInputs;
        return ancestry;
    }
    if (scope->sampled_images != 1) {
        ancestry.terminal = ImageColorScopeAncestryTerminal::MultipleInputs;
        return ancestry;
    }
    if (scope->ancestry.depth == 0) {
        ancestry.terminal = ImageColorScopeAncestryTerminal::HistoryUnavailable;
        return ancestry;
    }
    const u32 available = MaxImageColorScopeAncestryDepth - ancestry.depth;
    const u32 copied = scope->ancestry.depth < available ? scope->ancestry.depth : available;
    for (u32 index = 0; index < copied; ++index) {
        ancestry.nodes[ancestry.depth + index] = scope->ancestry.nodes[index];
    }
    ancestry.depth += copied;
    if (copied != scope->ancestry.depth || scope->ancestry.truncated) {
        ancestry.terminal = ImageColorScopeAncestryTerminal::DepthCap;
        ancestry.truncated = true;
    } else {
        ancestry.terminal = scope->ancestry.terminal;
        ancestry.terminal_draws = scope->ancestry.terminal_draws;
        ancestry.terminal_draw_count = scope->ancestry.terminal_draw_count;
        ancestry.terminal_draws_truncated = scope->ancestry.terminal_draws_truncated;
        if (ancestry.terminal == ImageColorScopeAncestryTerminal::Unknown) {
            ancestry.terminal = ImageColorScopeAncestryTerminal::HistoryUnavailable;
        }
    }
    return ancestry;
}

class ImageColorScopeProducerState {
public:
    static constexpr u32 MaxTrackedDraws = 2048;
    static constexpr u32 MaxTrackedImageBindings = 32;

    void BeginScope(u64 scope_serial_, bool clear_at_begin_) noexcept {
        if (scope_serial_ == 0) {
            observation = {};
            scope_serial = 0;
            return;
        }
        if (scope_serial == scope_serial_) {
            return;
        }
        scope_serial = scope_serial_;
        observation = {
            .clear_at_begin = clear_at_begin_,
            .valid = true,
        };
    }

    void MarkDraw(u64 scope_serial_, ImageColorScopeDrawDescriptor descriptor) noexcept {
        if (scope_serial == 0 || scope_serial_ != scope_serial ||
            descriptor.kind == ImageColorScopeDrawKind::Unknown) {
            observation.valid = false;
            return;
        }
        observation.last_draw = descriptor.kind;
        observation.indexed = descriptor.indexed;
        observation.element_count = descriptor.element_count;
        observation.instance_count = descriptor.instance_count;
        observation.sampled_bindings = descriptor.sampled_bindings;
        observation.sampled_images = descriptor.sampled_images;
        observation.storage_writes = descriptor.storage_writes;
        observation.sampled_input_producer = descriptor.sampled_input_producer;
        observation.sampled_input_fresh = descriptor.sampled_input_fresh;
        observation.sampled_input_alias = descriptor.sampled_input_alias;
        observation.sampled_input_valid = descriptor.sampled_input_valid;
        observation.sampled_input_image = descriptor.sampled_images == 1
                                              ? descriptor.sampled_input_image
                                              : ImageColorScopePrivateLink{};
        observation.sampled_input_scope_draw_count = descriptor.sampled_input_scope_draw_count;
        observation.sampled_input_scope_last_draw = descriptor.sampled_input_scope_last_draw;
        observation.sampled_input_scope_indexed = descriptor.sampled_input_scope_indexed;
        observation.sampled_input_scope_element_count =
            descriptor.sampled_input_scope_element_count;
        observation.sampled_input_scope_instance_count =
            descriptor.sampled_input_scope_instance_count;
        observation.sampled_input_scope_sampled_images =
            descriptor.sampled_input_scope_sampled_images;
        observation.sampled_input_scope_storage_writes =
            descriptor.sampled_input_scope_storage_writes;
        observation.sampled_input_scope_clear_at_begin =
            descriptor.sampled_input_scope_clear_at_begin;
        observation.sampled_input_scope_valid = descriptor.sampled_input_scope_valid;
        observation.sampled_input_scope_overflow = descriptor.sampled_input_scope_overflow;
        observation.sampled_input_scope_input_producer =
            descriptor.sampled_input_scope_input_producer;
        observation.sampled_input_scope_input_fresh = descriptor.sampled_input_scope_input_fresh;
        observation.sampled_input_scope_input_alias = descriptor.sampled_input_scope_input_alias;
        observation.sampled_input_scope_input_valid = descriptor.sampled_input_scope_input_valid;
        observation.ancestry = descriptor.ancestry;
        if (observation.draw_summary_count < MaxImageColorScopeTerminalDraws) {
            observation.draw_summaries[observation.draw_summary_count++] = {
                .kind = descriptor.kind,
                .indexed = descriptor.indexed,
                .element_count = descriptor.element_count,
                .instance_count = descriptor.instance_count,
                .sampled_images = descriptor.sampled_images,
                .storage_writes = descriptor.storage_writes,
                .sampled_input_producer = descriptor.sampled_input_producer,
                .sampled_input_fresh = descriptor.sampled_input_fresh,
                .sampled_input_alias = descriptor.sampled_input_alias,
                .sampled_input_valid = descriptor.sampled_input_valid,
            };
        } else {
            observation.draw_summaries_truncated = true;
        }
        if (descriptor.sampled_images > descriptor.sampled_bindings ||
            descriptor.sampled_bindings > MaxTrackedImageBindings ||
            descriptor.storage_writes > MaxTrackedImageBindings) {
            observation.valid = false;
        }
        if (observation.draw_count == MaxTrackedDraws) {
            observation.overflow = true;
            observation.valid = false;
            return;
        }
        ++observation.draw_count;
    }

    [[nodiscard]] ImageColorScopeProducerObservation Observe() const noexcept {
        return observation;
    }

    void Reset() noexcept {
        scope_serial = 0;
        observation = {};
    }

private:
    u64 scope_serial{};
    ImageColorScopeProducerObservation observation{};
};

} // namespace VideoCore
