// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <limits>
#include <utility>

#include "video_core/texture_cache/image_info.h"

namespace VideoCore {

enum class ImageSubresourceLookupRelation : u8 {
    None = 0,
    Exact = 1,
    CanonicalRejected = 2,
    CanonicalMisselected = 3,
    FalseSubresource = 4,
    Invalid = 5,
};

enum class ImageSubresourceLookupDisposition : u8 {
    Idle = 0,
    NoOverlap = 1,
    ExactSubresource = 2,
    CanonicalRejected = 3,
    CanonicalMisselected = 4,
    FalseSubresource = 5,
    AmbiguousExact = 6,
    InvalidTransition = 7,
};

enum class ImageSubresourceLookupPath : u8 {
    Idle = 0,
    Perfect = 1,
    RightOverlap = 2,
    LeftOverlap = 3,
    CreatedStandalone = 4,
    Invalid = 5,
};

struct ImageSubresourceLookupActual {
    s32 mip{-1};
    s32 slice{-1};

    auto operator<=>(const ImageSubresourceLookupActual&) const = default;
};

/// A deliberately bounded diagnostic fact. In particular, this type must never retain an image
/// identity, guest address, byte size, or numeric mip/layer index.
struct ImageSubresourceLookupFact {
    ImageSubresourceLookupPath path{ImageSubresourceLookupPath::Idle};
    ImageSubresourceLookupRelation relation{ImageSubresourceLookupRelation::None};
    u8 perfect_candidate_count{};
    u8 overlap_candidate_count{};
    bool actual_accepted{};
    bool canonical_accepted{};
    bool actual_canonical_equal{};
    bool nonzero_mip{};
    bool nonzero_layer{};
    bool logical_single_layer{};
    bool parent_stride_aligned{};
    bool bounded_by_parent{};
    bool parent_gpu_authoritative{};
    bool created_standalone{};
    bool candidate_drift{};
    bool overflowed{};
    bool malformed{};
    bool provenance_current{};
    bool observed{};
    ImageSubresourceLookupDisposition disposition{ImageSubresourceLookupDisposition::Idle};

    auto operator<=>(const ImageSubresourceLookupFact&) const = default;
};

/// Transient lookup detail. Callers must copy only `fact` (and, when explicitly required, the
/// already-sanitized selected origin) beyond the immediate lookup call.
struct ImageSubresourceLookupObservation : ImageSubresourceLookupFact {
    ImageSubresourceLookupActual actual_result{};
    ImageSubresourceLookupActual canonical_result{};
    ImageSubresourceLookupFact fact{};
    ImageSubresourceLookupFact selected_origin{};
};

[[nodiscard]] constexpr ImageSubresourceLookupRelation
ClassifyImageSubresourceLookupRelation(const bool actual_accepted,
                                       const bool canonical_accepted,
                                       const bool actual_canonical_equal) noexcept {
    if (actual_canonical_equal && (!actual_accepted || !canonical_accepted)) {
        return ImageSubresourceLookupRelation::Invalid;
    }
    if (!actual_accepted && !canonical_accepted) {
        return ImageSubresourceLookupRelation::None;
    }
    if (!actual_accepted && canonical_accepted) {
        return ImageSubresourceLookupRelation::CanonicalRejected;
    }
    if (actual_accepted && !canonical_accepted) {
        return ImageSubresourceLookupRelation::FalseSubresource;
    }
    return actual_canonical_equal ? ImageSubresourceLookupRelation::Exact
                                  : ImageSubresourceLookupRelation::CanonicalMisselected;
}

[[nodiscard]] constexpr bool ImageSubresourceLookupRelationMatches(
    const ImageSubresourceLookupFact& value) noexcept {
    return value.relation == ClassifyImageSubresourceLookupRelation(
                                 value.actual_accepted, value.canonical_accepted,
                                 value.actual_canonical_equal);
}

[[nodiscard]] constexpr ImageSubresourceLookupDisposition
ClassifyImageSubresourceLookup(const ImageSubresourceLookupFact& value) noexcept {
    using Disposition = ImageSubresourceLookupDisposition;
    using Path = ImageSubresourceLookupPath;
    using Relation = ImageSubresourceLookupRelation;

    if (!value.observed) {
        return value.path == Path::Idle && value.relation == Relation::None &&
                       value.perfect_candidate_count == 0 && value.overlap_candidate_count == 0 &&
                       !value.created_standalone && !value.candidate_drift && !value.overflowed &&
                       !value.malformed
                   ? Disposition::Idle
                   : Disposition::InvalidTransition;
    }
    if (value.path == Path::Invalid || value.overflowed || value.malformed ||
        value.candidate_drift || !ImageSubresourceLookupRelationMatches(value)) {
        return Disposition::InvalidTransition;
    }
    if (value.path == Path::Perfect) {
        if (value.overlap_candidate_count != 0 || value.perfect_candidate_count == 0 ||
            value.created_standalone || value.relation != Relation::None) {
            return Disposition::InvalidTransition;
        }
        return value.perfect_candidate_count == 1 ? Disposition::NoOverlap
                                                  : Disposition::AmbiguousExact;
    }
    if (value.path == Path::RightOverlap || value.path == Path::LeftOverlap) {
        if (value.overlap_candidate_count == 0 || value.perfect_candidate_count != 0 ||
            value.created_standalone || !value.provenance_current) {
            return Disposition::InvalidTransition;
        }
    } else if (value.path == Path::CreatedStandalone) {
        if (!value.created_standalone || !value.provenance_current) {
            return Disposition::InvalidTransition;
        }
        if (value.relation != Relation::None && value.overlap_candidate_count == 0) {
            return Disposition::InvalidTransition;
        }
    } else if (value.path != Path::Idle) {
        return Disposition::InvalidTransition;
    }

    switch (value.relation) {
    case Relation::None:
        return Disposition::NoOverlap;
    case Relation::Exact:
        return Disposition::ExactSubresource;
    case Relation::CanonicalRejected:
        return Disposition::CanonicalRejected;
    case Relation::CanonicalMisselected:
        return Disposition::CanonicalMisselected;
    case Relation::FalseSubresource:
        return Disposition::FalseSubresource;
    case Relation::Invalid:
        return Disposition::InvalidTransition;
    }
    return Disposition::InvalidTransition;
}

inline void SetImageSubresourceLookupObservationFact(ImageSubresourceLookupObservation& out,
                                                     ImageSubresourceLookupFact fact) noexcept {
    fact.disposition = ClassifyImageSubresourceLookup(fact);
    static_cast<ImageSubresourceLookupFact&>(out) = fact;
    out.fact = fact;
}

[[nodiscard]] inline bool IsImageSubresourceLookupParentDomainSafe(
    const ImageInfo& parent) noexcept {
    if (parent.resources.layers == 0 || parent.resources.levels == 0 ||
        parent.resources.levels > parent.mips_layout.size()) {
        return false;
    }
    for (u32 mip = 0; mip < parent.resources.levels; ++mip) {
        const u32 mip_size = parent.mips_layout[mip].size;
        if (mip_size == 0 || mip_size % parent.resources.layers != 0 ||
            mip_size / parent.resources.layers == 0) {
            return false;
        }
    }
    return true;
}

/// Compares the already-produced legacy MipOf/SliceOf result with the parent-stride relation. It
/// intentionally does not invoke either legacy helper and cannot alter their authoritative result.
[[nodiscard]] inline ImageSubresourceLookupObservation CompareImageSubresourceLookup(
    const ImageInfo& child, const ImageInfo& parent,
    const ImageSubresourceLookupActual actual_result) noexcept {
    ImageSubresourceLookupObservation out{};
    out.actual_result = actual_result;
    ImageSubresourceLookupFact fact{
        .actual_accepted = actual_result.mip >= 0 && actual_result.slice >= 0,
        .nonzero_mip = actual_result.mip > 0,
        .nonzero_layer = actual_result.slice > 0,
        .logical_single_layer = child.resources.layers == 1,
        .observed = true,
    };

    if (!IsImageSubresourceLookupParentDomainSafe(parent) || child.resources.layers == 0 ||
        child.guest_size == 0 || actual_result.mip >= static_cast<s32>(parent.resources.levels)) {
        fact.malformed = !IsImageSubresourceLookupParentDomainSafe(parent) ||
                         child.resources.layers == 0 || child.guest_size == 0 ||
                         actual_result.mip >= static_cast<s32>(parent.resources.levels);
        fact.relation = fact.malformed
                            ? ImageSubresourceLookupRelation::Invalid
                            : ClassifyImageSubresourceLookupRelation(fact.actual_accepted, false,
                                                                    false);
        SetImageSubresourceLookupObservationFact(out, fact);
        return out;
    }

    s32 canonical_mip = actual_result.mip;
    if (canonical_mip >= 0) {
        const auto& mip_layout = parent.mips_layout[canonical_mip];
        const u64 slice_stride = mip_layout.size / parent.resources.layers;
        const u64 mip_base = parent.guest_address + mip_layout.offset;
        if (child.guest_address >= mip_base) {
            const u64 address_delta = child.guest_address - mip_base;
            fact.parent_stride_aligned = address_delta % slice_stride == 0;
            if (fact.parent_stride_aligned) {
                const u64 canonical_slice = address_delta / slice_stride;
                const u64 child_layers = child.resources.layers;
                const bool layer_extent_bounded =
                    canonical_slice <= parent.resources.layers &&
                    child_layers <= parent.resources.layers -
                                        std::min<u64>(canonical_slice, parent.resources.layers);
                const bool byte_extent_bounded =
                    fact.logical_single_layer
                        ? child.guest_size <= slice_stride
                        : child_layers <= std::numeric_limits<u64>::max() / slice_stride &&
                              child.guest_size == slice_stride * child_layers;
                fact.bounded_by_parent = layer_extent_bounded && byte_extent_bounded;
                if (fact.bounded_by_parent &&
                    canonical_slice <= static_cast<u64>(std::numeric_limits<s32>::max())) {
                    out.canonical_result = {
                        .mip = canonical_mip,
                        .slice = static_cast<s32>(canonical_slice),
                    };
                    fact.canonical_accepted = true;
                    fact.nonzero_mip = fact.nonzero_mip || canonical_mip > 0;
                    fact.nonzero_layer = fact.nonzero_layer || canonical_slice > 0;
                }
            }
        }
    }

    fact.actual_canonical_equal =
        fact.actual_accepted && fact.canonical_accepted && actual_result == out.canonical_result;
    fact.relation = ClassifyImageSubresourceLookupRelation(
        fact.actual_accepted, fact.canonical_accepted, fact.actual_canonical_equal);
    SetImageSubresourceLookupObservationFact(out, fact);
    return out;
}

template <typename MipLookup, typename SliceLookup>
[[nodiscard]] inline ImageSubresourceLookupObservation ObserveImageSubresourceLookupOnce(
    const ImageInfo& child, const ImageInfo& parent, MipLookup&& mip_lookup,
    SliceLookup&& slice_lookup) {
    if (!IsImageSubresourceLookupParentDomainSafe(parent)) {
        ImageSubresourceLookupObservation out{};
        SetImageSubresourceLookupObservationFact(
            out, {.relation = ImageSubresourceLookupRelation::Invalid,
                  .malformed = true,
                  .observed = true});
        return out;
    }

    const s32 mip = std::forward<MipLookup>(mip_lookup)(child, parent);
    if (child.resources.layers == 0 || child.guest_size == 0 ||
        mip >= static_cast<s32>(parent.resources.levels)) {
        ImageSubresourceLookupObservation out{};
        out.actual_result.mip = mip;
        SetImageSubresourceLookupObservationFact(
            out, {.relation = ImageSubresourceLookupRelation::Invalid,
                  .nonzero_mip = mip > 0,
                  .malformed = true,
                  .observed = true});
        return out;
    }
    if (mip < 0) {
        return CompareImageSubresourceLookup(child, parent, {.mip = mip, .slice = -1});
    }
    const s32 slice = std::forward<SliceLookup>(slice_lookup)(child, parent, mip);
    return CompareImageSubresourceLookup(child, parent, {.mip = mip, .slice = slice});
}

[[nodiscard]] inline ImageSubresourceLookupObservation ObserveImageSubresourceLookupOnce(
    const ImageInfo& child, const ImageInfo& parent) {
    return ObserveImageSubresourceLookupOnce(
        child, parent,
        [](const ImageInfo& value, const ImageInfo& container) { return value.MipOf(container); },
        [](const ImageInfo& value, const ImageInfo& container, const s32 mip) {
            return value.SliceOf(container, mip);
        });
}

template <typename Range>
[[nodiscard]] inline ImageSubresourceLookupFact CollectImageSubresourceLookupCandidates(
    const Range& candidates) noexcept {
    ImageSubresourceLookupFact result{};
    bool saw_accepted_overlap{};
    bool saw_any{};
    for (const auto& candidate : candidates) {
        if (!candidate.observed) {
            continue;
        }
        saw_any = true;
        const bool perfect = candidate.path == ImageSubresourceLookupPath::Perfect;
        u8& count = perfect ? result.perfect_candidate_count : result.overlap_candidate_count;
        if (count == std::numeric_limits<u8>::max()) {
            result.overflowed = true;
        } else {
            ++count;
        }
        if (!perfect && saw_accepted_overlap && !candidate.actual_accepted) {
            result.candidate_drift = true;
        }
        if (!perfect && candidate.actual_accepted) {
            saw_accepted_overlap = true;
        }

        const u8 perfect_count = result.perfect_candidate_count;
        const u8 overlap_count = result.overlap_candidate_count;
        const bool overflowed = result.overflowed || candidate.overflowed;
        const bool candidate_drift = result.candidate_drift || candidate.candidate_drift;
        result = candidate;
        result.perfect_candidate_count = perfect_count;
        result.overlap_candidate_count = overlap_count;
        result.overflowed = overflowed;
        result.candidate_drift = candidate_drift;
        if (result.candidate_drift) {
            result.provenance_current = false;
        }
    }
    if (!saw_any) {
        return {};
    }
    result.observed = true;
    result.disposition = ClassifyImageSubresourceLookup(result);
    return result;
}

} // namespace VideoCore
