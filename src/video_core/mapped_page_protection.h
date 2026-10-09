// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/icl/interval_map.hpp>

#include "common/shared_first_mutex.h"
#include "common/types.h"

namespace VideoCore {

namespace Detail {

[[nodiscard]] constexpr bool IsValidAddressRange(VAddr address, u64 size) noexcept {
    return size != 0 && size <= std::numeric_limits<VAddr>::max() - address;
}

} // namespace Detail

template <typename Visitor>
[[nodiscard]] bool ForEachAddressRangeChunk(VAddr address, u64 size, u64 chunk_alignment,
                                            Visitor&& visitor) {
    if (!Detail::IsValidAddressRange(address, size) || chunk_alignment == 0) {
        return false;
    }

    const VAddr end_address = address + size;
    while (address < end_address) {
        const u64 offset = address % chunk_alignment;
        const u64 bytes_to_boundary = offset == 0 ? chunk_alignment : chunk_alignment - offset;
        const u64 chunk_size = std::min<u64>(end_address - address, bytes_to_boundary);
        std::invoke(visitor, address, chunk_size);
        address += chunk_size;
    }
    return true;
}

// Coalesces the protections that must be restored when a GPU mapping is published. A missing
// permission means the page has no active watcher and should retain the mapping's default access.
template <typename StateAtPage, typename Emit>
[[nodiscard]] bool ReplayActivePageProtectionRuns(VAddr address, u64 size, u64 page_size,
                                                  StateAtPage&& state_at_page, Emit&& emit) {
    if (!Detail::IsValidAddressRange(address, size) || page_size == 0 || address % page_size != 0 ||
        size % page_size != 0) {
        return false;
    }

    using OptionalPermission = std::remove_cvref_t<std::invoke_result_t<StateAtPage&, VAddr>>;
    using Permission = typename OptionalPermission::value_type;

    std::optional<Permission> pending_permission;
    VAddr pending_address{};
    u64 pending_size{};
    const auto release_pending = [&] {
        if (pending_permission) {
            std::invoke(emit, pending_address, pending_size, *pending_permission);
            pending_permission.reset();
            pending_size = 0;
        }
    };

    const VAddr end_address = address + size;
    for (VAddr page_address = address; page_address < end_address; page_address += page_size) {
        OptionalPermission permission = std::invoke(state_at_page, page_address);
        if (!permission) {
            release_pending();
            continue;
        }
        if (pending_permission && *pending_permission == *permission) {
            pending_size += page_size;
            continue;
        }
        release_pending();
        pending_permission = *permission;
        pending_address = page_address;
        pending_size = page_size;
    }
    release_pending();
    return true;
}

// Serializes the host tracking registration lifecycle with the mapped-range view used by page
// watcher updates. Publication happens only after registration and deferred protection replay;
// retirement happens before host tracking is unregistered.
class MappedPageProtectionState {
public:
    template <typename Register, typename Replay>
    [[nodiscard]] bool PublishMappedRange(VAddr address, u64 size, Register&& register_range,
                                          Replay&& replay) {
        if (!Detail::IsValidAddressRange(address, size)) {
            return false;
        }

        std::scoped_lock lifecycle_lock{lifecycle_mutex};
        const auto visibility_gains = UncoveredSubranges(address, size);
        for (const auto& [gain_address, gain_size] : visibility_gains) {
            std::invoke(register_range, gain_address, gain_size);
            std::invoke(replay, gain_address, gain_size);
        }
        {
            std::scoped_lock lock{mutex};
            mapped_ranges += std::make_pair(Interval::right_open(address, address + size), u32{1});
        }
        return true;
    }

    template <typename Unregister>
    [[nodiscard]] bool RetireMappedRange(VAddr address, u64 size, Unregister&& unregister_range) {
        if (!Detail::IsValidAddressRange(address, size)) {
            return false;
        }

        std::scoped_lock lifecycle_lock{lifecycle_mutex};
        std::vector<Range> visibility_losses;
        {
            std::scoped_lock lock{mutex};
            std::vector<CountedRange> covered_ranges;
            ForEachClipped(address, size, [&](VAddr mapped_address, VAddr mapped_end, u32 count) {
                covered_ranges.push_back({mapped_address, mapped_end, count});
                if (count == 1) {
                    AppendCoalesced(visibility_losses, mapped_address, mapped_end);
                }
            });
            for (const auto& covered : covered_ranges) {
                mapped_ranges -= std::make_pair(
                    Interval::right_open(covered.address, covered.end_address), u32{1});
            }
        }
        for (const auto& [loss_address, loss_size] : visibility_losses) {
            std::invoke(unregister_range, loss_address, loss_size);
        }
        return true;
    }

    [[nodiscard]] bool Contains(VAddr address, u64 size) const {
        if (!Detail::IsValidAddressRange(address, size)) {
            return false;
        }
        std::shared_lock lock{mutex};
        return boost::icl::contains(mapped_ranges, Interval::right_open(address, address + size));
    }

    template <typename Visitor>
    [[nodiscard]] bool ForEachMappedSubrange(VAddr address, u64 size, Visitor&& visitor) const {
        if (!Detail::IsValidAddressRange(address, size)) {
            return false;
        }
        std::shared_lock lock{mutex};
        bool has_pending{};
        VAddr pending_address{};
        VAddr pending_end{};
        const auto release_pending = [&] {
            if (has_pending) {
                std::invoke(visitor, pending_address,
                            static_cast<u64>(pending_end - pending_address));
                has_pending = false;
            }
        };
        ForEachClipped(address, size, [&](VAddr mapped_address, VAddr mapped_end, u32) {
            if (has_pending && pending_end == mapped_address) {
                pending_end = mapped_end;
                return;
            }
            release_pending();
            has_pending = true;
            pending_address = mapped_address;
            pending_end = mapped_end;
        });
        release_pending();
        return true;
    }

private:
    using CountedRanges = boost::icl::interval_map<VAddr, u32>;
    using Interval = typename CountedRanges::interval_type;
    using Range = std::pair<VAddr, u64>;
    struct CountedRange {
        VAddr address;
        VAddr end_address;
        u32 count;
    };

    static void AppendCoalesced(std::vector<Range>& ranges, VAddr address, VAddr end_address) {
        if (!ranges.empty() && ranges.back().first + ranges.back().second == address) {
            ranges.back().second = end_address - ranges.back().first;
            return;
        }
        ranges.emplace_back(address, end_address - address);
    }

    template <typename Visitor>
    void ForEachClipped(VAddr address, u64 size, Visitor&& visitor) const {
        const VAddr end_address = address + size;
        const Interval search = Interval::right_open(address, end_address);
        auto it = mapped_ranges.lower_bound(search);
        const auto end_it = mapped_ranges.upper_bound(search);
        for (; it != end_it; ++it) {
            const VAddr mapped_address = std::max(address, it->first.lower());
            const VAddr mapped_end = std::min(end_address, it->first.upper());
            if (mapped_address < mapped_end) {
                std::invoke(visitor, mapped_address, mapped_end, it->second);
            }
        }
    }

    [[nodiscard]] std::vector<Range> UncoveredSubranges(VAddr address, u64 size) const {
        std::shared_lock lock{mutex};
        std::vector<Range> gaps;
        VAddr cursor = address;
        const VAddr end_address = address + size;
        ForEachClipped(address, size, [&](VAddr mapped_address, VAddr mapped_end, u32) {
            if (cursor < mapped_address) {
                gaps.emplace_back(cursor, mapped_address - cursor);
            }
            cursor = std::max(cursor, mapped_end);
        });
        if (cursor < end_address) {
            gaps.emplace_back(cursor, end_address - cursor);
        }
        return gaps;
    }

    CountedRanges mapped_ranges;
    mutable Common::SharedFirstMutex mutex;
    std::mutex lifecycle_mutex;
};

// Keeps sparse-placeholder and host-mapped ownership distinct while exposing their union to GPU
// residency queries. Unmatched retirement of one owner kind must not consume the other kind.
class LogicalGpuMappingState {
public:
    [[nodiscard]] bool PublishHostMappedRange(VAddr address, u64 size) {
        return PublishOwner(host_owners, address, size);
    }

    [[nodiscard]] bool RetireHostMappedRange(VAddr address, u64 size) {
        return RetireOwner(host_owners, address, size);
    }

    [[nodiscard]] bool PublishSparsePlaceholderRange(VAddr address, u64 size) {
        return PublishOwner(sparse_owners, address, size);
    }

    [[nodiscard]] bool RetireSparsePlaceholderRange(VAddr address, u64 size) {
        return RetireOwner(sparse_owners, address, size);
    }

    [[nodiscard]] bool Contains(VAddr address, u64 size) const {
        return visible_ranges.Contains(address, size);
    }

    template <typename Visitor>
    [[nodiscard]] bool ForEachMappedSubrange(VAddr address, u64 size, Visitor&& visitor) const {
        return visible_ranges.ForEachMappedSubrange(address, size,
                                                    std::forward<Visitor>(visitor));
    }

    // Return only the ranges owned by the requested mapping kind.  GPU teardown callers must
    // not apply host-page alignment/retirement rules to sparse placeholders (or vice versa) when
    // the two logical owners overlap.
    template <typename Visitor>
    [[nodiscard]] bool ForEachHostMappedSubrange(VAddr address, u64 size,
                                                  Visitor&& visitor) const {
        return host_owners.ForEachMappedSubrange(address, size, std::forward<Visitor>(visitor));
    }

    template <typename Visitor>
    [[nodiscard]] bool ForEachSparsePlaceholderSubrange(VAddr address, u64 size,
                                                         Visitor&& visitor) const {
        return sparse_owners.ForEachMappedSubrange(address, size,
                                                   std::forward<Visitor>(visitor));
    }

    // Preserve offsets from the requested base, including any leading or interior holes, but
    // exclude trailing bytes that have no GPU mapping. Sparse placeholders are logical mappings
    // too: their unbacked bytes must remain in the extent until that placeholder is retired.
    [[nodiscard]] u64 MappedExtent(VAddr address, u64 size) const {
        u64 extent{};
        const bool valid = visible_ranges.ForEachMappedSubrange(
            address, size, [&](VAddr mapped_address, u64 mapped_size) {
                extent = mapped_address - address + mapped_size;
            });
        return valid ? extent : 0;
    }

private:
    [[nodiscard]] bool PublishOwner(MappedPageProtectionState& owners, VAddr address, u64 size) {
        bool transitions_succeeded = true;
        const bool owner_published = owners.PublishMappedRange(
            address, size,
            [this, &transitions_succeeded](VAddr gain_address, u64 gain_size) {
                transitions_succeeded &= visible_ranges.PublishMappedRange(
                    gain_address, gain_size, [](VAddr, u64) {}, [](VAddr, u64) {});
            },
            [](VAddr, u64) {});
        return owner_published && transitions_succeeded;
    }

    [[nodiscard]] bool RetireOwner(MappedPageProtectionState& owners, VAddr address, u64 size) {
        bool transitions_succeeded = true;
        const bool owner_retired = owners.RetireMappedRange(
            address, size, [this, &transitions_succeeded](VAddr loss_address, u64 loss_size) {
                transitions_succeeded &=
                    visible_ranges.RetireMappedRange(loss_address, loss_size, [](VAddr, u64) {});
            });
        return owner_retired && transitions_succeeded;
    }

    MappedPageProtectionState visible_ranges;
    MappedPageProtectionState host_owners;
    MappedPageProtectionState sparse_owners;
};

} // namespace VideoCore
