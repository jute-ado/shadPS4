// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/assert.h"
#include "common/types.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace Serialization {

template <typename T>
concept Container = requires(T t) {
    typename T::iterator;
    { t.begin() } -> std::same_as<typename T::iterator>;
    { t.end() } -> std::same_as<typename T::iterator>;
    { t.size() } -> std::convertible_to<std::size_t>;
};

struct Archive {
    void Alloc(size_t size) {
        container.resize(size);
    }

    void Grow(size_t size) {
        container.resize(container.size() + size);
    }

    void Merge(const Archive& ar) {
        container.insert(container.end(), ar.container.cbegin(), ar.container.cend());
        offset = container.size();
    }

    [[nodiscard]] size_t SizeBytes() const {
        return container.size();
    }

    u8* CurrPtr() {
        return container.data() + offset;
    }

    [[nodiscard]] size_t RemainingBytes() const {
        return offset <= container.size() ? container.size() - offset : 0;
    }

    bool TryAdvance(size_t size) {
        if (size > RemainingBytes()) {
            return false;
        }
        offset += size;
        return true;
    }

    void Advance(size_t size) {
        ASSERT_MSG(offset + size <= container.size(),
                   "Invalid or corrupted deserialization container/shader cache");
        offset += size;
    }

    std::vector<u8>&& TakeOff() {
        offset = 0;
        return std::move(container);
    }

    [[nodiscard]] bool IsEoS() const {
        return offset >= container.size();
    }

    Archive() = default;
    explicit Archive(std::vector<u8>&& v) : container{v} {}

private:
    size_t offset{};
    std::vector<u8> container{};

    friend struct Writer;
    friend struct Reader;
};

struct Writer {
    template <typename T>
    void Write(const T* ptr, size_t size) {
        if (size > std::numeric_limits<size_t>::max() - ar.offset) {
            throw std::length_error{"Serialization archive size overflow"};
        }
        const size_t end_offset = ar.offset + size;
        if (end_offset > ar.container.size()) {
            ar.container.resize(end_offset);
        }
        std::memcpy(ar.CurrPtr(), reinterpret_cast<const void*>(ptr), size);
        ar.Advance(size);
    }

    template <typename T>
        requires(!Container<T>)
    void Write(const T& value) {
        const auto size = sizeof(value);
        Write(&value, size);
    }

    void Write(const auto& v) {
        Write(v.size());
        for (const auto& elem : v) {
            Write(elem);
        }
    }

    void Write(const std::string& s) {
        Write(s.size());
        Write(s.c_str(), s.size());
    }

    Writer() = delete;
    explicit Writer(Archive& ar_) : ar{ar_} {}

    Archive& ar;
};

struct Reader {
    static constexpr size_t DefaultMaxContainerElements = 1U << 20;
    static constexpr size_t DefaultMaxStringBytes = 1U << 20;

    template <typename T>
    bool Read(T* ptr, size_t size) {
        if (!valid || size > ar.RemainingBytes()) {
            valid = false;
            return false;
        }
        std::memcpy(reinterpret_cast<void*>(ptr), ar.CurrPtr(), size);
        valid = ar.TryAdvance(size);
        return valid;
    }

    template <typename T>
        requires(!Container<T>)
    bool Read(T& value) {
        const auto size = sizeof(value);
        return Read(&value, size);
    }

    bool Read(bool& value) {
        u8 encoded{};
        if (!Read(encoded) || encoded > 1) {
            valid = false;
            value = false;
            return false;
        }
        value = encoded != 0;
        return true;
    }

    template <Container T>
    bool Read(T& v, size_t max_elements = DefaultMaxContainerElements) {
        size_t num_elements{};
        if (!Read(num_elements)) {
            return false;
        }
        using Value = typename T::value_type;
        constexpr size_t MinEncodedElementBytes =
            Container<Value> ? sizeof(size_t) : std::max<size_t>(sizeof(Value), 1);
        const size_t remaining_capacity = ar.RemainingBytes() / MinEncodedElementBytes;
        if (num_elements > max_elements || num_elements > remaining_capacity ||
            num_elements > v.max_size()) {
            valid = false;
            return false;
        }
        v.clear();
        try {
            for (size_t i = 0; i < num_elements; ++i) {
                v.emplace_back();
                if (!Read(v.back())) {
                    v.clear();
                    return false;
                }
            }
        } catch (const std::exception&) {
            valid = false;
            v.clear();
            return false;
        }
        return true;
    }

    bool Read(std::string& s, size_t max_length = DefaultMaxStringBytes) {
        size_t length{};
        if (!Read(length) || length > max_length || length > ar.RemainingBytes()) {
            valid = false;
            s.clear();
            return false;
        }
        try {
            s.resize(length);
        } catch (const std::exception&) {
            valid = false;
            s.clear();
            return false;
        }
        if (!Read(s.data(), length)) {
            s.clear();
            return false;
        }
        return true;
    }

    [[nodiscard]] bool IsValid() const {
        return valid;
    }

    Reader() = delete;
    explicit Reader(Archive& ar_) : ar{ar_} {}

    Archive& ar;
    bool valid{true};
};

} // namespace Serialization
