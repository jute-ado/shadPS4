// CPU-only RED specification for a draw-resource admission bug.
//
// BindBuffers currently copies Flatbuf/clip-plane/LDS payloads into the recyclable stream ring
// before the draw's vertex/index/indirect preparation has completed.  A later preparation can
// wrap that ring and overwrite the descriptor payload.  This test deliberately contains no
// Vulkan or game fixture; it records the smallest ring-buffer sequence that makes the defect
// observable and checks the production source for the missing deferred-binding seam.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

class FakeStream final {
public:
    explicit FakeStream(std::size_t capacity) : storage_(capacity, std::byte{0xa5}) {}

    struct Reservation {
        std::size_t offset{};
        std::size_t size{};
    };

    [[nodiscard]] Reservation Copy(std::span<const std::byte> bytes) {
        if (offset_ + bytes.size() > storage_.size()) {
            // The real StreamBuffer waits for the old generation, then reuses the ring from 0.
            offset_ = 0;
        }
        std::memcpy(storage_.data() + offset_, bytes.data(), bytes.size());
        const Reservation result{offset_, bytes.size()};
        offset_ += bytes.size();
        return result;
    }

    [[nodiscard]] std::span<const std::byte> Read(const Reservation reservation) const {
        return std::span<const std::byte>{storage_}.subspan(reservation.offset, reservation.size);
    }

private:
    std::vector<std::byte> storage_;
    std::size_t offset_{};
};

[[nodiscard]] std::string ReadFile(const char* path) {
    std::ifstream input{path, std::ios::binary};
    if (!input) {
        return {};
    }
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

[[nodiscard]] std::string FunctionBody(const std::string& source, std::string_view begin,
                                       std::string_view end) {
    const auto first = source.find(begin);
    const auto last = source.find(end, first == std::string::npos ? 0 : first + begin.size());
    if (first == std::string::npos || last == std::string::npos) {
        return {};
    }
    return source.substr(first, last - first);
}

void Expect(bool condition, std::string_view message, int& failures) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %.*s\n", static_cast<int>(message.size()), message.data());
        ++failures;
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: test_stream_special_binding_admission_red.exe <vk_rasterizer.cpp>\n");
        return 2;
    }

    int failures = 0;
    FakeStream stream{16};
    constexpr std::array flat_data{std::byte{'F'}, std::byte{'L'}, std::byte{'A'}, std::byte{'T'},
                                   std::byte{'D'}, std::byte{'E'}, std::byte{'S'}, std::byte{'C'}};
    constexpr std::array vertex_data{std::byte{'V'}, std::byte{'E'}, std::byte{'R'},
                                     std::byte{'T'}, std::byte{'E'}, std::byte{'X'},
                                     std::byte{'P'}, std::byte{'R'}, std::byte{'E'},
                                     std::byte{'P'}, std::byte{'A'}, std::byte{'R'}};

    // RED setup: an eager utility descriptor is captured before a later wait-capable preparation
    // wraps the stream.  The reservation aliases the newly prepared vertex bytes.
    const auto flat = stream.Copy(flat_data);
    static_cast<void>(stream.Copy(vertex_data));
    Expect(!std::ranges::equal(stream.Read(flat), flat_data),
           "eager utility stream reservation is overwritten by a later preparation wrap", failures);

    const auto source = ReadFile(argv[1]);
    if (source.empty()) {
        std::fprintf(stderr, "SOURCE ERROR: could not read %s\n", argv[1]);
        return 2;
    }
    const auto bind = FunctionBody(source, "void Rasterizer::BindBuffers(",
                                   "void Rasterizer::BindTextures(");
    Expect(!bind.empty(), "Rasterizer::BindBuffers source section exists", failures);

    // Desired GREEN wiring: BindBuffers must snapshot utility payloads, and a later owner phase
    // must reserve/copy them only after all wait-capable vertex/index/indirect preparation.  The
    // clean baseline has direct Copy/Map calls here, so this source contract is expected to RED.
    for (const auto forbidden : {"vk_buffer.Copy(stage.flattened_ud_buf.data()",
                                "vk_buffer.Copy(planes.data()", "lds_buffer.Map(lds_size"}) {
        Expect(bind.find(forbidden) == std::string::npos,
               "utility descriptor payload is still eagerly copied in BindBuffers", failures);
    }
    Expect(source.find("FinalizeUtilityBufferBindings") != std::string::npos,
           "production has a late utility-buffer finalization seam", failures);

    if (failures != 0) {
        std::fprintf(stderr,
                     "RED: %d draw-resource admission assertion(s) failed; no GPU was launched\n",
                     failures);
        return 1;
    }
    std::puts("GREEN: utility descriptors are deferred past draw-buffer preparation");
    return 0;
}
