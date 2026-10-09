// CPU-only RED specification for the PM4-to-rasterizer draw admission seam.
//
// This is intentionally kept out of the shared checkout.  It models the only safe shape for
// retrying a transient pipeline/resource miss: snapshot the draw after PM4 state writes, retain it
// at the owner boundary, and retry the snapshot later.  Replaying the raw PM4 packet is forbidden
// because it repeats register/marker/diagnostic side effects.

#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace {

enum class Geometry : std::uint8_t {
    Actor,
    Horse,
};

enum class Admission : std::uint8_t {
    Ready,
    Retry,
};

struct DrawSnapshot {
    Geometry geometry;
    std::uint32_t state_generation;
};

// Intended owner-side behavior.  It deliberately has no Vulkan, guest pointer, or PM4-header
// dependency, and therefore cannot replay a packet accidentally.
class GeometryOwnerModel {
public:
    void SetContextState() {
        ++state_generation_;
        ++state_writes_;
    }

    void SnapshotDraw(Geometry geometry) {
        pending_.push_back({.geometry = geometry, .state_generation = state_generation_});
        ++packet_advances_;
    }

    void ProcessOwnerBoundary(const std::function<Admission(const DrawSnapshot&)>& admit) {
        while (!pending_.empty()) {
            const DrawSnapshot draw = pending_.front();
            if (admit(draw) == Admission::Retry) {
                return;
            }
            emitted_.push_back(draw.geometry);
            pending_.erase(pending_.begin());
        }
    }

    [[nodiscard]] std::size_t PendingCount() const {
        return pending_.size();
    }

    [[nodiscard]] const std::vector<Geometry>& Emitted() const {
        return emitted_;
    }

    [[nodiscard]] std::uint32_t StateWrites() const {
        return state_writes_;
    }

    [[nodiscard]] std::uint32_t PacketAdvances() const {
        return packet_advances_;
    }

private:
    std::uint32_t state_generation_{};
    std::uint32_t state_writes_{};
    std::uint32_t packet_advances_{};
    std::vector<DrawSnapshot> pending_;
    std::vector<Geometry> emitted_;
};

[[nodiscard]] std::string ReadFile(const char* path) {
    std::ifstream input{path, std::ios::binary};
    if (!input) {
        return {};
    }
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

[[nodiscard]] bool ContainsAll(std::string_view source,
                               std::initializer_list<std::string_view> markers) {
    for (const auto marker : markers) {
        if (source.find(marker) == std::string_view::npos) {
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: test_shadow_geometry_admission_red.exe <rasterizer.cpp> "
                     "<liverpool.cpp>\n";
        return 2;
    }

    // Synthetic actor/horse progression: the first actor attempt misses transiently.  The owner
    // must stop at that draw, then emit actor and horse on the next boundary with each PM4 state
    // side effect applied exactly once.
    GeometryOwnerModel model;
    model.SetContextState();
    model.SnapshotDraw(Geometry::Actor);
    model.SetContextState();
    model.SnapshotDraw(Geometry::Horse);

    std::uint32_t actor_attempts{};
    model.ProcessOwnerBoundary([&](const DrawSnapshot& draw) {
        if (draw.geometry == Geometry::Actor && actor_attempts++ == 0) {
            return Admission::Retry;
        }
        return Admission::Ready;
    });
    if (model.PendingCount() != 2 || !model.Emitted().empty() || model.StateWrites() != 2 ||
        model.PacketAdvances() != 2) {
        std::cerr << "model contract failed at transient owner boundary\n";
        return 3;
    }

    model.ProcessOwnerBoundary([](const DrawSnapshot&) { return Admission::Ready; });
    if (model.PendingCount() != 0 || model.Emitted().size() != 2 ||
        model.Emitted()[0] != Geometry::Actor || model.Emitted()[1] != Geometry::Horse ||
        model.StateWrites() != 2 || model.PacketAdvances() != 2) {
        std::cerr << "model contract failed while advancing actor/horse\n";
        return 3;
    }

    const auto rasterizer = ReadFile(argv[1]);
    const auto liverpool = ReadFile(argv[2]);
    if (rasterizer.empty() || liverpool.empty()) {
        std::cerr << "could not read production source\n";
        return 2;
    }

    // RED assertion: production currently has void Draw/DrawIndirect calls with terminal early
    // returns and ProcessGraphics advances dcb after the call.  A future safe implementation must
    // expose an explicit retry admission and a draw snapshot/owner handoff; it must not replay the
    // raw PM4 packet.  Keep this assertion source-level so this test never launches a GPU.
    const bool safe_admission_seam =
        ContainsAll(rasterizer, {"DrawAdmission", "Retry"}) &&
        ContainsAll(liverpool, {"DrawAdmission", "Retry", "SnapshotDraw"});
    if (!safe_admission_seam) {
        std::cerr << "RED: no explicit draw-admission retry/snapshot seam in vk_rasterizer.cpp "
                     "and liverpool.cpp; actor/horse model passes but production drops are "
                     "terminal at the current call boundary\n";
        return 1;
    }

    std::cout << "GREEN: explicit draw-admission retry/snapshot seam present\n";
    return 0;
}
