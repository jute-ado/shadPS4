# U1 upstream contribution boundaries

This is a concise map of the generic runtime work currently needed for U1 to
reach and remain stable in 3D gameplay. It is not a request to open pull
requests, and the commits should not be submitted as one large series.

## Runtime boundaries

1. **System-managed executable mappings**
   - Ensure the main executable's system mappings are admitted consistently.
   - Representative fork commit: `f840fd3c`.

2. **Depth association format safety**
   - Reject incompatible depth associations instead of reusing an unsafe view.
   - Representative fork commit: `84b6b1f2`.

3. **GPU buffer residency and page publication**
   - Preserve CPU/GPU page state, mapped-range visibility, and DMA/BDA buffer
     residency across the actual ownership handoff.
   - Representative commits include `3eb52128`, `07bec1f5`, `a65b0597`,
     `08a77ed5`, and `d0b38d50`.

4. **Graphics-event edge semantics and EOP ordering**
   - Publish guest events at the correct parser boundary while keeping queued
     submission ownership alive.
   - Representative commits include `f83e9780`, `919b2ac9`, `4213d059`, and
     `e227a31f`.

5. **Bound static-buffer ranges**
   - Prevent descriptors from exposing bytes outside the guest-declared static
     range.
   - Representative fork commit: `3f71b597`.

6. **Presentation/readable-frame ownership**
   - Keep presented images alive and readable until the consumer is finished.
   - Representative commits: `69f9a6ac` and `4699c660`.

7. **ADDR64 decoding and descriptor-relative reads**
   - Decode MUBUF ADDR64 correctly, bind the read to the selected storage
     descriptor, and use a robust descriptor-relative 64-bit offset.
   - The progression is `59bbd421` -> `7b182423` -> `5bebf5ab`.
   - The final step fixes the intermittent white hair/web frame without
     reverting the earlier orange-hair lighting correction.

8. **Queued and nested command-buffer lifetime**
   - Retain top-level and nested indirect command buffers until their queued
     work has completed.
   - Representative commits: `9fef58e6` and `5d693397`.

9. **Synchronous host-compilation lifecycle guard**
   - Pause guest lifecycle time only while synchronous pipeline/module
     compilation blocks the host; keep async workers unpaused.
   - Representative commit: `ae945e60`.

## Separate features

- Test Lab controller replay, route recording, diagnostics, and RenderDoc
  integration are fork tooling. They are valuable for validation but are not
  required runtime changes for U1 3D gameplay.
- Native 2x internal resolution is an optional renderer feature and should be
  reviewed independently from compatibility fixes.
- Async graphics compilation is a usability/performance feature with its own
  fallback and lifetime tests.

Each upstreamable boundary should be rebased independently, reduced to its
smallest generic form, and protected by focused tests plus an upstream-control
game route. No private games, saves, captures, maps, or raw logs belong in an
upstream contribution.
