# Uncharted resolution-scaling investigation

Status: 3840x2160 output works and presents a correct moving image, but the
current configuration is not proven native rendering. Do not describe it as
native 4K.

## Proven boundary

- `internalScreenWidth` and `internalScreenHeight` are exposed through
  `sceVideoOutGetResolutionStatus` and can request a 3840x2160 output mode.
- A PS4 Pro U2 probe cleanly presented a 3840x2160 screenshot at 60.00 FPS in
  a stationary scene, with no frame interval above 25 ms in the measured
  300-frame window.
- The same run registered guest VideoOut buffers at 1920x1080. The 4K capture
  therefore proves 4K presentation, not a 4K guest render target.
- A visual comparison retained the same scene structure and brightness
  invariants, but correctly reported changed dimensions relative to the pinned
  1280x720 reference.

## Rejected shortcuts

- Do not key resolution, memory allocation, or shader behavior to a title ID,
  executable address, route, or asset.
- Do not call a larger swapchain or screenshot native rendering when guest
  color/depth targets remain at their original dimensions.
- Do not import Bloodborne-specific expanded-memory or executable-patch hacks.
  That fork's own history records that high-resolution rendering depends on a
  guest patch; its older emulator-side approach special-cased game IDs and
  multiplied console memory limits.

## Acceptance checklist

- [x] Log the configured internal resolution at startup with a focused
  synthetic test.
- [x] Prove the 3840x2160 request, clean process outcome, visible output, and
  stationary frame timing.
- [x] Inspect guest VideoOut registration and reject the current run as native
  4K because the guest buffer is 1920x1080.
- [ ] Design a general render-scale transform covering render targets,
  depth targets, viewports, scissors, resolves, copies, and presentation while
  preserving guest memory layout and readback semantics.
- [ ] Add synthetic RED tests for coordinate and subresource transforms before
  changing renderer logic.
- [ ] Prove major guest color/depth attachments are 3840x2160 in a separate
  RenderDoc diagnostic; the swapchain size alone is insufficient.
- [ ] Validate temporal output and performance on short U1/U2/U3 routes before
  any broad PS4 gate.
- [ ] Investigate higher refresh rates only after native-resolution correctness.
  A vblank/presentation rate is not proof that guest simulation, animation,
  audio, and render cadence are correct at 120 FPS.
