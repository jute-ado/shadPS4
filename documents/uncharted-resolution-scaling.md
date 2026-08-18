# Uncharted resolution scaling

Status: the emulator now exposes the existing guest-display dimensions as a
per-game setting with 720p, 1080p, 1440p, 4K, and custom presets. This is not
yet native-resolution rendering.

## Proven boundary

- `internalScreenWidth` and `internalScreenHeight` feed
  `sceVideoOutGetResolutionStatus`. A 3840x2160 setting therefore asks the
  guest for a 4K display mode and permits 4K presentation.
- The setting is independent from the window size and is saved as a per-game
  override. Startup logging records the exact requested guest-display size.
- An earlier PS4 Pro U2 probe presented a correct moving 3840x2160 image, but
  the guest registered 1920x1080 VideoOut buffers. That run proved 4K output,
  not a 4K game render target.
- PS4 Pro mode changes guest-visible hardware behavior, libraries, tiling, and
  memory limits. It must not be enabled merely to enlarge the host output.

## Terminology

- **Window resolution** is the host window size.
- **Guest-display resolution** is the mode reported through VideoOut.
- **Presented resolution** is the final swapchain or captured-frame size.
- **Native render resolution** is the size of the guest color/depth rendering
  work after a complete, coherent host-side scale transform.

Only the first three are currently configurable. A larger window, swapchain,
or screenshot must never be described as native rendering when the source
attachments remain at their original size.

## Native scaling requirements

A generic renderer scale cannot change guest memory layout. It must instead
maintain a host extent alongside every scalable image and consistently apply
the same transform to:

1. color and depth attachment allocation;
2. viewport, scissor, render area, and clear rectangles;
3. sampled views and attachment-feedback paths;
4. image copies, resolves, blits, uploads, downloads, and readbacks;
5. presentation and screenshot metadata;
6. aliasing, subresource ranges, and synchronization.

Buffers, compressed/video surfaces, one-dimensional resources, and resources
whose guest byte layout is externally observed must remain unscaled unless a
separate conversion path proves correctness. Scale identity must also be part
of every affected host-image/view cache key.

## Test-first acceptance plan

- [x] Presets recognize 720p, 1080p, 1440p, and 4K while preserving custom
  dimensions.
- [x] Guest-display dimensions are per-game overrideable.
- [x] Startup logging reports the exact requested dimensions.
- [x] Full settings tests and a fresh application build pass.
- [x] Re-run Nvidia PS4 and PS4 Pro probes at 3840x2160 and record the actual
  registered guest buffers, presented dimensions, stability, and timing.
- [ ] Add synthetic RED tests for scalable-image eligibility and host extents.
- [ ] Add RED tests for viewport/scissor, copy/resolve, sampling, and readback
  transforms before changing renderer behavior.
- [ ] Prove major game color and depth attachments use the scaled host extent;
  the swapchain size alone is insufficient.
- [ ] Validate U1, U2, and U3 Nvidia routes against their native-resolution
  controls, including temporal visual checks and performance metrics.
- [ ] Merge only after clean unit, application, and PS4 Test Lab gates.

## Rejected shortcuts

- No title-ID, executable-address, route, asset, or scene-specific scaling.
- No guest-memory multiplication presented as a generic renderer feature.
- No PS4 Pro requirement unless current Nvidia evidence shows that the game
  actually selects a useful Pro render path and remains at least as stable.
- No AMD 4K gate for this work; the requested validation target is Nvidia.

## Nvidia mode comparison (2026-08-18)

Both probes used the same U2 sewer checkpoint, controller route, game/seed
assets, 3840x2160 guest-display request, 3840x2160 presented-frame checkpoint,
and RTX 4080. Both exited with code 0 in about 111.2 seconds with no device
loss or known renderer assertion.

| Profile | Guest VideoOut buffers | Visual result | Duration |
| --- | --- | --- | --- |
| PS4 | 1920x1080, three buffers | changed strongly from the retained checkpoint | 111.141 s |
| PS4 Pro | 1920x1080, three buffers | passed the retained 4K presentation checkpoint | 111.187 s |

The PS4 Pro capture had mean absolute difference 0.00297 and cosine difference
0.00237 from the retained reference. The base-PS4 capture represented a
different visual state and had mean absolute difference 0.16316. This makes
PS4 Pro the better current profile for stable 4K presentation in this
collection, with no meaningful performance difference in the measured route.
It does **not** prove native 4K: both profiles still registered 1920x1080 guest
buffers.
