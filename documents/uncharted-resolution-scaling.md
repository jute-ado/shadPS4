# Uncharted resolution scaling

Status: guest-display dimensions remain independently configurable. A separate
experimental host-render scale now has a strict 100%/200% policy. Its corrected
200% path has passed the focused renderer contracts, the full unit suite, a
reviewed 100-frame Nvidia U1 cave replay, and a U2 4K correctness/performance
pair. U3 reaches the same pre-existing EOP lifecycle crash at native and 200%,
so this is not yet a collection-wide compatibility claim.

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

The first three and the experimental host-render scale are configurable. A
larger window, swapchain, or screenshot must never be described as native
rendering when the source attachments remain at their original size.

The development branch additionally exposes an **Internal Render Resolution**
choice (`Native (100%)` or `2x (200%)`) as a global or per-game override and a
matching `--internal-resolution-scale 100|200` CLI option. This setting is
experimental until the full acceptance list below passes.

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
- [x] Add synthetic RED tests for scalable-image eligibility and host extents.
- [x] Add RED coverage for the current viewport/scissor, copy/resolve,
  sampling, readback, coordinate-remapping, exact-binding fallback, and
  atomic-attachment contracts before accepting their renderer behavior.
- [x] Prove major U1 color and depth attachments are allocated at the scaled
  host extent and remain one coherent post-refresh scale transaction.
- [x] Complete one reviewed U1 200% temporal route with coherent scaled
  attachments; the swapchain size alone is not counted as evidence.
- [x] Validate U1 and U2 Nvidia routes against their native-resolution controls,
  including temporal visual checks and performance metrics.
- [ ] Re-run U3 200% correctness after the scale-independent EOP lifecycle crash
  is fixed; native and 200% currently fail at the same boundary.
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

## Host-scaling investigation (2026-08-19)

The first renderer implementation keeps guest allocation and addressing at the
original size while maintaining alternate Vulkan backings for eligible 2D,
single-level, single-sample, uncompressed color/depth attachments. Unsupported
images and mixed attachment sets fall back transactionally to native scale.
Uploads, downloads, storage access, integer-coordinate access, and externally
visible copies also force a native backing unless their transform is proven.

Two early U1 cave failures established reusable renderer requirements:

1. Vulkan `FragCoord` is in host pixels. Passing the doubled value directly to
   PS4 shaders corrupted screen-space passes. The branch now uses a draw-local
   inverse-scale push constant and maps only X/Y back into guest pixel space;
   the complete push block remains within Vulkan's guaranteed 128-byte limit.
2. Target promotion must occur after the real refresh/upload and must be
   reapplied as one attachment transaction. Promoting during preparation alone
   allowed a later color upload to return one target to native size while its
   depth attachment remained 2x, producing stale silhouettes and invalid
   lighting. A focused scale-transaction RED now protects the all-or-native
   rule, and the runtime re-applies the chosen scale after refresh.

The Test Lab route itself has a separate cold-compilation constraint. A capture
scheduled by wall clock can observe only the explicit `Emulation Paused`
overlay or no visible frame while host pipelines compile. Such runs are
unavailable, not visual passes or failures. Resolution acceptance therefore
requires reviewed gameplay membership plus temporal evidence; raw timeout or
checkpoint counters are insufficient.

### Accepted U1 evidence

After the post-refresh attachment transaction repair, a fresh RTX 4080 cave
replay completed normally and produced 100 distinct temporal frames. The
temporal visual contract passed with zero abrupt or invisible returns. Reviewed
first, middle, and final frames retained the native composition without the
earlier stale silhouette, overbright wet geometry, or mixed color/depth extent.
Checkpoint mean-absolute differences ranged from 0.0173 to 0.0191 and cosine
differences from 0.095 to 0.106 against the retained native control.

This establishes the tested U1 cave boundary only. It does not prove every U1
scene and does not replace a manual playthrough. The U2 and U3 boundaries are
recorded separately below.

### Current Nvidia U2 and U3 evidence (2026-08-20)

The rebuilt post-upstream branch passed the U2 sewer 200% visual checkpoint on
an RTX 4080. The run used the base PS4 profile, logged the 200% renderer mode,
and promoted the primary 1920x1080 targets to 3840x2160. Its retained checkpoint
had mean-absolute difference 0.00740 and cosine difference 0.01537, with no
device loss or renderer assertion.

The matching three-trial U2 performance pair was effectively refresh-limited:

| Host-render scale | Mean FPS | Median frame time | p95 | p99 | Stutters |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100% | 59.999 | 16.554 ms | 17.531 ms | 17.600 ms | 0% |
| 200% | 60.003 | 16.575 ms | 17.522 ms | 17.798 ms | 0% |

The U3 bottle route is not valid 4K acceptance evidence yet. Both the native
control and 200% run entered the bar scene, then hit the same guest
`FrameBegin` EOP assertion and host access violation after about 55 seconds.
The nearly identical native/200% failure boundary rules out a scale-specific
regression on this route, but it does not establish later-scene correctness.
