# Uncharted 1 cave hair investigation

Status: the original orange hair-card lattice is fixed. The follow-on
intermittent white hair/web return passed the historical gates below, but a
newer motion-bearing route reproduced it twice and supersedes the old
"fixed" conclusion. No new U1 renderer fix is accepted. The current upstream
integration branch is also blocked separately by an AMD frame-EOP assertion.

## Target

Remove the bright orange/zebra hair-card lattice in the Chapter 2 cave while
preserving the compact dark hair silhouette. Fix a general renderer invariant,
not a title, shader, draw, or asset special case.

## Checklist

- [x] Confirm the defect against PS4 footage and a repeatable checkpoint capture.
- [x] Localize it to the hair draw's sign-sensitive direct/anisotropic lighting.
- [x] Rule out corrupt descriptors, textures, vertex normals/tangents, fragment
  interpolation, mip selection, shadow cascade selection/comparison, albedo
  degamma loss, normal-map channel/sign convention, and simple I/J swaps.
- [x] Rule out fused host lowering of guest MAD/MAC as the hair cause. Replaying
  all 243 FMAs as explicit multiply-add produced an identical frame hash.
- [x] Show that forcing the opposite front-face sign hides the artifact, but do
  not treat that diagnostic toggle as a valid fix.
- [x] Rule out a global negative-viewport/front-face remap. Vulkan and Mesa's
  AMD path keep the signed viewport transform and map `frontFace` directly to
  `PA_SU_SC_MODE_CNTL.FACE`, matching the current shadPS4 state conversion.
- [x] Rule out the logged `ClampHalfBorder` fallback as the hair cause. The
  hair draw uses two wrapping material samplers and one clamp-to-edge shadow
  sampler, so none uses the warning's clamp-to-border fallback.
- [x] Rule out the earlier scalar `ReadConst` fallback mismatch as the cause.
  The CLI resolved all 124 immediate reads; the mismatches are one zeroed
  SRT-walker page while BDA holds coherent material constants.
- [x] Audit the vertex-stage tangent-frame transform and interface against guest
  ISA and captured data. Position, normal, tangent, handedness, both UV sets,
  all six exports, and perspective interpolation match the guest program.
- [x] Reproduce the capture on a second replay path. NVIDIA CLI replay is bit
  exact; AMD replay cannot open this capture because that driver lacks the
  captured `VK_KHR_maintenance8` requirement, so it supplies no counterexample.
- [x] Rule out index assembly and an ignored guest swap mode. The exact
  checkpoint reports `VGT_DMA_INDEX_TYPE=0` (16-bit, no swap), and the captured
  bad triangle intentionally reaches Vulkan as `1435,1434,1436`.
- [x] Trace the first divergent lighting value to a general emulator invariant.
  The guest uses MUBUF `addr64`, but the decoder drops bit 15 and the translator
  ignores the 64-bit VGPR address. RenderDoc proves the resulting 64-byte
  "lighting table" is byte-for-byte the shader binary's first 64 bytes.
- [x] Add a focused synthetic RED test. With the production userdata-flattening
  pass enabled, it first exits 1 on `ReadConst not from constant memory`.
- [x] Decode MUBUF bit 15 and translate raw `addr64` loads through dynamic guest
  memory. Leave VGPR-derived reads out of SRT flattening; all 47 GCN tests pass.
- [x] Verify the 139-second cave checkpoint and separate RenderDoc diagnostic.
  The bright lattice is gone. Shader `cc2d0c16` no longer binds the erroneous
  64-byte shader-code buffer and instead uses the physical dynamic-read path.
- [x] Pass `local-ps4-uncharted-focus` (3/3) and `local-ps4-regression`
  (12/12). RenderDoc measures the exact 9,804-index draw at 8.2-9.2 us versus
  7.2-8.2 us before the fix, a roughly 2 us absolute cost.
- [x] Record and sync clean PS4 fork observation
  `679bd79f-7b87-4fb7-8018-f85bc346d09e` at emulator commit `59bbd421`.

## Intermittent white material flash follow-up

The corrected cave rendering later exposed a separate, intermittent material
residency defect: hair and nearby spider-web geometry could briefly turn white.
The repeatable temporal signature is a dark frame, one or two bright-white
frames, then an immediate return to dark. The accepted integration branch
produced two strict abrupt-return violations in a 300-frame capture, with a
maximum adjacent-frame difference of `0.06362248369404`.

- [x] Reduce the defect to a synthetic RED. An ADDR64 MUBUF load retained its
  64-bit guest address for the shader, but resource tracking exposed only one
  guest buffer instead of the required address-source buffer plus destination.
- [x] Add an explicit `ReadConstBufferAddr64` IR operation carrying the source
  descriptor, 64-bit address, and dword offset. Resource tracking now registers
  the source guest buffer before the draw; the backend continues to perform the
  direct buffer-device-address read.
- [x] Pass all 48 focused GCN tests, including
  `mubuf_addr64_tracks_source_buffer_residency`.
- [x] Pass five consecutive strict cave trials: 200/200 frames were distinct,
  every trial reported zero abrupt returns and zero invisible flashes, and the
  worst adjacent-frame difference was `0.0317647030455701`.
- [x] Pass a supplemental 300-frame/30-second capture with zero abrupt returns
  and maximum adjacent-frame difference `0.027548541609931`.
- [x] Pass `local-ps4-uncharted-focus` (3/3) and the separate GPU diagnostic.
  The GPU capture completed without a finding.
- [x] Preserve the cross-game performance result: five valid PT trials report
  `29.9989` mean FPS, `34.09054 ms` p95 frame time, and zero measured stutter.
- [x] Pass the complete PS4 regression gate. Two earlier 11/12 runs exited with
  `0x80000003` at `image.cpp:258 GetBarriers` after 76.682 and 79.967 seconds.
  The underlying texture-cache selection used lexicographic comparison for a
  two-dimensional mip/layer extent, allowing an image with too few mip levels
  to be reused when it happened to have more layers. Synthetic tests now require
  component-wise containment, and `FindImage` rejects an image unless both
  dimensions contain the requested extent. The final 12/12 run is
  `f16aef8f-47ec-48f4-9674-cbb49a7db9b4`; late PT completed in 115.887 seconds,
  Uncharted audio passed, and five performance trials retained 29.9993 mean FPS,
  34.0785 ms p95 frame time, and zero measured stutter.

Five focused late-PT trials produced four passes and one preserved visual-only
`changed` result (`530ee00e-10a4-4a8d-9fe4-767a3b7b2c8e`). That fifth run had
no crash or forbidden marker and passed all temporal invariants, but its 105-second
checkpoint caught a different phase of the animated options highlight. The
accepted visual reference was not changed and the result is not counted as a
clean pass.

## 2026-08-18 upstream-integration revalidation

The upstream-integration campaign invalidated an initially reassuring U1
recheck. The legacy elapsed-time replay stopped sending confirmation input near
30 seconds. On the current build U1 reaches its offline and title dialogs later,
so the nominal cave runs stayed in menus. Their clean temporal results are not
evidence about cave rendering and must not be counted as white-flash trials.

A replacement private route sends bounded confirmation pulses throughout the
slow menu phase. It reaches the saved jungle checkpoint at native 1920x1080 and
produces 300/300 distinct gameplay frames. That stationary 30-second window has
zero invisible flashes, zero abrupt A-B-A returns, and a maximum adjacent-frame
difference of `0.06269234705130189`; its visual oracle passes. The containing
run is classified separately because bounded stdout truncation makes the smoke
result fail closed.

The first route that also replayed the preserved human camera/gameplay input
entered a long cold shader/pipeline-compilation pause at the start of capture.
The sole captured frame is the explicit `Emulation Paused` overlay, the process
continued compiling without a device-loss/assertion signature, and Test Lab
terminated it at the scenario deadline. A second bounded run reproduced the
same condition for its full 905-second lifetime: the emulator stayed responsive
and accumulated CPU time, but no second gameplay frame was produced and no
device-loss or renderer assertion appeared.

That result exposed a testability and liveness tradeoff in fork commit
`90d54e7d`: shader and pipeline creation wraps driver work in
`ScopedHostCompilationGuestPause`, suspending guest threads while the uncached
host work completes. The same design had previously been reverted by
`69563239`, so the integration temporarily removed the three pause scopes.
That experiment was rejected after matched AMD runs repeatedly reached the
guest `m_gfxEopTick` assertion and terminated with host access violation
`0xC0000005`. The accepted-main control retained the guard and survived to its
configured timeout without the assertion. The integration therefore preserves
the scoped pause and adds focused lifetime, pre-existing-pause, nested-pause,
and production-wiring tests. Restoring it did not, by itself, make the merged
AMD U1 route reliable, so neither policy is an accepted U1 fix.

The rejected no-pause experiment let the identical moving route complete for
900.951 seconds with the emulator and runner both exiting cleanly. It produced
300 screenshots,
298 distinct frames, and actual motion-bearing jungle gameplay throughout the
capture window. That run reproduced the reported defect twice. In each case,
Drake and Sully's hair and several nearby spider-web-like surfaces became bright
white for exactly one frame and returned to their normal material on the next
frame. The strict temporal evaluator reported two abrupt A-B-A returns and a
maximum adjacent-frame difference of `0.5599515`.

The first white frame followed creation of vertex shader `0x767c63ee` and
graphics pipeline `0x9ab644acaf87479f`. The second white frame had no intervening
shader or pipeline compilation. Host compilation can therefore coincide with
the defect but is not its necessary cause. The no-pause experiment improved
this route's liveness but is not retained because it weakens AMD frame-lifecycle
reliability; it also did not fix the white material return.

This motion-bearing replay supersedes the earlier five-clean-trial conclusion.
Those trials remain useful historical evidence for the ADDR64 residency repair,
but they did not exercise this route strongly enough to establish that the
intermittent white return was gone. The current leading boundary is queued GPU
memory ownership: the private seed has `copy_gpu_buffers=false`. A controlled
replay kept the route and renderer unchanged while enabling queued guest
command-buffer copying. Its 300/300 frames were distinct and the temporal oracle
reported zero abrupt returns, zero invisible flashes, and no findings. The run
exited normally; only bounded stdout truncation prevented its outer smoke result
from being fully green.

That single clean comparison is a strong discriminator, not an accepted fix.
The optional setting copies only the top-level graphics DCB/CCB, and changing it
globally could affect performance and other titles. A generic ownership RED,
multiple clean U1 trials, and cross-game controls are still required. Per the
upstream-first integration boundary, implementation work is deferred until the
upstream merge and regression campaign are complete.

Preserve the distinction between:

- a renderer-visible white material return in completed gameplay frames;
- a menu/route timing miss;
- a host-compilation pause frame; and
- bounded runner-output truncation.

Private scenarios, input routes, screenshots, saves, and raw logs remain in the
local Test Lab evidence store. The reusable lesson is that a strict temporal
policy is only authoritative after actual scene membership is reviewed.

White-flash candidate commits are `845f6391` (RED) and `7b182423` (GREEN).
Texture-containment commits are `e7fff725` (RED) and `eaa8c93b` (GREEN).

## Working rules

Prefer deterministic RenderDoc Python/CLI reports over GUI inspection. Keep
private captures, screenshots, saves, maps, and machine paths outside Git.
Update this file when an item is proved or ruled out; do not check off a fix on
visual improvement alone.
