# Uncharted 1 cave hair investigation

- **Status:** Native-scale ADDR64 pointer-table residency repair validated; broader promotion pending
- **Last verified:** 2026-08-20
- **Verified revision:** candidate branch `fix/u1-cave-material-residency-20260820`
- **Scope:** MUBUF `addr64`, source-buffer residency, and texture containment

Status: the original orange hair-card lattice and the later intermittent white
hair/web return are both fixed. They were separate defects in the same ADDR64
path: the first correction decoded the guest resource-relative address; the
final correction stopped lowering that bounded descriptor access through the
fault-on-first-use BDA path. The final fix deliberately builds on rather than
reverts the original lighting correction.

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
`0xC0000005`. The accepted-main control retained the guard and survived to its configured timeout
without the assertion. A later Nvidia validation then remained in the visible `Emulation Paused`
state for more than ten minutes. Its log stopped between shader-module creation and compute-pipeline
creation while the process remained responsive with little additional CPU time. This demonstrates
that suspending arbitrary guest threads from the GPU compilation path can deadlock on a lock or
dependency held by a suspended thread. The scoped pause is therefore no longer accepted as a
liveness mechanism.

The replacement synchronous fallback compiles without OS-suspending guest threads. An experimental
asynchronous graphics-pipeline stage is documented in `async-graphics-compilation.md`; it is not an
accepted U1 or AMD fix until matched live controls pass. Compute and shader-module compilation remain
synchronous in this first stage because silently dropping compute dispatches or borrowing mutable
shader state would be incorrect.

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

### 2026-08-18 immutable graphics-submission ownership

After the upstream promotion completed, the command-buffer discriminator was
implemented as a generic lifetime contract rather than a game override. A
focused RED now requires every queued graphics submission to own immutable
copies of its top-level DCB and CCB. `Liverpool::SubmitGfx` creates that owner
before the guest can mutate or reuse the source memory, and the existing
graphics coroutine retains it until command decoding finishes. The old
`copyGPUBuffers` setting and per-game override were removed because permitting
borrowed queued command spans is not a safe behavior choice; legacy TOML keys
are ignored.

The focused graphics-submission gate passes `13/13`. A fresh promoted-source
build discovers and passes `560/560` tests, with one expected host-environment
skip, and the Release application links. The original private motion seed still
contains `copy_gpu_buffers=false`, so it verifies the code contract rather than
the earlier modified seed. Its complete 300-frame replay produced 300/300
distinct gameplay frames, zero invisible flashes, zero abrupt A-B-A returns,
and a longest repeated run of one frame. This matches the earlier copied-buffer
control and differs from the borrowed-buffer replay's two one-frame white
returns.

Repeated motion attempts also exposed an independent route/bootstrap problem:
several runs remained in early asset loading, produced no screenshots, and
timed out without a device-loss or forbidden assertion. Those unavailable runs
are retained but are not counted as either visual passes or failures. A
separate 600-frame post-observation gate initially failed before triage because
the task-local asset map supplied baseline `v3` while the visual contract pins
baseline `v2`; a diagnostic Test Lab build identified the exact mismatch. The
corrected map is retained privately. Subsequent zero-frame starts remained the
same route issue, not captured white returns.

The maintained Nvidia parity suite provides the cross-game gate for the same
binary: U1 audio passes, U2 smoke passes, and U2 rooftop performance passes all
three trials at approximately 60 FPS with zero measured stutter. Against the
accepted upstream-integration build, median frame time changed from 16.5532 ms
to 16.5667 ms, p95 improved from 17.5425 ms to 17.5227 ms, p99 improved from
17.6314 ms to 17.6161 ms, and stutter remained zero. The suite's U1 visual entry
is still the inherited route/seed timeout and is classified inconclusive.

This evidence supports immutable queued graphics command ownership as the U1
white-return repair without a measurable Nvidia cross-game performance
regression. It does not claim that the separate U1 route-start reliability
problem is fixed.

Preserve the distinction between:

- a renderer-visible white material return in completed gameplay frames;
- a menu/route timing miss;
- a host-compilation pause frame; and
- bounded runner-output truncation.

Private scenarios, input routes, screenshots, saves, and raw logs remain in the
local Test Lab evidence store. The reusable lesson is that a strict temporal
policy is only authoritative after actual scene membership is reviewed.

## 2026-08-20 dynamic fragment ADDR64 publication boundary

A later full-motion replay superseded the immutable-command-buffer conclusion.
The ownership fix remains a valid general lifetime repair, but the same moving
cave route again produced one-frame white hair and web returns. The immutable
submission change was therefore not sufficient to fix this renderer defect.

The earlier MUBUF `addr64` correction must not be reverted. Before that fix the
hair fragment shader read its own shader bytes as lighting constants, which is
the proven cause of the orange/zebra hair-card lattice. The follow-up
`ReadConstBufferAddr64` resource-tracking change also remains correct, but its
source-descriptor residency cannot enumerate the final address because that
address is computed dynamically by fragment invocations.

The direct-memory shader path made the remaining failure mode explicit. The
backend ignored the bound buffer handle for `ReadConstBufferAddr64` and instead
consulted the buffer-device-address page table. When the page was not resident,
the invocation recorded a fault and returned zero for that read. The normal
asynchronous fault consumer made the page resident only after the draw, so the
first published draw could use zero material constants and the next frame
immediately recovered. That sequence matches the observed dark -> white -> dark
temporal signature.

That backend behavior was not faithful to the guest ISA. AMD's GCN3 ISA defines
MUBUF's final address as resource-base + SGPR offset + buffer offset, keeps range
checking relative to the resource descriptor, and specifies zero for
out-of-range reads. The relevant primary reference is AMD GCN3 ISA revision 1.1,
sections 8.1.5 and 8.1.5.1:
`https://gpuopen.com/download/AMD_GCN3_Instruction_Set_Architecture_rev1.1.pdf`.

The production correction therefore does not add a discovery draw. Translation
keeps ADDR64's 64-bit vector/scalar/instruction byte offset descriptor-relative;
the bound storage-buffer descriptor supplies the runtime base. The backend
converts that offset to a dword index, applies the existing Vulkan binding
offset, and loads through the descriptor. Vulkan robust-buffer access then
provides the guest's out-of-range zero behavior. The generated shader does not
specialize on the resource base, so rebinding remains correct. ADDR64 no longer
enables the BDA page table or fault buffer and produces identical SPIR-V with
emulator DMA mode enabled or disabled. Ordinary unbounded `ReadConst` operations
retain the existing BDA/fault path. The persistent shader-binary cache version
is bumped so an installation cannot silently reuse SPIR-V generated by the
faulty backend.

The first broad diagnostic implementation deliberately preflighted every dynamic
fragment DMA draw. It caught 130 first-use fault epochs in a long replay but was
too expensive: 262,144 discovery draws reduced the observation window to about
10 presented frames per second. Even so, two independent moving windows produced
436 valid frames each with no white return. Their strongest abrupt-bright-return
scores were 17.7 and 19.9, both ordinary character motion, versus 83 and 117 for
the preserved failing frames.

An intermediate refinement restricted preflight to ADDR64 and reduced its cost,
but it remained a workaround. Its third moving window produced 430 valid frames
with no white hair/web return; the highest detector scores (27.6 and 24.1) were
flashlight motion on a companion, not the preserved defect. All fragment
preflight code and per-fault diagnostic logging were removed after the
descriptor defect was identified.

Focused RED/GREEN coverage now proves that ADDR64 keeps the source guest buffer
bound, does not set `uses_dma`, emits identical SPIR-V in both DMA modes, and
does not specialize on the descriptor base. The complete GCN test executable
passes 54/54. The related DMA-publication, fault-download, and asynchronous
pipeline/cache gates pass 7/7, 6/6, and 7/7 respectively, and the Release
application links.

Two fresh descriptor-only moving-window replays then exercised the original
cave camera motion on an RTX 4080 with asynchronous graphics-pipeline
compilation enabled and normal `readbacks_mode=0`. The first produced 419 valid
frames plus one truncated tail frame; its highest abrupt-return score was 8.47.
The final rebinding-safe/cache-invalidating build produced 423 valid frames plus
one truncated tail frame; its highest score was 16.28. Review of both candidate
contact sheets found only normal character and flashlight motion. Neither run
contained a white hair or spider-web return. The preserved bad sequence scores
83.02 and 117.24 under the same detector, so the final run is separated from the
known defect by a wide margin. The outer Test Lab visual result is unavailable
only because its legacy contract requests 600 frames while this deterministic
route exits after roughly 420; the full available sequence was retained and
analyzed rather than treating the count mismatch as a visual failure.

This evidence promotes descriptor-relative ADDR64 lowering from a diagnostic
candidate to the U1 white-frame repair. It fixes the fault-on-first-dynamic-read
mechanism without reverting the earlier orange-hair correction, without adding
a preflight draw, and without retaining the temporary fault instrumentation.
The repair passed the normal branch-level regression suite and was promoted to
both `dev` and clean `main`. Any future white return must still be compared
against the preserved failing frames and not classified from runner completion
alone.

White-flash candidate commits are `845f6391` (RED) and `7b182423` (GREEN).

### 2026-08-19 superseded moving-scene revalidation

A new visible moving-scene check invalidated the conclusion that immutable top-level DCB/CCB
ownership was sufficient. The white hair and spider-web flicker remained visible at native 1080p
with both correctness-preserving asynchronous graphics-pipeline compilation and the matched
synchronous control. Async pipeline misses are therefore not a necessary cause, and internal
resolution scaling is not involved in this reproduction.

The ownership audit found a narrower lifetime hole below `SubmitGfx`: the submission retained copied
top-level DCB/CCB storage, but `INDIRECT_BUFFER` and `INDIRECT_BUFFER_CONST` recursively decoded raw
guest pointers and could yield while those nested command streams remained borrowed. A focused RED
mutates both a nested graphics IB and constant-engine IB after submission; it now requires the
submission owner to preserve both recursively and requires both Liverpool recursion sites to resolve
through that owner. The focused submission suite passes 15/15 after the repair. This lifetime repair
remains useful, but later live evidence showed it was not the white-return fix; the descriptor-relative
ADDR64 correction above is the accepted repair.
Texture-containment commits are `e7fff725` (RED) and `eaa8c93b` (GREEN).

### 2026-08-20 unresolved 200% host-scaling regression

A live manual review invalidated the earlier acceptance claim for the U1 cave at
200% internal render scale. Hair and spider webs were not intermittently wrong
for one or two frames; they were persistently and uniformly rendered with the
wrong material appearance throughout the observed scene. This is a distinct,
stronger failure than the preserved native-scale white-return sequence.

The reviewed Test Lab launch did pass `--internal-resolution-scale 200`, and the
emulator's own startup log confirmed `GPU internalRenderScale: 200%` on the RTX
4080. The guest display and captured output remained 1920x1080. The current
implementation scales eligible host attachments transactionally and permits
ineligible images or mixed attachment sets to fall back to native scale; it
does not render a native 3840x2160 guest framebuffer or present a 3840x2160
image. The run also ended as invalid Test Lab evidence rather than an accepted
visual result. Without per-draw effective-scale evidence, it must not be
described as proof that every attachment was rendered at 3840x2160.

Status: **open and deliberately not being fixed in this pass**. The earlier
native-scale descriptor-relative ADDR64 result remains separate evidence, but
the 200% cave acceptance in the resolution-scaling notes is superseded. A
future investigation should replay the exact same seed, route, async-pipeline
mode, and camera path at 100% and 200%; publish the effective scale selected for
each target transaction; and compare the persistent material failure against
the native composition. Do not reduce this report to sharpness or fullscreen
window size.

## 2026-08-20 native-scale pointer-table residency correction

Later native-scale review invalidated the statement that descriptor-relative
lowering had completely solved the cave material path. The guest instruction
uses an unbounded 64-bit address, and restoring that ISA behavior was necessary,
but the address itself comes from a small guest pointer table reached through
dynamic `ReadConst` operations. The table was not necessarily registered in the
buffer-device-address page table before the first fragment draw. A missed first
read therefore still returned zero, which can produce the stable-white material
or a one-frame white return depending on when the table becomes resident.

The current candidate fixes the ownership boundary without changing generated
SPIR-V. Resource tracking walks the exact dynamic ADDR64 address expression and
records bounded flattened-userdata roots. Before normal buffer binding, the
rasterizer reads each root pointer already produced by the SRT walker, validates
the resulting 256-byte guest range against the 40-bit GPU address space and the
live GPU mapping, and sends it through the existing `FindBuffer` plus
`SynchronizeBuffersInRange` path. The direct-memory shader then sees the same
runtime-rebindable address and the normal buffer cache publishes current data
before the draw.

The compiler analysis is deliberately bounded and fail-closed: at most eight
roots, 32 `ReadConst` dependencies, and 256 traversed IR nodes are retained;
duplicates are removed and overflow discards the observation. Runtime rejects
zero, wrapping, out-of-address-space, out-of-flat-buffer, and unmapped ranges.
There is no title ID, shader hash, asset, draw ordinal, readback override, extra
GPU draw, or shader-data snapshot in the fix.

A rejected prototype copied the 256-byte table into appended userdata and added
a SPIR-V fallback. Although logically bounded, that changed the affected shader
enough to make NVIDIA async pipeline compilation stall for minutes. It was
removed. A second rejected refinement required the two pointer words to share
one exact IR producer node; the real guest expression uses semantically related
but non-identical nodes, and this overconstraint reproduced the known stable-
white signature. The production candidate relies on adjacent flattened pointer
words plus the strict runtime address/mapping checks instead.

Focused evidence currently includes seven buffer-residency tests and four GCN
ADDR64 tests, all passing. The complete generated test suite passes 586/586
(with the one documented environment-only skip), and the Release application
links successfully. The authoritative
native 1920x1080 async replay completed 190/190 distinct cave frames with zero
abrupt returns and zero invisible flashes. It matched the correct-material
signature and rejected both the orange-lattice and white-material signatures.
The deliberately overconstrained build completed the same scene and matched the
white-material signature, providing a live negative control. Subsequent fresh
starts used the same `async_graphics_pipeline_compilation=true` configuration.
Because the private seed deliberately disables the pipeline cache, several of
those cold starts remained in shader compilation or asset loading through the
95--114 second window. A late 175--194 second variant and a longer resilient
route likewise returned no screenshots before their deadlines. They are
retained as capture-unavailable evidence, not renderer failures; neither
produced a contradictory cave frame. The final binary differs from the clean
live candidate only by the required shader-metadata serialization-version bump
and has the same tested renderer behavior.

This section supersedes the earlier claim that descriptor-relative lowering by
itself was the final native-scale repair. Promotion still requires a second
reached-scene pass plus the broader regression gates. The separate 200% scaling
failure above remains open and must not be conflated with this native-scale
result.

## Working rules

Prefer deterministic RenderDoc Python/CLI reports over GUI inspection. Keep
private captures, screenshots, saves, maps, and machine paths outside Git.
Update this file when an item is proved or ruled out; do not check off a fix on
visual improvement alone.
