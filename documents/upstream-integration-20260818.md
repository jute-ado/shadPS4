# Upstream integration audit: 2026-08-18

This document records the review and validation of the upstream merge that
started from fork commit `54888b82` and was refreshed through upstream commit
`11a5e47a`. The merge base is `dc5a33d6`; the fork was 59 upstream commits
behind when the work began and the review branch is now zero commits behind.
Private games, saves, routes, captures, machine maps, and raw logs are not part
of this repository.

## Integration policy

The merge is not a blanket preference for either side. For every overlap:

1. keep upstream's newer ownership, locking, and data model when it is stronger;
2. reapply fork-only Test Lab and compatibility behavior on top of that model;
3. add or retain a focused RED for each fork invariant;
4. build and run the complete unit suite;
5. run platform-scoped Test Lab checks on Nvidia and AMD before accepting the
   merge.

Upstream remains read-only. Only the fork remote may receive the final merge.

## Reviewed overlaps

### Controller state

Upstream's locked controller history and single-sample access replace the
fork's older direct-state access. Test Lab virtual-controller replay and route
recording are layered onto that implementation rather than restoring the old
model. Physical-device disconnect clears physical state without discarding an
active virtual replay state.

Replay routes remain timing-sensitive when they use elapsed milliseconds. A
route recorded against a fast startup can spend all of its menu input before a
cold build reaches the title screen. Scene membership must therefore be
reviewed; a temporally clean menu is not a gameplay result.

### Kernel threads and exceptions

The merged pthread implementation keeps upstream's `unique_ptr` ownership and
locking while retaining the fork's guest-priority and C++ exception behavior.
The Windows exception policy continues to preserve MSVC C++ exceptions and the
guest red-zone contract.

### Module mapping and video decode

The upstream module-mapping system is the base. Game libraries remain below
4 GiB at `0x80000000`; system libraries remain at `0x800000000`. Video-decoder
changes retain both upstream lifetime work and the fork's packet/picture
ownership invariants.

### Shader and pipeline passes

Both `BufferAccessRange` and `LowerUserClipPlanes` run in the merged pass order.
Pipeline serialization remains at the fork's versioned format required by its
additional state. Focused GCN coverage includes separate guest multiply-add
rounding and the retained Uncharted buffer-access contracts.

## TDD corrections discovered by the merge

### Windows shutdown handler lifetime

Upstream removed process exception/signal handlers at the start of
`Emulator::Shutdown`. Test Lab can request a stop while guest/renderer threads
are still unwinding; removing handlers at that point caused an Nvidia U1 audio
run to terminate with Windows access violation `0xC0000005`.

Focused tests now require:

- `Emulator::Shutdown` does not remove process handlers while worker threads may
  still be alive;
- `SignalDispatch` destruction performs final normal-path removal; and
- the assertion path removes handlers immediately before its breakpoint crash
  so the handler cannot recursively consume the deliberate trap.

The implementation removes the early shutdown call and leaves final cleanup in
the dispatcher destructor, with the assertion-only exception above. The exact
Nvidia audio route then returned to a normal bounded timeout with all required
markers and no crash signature.

### Depth/stencil dynamic-state policy

A small pure policy surface covers merged depth clear and stencil-reference
selection before rasterizer wiring. The focused matrix passes 4/4 and prevents
state-selection changes from being hidden inside large Vulkan integration
tests.

### Host pipeline compilation pause and EOP flip ordering

The inherited fork commit `90d54e7d` suspends guest threads around shader,
graphics-pipeline, and compute-pipeline compilation. A motion-bearing U1 route
once remained on the explicit `Emulation Paused` overlay until its deadline, so
the integration initially removed this wrapper. That removal was rejected.

On AMD, the unpaused integration repeatedly reached the guest
`m_gfxEopTick` assertion near 113 seconds and then terminated with host access
violation `0xC0000005`, while the accepted-main control retained the guard and
survived to its configured timeout without the assertion. The fork guard is
therefore preserved on top of upstream's controller model. Focused tests cover
normal lifetime, a pre-existing user/debugger pause, nested compilation, and
all three production compilation sites.

Restoring the guard did not make the merged AMD U1 route reliable: matched runs
still reached the same assertion. Reversing upstream's signal-emulation rewrite
on an isolated control branch also reproduced the assertion at 115 seconds, so
that upstream commit was ruled out.

The actual race was at the EOP-associated presentation boundary. The existing
fork tracker published the graphics EOP interrupt before queueing its flip, but
the presentation thread could consume that new flip during the same vblank
iteration. U1's woken job thread could therefore observe the flip/free event
before it had processed the EOP event and set `m_gfxEopTick`.

The generic correction stamps each submitted flip with the current vblank
counter. CPU flips remain immediately eligible; an EOP-associated flip becomes
eligible only after the counter advances. Focused coverage pins both cases. The
exact AMD U1 route subsequently completed after 333 seconds with a passing
visual result and no EOP assertion, and the full AMD suite repeated that pass.
This is a presentation-order correction, not a vendor or title override.

Removing the compilation guard previously enabled a completed Nvidia
motion-bearing capture but exposed two one-frame white material returns. The
guard remains preserved; those visual returns are a separate U1 correctness
task after upstream promotion.

The next controlled boundary was queued GPU memory ownership. The preserved U1
seed disables command-buffer copying; an otherwise identical short replay with
copying enabled produced 300/300 distinct frames and zero abrupt returns or
invisible flashes. This is a strong discriminator but not a configuration fix
or merge candidate. The optional path owns only the top-level graphics DCB/CCB,
and a single clean trial does not establish a generic lifetime contract. The
production experiment was removed from the integration worktree. Further U1
work is explicitly deferred at the user's requested stop boundary.

### Independent sysmodule preload requirements

Upstream commit `db2f820a` correctly separated missing-libc and missing-Fios2
checks but assigned the Fios2 branch to the libc flag. A pure classifier and
focused 3/3 matrix now keep the two preload requirements independent before the
linker mutates global preload state.

### Assertion and normal-shutdown handler lifetime

Upstream commit `11a5e47a` removes process handlers from the assertion path
immediately before the breakpoint crash, while normal shutdown leaves them
installed until `SignalDispatch` destruction. The merge adds a focused source
contract for that ordering. Because `common/assert.cpp` now references
`SignalDispatch`, the settings, NGS2, and HTTP focused executables also receive
the existing signal stub; without that integration support their link failed.

## Validation snapshot

The fresh promoted-source unit build discovers and passes `560/560`, with one
expected environment-dependent host-override skip. An earlier `649/649` count
came from a reused build directory that still registered 89 removed diagnostic
tests; it was not the clean promoted-source inventory. The Release application
builds and links.

Final Nvidia platform results:

- U1 visual: bounded timeout with no crash or forbidden marker, but the retained
  Nvidia seed/route did not reach the cave checkpoint; accepted `main` has the
  same route gap, so this remains an inherited inconclusive rather than a passed
  visual gate;
- U1 audio: pass;
- U2 smoke: pass;
- U2 performance: pass.

Final AMD platform results:

- U1 visual: pass in both the focused 333-second replay and the full suite;
- U1 audio: known incomplete-window inconclusive, without a crash signature;
- U2 smoke: pass;
- U2 performance: pass on a complete retry quorum, approximately 32 FPS with
  zero measured stutter. The first suite invocation had two valid trials and
  one trace-less timeout, which was retained as infrastructure evidence.

The upstream integration is accepted for promotion. The requested 4K campaign
remains Nvidia-only and has not started.

## Follow-up work after upstream promotion

- isolate and fix the two reproduced one-frame U1 white material returns before
  4K work;
- repair or replace the inherited Nvidia U1 seed/route so it reaches the cave
  deterministically;
- keep cold compilation stutter, guest-pause liveness, frame lifecycle, and
  renderer flicker as separate classifications;
- update the corpus only after a route proves it reached the intended scene;
- rerun the complete unit suite and relevant platform-scoped Test Lab suites
  after each U1, resolution, or performance change.

After the U1 gate, add generic per-game internal-resolution configuration and
UI rather than title-specific patches. Evaluate PS4 versus PS4 Pro behavior on
Nvidia for U1/U2/U3, then document visual accuracy, stability, and performance
before enabling or recommending 4K profiles.

## 2026-08-19 incremental upstream review

Upstream advanced from `11a5e47a` through `dd968182`. The integration keeps all
five commits in its ancestry, but deliberately reverts the storage-buffer
cleanup from `dd968182` after a matched U3 runtime regression. The remaining
changes are retained:

- Abseil is consumed from the upstream submodule;
- the upstream signal-test stub is not used where the fork's focused binaries
  require the functional signal-dispatch stub;
- non-AMD fragment barycentrics use the KHR barycentric path; and
- GDS offsets are computed dynamically.

The merge initially duplicated the signal stub in several focused targets
because the fork already supplied it. Those duplicate source entries were
removed. A new `BarycentricSpirv` test exercises direct, smooth, centroid,
sample, sample-ID, interpolation-function, and sample-rate-shading behavior on
the non-AMD path. The clean merged test inventory passes `570/570`, with 569
executed tests and the expected environment-dependent host-override skip.

### Storage-buffer cleanup hold

`dd968182` looked mechanically compatible because the fork's existing
`IsStorage` policy already classifies guest buffers as storage buffers.
Nevertheless, the exact U3 relaxed bottle route exposed a runtime regression:

- the first cold run terminated at the guest `m_gfxEopTick` assertion during
  shader compilation;
- a repeat completed but emitted only 19 temporal frames before timeout; and
- reverting only `dd968182` restored a normal exit and all 300 requested
  temporal frames.

The revert preserves the fork's bounded `GetBindingSize` contract as well as
the pre-cleanup buffer specialization and Vulkan feature/layout path. The
upstream commit remains in history so the fork is not ancestry-behind, but its
behavior is held until a smaller independently tested subset can be admitted.
The staging-buffer alignment sub-change may be reconsidered separately; it is
not bundled into the shader storage-class change.

The restored 300-frame U3 run still fails the bottle temporal oracle (216
distinct frames plus repeated-frame and adjacent-difference violations). A
dev-equivalent shader control can also hit the pre-existing guest EOP assertion,
while older U3 graphics-integration controls passed 300/300. This is recorded as
the known U3 correctness/liveness campaign, not evidence that the remaining
2026-08-19 upstream commits regress U3.

### Nvidia game gates

- U1 cave/hair visual: pass, 300/300 distinct frames, no invisible flash, no
  abrupt return, and no visual violation. The longer delayed-human route also
  reproduced its known route flakiness and produced no capture, so it is not
  used as positive evidence.
- U2 sewer visual: pass on the exact retained corpus scenario and baseline.
- U3 relaxed bottle: required scene markers reached. With `dd968182` active it
  regressed as described above; with the revert it exits normally and captures
  300 frames, while the existing temporal defect remains visible to the oracle.

### Resolution status

The separate resolution campaign now exposes per-game guest VideoOut requests
and UI/config persistence, but does not yet provide native internal 4K
rendering. Base and Pro-mode probes both observed 1920x1080 guest buffers. Pro
mode is the preferred stable 4K-presentation profile on Nvidia, but a genuine
internal-resolution implementation must scale the complete guest render-target,
viewport/scissor, texture-view, resolve/copy, screenshot, and presentation
pipeline coherently. Partial render-area scaling is explicitly rejected.
