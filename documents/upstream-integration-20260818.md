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

### Host pipeline compilation pause remains an unresolved tradeoff

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

Restoring the guard did not make the merged AMD U1 route reliable: a final
matched run still reached the same assertion. Conversely, removing it enabled
a completed Nvidia motion-bearing capture but exposed two one-frame white
material returns. The pause is neither an accepted U1 fix nor sufficient proof
of the AMD crash cause. Resolving this cross-vendor timing/lifecycle boundary is
explicitly deferred; a vendor-specific bypass is not authorized.

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

The complete discovered unit suite passes `648/648`, with one expected
environment-dependent host-override skip. The Release application builds and
links.

The earlier final Nvidia platform suite passed all four entries. The exact
post-refresh rerun remained functionally healthy but the suite wrapper reported
failure because bounded stdout was truncated for U1 visual and U1 audio was
classified inconclusive at its normal bounded timeout:

- U1 visual: process exit 0, temporal oracle pass, no forbidden marker;
- U1 audio: bounded timeout, expected progress milestone, no crash signature;
- U2 smoke: pass;
- U2 performance: pass.

AMD platform suite and controls:

- U1 visual integration runs: repeated guest EOP assertion followed by host
  access violation near 113 seconds;
- current accepted-main exact control: no assertion/crash and a clean bounded
  timeout, but it did not reach the visual checkpoint;
- restoring the fork compilation guard did not remove the integration crash;
- U1 audio: pass;
- U2 smoke: pass;
- U2 performance: approximately 32 FPS with zero measured stutter.

The earlier single passing integration retry is retained as timing evidence but
does not override the repeated failures. AMD is required for ordinary
integration regression, so the review branch is not accepted into fork `main`.
The requested 4K campaign remains Nvidia-only and has not started.

## Remaining gates before merge acceptance

- isolate the AMD U1 EOP/assertion reliability regression without weakening the
  retained host-compilation lifetime contract;
- isolate and fix the two reproduced one-frame U1 white material returns before
  4K work;
- keep cold compilation stutter, guest-pause liveness, AMD frame lifecycle, and
  renderer flicker as separate classifications;
- update the corpus only after a route proves it reached the intended scene;
- rerun the complete unit suite and both vendor-scoped Test Lab suites after the
  final U1 change;
- merge the review branch to fork `main` only after both vendor-scoped gates are
  functionally green.

After the U1 gate, add generic per-game internal-resolution configuration and
UI rather than title-specific patches. Evaluate PS4 versus PS4 Pro behavior on
Nvidia for U1/U2/U3, then document visual accuracy, stability, and performance
before enabling or recommending 4K profiles.
