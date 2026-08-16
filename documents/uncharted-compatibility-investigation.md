# Uncharted compatibility investigation guide

This document records the reusable engineering method and accepted findings from
the Uncharted compatibility campaign. It is intentionally free of games, saves,
captures, machine paths, private manifests, and raw logs. Those artifacts remain
in the local Test Lab evidence store.

The goal is not to accumulate title-specific workarounds. Each accepted change
must repair a general emulator invariant, begin with a synthetic failing test,
and pass PS4-only regression gates.

## Campaign status

| Area | Status | General invariant |
| --- | --- | --- |
| Video decode picture metadata | Accepted | Picture metadata remains owned until the consumer has finished with it. |
| Video decode packet data | Accepted | Packet storage outlives asynchronous decode and presentation work. |
| Buffer lookup | Independently audited | Selection and containment behavior is covered by focused tests and PS4 gates. |
| Uncharted 1 cave hair | Accepted | MUBUF `addr64` decoding and source-buffer residency are preserved. |
| Uncharted 3 pub geometry corruption | Operationally fixed for CUSA02320 | The integration build plus a per-game `Relaxed` GPU-readback override passed three fresh full-scene trials. The global default remains unchanged; a general automatic readback trigger is still future work. |
| Uncharted 1 production visual and audio checkpoints | Passed on the integration build | The production-shaped seed keeps DMA enabled and scopes `Relaxed` readbacks to CUSA02320. Reviewed title and cave frames are correct, and the preserved stereo PCM audio contract passes. |
| Uncharted 2 checkpoint and performance route | Passed on the integration build | The 180-second checkpoint completed without the historical device-loss, buffer-lookup, EOP, or offset failures. Three performance trials held approximately 60 FPS with no measured stutter. |
| ReadConst with DMA disabled | Candidate code repair | Dynamic and immediate ReadConst accesses now follow the global DMA boundary consistently; focused tests, the complete unit suite, application linking, and title checkpoints pass. |

Compatibility claims are checkpoint-specific until the complete campaign
playthrough and regression matrix is green.

## Required development loop

1. Reproduce the defect at a deterministic checkpoint on clean accepted main.
2. Preserve the immutable run and classify process, marker, visual, temporal,
   audio, and performance health independently.
3. Localize the first broken emulator invariant. Do not infer a cause from the
   last subsystem that happened to log a warning.
4. Add a focused synthetic RED that fails for the missing contract.
5. Implement the smallest general GREEN without title, shader-hash, draw, or
   asset special cases.
6. Run the focused test, affected subsystem tests, Uncharted-focused PS4 suite,
   and complete PS4 regression suite.
7. Run repeated clean checkpoint trials and review the actual rendered frames
   or video manually.
8. Update this document with proven, rejected, and still-unknown findings.

Keep code, corpus metadata, and private evidence in separate repositories and
worktrees. See [Emulator Test Lab workflow](emulator-test-lab.md) for the
cross-repository rules.

## Validation configuration is part of test authority

A visually bad run is not evidence of a renderer regression until the candidate
and accepted control use the same authoritative seed configuration. This
campaign exposed a particularly misleading case: an intentionally diagnostic
Uncharted 1 seed had direct memory access disabled. It rendered an all-black
scene on both the accepted pre-stack build and the current integration build,
while the historical visible reference and the production-shaped seed had DMA
enabled.

The correct production validation therefore uses:

- the normal global DMA setting used by the accepted reference;
- the title-scoped CUSA02320 `Relaxed` readback override;
- the exact checkpoint, input route, asset set, and scenario revision;
- a reviewed scene reference from that same configuration.

Do not label a black, corrupt, or unexpectedly stable diagnostic run as a new
renderer failure until these inputs match. Record configuration differences as
first-class evidence rather than silently cloning a convenient scenario.

Test Lab provenance has two related operational requirements:

- scenario files must live under a Git repository or worktree so their revision
  can be recorded; coordinator-local scenarios are not authoritative inputs;
- temporary bisect worktrees must be clean and commit-specific. Nested or
  untracked repositories can make bounded provenance hashing ambiguous or
  excessively large.

These rules prevent a seed mismatch or provenance failure from being mistaken
for an emulator result.

## ReadConst policy must follow DMA availability

The broad U1 diagnostic route uncovered a real fail-fast defect even though its
black output was a seed-configuration issue. A prior shader change forced
dynamic ReadConst offsets onto the DMA path even when direct memory access was
globally disabled. Shader collection also retained DMA descriptor metadata in
that disabled state. The generated resource contract and runtime bindings could
therefore disagree, leading to a Windows fail-fast termination.

The repaired invariant is general:

- when DMA is disabled, both immediate and dynamic ReadConst accesses use the
  flat-buffer path;
- when DMA is enabled, dynamic offsets use DMA and immediate offsets retain the
  established DMA-with-flat-buffer-fallback behavior;
- shader collection clears `uses_dma` and ReadConst descriptor metadata when
  DMA is disabled;
- production behavior with DMA enabled is unchanged.

The focused TDD matrix covers both offset classes, both DMA states, and metadata
retention. This is not a title, shader, address, or draw special case. The
DMA-disabled route now completes rather than terminating, but it remains a
diagnostic configuration and is not a visual acceptance route for U1.

### 2026-08-16 broad validation snapshot

After the ReadConst repair, the integration build passed these sanitized gates:

- **U1 visual:** reviewed title and cave frames at 60, 90, and 120 seconds show
  the expected characters, geometry, lighting, and hair;
- **U1 audio:** the preserved stereo PCM policy passes;
- **U2 checkpoint:** the 180-second route completes without device loss,
  `FindBuffer`, EOP, or offset-assertion signatures;
- **U2 performance:** three valid trials average approximately `60.004 FPS`,
  with `16.543 ms` median, `17.522 ms` p95, `17.594 ms` p99, and zero measured
  stutter;
- **U3 visual:** the exact CUSA02320 temporal route passes again with the global
  readback mode disabled and only the per-game `Relaxed` override active.

Focused ReadConst tests pass `3/3`, the shader/GCN target passes `59/59`, the
Release application builds and links, and the complete discovered unit suite
passes `553/553` with one intentional environment-dependent skip.

The full-suite run required two test-infrastructure corrections that leave
emulator warning policy unchanged. Clang 22 reports a `char8_t` conversion in
GoogleTest 1.17, so the dependency-only suppression must be applied to `gtest`,
`gtest_main`, `gmock`, and `gmock_main`; emulator targets retain warnings as
errors. CMake 4.4 also omits the target key from its parallel post-build
GoogleTest discovery script, making targets race on one empty-name JSON file.
Using pre-test discovery for CMake 4.4 serializes that dependency step and
restores deterministic discovery. These are test-harness compatibility fixes,
not renderer behavior changes.

## Visual acceptance is multidimensional

A temporal oracle answers whether frames change unexpectedly. It does not prove
that the frame is the correct scene. A completely wrong but stable image can
pass a temporal-only test.

Every visual fix therefore needs all of these gates:

- **Scene identity:** compare against a reviewed reference from the same
  checkpoint and camera state.
- **Static correctness:** use bounded full-frame difference and structural or
  cosine difference thresholds.
- **Temporal correctness:** check abrupt returns, localized returns, visibility,
  distinct-frame count, and adjacent-frame differences.
- **Manual review:** inspect the first frame, representative transitions, and a
  playable video of the full checkpoint.
- **Process health:** keep exit, timeout, signal, required-marker, and
  forbidden-marker results separate from image quality.

Do not widen a visual threshold to admit a candidate. If animation makes a
single static baseline unsuitable, improve the oracle or capture contract.

### Rejected U3 false positive

An experiment that changed immediate constant reads and stream-buffer watch
lifetime made the temporal test pass, but manual review showed a large dark
polygon covering most of the scene. The oracle had compared it to an unrelated
loading-state image with permissive static thresholds, so a stable corruption
was incorrectly classified as a fix.

That experiment is rejected. It must not be merged or used as evidence that U3
is fixed. The corrected oracle uses a reviewed full pub-scene reference and
rejects the stable polygon by both mean absolute and cosine difference.

The lesson is general: a temporal pass is only meaningful after scene identity
and static correctness are established.

## U3 failure signature

Clean accepted main renders the intended pub scene, but adjacent frames can
alternate between:

1. a complete frame;
2. a frame with large rectangular regions of geometry or HUD missing or dark;
3. a return to the complete frame.

This is an A-B-A corruption pattern in an otherwise correct scene. A candidate
that replaces it with a stable wrong scene is a regression, even if every
temporal metric improves.

Current evidence rules out several sampled-image, sampler, cube-coordinate, and
content-lineage hypotheses as unique discriminators. These diagnostics remain
useful negative evidence, but they do not authorize renderer behavior changes.

## Validated CUSA02320 operational fix

The decisive control was GPU readback policy, not another shader, sampler, or
attachment mutation. With the global `readbacks_mode` left at `Disabled`, the
reviewed full-scene route reproduced one-frame geometry expansion and object or
character disappearance. Setting the effective mode to `Relaxed` removed the
corruption. `Precise` also passed, but was not required.

Use a game-specific override so unrelated titles keep their existing global
policy:

```json
{
  "GPU": {
    "readbacks_mode": 1
  }
}
```

Store this as `custom_configs/CUSA02320.json` in the shadPS4 user directory, or
select **Readbacks Mode: Relaxed** in the CUSA02320 game-specific settings UI.
Do not change the global default solely for this title.

### Acceptance evidence

The production-shaped validation kept the seeded global configuration at mode
`0` and supplied only the per-game override. The emulator reported that a
game-specific configuration was active and that the effective mode was `1`.
Three independent full-scene trials then produced:

- normal process exit in all three trials;
- all required scene markers and no forbidden markers;
- 300 captured frames and 300 distinct frames per trial;
- zero global abrupt returns per trial;
- zero localized abrupt returns per trial;
- maximum adjacent differences of approximately `0.00131`, `0.00136`, and
  `0.00106`, all below the `0.007` contract threshold;
- manual review of the complete 30-second frame sequence with no expanding
  table, disappearing character, large dark polygon, or scene replacement.

The same integration build failed with mode `0`, so the readback control is a
real A/B discriminator rather than a lucky replay. A near-upstream control with
mode `1` still stopped at the older `FindBuffer` assertion before the scene,
which proves the per-game setting is not a substitute for the integration
branch's generic buffer lifetime and lookup repairs.

This is an accepted operational fix for the tested CUSA02320 checkpoint. It is
not yet a general code repair that automatically discovers which GPU-produced
pages the guest will read. Future code work should preserve the successful
semantics while finding a narrow, data-driven trigger; it must not hard-code a
title, shader, draw, address, or captured event.

### Rollback and diagnostics

To roll back the operational fix, remove the CUSA02320 custom configuration or
set its Readbacks Mode back to Disabled. Keep that rollback scoped to the game;
do not delete the global user configuration.

When validating another region or release:

1. confirm the title ID and create a separate per-game override;
2. verify the startup log reports both game-specific configuration use and the
   intended effective readback mode;
3. run at least three clean full-scene trials;
4. inspect the complete video manually, not only the temporal verdict;
5. record performance separately, because this fixed-duration visual route is
   not a frame-time benchmark.

## Cached shader permutation generation

The strongest scene-preserving lead is a generation-coherence invariant in the
shader cache. When an existing shader program needs a new specialization:

1. the cached `Info` is refreshed from current user data;
2. specialization parses flattened descriptors and the fetch shader;
3. a fresh `Info` is constructed to compile the new permutation;
4. translation may otherwise walk mutable guest descriptors and fetch code a
   second time.

If guest memory changes between steps 2 and 4, the specialization key can
describe generation A while the compiled module consumes generation B. The
module is then cached under a key that does not match its compiled resources.

The candidate contract is deliberately narrow:

- capture the exact flattened descriptor image after the existing refresh;
- retain the parsed fetch-shader result used by specialization;
- pass both snapshots only through the synchronous new-permutation compile;
- replay them before translation can read live guest memory again;
- keep the snapshots transient and out of serialized shader metadata;
- preserve the normal live path for initial compilation and cache hits.

This contract requires focused RED tests for generation replay, malformed
snapshot rejection, fetch replay, call ordering, and transient ownership. It is
implemented on the integration branch and remains useful hardening, but it did
not eliminate the full-scene defect. It is not, by itself, the U3 fix.

## U3 boundaries measured so far

The investigation now has sequence-joined, bounded observations at several
layers. These results are important because they prevent future work from
repeating attractive but non-discriminating hypotheses.

| Boundary | Result | Interpretation |
| --- | --- | --- |
| Final sampled-image lookup and view | Uniform across event and control frames | The selected image, direct view, descriptor, and subresource relation do not uniquely explain the event. |
| Sampler realization and descriptor | Uniform and exact | LOD, compare, reduction, association, cache, and sampler descriptor state did not separate event frames. |
| Cube lowering | Uniform `NonCube` | The measured source is not a cube resource; a cube-array face remap is not authorized. |
| Sampled content lineage | Clean refresh, current direct producer, no transfer | Repeating the same lineage receipt would be redundant. |
| Instrumented sampled values | Stable in localized event rows | The selected sampled value is not the first observed unstable boundary. |
| Tee-authorized target pre/post temporal class | Event transitions also occur in controls | This localizes some changes to the draw-output side but does not establish causality. |
| Same-frame target pre/post equality | Changed pixels are enriched in current event members but also common in controls | Useful localization only; not a unique repair condition. |

All observations retain scalar classifications only. Raw shader values, image
identities, routes, coordinates, captures, and private logs remain outside the
repository.

### Target pre/post interpretation

The target pre/post reuse path is authorized by the existing sampled-result
tee, then compares CPU-owned pre-draw and post-draw planes for the exact selected
window. It does not add a GPU copy, command, barrier, slot, wait, or release.

This boundary must not be described as sample-to-export identity. The post plane
also includes shader control flow, other inputs, depth/stencil tests, blending,
and attachment writeback. A temporal-class transition or same-frame byte change
therefore localizes the unresolved work but cannot select a renderer mutation.

## Rejected U3 experiments

The following changes were each tested on a fresh full-scene route and rejected.
Keep the explicit reverts in history and do not retry them without a new
discriminating test.

- **Merged vertex-descriptor clamping:** satisfied its synthetic range test but
  increased full-scene localized corruption.
- **One- and two-MiB fault readback windows:** a one-MiB trial produced one lucky
  clean run, but immediate repetition showed a large one-frame dark geometry
  obstruction. Two MiB also failed. Window size changes timing rather than the
  root invariant; 512 KiB remains the bounded baseline.
- **Broad read/write stream mapping:** corrupted the whole scene and violated
  the narrow ownership model.
- **Large readonly stream snapshots:** produced stable large black polygons and
  greatly reduced distinct frames.
- **Write-after-write barrier broadening:** produced giant triangles and many
  global returns.
- **ADDR64 DMA descriptor fallback:** on a missing BDA page, substituting the
  already-bound descriptor value instead of zero is unsafe because the binding
  can represent a different residency generation. The candidate was unit- and
  SPIR-V-validator-green but worsened the full scene to 16 global and 18
  localized returns; manual review showed tutorial text stretching into long
  shards for one frame. The experiment was reverted.

The ADDR64 experiment also exposed a reusable SPIR-V rule: passing a
`StorageBuffer` pointer as a function argument requires variable-pointer
capability. A validator-clean implementation must either declare and support
that capability or keep the conditional descriptor load inline. Passing unit
tests alone did not prove the generated module legal.

## Next code boundary if the defect reproduces with Relaxed readbacks

Do not add more fragment-export instrumentation while the validated
game-specific readback configuration remains clean. If the defect reproduces
under `Relaxed`, first determine which GPU-owned page was read by the guest and
why the relaxed protection/flush path missed it. Preserve the successful
readback semantics with a focused synthetic RED before changing policy.

If that readback path is exact and the visual defect still reproduces, the next
rendering boundary is the fragment color/MRT export before fixed-function
depth/stencil, blend, and attachment writeback, joined to the existing
target-post plane. Instrument it without adding a second render pass or changing
renderer behavior. The test must prove that it consumes the existing export
SSA, uses the existing sequence/writer authority, retains only bounded
classifications, and adds no image operation, copy, barrier, command, slot,
wait, or release.

If export is stable while target-post is unstable, isolate attachment feedback,
write classification, blend, and writeback contracts one at a time. If export
itself is unstable, continue backward through shader inputs and control flow.
Do not bundle multiple renderer changes or revive a historical patch solely
because it is plausible.

## Evidence classifications

Use precise language in reviews and handoffs:

- **Accepted:** synthetic RED/GREEN, clean checkpoint repetition, focused PS4
  suite, full PS4 regression, and manual visual review all pass.
- **Candidate:** tests pass and the checkpoint improves, but one or more required
  gates remain.
- **Negative evidence:** a measured boundary is uniform across positives and
  controls, so it does not authorize a behavior change.
- **Rejected:** the candidate regresses correctness, relies on a title special
  case, or passed an invalid oracle.
- **Unexercised:** the observed route did not use the hypothesized behavior.

Never convert correlation, timing changes, or a reduced event count into a
causal fix claim without a discriminating invariant.

## Privacy and publication boundaries

Safe to commit:

- general invariants and architecture descriptions;
- synthetic tests and fixtures;
- portable scenario contracts that contain no private assets;
- aggregate pass/fail and bounded performance summaries;
- public commit identifiers.

Keep private and outside Git:

- commercial game files and extracted shaders;
- saves, controller recordings, screenshots, videos, GPU captures, and memory
  dumps;
- machine maps, absolute local paths, credentials, and private service URLs;
- raw logs that may contain paths, identities, or guest content.

## Merge checklist

- [ ] The defect reproduces on clean accepted main with a reviewed full-scene
      reference.
- [ ] A focused synthetic test is RED before production edits.
- [ ] The GREEN repairs a general invariant without a title or shader special
      case.
- [ ] Focused tests and the affected subsystem test target pass.
- [ ] At least three fresh checkpoint trials pass static, temporal, process, and
      manual-review gates.
- [ ] The Uncharted-focused PS4 suite passes.
- [ ] The complete PS4 regression suite passes.
- [ ] Audio and performance results do not regress.
- [ ] Emulator and corpus revisions are recorded together when test intent
      changes.
- [ ] Candidate and accepted-control seeds have matching DMA, readback, title,
      route, asset, and checkpoint configuration.
- [ ] Scenario provenance comes from a clean, commit-specific Git worktree; no
      coordinator-local or nested untracked repository is treated as an
      authoritative input.
- [ ] Documentation distinguishes accepted, rejected, negative, and unknown
      findings.
- [ ] Only source, tests, and sanitized Markdown are committed; private evidence
      remains external.
