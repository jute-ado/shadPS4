# Uncharted 1 cave hair investigation

Status: the original orange hair-card lattice is fixed. A follow-on intermittent
white hair/web flash has a focused TDD candidate, but that candidate is not yet
promoted because the full PS4 regression gate is 11/12.

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
- [ ] Pass the complete PS4 regression gate. The current run is 11/12: one PT
  late-boot trial exited with `0x80000003` at `image.cpp:258 GetBarriers` before
  its 105-second checkpoint and produced zero frames. A bounded rerun of the
  candidate and a matching accepted-branch control both reached the checkpoint
  cleanly, so the assertion is currently intermittent and is not attributed to
  this fix without repeatable evidence. Do not suppress or clamp the assertion.

Candidate commits are `845f6391` (RED) and `7b182423` (GREEN). The source branch
must remain separate from the Uncharted integration branch until the applicable
regression gate is green or the independent `GetBarriers` invariant is proved
and fixed.

## Working rules

Prefer deterministic RenderDoc Python/CLI reports over GUI inspection. Keep
private captures, screenshots, saves, maps, and machine paths outside Git.
Update this file when an item is proved or ruled out; do not check off a fix on
visual improvement alone.
