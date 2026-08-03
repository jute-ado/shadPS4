# Uncharted 1 cave hair investigation

Status: candidate fix proven at the cave checkpoint; merge gates pending.

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
- [ ] Run focused PS4 suites, performance, and cross-game regression gates
  before merge or push.

## Working rules

Prefer deterministic RenderDoc Python/CLI reports over GUI inspection. Keep
private captures, screenshots, saves, maps, and machine paths outside Git.
Update this file when an item is proved or ruled out; do not check off a fix on
visual improvement alone.
