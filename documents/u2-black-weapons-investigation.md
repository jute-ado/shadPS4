# Uncharted 2 black weapons investigation

Status: a current deterministic route now exposes a held, visually near-black
pistol for more than 100 seconds. A guest-frame capture traces the appearance
to the game's own material and lighting calculation rather than a failed
texture, missing draw, final composite, or presentation operation. This route
does not yet prove a general emulator defect, so do not claim a fix.

Current evidence:

- The first post-ADDR64 visual run accidentally used the segment-3 recording-source seed. It reached the snowy train scene without an emulator assertion, but it did not exercise the reported sewer weapon view and Test Lab rejected the incomplete visual evidence package.
- Direct replay of the intended post-cutscene seed completed the 115-second route with exit code zero and no forbidden crash marker.
- Test Lab requires the scenario path to be inside the paired corpus worktree for revision provenance. The task-local scenario therefore reports `visual_run_unavailable`; the paired candidate now references the intended post-cutscene seed.
- The provenance-correct post-cutscene visual run passed with six distinct moving frames and no forbidden marker. The holstered sidearm is colored, and the L2 window does not expose a held gun, so this route does not reproduce the human-reported all-black weapons defect.
- Next evidence attempt: replay only the first 100 seconds of the preserved segment-3 recording and inspect its existing 68--88 second temporal window.
- The recorded 68--88 second window remained in the title menus on the current build; it does not reach the old gameplay timing. Historical seed screenshots do show stable opaque-black circular regions at Drake's left torso/holster in the snowy train scene.
- A derived private seed removes only the 19 stale screenshots from the recording-source user data. Use it for a same-route A/B between the pre-ADDR64 binary and accepted post-ADDR64 `main`; never commit the seed or its machine-map entry.
- The exact historical trace was retained, but its menu inputs occur before the current cold-start menu is ready. A hybrid diagnostic uses the proven 30/45/60-second menu prefix, then preserves every historical gameplay state from its original first movement onward, shifted so gameplay input begins at 80 seconds. The focused visual window is 240--265 seconds.
- The provenance-correct hybrid run on accepted `main` reached the snowy train scene and showed stable dark debris regions across the 245--255-second frames. The opposite hip sidearm remained metallic gray, and no held weapon was visible. The process exited cleanly with six distinct moving screenshots and no forbidden crash marker. The run was red only because the historical seed's verbose output exceeded the bounded log limit; a derived private seed now changes only logging to bounded normal settings.
- A separate accepted-main RenderDoc diagnostic captured the reproduced frame at 250 seconds (`966a7b18...d5483`, 319,949,925 bytes), but the capture contains only ten presenter/composite actions through event 32. It proves the bad image reached presentation, not which guest draw first produced it.
- The existing general guest-frame capture implementation was replayed TDD onto the current branch. Its synthetic test first failed because capture state lacked an active-through-present phase, then passed after the capture began before guest-frame submission and ended after presentation. The prior implementation produced a 1.27-GB U2 guest capture, so the current hybrid route is the authoritative validation target.
- The current implementation produced a clean 1,625,402,989-byte capture (`637daa61...b0cfef`) containing 4,337 actions through event 15,699. The older presenter-only capture contained ten actions through event 32.
- RenderDoc pixel history traced representative dark snow-field regions through the presentation and tone-map chain to geometry draw 13,351. That draw uses valid rusty-metal color and normal textures and writes actual debris/wreck geometry in front of the later snow draw, which correctly fails depth. This is not sufficient evidence of the reported all-black-guns defect.
- Historical screenshots likewise show a metallic/colored holstered sidearm and never isolate a held gun. A future focused recording must visibly hold or aim a weapon before renderer logic changes.
- The prompt-safe route from the later human playthrough presses L2 at 101.170
  seconds. Dense 250-ms temporal evidence shows the held pistol continuously
  from 101.5 through 104.5 seconds while the HUD icon and nearby architecture
  remain colored.
- A timing-tolerant derivative preserves the human route through the fully
  pressed L2 event, then holds neutral sticks plus L2 through 205 seconds. Its
  20-frame temporal run passed with 20 distinct frames, zero abrupt returns,
  no forbidden marker, and the pistol visible at every 120--205-second
  checkpoint. Run ID: `d649cd5b-225c-481d-98f3-5388fb8cd81e`.
- The first RenderDoc trigger at 103.5 seconds captured the game's loading
  dagger because injection overhead delayed the loaded scene. It is not weapon
  evidence. The long-L2 route moved the trigger to the proven late window.
- The late guest-frame diagnostic captured 6,782 actions through event 24,417.
  Run ID: `95351f0c-3d24-4404-9142-14daab0c5e44`; capture SHA-256:
  `2660fb7adc8fd1d5eaf06f336d29b604143dd149ef2b891499efb9876e1b119a`.
- The black value is already present in the game's 1920x1080 HDR image. It is
  not introduced by the final copy, sRGB conversion, HUD pass, output scaling,
  or presentation.
- Pixel history identifies draw event 19,376 and translated fragment shader
  `fs_0x00000000128e945b_0`. The draw's 1024x1024 sRGB albedo contains intact
  gray metal and wood detail; its companion material texture and environment
  map are also populated.
- The emulator's dumped SPIR-V reproduces RenderDoc's original shader output
  bit-for-bit. Surgical output replacements isolate the dark body pixel:
  albedo `(0.0407, 0.0359, 0.0324)`, material/specular sample about `0.243`,
  environment sample `(0.125, 0.165, 0.198)`, and normalized world normal
  `(0.988, -0.107, 0.111)`. The game computes a normal-light term of `0.207`
  and a curved diffuse multiplier of `0.059`.
- A correctly brighter cylinder pixel from the same draw, shader, and weapon
  has a world normal `(0.093, 0.968, 0.235)`, normal-light term `0.974`, and
  diffuse multiplier `0.954`. The observed contrast is therefore explained by
  surface orientation and the game's lighting path in this nighttime scene.

## Target

Restore correctly shaded weapon geometry without a title, route, shader, draw,
or asset special case.

## Checklist

- [x] Reproduce the reported dark appearance with a deterministic held-weapon route.
- [x] Confirm an actually visible gun across temporal frames and distinguish this route from a failed draw or texture.
- [x] Capture the affected guest-rendered frame separately and inspect it with RenderDoc CLI.
- [x] Identify the exact draw, shaders, resources, and source of the dark value for this route.
- [ ] Reduce the renderer invariant to a synthetic failing test.
- [ ] Implement the smallest general fix and make the synthetic test green.
- [ ] Repeat the exact U2 route and visually verify the weapon.
- [ ] Run relevant GPU, Uncharted-focus, PS4-regression, and campaign gates.

## Hypotheses

- Texture descriptor, format, swizzle, or subresource selection is wrong.
- Material data is read from the wrong guest address or buffer range.
- Alpha/blend, sRGB conversion, or fragment export loses the lit material.
- A required weapon pass is missing or loses depth/order against the character.
- Vertex normal, tangent, or front-face input is decoded incorrectly.

## Findings

- Older captures show stable opaque-black weapon-shaped regions, but were made
  before the accepted U1 `ADDR64` change and are not the current baseline.
- The current late capture rejects the texture, material-address, missing-pass,
  final-composite, and sRGB hypotheses for the aimed pistol. It also shows a
  coherent full-scene world-normal buffer. A future claim still needs a route
  where the gun is implausibly black under lighting that makes the same
  surfaces bright on original hardware; this nighttime route is not that
  oracle by itself.
