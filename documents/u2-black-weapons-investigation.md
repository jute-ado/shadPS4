# Uncharted 2 black weapons investigation

Status: the historical human report is preserved, but the current deterministic
route does not expose a held black weapon. Do not claim reproduction or a fix.

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

## Target

Restore correctly shaded weapon geometry without a title, route, shader, draw,
or asset special case.

## Checklist

- [ ] Reproduce the defect with the shortest deterministic recorded U2 weapon route.
- [ ] Confirm an actually visible gun across temporal frames and distinguish it from deep shadow.
- [ ] Capture the affected guest-rendered frame separately and inspect it with RenderDoc CLI.
- [ ] Identify the exact draw, shaders, resources, and first incorrect pipeline stage.
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
