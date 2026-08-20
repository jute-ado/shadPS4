# PS4 game test-suite audit (2026-08-20)

- **Status:** Inventory and cleanup proposal; no suite changes made
- **Corpus revision:** `2db4a6ae`
- **shadPS4 unit-test snapshot:** 579 discovered CTest cases

This document separates code-level tests, committed end-to-end game tests, and
private investigation runs. Their results are not interchangeable.

## Layers

1. **CTest/synthetic tests:** 579 cases in the inspected build. They exercise
   settings, GCN translation, renderer policies, synchronization, controller
   replay, Test Lab IPC, audio capture, HTTP/HLE behavior, and other bounded
   contracts. They do not boot a commercial game.
2. **Committed Test Lab corpus:** 31 PS4 scenario files across P.T., Shadow of
   the Colossus, The Last Guardian, and Uncharted: The Nathan Drake Collection.
   Sixteen PS4 suite files contain 48 entries referring to 28 unique scenarios;
   overlap is intentional but currently substantial.
3. **Private task-local investigations:** one-off routes, seeds, temporal
   baselines, diagnostics, and captures used during U1/U2/U3 investigations.
   These are valuable evidence but are not automatically part of the committed
   pre-merge regression gate.

## Canonical broad gates

| Suite | Entries | What it currently proves |
| --- | ---: | --- |
| `local.ps4-quick` | 1 | Shadow of the Colossus boot visual only. |
| `local.ps4-smoke` | 4 | Basic boot/title markers for P.T., Shadow, The Last Guardian, and Uncharted. |
| `local.ps4-visual` | 4 | P.T. late boot, Shadow boot, The Last Guardian boot, and Uncharted seeded intro. |
| `local.ps4-regression` | 12 | The four boot smokes plus P.T. settings/profile variants, three boot/intro visuals, Uncharted intro audio, and one P.T. performance test. |
| `local.ps4-performance` | 1 | P.T. boot performance only. |

The broad regression suite is useful as a cross-game crash/boot guard, but it
does not cover deep Uncharted gameplay, U1 cave materials, U2 field of view, or
U3 bar-scene correctness. Its name currently suggests more coverage than its
12 entries provide.

## Canonical Uncharted-focused gates

| Suite | Entries | Coverage |
| --- | ---: | --- |
| `local.ps4-uncharted-focus` | 3 | Seeded smoke in base/Pro configurations and seeded intro audio. |
| `local.ps4-uncharted-transition-diagnostic` | 6 | Base/Pro intro-to-gameplay smoke, visual, and audio variants. |
| `local.ps4-uncharted-seeded-visual-diagnostic` | 1 | Seeded intro visual/GPU-diagnostic scenario. |
| `local.ps4-uncharted-nvidia-parity` | 4 | U1 cave visual, U1 intro audio, U2 checkpoint smoke, and U2 rooftop performance on Nvidia. |
| `local.ps4-uncharted-amd-parity` | 4 | The same four intents with AMD-specific seeds/profile/reference. |

The vendor-parity pair is the strongest committed gameplay matrix. It still
has important omissions:

- no committed U3 scenario;
- no U2 visual/FoV oracle;
- no 200% host-scale scenario;
- U1 is one 720p, 165-second visual route and does not encode the newly reported
  persistent 200% hair/web failure;
- audio is U1 intro-only, not longer gameplay;
- performance is U2-only and uses adapter-specific references.

## What the current Uncharted tests actually assert

The scenario timeout is not the assertion. A 185-second process can still pass
with a very shallow oracle.

### Boot and seeded smoke

- `boot` runs for up to 60 seconds and requires only that CUSA02320 is
  identified, with no device loss. It does not select U1/U2/U3 or prove a menu
  frame.
- `seeded-smoke` and its PS4 Pro variant install the after-new-game user seed
  but have no controller route or gameplay milestone. They mainly prove that
  the seeded profile can launch without an immediate crash.

These are useful when diagnosing whether startup itself broke, but they add
little pre-merge confidence when a stronger scenario uses the same
configuration and demonstrably reaches gameplay.

### U1 intro and cave

- The intro-to-gameplay family replays one route from the seeded profile and
  checks a route-stage ordinal near the intro/gameplay transition. Base PS4 and
  PS4 Pro variants repeat smoke, visual, and audio operations around the same
  route.
- The audio oracle checks for mostly audible, non-clipped 48 kHz stereo PCM in
  a fixed window. It does not validate dialogue/music identity.
- The committed Nvidia U1 cave visual runs at 1280x720 for up to 165 seconds;
  the AMD variant permits 330 seconds. It samples three elapsed-time frames and
  a 13-frame temporal window. The static comparison thresholds allow the full
  possible image difference, so the meaningful assertions are visibility,
  temporal diversity, repeated-frame bounds, and a coarse abrupt-return count.
  It does not recognize correct hair/web materials and therefore cannot catch
  the persistent 200% regression reported on 2026-08-20.

### U2 checkpoint and performance

- The Nvidia/AMD checkpoint smokes use a U2-direct launch, a private checkpoint
  seed, and a 180-second Continue/confirm route. Their required markers prove
  CUSA02320 started and U2 data—not U1 data—was selected. The minimum stage is
  still only `title_identified`; there is no committed chapter identity,
  post-load gameplay screenshot, movement proof, or level-transition oracle.
- The rooftop performance scenarios reuse that seed and a 60-second route.
  Nvidia requires three valid trials around 60 FPS after presented frame 1500;
  the AMD iGPU has a separate functional 30-FPS-class window. These are useful
  performance gates but not visual-correctness or progression tests.

### U3

There is no committed U3 scenario in the canonical corpus. The bar/bottle,
post-bar crash, and readback investigations currently exist only as private
task-local evidence. Consequently, the normal corpus suites can be green while
U3 gameplay is completely broken.

## Chapter/checkpoint progression suite

The highest-value expansion is a save/checkpoint matrix for U1, U2, and U3.
Each row should represent one chapter or historically fragile level load and
use a private user-data seed plus portable hashed metadata:

1. start from a deterministic save at the beginning of the chapter, or just
   before a historically crashing transition;
2. replay only the inputs needed to Continue/confirm and move through 20-60
   seconds of gameplay;
3. require the correct title and chapter-specific load evidence;
4. require a visible, non-loading gameplay frame and a short distinct-frame
   temporal window after loading;
5. require a bounded gameplay action or camera/movement result, rather than
   accepting a static menu;
6. forbid device loss, access violations, known renderer assertions, and
   premature process exit;
7. keep raw saves and captures private while committing only logical IDs,
   hashes, routes, and sanitized expectations.

For crashes that historically occurred *between* levels, a save already inside
the destination level is not sufficient. Add a smaller transition subset with
the save shortly before the boundary, then cross the boundary under controller
replay. This exercises old-level teardown, streaming, new-level initialization,
and first gameplay together.

Recommended suite tiers:

- `local.ps4-uncharted-chapter-smoke`: one short load-and-move scenario per
  available chapter/checkpoint, run nightly or before promotion;
- `local.ps4-uncharted-transition-regression`: only historically crashing
  boundaries, suitable for a stronger pre-merge gate;
- existing visual/audio/performance suites remain specialized and are not
  substitutes for progression coverage.

One composite checkpoint scenario can cover startup, title selection, save
loading, and gameplay. Its smoke policy reports process/required-marker failure,
individual visual checkpoints report which post-load boundary was unavailable
or changed, and the final gameplay assertion is evaluated only after all prior
boundaries succeed. Separate boot coverage is therefore not inherently
necessary. Once the chapter suite reliably traverses the collection menu, the
standalone Uncharted `boot` and un-routed `seeded-smoke` entries should leave the
main pre-merge gate. Keep a tiny startup-only scenario solely when its materially
shorter runtime improves local failure triage; do not count it as independent
compatibility coverage.

## Existing save-seed inventory and storage

The canonical Uncharted scenarios currently reference eight distinct logical
user-data seeds with eight distinct tree digests:

- one `after-new-game` seed reused by twelve intro/smoke/visual scenarios;
- separate Nvidia and AMD U1 cave seeds;
- separate Nvidia and AMD U1 intro-audio seeds;
- one U1 second-gameplay/chapter-2 load seed, currently not selected by a suite;
- separate Nvidia and AMD U2 checkpoint seeds, each reused by its smoke and
  performance scenario.

This is not yet a per-level save matrix. It is a small collection of historical
investigation checkpoints, and several vendor-specific seeds likely represent
the same gameplay intent with different run-local configuration state.

The save bytes are deliberately absent from Git. A scenario commits only a
logical asset ID and exact `sha256-tree-v1` digest. An untracked machine asset
map resolves that ID to a read-only private bundle on the executing machine.
For every run, Test Lab verifies the source tree, copies it into a unique
run-local shadPS4 `user` profile, verifies the copy, and lets the emulator mutate
only that copy. Performance trials receive independent copies. Another machine
can execute the scenario only after installing a byte-identical private bundle;
otherwise asset validation fails before launch.

This existing mechanism is sufficient for chapter checkpoint tests. We need to
curate and register additional save candidates, not invent save-state support.
Complete CPU/GPU emulator snapshots remain unsupported and are intentionally a
separate, more fragile future capability.

## Reusable controller-route fragments

The controller contract is not DRY today. Each scenario references one complete
schema-1 route: a normalized full-controller-state timeline with absolute
elapsed-millisecond offsets and one exact SHA-256. Test Lab can record, import,
validate, and replay that route, but it cannot currently reference or compose
route fragments. Copying a common menu prefix into many route JSON files would
work operationally while creating maintenance drift.

Add route composition as a small, separately tested framework/corpus feature:

- define immutable fragments such as `collection-select-u1`,
  `collection-select-u2`, `collection-select-u3`, `continue-confirm`, and a
  short chapter-specific movement tail;
- require every fragment to use the same controller profile and clock and to
  begin/end in a neutral full state;
- store fragment-relative offsets, rebase them during composition, and reject
  overlap, overflow, non-monotonic time, or incompatible boundary state;
- pin every fragment by logical ID and SHA-256, then materialize and pin the
  final composed route digest used by the scenario;
- record component and final digests in provenance so changing a shared prefix
  cannot silently change historical tests;
- make corpus validation report every dependent scenario when a fragment
  changes and require its composed route to be regenerated/reviewed.

Standardize chapter-save candidates so they all arrive at the same predictable
Continue UI state. Then most chapter scenarios can reuse one title-specific
menu/Continue prefix and vary only their post-load movement/checkpoint tail.
U2/U3 direct-entry scenarios may deliberately bypass the collection menu, but
the main progression matrix should preserve at least one composite route that
tests the collection selection path.

Do not optimize for the fastest human menu traversal. Current routes are driven
by elapsed time, and cold pipeline compilation plus vendor performance changes
when a menu becomes ready. Record the shortest route with conservative,
measured timing on the slow supported profile, or introduce a future bounded
marker/checkpoint wait between fragments. Nvidia and AMD may still need pacing
variants while sharing the same logical action sequence.

Three committed Uncharted scenarios are not selected by any PS4 suite:
`seeded-intro-history`, `seeded-intro-validation`, and `u1-chapter2-load`.
They should either gain a clearly named diagnostic suite or be marked as
fixtures intentionally invoked directly; silent orphaning makes maintenance
unclear.

## Recommended cleanup

Do not delete historical private evidence. Clean the *active selection layer*:

1. Define three explicit tiers:
   - `ps4-pr-fast`: synthetic tests plus a short multi-game smoke;
   - `ps4-nightly`: the existing 12-entry broad regression plus the Nvidia
     Uncharted functional matrix;
   - `ps4-diagnostic-*`: long, vendor-specific, RenderDoc, history, and known-
     failing routes, never implied to be a general pass gate.
2. Make suites compose or generate from one manifest instead of repeating the
   same scenario in up to three hand-maintained suites.
3. Add owner, purpose, expected duration, hardware scope, and last-reviewed
   date to every suite/scenario. A suite name alone is not an acceptance claim.
4. Add a validator that reports orphan scenarios, duplicate intent, stale
   hashes, missing machine capabilities, and accidental commercial/private
   artifacts before launch.
5. Keep expected failures explicit. The U1 200% cave case belongs in a known-
   failing visual diagnostic until fixed; weakening its oracle to make a broad
   suite green would destroy its value.

## Recommended expansion order

1. **U1:** matched native/200% cave routes with effective per-draw scale
   evidence and a temporal/material oracle. Keep native and scaled results
   distinct.
2. **U2:** add a reproducible visual FoV/aiming checkpoint; retain the existing
   checkpoint smoke and rooftop performance controls.
3. **U3:** promote one short bar/bottle correctness route and one post-bar
   crash/progression route into the committed corpus, with `readbacks_mode`
   stated explicitly.
4. **Reliability:** require repeat runs for temporal defects and performance;
   one visually lucky run is not acceptance.
5. **Coverage accounting:** publish a generated matrix of game, scene, vendor,
   profile, resolution mode, operation, and status so gaps are visible without
   reading JSON by hand.

The immediate conclusion is to reorganize before adding many more scenarios.
The framework is capable, but the active gates underrepresent the exact bugs
the fork is trying to protect and overrepresent boot/intro variants.
