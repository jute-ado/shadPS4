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
