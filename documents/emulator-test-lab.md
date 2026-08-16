# Emulator Test Lab workflow

shadPS4 game regressions are exercised by an external, local-only test
framework. The external framework owns orchestration, portable scenario
contracts, controller routes, evidence classification, and immutable run
reports. This repository owns shadPS4-specific behavior and synthetic
regression tests.

A Test Lab-enabled shadPS4 build must expose a capability probe. The reported
capabilities are authoritative: a test must fail its capability gate before
launch when the build cannot provide the requested controller, presented-frame,
timing, diagnostic, configuration, save-data, state, or audio-capture behavior.

## Repository boundaries

- shadPS4 source and synthetic tests belong in this repository.
- Portable game scenarios, controller routes, and expectation or baseline
  metadata belong in the external corpus repository.
- Games, saves, screenshots, videos, GPU traces, memory images, and machine
  maps remain outside Git in a private local store.
- Framework source changes require a framework branch only when the task
  changes orchestration or a versioned contract.

Because the emulator repositories are public, their committed documentation
must use placeholders—never your `F:\...` paths, private Forgejo address,
credentials, game identities unnecessarily, or vault layout. Exact
machine-specific commands belong in a private local runbook.

Public regression tests must use synthetic or legally redistributable inputs.
Reduce behavior learned from a private commercial-game run to a synthetic test
whenever possible.

For the stricter visual-correctness rules and a worked compatibility campaign,
see [Uncharted compatibility investigation guide](uncharted-compatibility-investigation.md).

## Starting an emulator task

Create one emulator feature worktree and one unique run root. Use the accepted
corpus branch by default:

```text
task/
├── shadPS4 worktree on feature/<change>
├── machine-emulators.json
└── runs/<unique-run-id>/

shared, read-only
├── released emu-test executable
├── accepted corpus checkout
└── private asset vault
```

The task-local `machine-emulators.json` must point to the executable built from
the task's shadPS4 worktree and identify that worktree as `repositoryPath`.
This records the exact emulator commit and, when necessary, a dirty-content
hash. Each run must receive isolated writable data rather than mutating a
developer profile.

Run a platform-isolated suite with explicit, portable arguments:

```text
emu-test suite run <suite> <corpus-root> <machine-assets.json> \
  <machine-emulators.json> <machine-profile-or-dash> <suite-runs-root> ps4
```

Use `local.ps4-quick` while iterating and `local.ps4-regression` for the
complete pre-merge game gate. Other `local.ps4-*` suites select a focused
operation such as audio, visual, performance, settings, or GPU diagnostics.
Do not use a `cross_platform` aggregate for routine shadPS4 development.

The explicit `ps4` argument is a fail-closed platform guard. The runner checks
it against the suite's declared scope before corpus resolution, run-directory
creation, emulator-map loading, or process launch. The corpus also rejects a
PS4 suite containing any SharpEmu scenario. The command's explicit corpus root
isolates test intent; it does not search for another task's corpus branch.

## When game progress changes

If the emulator branch still satisfies the accepted expectation, no corpus
branch is required. Add or strengthen a synthetic shadPS4 test in this branch
and continue using corpus `master`.

Create a paired corpus worktree when reviewed test intent changes, including:

- a new compatibility or progress floor;
- a changed DualShock/controller route;
- a visual or temporal candidate;
- a performance reference;
- a private save-data or future snapshot pin;
- a RenderDoc or guest-GPU diagnostic policy;
- an audio health policy.

The workflow is:

1. run this branch against the accepted corpus;
2. preserve and review the immutable evidence;
3. create a corpus feature worktree;
4. generate a candidate instead of overwriting the accepted object;
5. commit only portable corpus metadata;
6. run focused and regression suites against clean paired revisions;
7. cross-reference the proven emulator and corpus commits;
8. merge the emulator and corpus branches consecutively.

Matching branch names help people recognize the pair, but they do not establish
compatibility. Exact commit and content identities recorded in the run do.

## Concurrent worktrees

Another shadPS4 worktree continues using corpus `master` or its own corpus
worktree, so it cannot see unmerged expectations.

If two tasks edit the same scenario, route, baseline, or policy, merge the
first reviewed pair and rebase the second pair. The second task must rerun
against the newly accepted state and create fresh evidence. Never choose a
digest conflict mechanically or widen a visual/performance threshold merely
to make the branch pass.

Git cannot atomically merge two repositories. Local development therefore uses
a final clean paired run followed by consecutive merges. Cross-repository CI
coordination is a separate future concern.

## Audio capture contract

When `EMULATOR_TEST_LAB_AUDIO_PCM16` contains an absolute output path,
shadPS4 writes the normalized main AudioOut stream there as append-only
48 kHz stereo signed PCM16. Capture occurs before the host audio backend, so
it is independent of the selected sound device, speaker volume, and host
mixer. The file is private run evidence and must never be committed.
