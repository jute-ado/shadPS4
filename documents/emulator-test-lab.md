<!--
SPDX-FileCopyrightText: 2026 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# Emulator Test Lab workflow

shadPS4 supports an external, local-only regression framework for exercising
privately owned games. The external framework owns orchestration, portable
scenario contracts, controller routes, evidence classification, and immutable
run reports. This repository owns shadPS4-specific instrumentation and
synthetic regression tests.

The capability probe emitted by the build is authoritative. A test must fail
its capability gate before launch when the current build cannot provide the
requested controller, presented-frame, timing, diagnostic, configuration, or
snapshot behavior.

## Repository boundaries

- shadPS4 source and synthetic tests belong in this repository.
- Portable game scenarios, controller routes, and expectation or baseline
  metadata belong in the external corpus repository.
- Games, saves, screenshots, videos, RenderDoc captures, memory images, and
  machine maps remain outside Git in a private local store.
- Framework source changes require their own framework branch only when the
  task changes orchestration or a versioned contract.

Because the emulator repositories are public, their committed documentation
must use placeholders—never your `F:\...` paths, private Forgejo address,
credentials, game identities unnecessarily, or vault layout. Exact
machine-specific commands belong in a private local runbook.

Public regression tests must use synthetic or legally redistributable inputs.
Reduce behavior learned from a private commercial-game run to a synthetic test
whenever possible.

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
the task's shadPS4 worktree and identify that same worktree as
`repositoryPath`. The runner uses it to record the exact emulator commit and,
when necessary, a dirty-content hash.

Every scenario selects `ps4` or `ps4_pro` plus explicit output and internal
render dimensions. Do not rely on a developer's global configuration. Every
run receives an isolated writable profile.

Run an accepted suite with explicit, portable arguments:

```text
emu-test suite run <suite> <corpus-root> <machine-assets.json> \
  <machine-emulators.json> <machine-profile-or-dash> <suite-runs-root>
```

The command's explicit corpus root is what isolates test intent. It does not
search for or automatically use another task's corpus branch.

## When game progress changes

If the emulator branch still satisfies the accepted expectation, no corpus
branch is required. Add or strengthen a synthetic shadPS4 test in this branch
and continue using corpus `master`.

Create a paired corpus worktree when reviewed test intent changes, including:

- a new compatibility or progress floor;
- a changed controller route;
- a visual or temporal candidate;
- a performance reference;
- a private save-data or snapshot pin;
- a RenderDoc or other GPU diagnostic policy.

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
worktree, so it cannot see these unmerged expectations.

If two tasks edit the same scenario, route, baseline, or policy, merge the
first reviewed pair and rebase the second pair. The second task must rerun
against the newly accepted state and create a fresh candidate. Never choose a
digest conflict mechanically or widen a visual/performance threshold merely
to make the branch pass.

Git cannot atomically merge two repositories. Local development therefore uses
a final clean paired run followed by consecutive merges. Cross-repository CI
coordination is a separate future concern.
