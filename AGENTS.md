# Repository Instructions

Work test-first.

For behavior changes and bug fixes, add or strengthen a test that demonstrates
the problem, confirm that it fails for the expected reason, then implement the
smallest coherent fix and run the focused and relevant broader tests.

Do not commit game dumps, extracted game content, private manifests, local
filesystem paths, credentials, proprietary captures, or recordings.

Use synthetic or legally redistributable fixtures for public regression tests.
Observations from private game testing should be reduced to a synthetic test
before production behavior is changed whenever possible.

`main` is the downstream integration branch. Merge updates from
`upstream/main` into `main` after validation. Create branches intended for
upstream pull requests directly from `upstream/main`.

## External game regression workflow

Use the accepted external Emulator Test Lab corpus unchanged unless the
emulator task changes game-test intent. When a change advances or alters a
scenario, controller route, expectation, visual/performance baseline metadata,
save-data pin, or GPU policy, create a separate corpus feature worktree paired
with this emulator worktree.

Point the task-local emulator map at the executable built from this worktree
and at this same worktree as its source repository. Run the exact emulator,
framework, and corpus revisions together before merging. Matching branch names
are only a convenience; the immutable revision identities in the run are the
authoritative pairing.

Other emulator worktrees must continue using corpus `master` or their own
corpus worktree. If two branches change the same accepted test object, merge
the first pair, rebase the second, and produce fresh evidence. Never resolve a
baseline or scenario digest conflict mechanically.

Because the emulator repositories are public, their committed documentation
must use placeholders—never your `F:\...` paths, private Forgejo address,
credentials, game identities unnecessarily, or vault layout. Exact
machine-specific commands belong in a private local runbook.

See `documents/emulator-test-lab.md` for the complete local workflow.
