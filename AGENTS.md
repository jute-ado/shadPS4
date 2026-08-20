# Repository instructions

Work test-first.

For behavior changes and bug fixes, add or strengthen a test that demonstrates
the problem, confirm that it fails for the expected reason, then implement the
smallest coherent fix and run the focused and relevant broader tests.

Never commit game packages, extracted game content, private manifests,
commercial-game save data, screenshots, videos, GPU captures, memory dumps,
credentials, personal information, or machine-local filesystem paths. Public
regression tests must use synthetic or legally redistributable fixtures.
Reduce observations from private game testing to a synthetic test whenever
possible.

## External game regression workflow

shadPS4 development must use only `local.ps4-*` suites and must pass the
explicit `ps4` scope guard to `emu-test suite run`. Never use a
`cross_platform` aggregate as a routine shadPS4 gate: PS4 and PS5 development
are independent, and a shadPS4 task must not launch or depend on SharpEmu.

Use the accepted external Emulator Test Lab corpus unchanged unless the
emulator task changes game-test intent. When a change advances or alters a
scenario, controller route, expectation, visual/performance baseline metadata,
save-data pin, GPU policy, or audio policy, create a separate corpus feature
worktree paired with this emulator worktree.

Point the task-local emulator map at the executable built from this worktree
and identify this same worktree as its source repository. Run the exact
emulator, framework, and corpus revisions together before merging. Matching
branch names are only a convenience; immutable revision identities in the run
are the authoritative pairing.

Other emulator worktrees must continue using corpus `master` or their own
corpus worktree. If two branches change the same accepted test object, merge
the first pair, rebase the second, and produce fresh evidence. Never resolve a
baseline or scenario digest conflict mechanically.

Because the emulator repositories are public, their committed documentation
must use placeholders—never your `<drive>:\...` paths, private service address,
credentials, game identities unnecessarily, or vault layout. Exact
machine-specific commands belong in a private local runbook.

Start with `documents/engineering-knowledge-base.md` for the sanitized
engineering index and `documents/emulator-test-lab.md` for the complete local
workflow.
