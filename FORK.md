# jute-ado shadPS4 Fork

This is a downstream development fork of
[shadps4-emu/shadPS4](https://github.com/shadps4-emu/shadPS4).

## Focus

The fork prioritizes:

- test-driven development;
- stronger unit, integration, and regression coverage;
- reproducible diagnosis of compatibility failures;
- converting observations from a small private compatibility corpus into
  synthetic, legally redistributable regression tests;
- changes that remain understandable and reviewable for possible upstream
  contribution.

Private game dumps are evidence, not repository fixtures. Game files, local
paths, private manifests, recordings, and captures containing proprietary
content must not be committed.

## Branch model

- `main` is the downstream integration branch.
- `feature/<topic>` and `fix/<topic>` branches target downstream `main`.
- `upstream-fix/<topic>` branches start from `upstream/main` and contain
  focused changes intended for an upstream pull request.

Upstream changes are periodically merged into downstream `main`. The merge is
accepted only after the relevant tests pass and conflicts are resolved without
weakening established behavior.

## Development model

For behavior changes and bug fixes:

1. Add or strengthen a test that fails for the expected reason.
2. Implement the smallest coherent production change.
3. Run the focused test and the relevant broader test suite.
4. Keep the test and implementation together in the same reviewable change.

When a private game reveals a defect, reduce it to a synthetic test whenever
possible. Keep machine-specific end-to-end configuration outside Git.
