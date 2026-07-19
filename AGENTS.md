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
