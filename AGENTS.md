# Repository instructions

Work test-first. Add or strengthen a focused test before changing emulator
behavior, then implement the smallest coherent fix and run the affected and
broader validation gates.

Never commit game packages, extracted game content, commercial-game saves,
private manifests or routes, screenshots, videos, GPU captures, memory dumps,
raw game logs, credentials, personal information, or machine-local filesystem
paths. Public tests must use synthetic or legally redistributable fixtures.

`main` is the usable public fork branch. Keep investigation journals,
diagnostic schemas, AI-development notes, and detailed Test Lab procedures on
`dev`. Promote only working code, essential regression tests, and concise
user-facing documentation to `main`; merge `main` back into `dev`, never `dev`
wholesale into `main`.
