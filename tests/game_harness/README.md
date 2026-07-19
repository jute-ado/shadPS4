<!--
SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# Private game regression harness

This harness turns local game checks into isolated, repeatable subprocess tests.
The normal CTest suite uses `fake_emulator.py`; it never needs a copyrighted
game dump. Real manifests, game paths, logs, and captures stay outside the
repository.

## Manifest

Copy `games.example.json` somewhere private and edit it there. Paths are
resolved relative to the manifest. `gamePath` must point to an installed game
directory or bootable ELF, not a PKG archive.

Each case supports:

- `name`: unique, human-readable case name.
- `gamePath`: installed game directory or ELF path.
- `timeoutSeconds`: positive hard process timeout.
- `useIpc`: enable shadPS4 IPC start/stop control for graceful log flushing.
- `screenshotSeconds`: optional increasing list of times, in seconds after
  launch, at which to capture game-only frames. Requires `useIpc`; every
  requested capture must produce a valid PNG for the case to pass.
- `minimumDistinctScreenshots`: optional minimum number of unique screenshot
  contents required for the case to pass. This detects frozen or repeatedly
  blank output when multiple frames are scheduled.
- `buttonEvents`: optional increasing list of player-one button transitions.
  Each entry has `seconds`, a supported `button` name, and a Boolean `pressed`
  state. Requires `useIpc`. Model a tap with a press followed by a release.
  Supported names are `cross`, `circle`, `square`, `triangle`, `options`,
  `dpad_up`, `dpad_right`, `dpad_down`, `dpad_left`, `l1`, `r1`, `l3`, `r3`,
  and `touchpad`.
- `args`: arguments inserted before the game path.
- `allowedOutcomes`: any of `exited_zero`, `exited_nonzero`, or `timed_out`.
- `requiredLogPatterns`: literal strings that must occur in stdout, stderr, or
  `user/log/shad_log.txt`.
- `forbiddenLogPatterns`: literal strings that must not occur in those logs.

The optional top-level `emulator` path can be overridden with `--emulator`.

## Run

```powershell
python scripts/game_test_runner.py `
  --manifest path\to\private-games.json `
  --artifacts artifacts\game-tests\local-run
```

Alternatively, set `SHADPS4_GAME_TEST_MANIFEST` and omit `--manifest`.
Use a fresh artifact directory for each run. Every case gets its own working
directory and empty `user` directory, which activates shadPS4's portable-user
mode. The runner writes capped stdout and stderr logs, preserves the emulator
log, and writes `game-test-report.json`. With `useIpc` enabled it sends `RUN`
and `START`, then requests a graceful `STOP` at the deadline before falling
back to complete process-tree termination. Scheduled screenshot paths are
included in the report together with their SHA-256 content hashes. Button
events and screenshot requests share the same monotonic launch-relative
timeline, allowing deterministic menu navigation and visual checkpoints.

An initial smoke milestone can intentionally allow `timed_out`: reaching the
deadline without a forbidden crash marker proves the game survived the tested
boot window. Tighten the outcome and log expectations as compatibility
improves.

## Harness tests and coverage

```powershell
python -W error::ResourceWarning -m unittest discover `
  -s tests\game_harness -p "test_*.py" -v

python -m coverage run --branch -m unittest discover `
  -s tests\game_harness -p "test_*.py"
python -m coverage report -m --include="scripts/game_test_runner.py"
```

The synthetic tests cover manifest validation, relative paths, working
directory isolation, allowed outcomes, required and forbidden log markers,
bounded output, JSON reports, safe artifact names, IPC-controlled shutdown,
scheduled controller input, screenshot capture and validation, and hard
timeout behavior.
