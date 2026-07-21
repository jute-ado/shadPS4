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
- `userConfig`: optional path to a private shadPS4 `config.json`. The runner
  validates it as a JSON object and copies it into the case's isolated portable
  user directory before launch. This makes experimental settings and
  game-specific workarounds reproducible without committing personal config.
  It cannot be combined with `--config-clean`, which intentionally forces
  factory defaults and would ignore every supplied value.
- `userDataSeed`: optional path to a private shadPS4 portable user directory.
  The runner copies the complete directory into the case's isolated workspace
  before launch, allowing a known save/profile checkpoint to seed a repeatable
  route without modifying the source checkpoint. Copied files are made writable
  so the emulator can update its private copy even when the source is archived
  read-only. If `userConfig` is also set, it replaces the copied `config.json`
  after the seed is installed.
- `useIpc`: enable shadPS4 IPC start/stop control for graceful log flushing.
  The emulator's IPC capability handshake is required before the runner sends
  `RUN` and `START`; a missing handshake or required capability fails the case.
  All IPC cases require `ENABLE_EMU_CONTROL`, scheduled screenshots require
  `ENABLE_SCREENSHOT`, scheduled RenderDoc captures require
  `ENABLE_RENDERDOC_CAPTURE`, and controller events require `ENABLE_GAMEPAD`.
- `screenshotSeconds`: optional increasing list of times, in seconds after the
  IPC handshake, at which to capture game-only frames. Requires `useIpc`; every
  requested capture must produce a valid PNG for the case to pass.
- `renderdocCaptureSeconds`: optional increasing list of times, in seconds after
  the IPC handshake, at which to capture a complete RenderDoc frame. Requires
  `useIpc` and a process launched with RenderDoc loaded; every requested capture
  must produce a non-empty `.rdc` file for the case to pass.
  Vulkan runs must also have `VK_LAYER_RENDERDOC_Capture` active. A portable
  setup can point `VK_LAYER_PATH` or `VK_ADD_LAYER_PATH` at the directory that
  contains `renderdoc.json`, set `VK_INSTANCE_LAYERS` to that layer name, and
  set `ENABLE_VULKAN_RENDERDOC_CAPTURE=1` before launching the runner.
- `minimumDistinctScreenshots`: optional minimum number of unique screenshot
  contents required for the case to pass. This detects frozen or repeatedly
  blank output when multiple frames are scheduled.
- `screenshotComparisons`: optional pixel-level relationships between captures,
  referenced by their zero-based positions in `screenshotSeconds`. Each entry
  names `firstScreenshot` and `secondScreenshot` and supplies
  `minimumDifference`, `maximumDifference`, or both. The optional
  `differenceMode` is `mean_absolute` by default, which measures mean absolute
  RGB-channel change from 0 (identical) to 1 (maximum change). Set it to
  `cosine` to compare pixel-vector direction instead, making sparse movement in
  very dark scenes observable without amplifying a uniform exposure change.
  Alpha is ignored. A low maximum before input establishes a stable baseline,
  while a meaningful minimum after input proves a visual response. Thresholds
  are mode-specific and should come from repeat runs of the particular scene.
- `buttonEvents`: optional increasing list of player-one button transitions.
  Each entry has `seconds`, a supported `button` name, and a Boolean `pressed`
  state. Requires `useIpc`. Model a tap with a press followed by a release.
  Supported names are `cross`, `circle`, `square`, `triangle`, `options`,
  `dpad_up`, `dpad_right`, `dpad_down`, `dpad_left`, `l1`, `r1`, `l3`, `r3`,
  and `touchpad`.
- `screenshotButtonEvents`: optional ordered visual checkpoints that avoid
  racing variable startup and loading times. Each entry repeatedly requests a
  screenshot until either `screenshotSha256` matches exactly or the frame is
  within `maximumDifference` of `referenceScreenshot`, then taps `button`.
  Reference matching supports the same `differenceMode` values as screenshot
  comparisons and tolerates harmless animation or encoding variation. An
  optional pixel `comparisonRegion` object (`left`, `top`, `width`, `height`)
  limits reference comparison to a stable part of the frame, such as a menu
  control, when unrelated animation or rendering defects should not block
  navigation. The region must fit both equal-sized images and is not valid with
  an exact hash. Bound each checkpoint with `timeoutSeconds`; `pollSeconds`
  defaults to 0.25 and `holdSeconds` defaults to 0.1. Requires both screenshot
  and gamepad IPC capabilities. These state-driven events cannot be combined
  with timed screenshots, captures, or controller events in the same case.
  Keep hashes and reference frames for owned games private, not in the
  repository.
- `axisEvents`: optional increasing list of player-one analog-axis updates.
  Each entry has `seconds`, an `axis` name, and an integer `value` from 0
  through 255. Requires `useIpc`. Supported names are `left_x`, `left_y`,
  `right_x`, `right_y`, `l2`, and `r2`. Stick axes are centered at 128;
  triggers are released at 0. Schedule a return-to-center or release event
  explicitly when an action should end.
- `touchEvents`: optional increasing list of player-one touchpad updates. Each
  entry has `seconds`, a `finger` index of 0 or 1, a Boolean `down` state, and
  native PS4 `x`/`y` coordinates. Valid coordinates are 0 through 1919
  horizontally and 0 through 941 vertically. Requires `useIpc`; model a tap
  with a down event followed by an up event at the same coordinates.
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
mode. If `userConfig` is supplied, only that case receives the copied
`config.json`. The runner writes capped stdout and stderr logs, preserves the
emulator log, and writes `game-test-report.json`. With `useIpc` enabled it waits for the
IPC capability handshake, sends `RUN` and `START`, then requests a graceful
`STOP` at the deadline before falling back to complete process-tree
termination. The hard timeout remains relative to process launch, while
scheduled actions are relative to acknowledged IPC startup. Scheduled
screenshot paths are included in the report together with their SHA-256
content hashes and requested pixel-difference measurements. RenderDoc capture
paths and hashes are included as well. Each `screenshotButtonEvents` poll also
adds a `visual_checkpoint_attempts` report entry with its event index, captured
path and hash, measured reference difference, normalized mean intensity,
non-black pixel fraction, and match result. These measurements make blank or
uniform output visible in machine-readable failure evidence without embedding
the private reference-image path. Button, axis, touch, screenshot, and RenderDoc
events share the same monotonic post-handshake timeline, allowing deterministic
navigation, movement, gestures, causal visual assertions, and frame-level GPU
diagnosis.

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
bounded output, JSON reports, safe artifact names, IPC handshake and controlled
shutdown, scheduled digital, analog, and touchpad controller input, screenshot
and RenderDoc capture validation, pixel-level stable/change relationships, and
hard timeout behavior.
