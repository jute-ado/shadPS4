# U3 host-compilation and EOP lifecycle investigation

## Scope

This note records the generic renderer defect behind the reproducible U3
`m_gfxEopTick` failure seen while entering the London pub sequence. It does not
claim that every U3 rendering issue is fixed. In particular, U3 still needs
relaxed readbacks for the currently known bottle-scene rendering path.

## Symptom

With synchronous graphics pipeline compilation, U3 could fail after roughly
55 seconds at `FrameBegin` while freeing a three-frame render ring:

```text
GetRenderFrameParams(frameToFreeInstancesFor)->m_gfxEopTick
ndlib/render/fg-draw-mgr.cpp:234
```

The host was still compiling shaders and pipelines when guest CPU execution
continued far enough to retire a frame whose EOP had not yet been parsed. The
failure was timing-sensitive and therefore appeared as a U3 crash rather than
as a deterministic Vulkan error.

## Root cause

Commit `5bebf5ab` intentionally enabled asynchronous graphics compilation and
removed the older host-compilation guest pause. That is correct for the async
worker, but it also removed the lifetime guard from the synchronous fallback.
The two paths need different policies:

- synchronous compilation runs on the guest-facing path and must not advance
  guest frame lifecycle time while the host is blocked;
- asynchronous worker compilation must never pause guest threads.

This is independent of the U1 ADDR64 correction in the same commit. The U1
lighting fix remains valid: it binds ADDR64 constant-buffer reads to the decoded
descriptor and eliminates transient page-miss zero reads. Reverting that fix
would restore the earlier orange/zebra hair defect.

## Repair

`PipelineCache` now owns a `ScopedHostCompilationGuestPause` for only:

- synchronous graphics pipeline construction;
- synchronous compute pipeline construction;
- synchronous shader-module compilation.

`CompileGraphicsPipelineAsync` remains explicitly unpaused. The RAII helper
also preserves an already-paused guest and resumes only a pause that it created.

The repair does not change EOP signaling, submission completion, Vulkan queue
ownership, renderer commands, or async-worker behavior.

## TDD contract

`test_host_compilation_pause.cpp` first failed against the missing production
guards, then protects:

- pause/resume lifetime and exactly-once ownership;
- preservation of a pre-existing pause;
- all three synchronous compilation seams;
- absence of the pause in the async worker.

## Validation

Validated on RTX 4080 with the U3 direct-entry bottle route, native internal
resolution, relaxed readbacks, and a task-local async seed:

- synchronous control: survived the full 135-second process lifetime without
  the EOP assertion, but host compilation delayed route progress too much for
  normal play;
- asynchronous route: completed in 130.9 seconds, exit code 0, all required
  game/scene/configuration markers present, and all forbidden crash/device-loss
  markers absent;
- Release application link: pass;
- focused host-compilation and async-queue tests: pass;
- full Windows unit suite: 581/581 pass, with the one expected HTTP fixture
  skip.

The current visual corpus has a fixed 90-second capture window. Cold async runs
can still reach the pub assets after that window, so a zero-frame visual result
is inconclusive rather than a rendering regression. Use a warmed deterministic
seed or a route-relative scene marker before treating that visual gate as
authoritative.

## Development guidance

- Prefer async graphics pipeline compilation for actual play and long Test Lab
  routes.
- Keep the synchronous pause as a correctness fallback; do not make it the
  performance strategy.
- Do not move EOP publication to physical queue completion or make
  `sceGnmSubmitDone` synchronously wait. Earlier investigation found those
  alternatives either semantically wrong or deadlock-prone.
- When changing compilation ownership, test both the synchronous lifetime and
  the async-worker no-pause invariant.

