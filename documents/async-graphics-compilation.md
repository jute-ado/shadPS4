# Asynchronous graphics compilation

## Goal

The user-facing goal is gameplay without a global `Emulation Paused` interval when the host first
encounters a shader or pipeline. Compilation must not suspend arbitrary guest threads: a suspended
thread can own a lock needed by the GPU or driver path and turn a temporary compile into a deadlock.

The complete goal is asynchronous shader translation, SPIR-V creation, and graphics-pipeline
creation. The first implementation stage intentionally covers only graphics-pipeline creation.
Shader translation and compute pipelines remain synchronous until their inputs and state effects can
be snapshotted without borrowing mutable guest or renderer state.

## Stage-one contract

`asyncGraphicsPipelineCompilation` is an experimental Vulkan setting and defaults to `false`.
When enabled:

- a graphics-pipeline cache miss submits one immutable job to one worker;
- duplicate misses for the same key are coalesced;
- the guest CPU remains runnable while the GPU command path waits for the requested pipeline;
- the owner thread publishes the completed pipeline and then submits the original draw, so no draw
  is omitted merely because its pipeline was still compiling;
- the worker creates Vulkan objects without the shared `VkPipelineCache`, whose host access requires
  external synchronization;
- the render thread alone publishes completed objects into renderer maps and persistent metadata;
- a failed job publishes no partial pipeline and does not terminate the worker;
- shutdown stops accepting work and joins the worker before renderer dependencies are destroyed;
- compute dispatch remains synchronous because its stateful effects require a separate ownership and
  ordering design. Graphics draws are not dropped either.

The setting is available in the advanced big-picture settings UI as
`Async Graphics Pipeline Compilation (Experimental)`. It can also be selected in TOML:

```toml
[Vulkan]
asyncGraphicsPipelineCompilation = true
```

## Validation policy

Every async test needs a synchronous control from the same source revision and route. A successful
startup is insufficient. Validation must establish:

1. no persistent pause or deadlock;
2. queued keys eventually publish;
3. no draw is omitted while its pipeline is pending;
4. duplicate requests compile once;
5. shutdown does not retain or access renderer objects;
6. visual and crash gates remain no worse than the synchronous control.

The initial live target is the U1 moving cave/jungle route on Nvidia. AMD remains a separate control;
the old global guest-thread pause had masked frame-lifecycle failures there, so removing it is not by
itself evidence that those failures are solved.

The 2026-08-19 U1 moving-scene comparison reproduced the hair/web flicker with both async enabled and
the matched synchronous control. The original draw-skipping async policy was still incorrect and was
replaced by the ordered wait described above, but async compilation is not the demonstrated cause or
fix for this U1 material defect.

## Remaining work

- Move shader translation and SPIR-V module creation onto an ownership-safe worker job. The current
  `GetProgram` path mutates binding allocation and program permutations, so it cannot be moved by
  merely wrapping the existing function in a thread.
- Measure and reduce the GPU-command-path wait without weakening draw ordering. A compatible
  fallback pipeline may be investigated separately, but silently skipping a draw is not acceptable.
- Design a compute policy that preserves dispatch effects. Do not reuse graphics draw skipping.
- Add queue depth, compile latency, failure, and recovery counters without retaining shader code,
  game identities, or private capture data.

## Privacy

Committed tests and logs use keys, counts, states, and timings only. Game binaries, shader dumps,
routes, saves, captures, screenshots, machine maps, and raw Test Lab logs remain private.
