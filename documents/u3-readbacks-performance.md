# Uncharted 3 GPU-readback accuracy and performance

- **Status:** accurate operational mode retained; speculative window expansion rejected
- **Last verified:** 2026-08-20
- **Hardware:** RTX 4080, Ryzen 7 9800X3D
- **Scope:** CUSA02320 pub route, native internal resolution, asynchronous graphics compilation

## Correctness boundary

The tested U3 corruption is a CPU/GPU buffer-coherence failure. After the GPU
owns a buffer page, the guest can partially write that page and later consume
neighboring GPU-produced bytes on the CPU.

`readbacks_mode=0` (`Disabled`) snapshots the CPU page before the partial write
and later uploads only changed CPU words. That preserves the GPU buffer but does
not make untouched GPU-produced bytes visible in the guest CPU backing.
`readbacks_mode=1` (`Relaxed`) synchronously downloads the GPU-owned window on
the write fault, so both the write and neighboring CPU reads observe coherent
content. This explains why the per-game `Relaxed` override is a real accuracy
control rather than a title-specific rendering tweak.

Matched 135.7-second async controls used the same executable, save, route,
resolution, and RTX 4080. `Relaxed` reached the pub scene marker; `Disabled`
remained process-healthy but did not. Historical reviewed full-scene A/B trials
also show that `Relaxed` removes the one-frame geometry corruption while
`Disabled` reproduces it.

## Performance baseline

The performance window starts at presented frame 4200, warms for 60 frames, and
measures 600 frames. The accurate 512 KiB window produced two highly consistent
valid trials:

- mean: 38.34 and 38.38 FPS;
- median: 32.53 and 32.57 ms;
- p95: 34.60 and 34.48 ms;
- p99: 50.59 and 50.27 ms.

One of three trials missed the fixed 130-second route bound, so the formal
three-trial report is inconclusive even though the two timing traces agree.

`Disabled` is not an acceptable replacement. One valid trace reached 56.07 FPS
with a 16.57 ms median, but two of three trials failed to reach the same frame
window even after extending the route to 180 seconds, and the mode is already
known to be visually incorrect.

## Rejected optimization

The synchronous write-fault path currently widens scattered GPU-dirty islands
to a 512 KiB download window. A TDD candidate increased only accurate write
faults to a bounded 2 MiB window, attempting to amortize more neighboring faults
per GPU synchronization. Unit coverage proved buffer-bound and request-bound
window planning before the live change.

The candidate was rejected and removed in full. Its two valid trials averaged
about 37.72 FPS, slightly below the 512 KiB control, one trial timed out, and one
trace had materially worse p95 latency. Copying more neighboring bytes does not
remove enough synchronization events to offset the added transfer/range work.

## Next safe optimization boundary

The remaining cost is synchronization frequency. A future improvement should
retain exact `Relaxed` semantics while moving known CPU-hot GPU ranges into an
already-submitted asynchronous download before the guest faults. It needs:

1. a generic, bounded hot-range policy rather than a title/address heuristic;
2. exact GPU-write generation and completion ownership;
3. a fault path that waits for or consumes the matching generation without a
   second copy or submission;
4. TDD for stale generations, overlapping writes, eviction, and no-consumer
   controls;
5. the same U3 visual A/B gate plus Nvidia cross-game performance controls.

Do not weaken page coherence, promote the rejected 2 MiB heuristic, or claim
that `Disabled` is faster based on its single surviving trace. Keep the
CUSA02320 per-game `Relaxed` override until a generic automatic coherence path
passes those gates.
