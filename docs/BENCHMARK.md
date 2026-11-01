# Comparing a Ply twin with another service

A comparison is only worth quoting if the two sides describe the same capture. The chapel comparison
was not one: the reference was a 28 m gothic church from a professional service, ours an 18 m chapel
from eight handheld clips. It still taught us the right lessons, but it is not a benchmark.

Rules for the next one:

1. **Same building, same walk.** If the other service cannot process our clips, re-shoot with the SOP and
   give both sides the same footage.
2. **Compare numbers first.** Frames per second of footage, poses registered, gaussians trained, gaussians
   shipped, file size, wall clock. Renders come after, because a viewer's tone mapping flatters anything.
3. **Fixed pose for images.** Same camera position, same field of view, same exposure setting in both
   viewers, or the picture is a comparison of tone curves.
4. **Record it here** with the date, the build (`report.json`), and the exact URLs, so nobody repeats it.

## Reference points

| Date | Run | Views | Poses | Gaussians in → out | SPZ | Wall clock |
|---|---|---|---|---|---|---|
| 2026-09 | aHolo classroom (their service) | 973 keyframes, 1024 px, 368 s (2.64/s) | not exposed | ~1.5 M | 14.5 MB | 32 min |
| 2026-10 | Ply `hq-classroom-1` | 542, 1600 px, 484 s (1.12/s) | 542/542, 0.82 px | 4.0 M → 1.35 M | 37.1 MB | 164 min |
| 2026-10 | Ply `hq-chapel-1` | 275, 515 s (0.53/s) | 271/275, 0.78 px | 4.0 M → 0.68 M | 20.7 MB | 50 min |

The two Ply rows were produced before the planning and gates landed; both would now fail the views check
and never reach a viewer.
