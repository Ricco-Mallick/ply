# Status

Updated 2026-11-11. What is deployed, what is running and what is next. Update this file at the end of every session,
so the next one starts here and not from an exported transcript.

## Deployed

| What | Version | Note |
|---|---|---|
| Site (https://ricco.codes/ply/) | `main` at `5df7b5f` (PR #21), 2026-11-11 | opens a twin in Walk when its `occupancy.json` has `walk.pass` |
| API (`ply-api`, Cloudflare) | `35c9c1d`, 2026-11-05 | read from `GET /` |
| GPU worker (Modal app `ply`, riccomarcelino) | not redeployed since before PR #21 | the mapper and evidence-upload changes of 2026-11-11 reach jobs only after a redeploy |
| Chapel twin `chapel-high-2` | walking grid replaced 2026-11-11 | `ply-data` commit `4c09354`; the commit before it, `e063f08`, holds the old grid. The splat and `scene.json` are unchanged |
| CUDA wheels | `wheels/cp312-cu128-sm89` at `e063f08` | L4 and L40S only; `pipeline/wheels.lock.json` |

## Running

- **Wheel rebuild for H100 and A100**: workflow run 43063630302, started 2026-11-11 16:58 UTC. `ppisp`, `fused-ssim`
  and `fused-bilagrid` are built; `gsplat` was still compiling at 18:11 UTC and has a 5-hour limit. Artifacts are kept
  for 30 days.

## Next

1. **Publish the rebuilt wheels** when the run is green: `gh run download 43063630302 -D /tmp/wh`, copy the four
   `.whl` files flat into `pipeline/wheels`, `modal run scripts/wheels.py::publish --tag <python-cuda-arch list>`,
   commit the lock file, then `::fetch` into an empty folder and compare checksums. If `gsplat` times out, build it
   alone per GPU type.
2. **Redeploy the worker** after the wheels are fetched, then run the smoke capture (0.6, about $0.15) before any long
   job. The first long ordered capture after that shows whether skipping the global mapper saves the expected
   32 minutes; record it in `IMPROVEMENT.md`.
3. **Make the `ci` checks required** on `main` (repository settings, owner).
4. **Rest of phase 0**: spend limits and the idempotency key (0.3), results only for published jobs and an owner per
   job (0.4, API side), the smoke capture (0.6), `plyctl` (0.7), the clips' colour transfer (0.11).
5. **Phase 1** (measure) once phase 0 is closed. Open: the date of the final.

## Known and left alone

- The capture app cannot ask for the global mapper on a long video: it sends `--mapper` only for `incremental`, and
  its "Global (fast, default)" label no longer describes what a 400-frame walk gets (`app/src/capture/lib/backend.ts`).
- `wheels.yml` still defaults to Python 3.13 and arch 7.5. Dispatch it with `python=3.12` and the arch list.
- The model page accepts a module from anywhere on the site's origin, not only under `/ply/`. There is no content
  security policy, and `?scene=` accepts any origin by design.
