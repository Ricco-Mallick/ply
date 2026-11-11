# Status

Updated 2026-11-11 (evening). What is deployed, what is running and what is next. Update this file at the end of every
session, so the next one starts here and not from an exported transcript.

## Deployed

| What | Version | Note |
|---|---|---|
| Site (https://ricco.codes/ply/) | `main` at `e7bddf1` (PR #22), 2026-11-11 | still opens the chapel in Walk and draws the grid in the minimap until the second batch is merged |
| API (`ply-api`, Cloudflare) | `35c9c1d`, 2026-11-05 | read from `GET /` |
| GPU worker (Modal app `ply`, riccomarcelino) | not redeployed since before PR #21 | the mapper choice, one-commit evidence upload, job limits and top view reach jobs only after a redeploy |
| Twins on Hugging Face (`ply-data`) | `4c46886`, 2026-11-11 | 13 twins have `topview.webp` and a `topview` block in `scene.json`; `4c09354` is the commit before. `chapel-high-2` has the rebuilt walking grid (`4c09354`; `e063f08` before it) |
| CUDA wheels | `wheels/cp312-cu128-sm89` at `e063f08` | L4 and L40S only; `pipeline/wheels.lock.json` |

## Accounts

| Modal account | Holds | Used for |
|---|---|---|
| riccomarcelino (about $6) | the deployed worker, `ply-bench` and `ply-handoff` Volumes, secrets `ply-hf` and `ply-worker` | live jobs |
| opservemakesalt ($30 on 2026-11-11) | Volume `ply-bench` with `chapel/high/job` (the solve and photos only), secret `ply-hf` (read token), the worker image | experiments (`pipeline/modal/ply_eval.py`) |
| ricco-mallick (about $1.70) | `ply-bench:drjohnson/prep` | nothing at present |

## Running

- **Wheel rebuild for H100 and A100**: workflow run 43063630302, started 2026-11-11 16:58 UTC, 5-hour limit. Three
  packages are built; `gsplat` was still compiling at 19:15 UTC. Artifacts are kept for 30 days.
- **Phase 1 baselines** on opservemakesalt: see `EXPERIMENTS.md` for what has finished.

## Next

1. **Publish the rebuilt wheels** when the run is green: `gh run download 43063630302 -D /tmp/wh`, copy the four
   `.whl` files flat into `pipeline/wheels`, `modal run scripts/wheels.py::publish --tag <python-cuda-arch list>`,
   commit the lock file, then `::fetch` into an empty folder and compare checksums. If `gsplat` times out, build it
   alone per GPU type.
2. **Phase 4 variants** from the same saved solve, one launch each, read against the noise floor:
   `MODAL_PROFILE=opservemakesalt modal run --detach pipeline/modal/ply_eval.py --variant q5a` (then `q5b`, `q2`,
   `q4a`); `python pipeline/modal/ply_eval.py collect --scenario chapel --out /tmp/ply-eval` and
   `python scripts/eval_table.py /tmp/ply-eval` for the table. Q1, Q3 and Q6 still need code.
3. **Redeploy the worker** (riccomarcelino) and run the smoke capture (0.6) before any long job. The first long
   ordered capture after that shows whether skipping the global mapper saves the expected 32 minutes.
4. **Make the `ci` checks required** on `main` (repository settings, owner).
5. **Rest of phase 0**: the API side of 0.3 and 0.4 (it needs a D1 migration and deploys on merge), the smoke capture
   and `plyctl` (0.6, 0.7; they need the team code as a project variable).
6. **Walk** (2.1): walking speed and sliding along edges first. Free is the default until the owner says otherwise.

## Known and left alone

- The capture app cannot ask for the global mapper on a long video: it sends `--mapper` only for `incremental`, and
  its "Global (fast, default)" label no longer describes what a 400-frame walk gets (`app/src/capture/lib/backend.ts`).
- `wheels.yml` still defaults to Python 3.13 and arch 7.5. Dispatch it with `python=3.12` and the arch list.
- The model page accepts a module from anywhere on the site's origin, not only under `/ply/`. There is no content
  security policy, and `?scene=` accepts any origin by design.
- Two old twins (`c09a4c82…`, `0ffc07fb…`) live on the bucket, not on Hugging Face, and have no top view.
- Capture and upload findings C1 to C5 of `PROJECT-REVIEW.md` are open.
