# Mistakes

Mistakes made while running and changing the pipeline, what they cost, and the rule that prevents each one.
Read the checklist before any paid GPU run. Add an entry when something costs credits or time it should not have.

## Before a paid run

- [ ] **One run, the one that was asked for.** Extra A/B runs need a yes first.
- [ ] **Right account.** Check which Modal account has credit and say which one the run uses.
- [ ] **Detached.** `modal run --detach ...`, and confirm the log says detached. Only the last function the
      entrypoint triggered survives a disconnect, so a detached entrypoint should start one long function, and start
      it with `.spawn()`: a call the client waits on (`.remote()`) is cancelled when the client dies.
- [ ] **Resumable.** Every finished stage is saved to a Volume before the next starts, and a relaunch picks up
      from there (`.done/<stage>.json` plus the job folder).
- [ ] **GPU fallback.** `gpu=["L4", "L40S"]`, and read `modal app logs` within two minutes: a queue for capacity
      looks exactly like a slow start.
- [ ] **Estimate from evidence.** Scale the closest earlier run's per-stage times by frame count (SfM grows faster
      than linearly), price the GPU *and* the CPU/memory asked for, state that basis, and report back if spend
      passes 1.5x the estimate.
- [ ] **Report spend from billing, not arithmetic.** `modal billing report --for today --json` per app.
- [ ] **Kill by PID, never by pattern.** Read `/proc/<pid>/cmdline` and match the target binary; a pattern in your
      own command line matches your own shell.
- [ ] **One launch at a time.** Modal refuses apps created within a few seconds of each other: let one `modal run`
      return before the next starts.
- [ ] **Inputs present.** `pipeline/wheels` filled by `modal run scripts/wheels.py::fetch` (the copy on Hugging Face;
      the `wheels` workflow's artifacts expire after 30 days); data on the Volume checked with `modal volume ls`,
      symlinked folders included.

## Log

### 2026-11-11: four launches at once, again ($0, about two minutes)

The entry below already says to start parallel runs a few seconds apart. The first wave on the small scenario was
started as four `modal run --detach` in one shell line, and Modal refused two of them (`App create rate limit
exceeded`). Nothing was billed; both were launched again one after the other.
**Rule:** it is in the checklist now: one launch at a time.

### 2026-11-11: an export cancelled with its client, `--detach` or not (~$0.25, two attempts)

`pipeline/modal/ply_eval.py` was launched with `modal run --detach` and its entrypoint waited on the run with
`.remote()`. The launching shell has a 10-minute limit; when it killed the client during the `q5b` export, Modal
logged a cancellation about 90 seconds later and the export's results were never saved. Killing the client by hand
on a second attempt reproduced it. Four other runs finished inside the 10 minutes and hid the problem. Four apps
started within a few seconds also hit Modal's app-creation rate limit, and one launch failed (no cost).
**Rule:** the entrypoint starts the long function with `.spawn()` and returns (`ply_eval.py` now does; `--wait`
blocks on purpose). `chapel_e2e.py`, `sfm_cpu.py` and `ply_bench.py` still wait with `.remote()`: launch them from a
shell that outlives the run, or change them the same way first. Start parallel runs a few seconds apart.

### 2026-11-05: three hand-off bugs found only by a live job, and `pkill -f` again (~$2.45)

The CPU-solve split's tests used stand-ins for Modal and the pipeline, so a deploy switch that never reached the
container, a `modal` import that fails in the pipeline subprocess, and a skipped `dense/` folder all passed them.
Two live chapel attempts found them. During cleanup, `pkill -f` and a `/proc` loop each matched the running shell
(and its `timeout` wrapper), losing the command.
**Rule:** an image check runs the same interpreter and subprocess `PYTHONPATH` as the job, and imports every module
the job imports lazily. To stop a process, anchor the pattern on the binary (`pgrep -f '^\S*python\S* \S*/modal run'`)
and print the matches before killing them.

### 2026-11-04: a failed copy, then a launch that ran anyway ($0.12)

Seeding the High chapel run from the CPU-solved job, `modal volume cp` failed ("No such file or directory": the
destination's parent did not exist) and the launch ran anyway, because the two were chained with `;`. The run found
no saved stages and started extracting frames, heading for a full camera solve on the GPU box. Stopped after 4 min.
**Rule:** a launch that depends on a copy or a check is chained with `&&` or guarded by an `if`, and the check reads
what the run will read (here the job's `.done/` list), not the copy's exit text.

### 2026-11-04: `pkill -f`-style match killed its own shell, twice more

Stopping the CPU solves with `for p in $(pgrep -f 'sfm_cpu.py'); do kill $p; done`, and later the dev server with
`pkill -f "vite --port 5199"`, both matched the shell running that very line: the mistake logged on 2026-11-03,
made two more times in one evening. The Modal apps were stopped separately, so no credit was lost.
**Rule (unchanged, now in the checklist):** match on the full command line from `/proc/<pid>/cmdline` and skip
anything that is not the target binary.

### 2026-11-04: chapel test cancelled after 95 minutes ($4.23, nothing saved)

Launched the end-to-end chapel run with plain `modal run`. When the local client disconnected at 06:05, Modal
cancelled the function in the last step of the camera solve (1,023 of 1,231 frames placed). The harness only saved
logs, not finished stages, so the relaunch had to start again from frame extraction.
**Rule:** detached and resumable (checklist), for anything longer than a few minutes.

### 2026-11-04: chapel test estimated at $1-2, cost $9.46

The estimate assumed the old run's timings, but the new planner keeps 4.5x the frames (1,236 vs 275). The relaunch
alone cost $5.22 for 119 min: the camera solve took 103 min, CPU-bound, while an L40S (no L4 was free) plus 8 cores
and 32 GB billed at about $2.60/h. In chat the first attempt was put at $3.30 and the relaunch at $2.50; billing
said $4.23 and $5.22.
**Rule:** estimate from evidence, report spend from billing (checklist). Don't run a CPU-bound stage on a GPU box.

### 2026-11-04: 11 minutes queued for an L4

The test function asked for `gpu="L4"` only; it waited for capacity with no container and no log. Noticed only
when checking why nothing had happened.
**Rule:** GPU fallback and an early log check (checklist).

### 2026-11-03: five training runs when one was asked for (~$2, most of an almost-empty account)

Asked for "a single higher quality run" of Dr Johnson; launched five parallel A/B variants on the Modal account with
the least credit left. Four were stopped part-way, and only the run that was asked for gave a result.
**Rule:** one run, right account (checklist).

### 2026-11-03: `modal volume put` dropped a symlink, one GPU scoring job wasted

Copied the Dr Johnson prep folder to the other account's Volume. `data/images` is a symlink to `dense/images`, and
`volume put` skips symlinks, so the scoring job found no images and returned nothing.
**Rule:** upload the real files (or dereference first) and list the Volume before spending GPU time on it.

### 2026-11-03: first launch failed on missing CUDA wheels

`pipeline/wheels` is gitignored and a fresh checkout has none, so every Modal image build failed at once.
Cheap this time; on a detached run it would have failed unseen.
**Rule:** inputs present (checklist): `gh run download <id> -D /tmp/wh` from the last green `wheels` run, then copy
the `.whl` files flat into `pipeline/wheels`.

### 2026-11-03: `pkill -f <pattern>` killed its own shell

`pkill -f 'tag -v360k'` matched the shell running the command, because the pattern was in its own command line,
so it killed itself and left the runs going.
**Rule:** find processes by PID (`/proc/<pid>/cmdline`) and kill those, or make sure the pattern cannot match
the command doing the killing.
