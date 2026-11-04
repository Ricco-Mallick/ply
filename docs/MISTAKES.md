# Mistakes

Mistakes made while running and changing the pipeline, what they cost, and the rule that prevents each one.
Read the checklist before any paid GPU run. Add an entry when something costs credits or time it should not have.

## Before a paid run

- [ ] **One run, the one that was asked for.** Extra A/B runs need a yes first.
- [ ] **Right account.** Check which Modal account has credit and say which one the run uses.
- [ ] **Detached.** `modal run --detach ...`, and confirm the log says detached. Only the last function the
      entrypoint triggered survives a disconnect, so a detached entrypoint should start one long function.
- [ ] **Resumable.** Every finished stage is saved to a Volume before the next starts, and a relaunch picks up
      from there (`.done/<stage>.json` plus the job folder).
- [ ] **GPU fallback.** `gpu=["L4", "L40S"]`, and read `modal app logs` within two minutes: a queue for capacity
      looks exactly like a slow start.
- [ ] **Estimate from evidence.** Scale the closest earlier run's per-stage times by frame count (SfM grows faster
      than linearly), price the GPU *and* the CPU/memory asked for, state that basis, and report back if spend
      passes 1.5x the estimate.
- [ ] **Report spend from billing, not arithmetic.** `modal billing report --for today --json` per app.
- [ ] **Inputs present.** `pipeline/wheels` downloaded from the last green `wheels` run; data on the Volume checked
      with `modal volume ls`, symlinked folders included.

## Log

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
