# Handoff: finish and prove the worker-PC pipeline on a real Windows PC

You are Claude Code, running on a Windows 11 PC that Ricco borrowed from a friend (or the college lab PC). The person
at the keyboard is Ricco, the repo owner. Your job: get this PC set up as a Ply worker with `start.bat`, then run every
test in **`TEST.md`**. When something breaks, fix the code, and record what happened so the next session (or Ricco's
main machine) can pick it up.

Read this whole file, then `TEST.md`, then `docs/LOCAL.md`. Begin with the first unchecked step in TEST.md.

## Where things stand (written 2026-11-28 on Ricco's main machine)

Branch **`feat/local-windows`**. All code is written and the unit tests pass on Linux (561 pass; `test_mushroom_seed`
and `test_solvebench` fail everywhere except Modal, which is expected). **None of it has run on Windows or a GPU.**

What the branch adds:

| Piece | Files | Proven? |
|---|---|---|
| Double-click setup: WSL 2, Ubuntu 24.04 user `ply`, `.wslconfig`, install for this GPU, pair, worker window | `start.bat`, `pipeline/local/start.ps1`, `doctor.ps1`, `ply.ps1`, `install_local.sh`, `plylocal.sh` | **No.** The PowerShell has never even been parsed. |
| Settings planned from this PC's GPU and RAM (splat start, packed raster, loader workers, SAM 3 or YOLO) | `pipeline/plypipe/device.py`, `plan.py`, `cli.py` | Unit tests only |
| Restore points: autosave every 10 min; on Ctrl+C/close/SIGTERM it saves and exits with code 75; the same command resumes at the exact step | `pipeline/plypipe/restore.py`, `train_launcher.py`, `stages/train.py`, `util.py` | CPU stand-in only |
| Growth: no splat ceiling. More splats each round while held-out dB goes up; otherwise it keeps the previous model | `pipeline/plypipe/growth.py`, `stages/train.py` | Stand-in trainer only |
| YOLO people masks where SAM 3 won't fit (YOLOE text prompts, COCO YOLO11-seg fallback) | `pipeline/plypipe/people_yolo.py`, `people.py` | Stand-in model only |
| Readable progress lines in the console | `pipeline/plypipe/console.py`, `log.py` | Unit tests only |
| Pairing: the PC shows a code, the team approves it, the PC gets its own revocable token and runs only jobs assigned to it | `pipeline/plypipe/enroll.py`, `worker.py`, `cloud/src/machines.ts`, `cloud/migrations/0003_machines.sql` | Local D1 stand-in. **The live API doesn't have it yet:** `/v1/pair` returns 404 until Ricco deploys. |

## Rules (Ricco's, not negotiable)

1. **This is someone else's computer.** Do not save any secret on it that outlives the session. Specifically:
   no `WORKER_TOKEN`, no Hugging Face token, no Cloudflare/wrangler login, no Modal token, no B2 keys. The only
   credential that may live here is the PC's own pairing token (`~/.ply/worker.json` inside Ubuntu, revocable).
   If GitHub needs signing in to clone or push, use `gh auth login` and run `gh auth logout` before the session ends
   (TEST.md, last step).
2. **Never push, merge, deploy or open a PR without asking Ricco first, each time.** Commit freely on
   `feat/local-windows`. Never commit to `main`.
3. **Never use `--max-frames`.** Settings are planned from the capture. After every run read `report.json`.
4. **Never spend money.** Don't start Modal or anything else that bills.
5. **The repo is proprietary.** Don't paste code or captures into outside services.
6. Ricco's email stays private: never put it in a commit, a log you share, or a request to an outside service.
7. Keep fixes small and explain them in the commit message. If a fix changes behaviour that `docs/LOCAL.md` or
   `docs/PROCESSING.md` describes, update the doc in the same commit.

## How this PC is laid out (important: two copies of the code)

- **The Windows clone** (where you were started, e.g. `C:\Users\<them>\ply`) is the git repo and the
  **only place you edit and commit**.
- **`start.bat` copies the clone into Ubuntu at `~/ply`.** It deletes and recopies that folder every time it runs, so
  any edit made only in Ubuntu's `~/ply` is lost. Jobs live in `~/ply-work`, the install in `~/.ply`. Re-running
  `start.bat` leaves both alone.
- After you edit pipeline code, copy it into Ubuntu without re-running the whole setup (PowerShell, from the clone):

  ```powershell
  $src = (wsl -d Ubuntu-24.04 -u ply -e wslpath -a ($PWD.Path -replace '\\','/')).Trim()
  wsl -d Ubuntu-24.04 -u ply -e bash -lc "rm -rf ~/ply.new && cp -r '$src' ~/ply.new && rm -rf ~/ply && mv ~/ply.new ~/ply && find ~/ply/pipeline/local -name '*.sh' -exec sed -i 's/\r$//' {} + && echo synced"
  ```
- Run pipeline commands inside Ubuntu: `wsl -d Ubuntu-24.04 -u ply -e bash -lc "plylocal doctor"` (or open `wsl` and
  work there). `~/.ply/env.sh` sets `PLY_PY` (pipeline venv) and `PLY_TRAIN_PYTHON` (training venv, py3.10).
- Logs: `%LOCALAPPDATA%\Ply\start.log` (setup), `~/ply-work/jobs/<name>/log.jsonl` and `report.json` (each run).

## Your memory across reboots: the results file

WSL setup can restart Windows, which ends this session. So **write results down as you go** in
`docs/test-runs/<pc>.md` (e.g. `docs/test-runs/college-lab-1650.md`). Copy the template at the bottom of `TEST.md`,
tick each step, and paste the important numbers and errors. Commit after every step. A new session starts by reading
this file, TEST.md and that results file, then carries on from the first unticked step.

## Things most likely to break (look here first)

1. **PowerShell 5.1 parsing and quoting** in `start.ps1`, `doctor.ps1`, `ply.ps1`. They were written blind. The
   `Linux` helper must not contain double quotes in its command. Paths with spaces or an apostrophe in the user name
   may break step 5.
2. **`install_local.sh` on a fresh Ubuntu 24.04**: the CUDA 12.4 apt repo for WSL, prebuilt gsplat 1.5.3 in the
   py3.10 venv, `pycolmap-cuda12`, YOLO weights and CLIP in that venv.
3. **Restore shim against the real gsplat trainer.** `restore.py` replaces `tqdm.tqdm(range(init_step, max_steps))` in
   `simple_trainer`. If the trainer's loop looks different, resume won't hook in. Check for the "restore point saved"
   and "resumed from the restore point at step N" lines.
4. **Does closing the console window save?** Unknown whether closing a `wsl.exe` window sends SIGHUP to the pipeline
   inside Ubuntu. If it doesn't, closing loses up to 10 minutes (last autosave). Record what really happens. A fix,
   if needed, belongs in the worker launcher (`start.ps1` step 8) or `plylocal.sh`.
5. **Growth reading the eval renders**: `growth.patches()` expects `renders/val_step<N>_*.png` canvases (photo left,
   render right) and `stats/val_step<N>.json`. Check the real file names in a run's result folder.
6. **The memory estimate**: `BYTES_PER_GAUSSIAN = 2400` in `device.py` is a guess. Calibrate it from
   `report.json → train.peakVramGB / peakReservedGB` against the splat count (TEST.md step 9).
7. **GTX 1650 specifics**: 4 GB, Turing sm_7.5 (same as Kaggle's T4). Packed rasterisation is on below 12 GB.

## When you finish (or have to stop)

- Results file complete and committed. Everything committed on `feat/local-windows`.
- Ask Ricco whether to push the branch. Don't push without a yes.
- Give a short summary: what passed, what you fixed (commit hashes), what still fails, and what Ricco must do
  from their main machine (deploy the API, apply migration 0003, B2 lifecycle).
- Leave the PC as TEST.md's cleanup step says, unless Ricco wants it kept as a worker.
