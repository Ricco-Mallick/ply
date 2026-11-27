# Worker PCs: a college lab PC, a friend's laptop

Any Windows 11 PC with an NVIDIA GPU (Turing or newer, 4 GB+) can become a Ply worker. It runs the pipeline inside
WSL 2 Ubuntu on its own GPU and takes only the jobs the team sends it by name.

## On the PC: one double-click

1. Get the repo as a zip and extract it anywhere.
   - The repo is private, so a friend can't use GitHub's Download ZIP.
   - Send them a zip yourself: `git archive -o ply-worker.zip HEAD`, or a release asset.
   - A zip has no git history. To update a PC, send a new zip and run `start.bat` again. That copies the new code
     in and keeps the install, the pairing and the jobs.
2. Double-click **`start.bat`**. It runs as you; Windows asks for admin rights only once, to turn on WSL.

`start.bat` then runs `pipeline\local\start.ps1`. Each step is skipped when it's already done, so running it again is
safe.

| Step | What happens |
|---|---|
| 1. Check | GPU, VRAM, driver, RAM, disk, virtualisation and the download hosts (`doctor.ps1`). Stops with a plain reason if the PC can't train. |
| 2. WSL 2 | Installed if missing. If Windows needs a restart, setup carries on by itself after you log back in. |
| 3. Ubuntu 24.04 | Installed with a fixed Linux user `ply`, so there are no questions. |
| 4. Memory | Linux gets all the RAM but 4 GB, plus 16 GB swap (`%USERPROFILE%\.wslconfig`; an existing file is backed up). |
| 5. Copy | The extracted folder is copied into Ubuntu (`~/ply`). |
| 6. Install | For this GPU (`install_local.sh`): CUDA 12.4 compiler, Node, torch, gsplat, pycolmap, splat-transform, then a GPU smoke test. Takes 20–40 min once. |
| 7. Pair | Shows a **pairing code** like `K7QM-3XTP`. Send it to the team. When it's approved, the PC receives its own token. |
| 8. Worker | Starts in a window titled "Ply worker" and keeps the PC awake while that window is open. It asks once whether to start the worker at every logon. |

Logs go to `%LOCALAPPDATA%\Ply\start.log`.

## On the team side: approve, look, assign

Use the team key (`PLY_TEAM_KEY`, or `--key-file`):

```bash
python scripts/plyctl.py machines                           # every paired PC: online?, GPU, RAM, what it can train
python scripts/plyctl.py machines approve K7QM-3XTP --label "college lab 3"
python scripts/plyctl.py assign <job> pc-desktop-lab3-ab12                                     # run the job as queued
python scripts/plyctl.py assign <job> pc-desktop-lab3-ab12 --args "--preset standard --from train"   # choose what runs
python scripts/plyctl.py assign <job> --shared              # back to the shared queue (Modal / Kaggle)
# a finished, in-review or running job is never overwritten: assign makes a copy (same clips, new id) and queues that
python scripts/plyctl.py machines revoke pc-desktop-lab3-ab12
```

- **A PC runs only what it is assigned.** The shared queue (Modal, Kaggle) never takes an assigned job, and the API
  doesn't wake Modal for one.
- **`--args` picks the stages.** With `--until occupancy` the PC runs only the camera solve and floor plan. With
  `--from train` it trains a job whose solve was done elsewhere. A failed or finished job can be assigned again and is
  re-queued.
- **What the PC's token can do:** claim its assigned jobs, report on them, read their inputs, and write
  `scenes/<that job>/`. Nothing else in the API answers it. `revoke` kills the token at the next request.
- **Pairing codes:** they expire after 24 h, and the API accepts at most 20 pairing requests an hour.
- **Results** from a PC go to the bucket, not Hugging Face, because a PC holds no HF token. The API serves them from
  either. Mind the bucket's 1 GB/day download cap for big twins.
- **Inputs on Hugging Face don't work on a PC.** A job that reads from it, such as `--scan hf:lidar/...` (the office
  route), fails there. Run those on Modal or Kaggle.
- **Deploy order:** migration `0003_machines.sql` must be in D1 before this API code runs. `api.yml` applies it before
  deploying and stops if it can't. Its Cloudflare token needs D1 edit rights, or you apply it by hand:
  `npx wrangler d1 execute ply-jobs --remote --file cloud/migrations/0003_machines.sql`.

API: `cloud/src/machines.ts`, migration `cloud/migrations/0003_machines.sql`, test `cloud/test/machines.test.mts`.

## What the PC plans for itself (plan.json → `device`)

The settings are planned from the capture (docs/PROCESSING.md) **and** from the machine. `plypipe.device` measures the
GPU and RAM on every run, and the PC reports the result when it claims, so `plyctl machines` shows it.

| | GTX 1650 4 GB, 16 GB RAM | L4 / L40S |
|---|---|---|
| Frames, frame rate | as planned from the clips | same |
| Working size | as planned (1280–2048 px); ≤1024 px only under 3.5 GB free | same |
| Splats | grow while held-out views improve, up to what the card holds (~1.1M on 4 GB). Past that the report says "needs a bigger GPU" | grow while held-out views improve |
| Rasterisation | packed (less memory, a bit slower) | default |
| Loader workers | 2 (RAM) | 4 |
| People masks | YOLO (YOLOE text prompts, COCO YOLO11-seg fallback) | SAM 3 |
| Object labels (SAM 3) | off | on |
| Out of memory | one retry from scratch at 0.7× the splat cap (`report.train.oomRetry`) | n/a |

- **The per-splat cost is an estimate.** Each run records `report.train.peakVramGB`, so tune `BYTES_PER_GAUSSIAN` in
  `plypipe/device.py` after the first real run on a 1650.
- **People:** small cards mask people with YOLO. SAM 3 masks made elsewhere (Kaggle, `--until occupancy`) are still used
  when a job continues on the PC.
- **Overrides:** `--device-profile off` plans as on a big cloud card. `PLY_DEVICE_VRAM_GB`, `PLY_DEVICE_RAM_GB` and
  `PLY_DEVICE_SM` override the probe.
- **Timing on a GTX 1650:** a ~300-frame room should take tens of minutes to solve and several hours to train at
  `standard`. These are guesses; measure the first run.

## By hand (no API), and Claude Code on the PC

Inside Ubuntu (`wsl`, then `cd ~/ply`), or `pipeline\local\ply.ps1 ...` from Windows:

```bash
plylocal doctor                                   # what this PC can do
plylocal run /mnt/c/Users/me/Videos/lab --preset standard       # resumable; same command again continues
plylocal status | jobs | results <name>           # results -> %USERPROFILE%\Ply\<name>
```

`--max-frames` is refused on purpose (docs/PROCESSING.md).

To use Claude Code on that PC, run it inside Ubuntu: `wsl`, then `cd ~/ply` and `claude`.

## Handing a borrowed PC back

- In Ubuntu, `plylocal forget` deletes jobs, the install and the caches.
- On Windows, `pipeline\local\ply.ps1 forget` removes the Ubuntu distro and the "Ply worker" logon task, and restores
  any earlier `.wslconfig`.
- On the team side, run `plyctl machines revoke <name>`.

Nothing else is left behind: the zip held no secrets, and the PC's token lived only in `~/.ply/worker.json` inside
Ubuntu.

## Why WSL, not native Windows

The GPU camera solve (`pycolmap-cuda12`) ships Linux wheels only. The training stack (gsplat 1.5.3 + fused-ssim) is
the route already proven on Kaggle's T4, and the GTX 1650 has the same architecture (Turing, sm 7.5).

## Stopping, resuming, and what the window shows

- **Closing the worker window or pressing Ctrl+C is safe.** Training saves a restore point at the end of its current
  step, the worker uploads it, and the job goes back to the queue as paused. The next start continues from that step.
- **Hard stops lose little.** If the PC just dies (power cut, lid closed, `wsl --shutdown`), the last autosave
  (every 10 min) is the restore point.
- **What the window shows:** one line per event, and for training every few seconds:
  `[########............]  42.1% | step 12,630/30,000 | loss 0.041 | 1.12M splats | VRAM 3.1 GB | 9.8 steps/s | ETA 29m`.
  Growth rounds print their comparison: `round 2: 1,680,000 splats, 29.41 dB (best 29.12) ... kept`.
