# Worker-PC test plan

Run these steps in order on the borrowed PC. Background and rules are in `HANDOFF.md`; read it first. Record each step in
`docs/test-runs/<pc>.md` (template at the bottom) and commit after each one. If a step fails, fix it, re-run that step,
and record both the failure and the fix.

Part A needs nothing from the API and can be done today. Part B needs Ricco to deploy the API first.

**Ricco brings:** one or two phone videos of a single room, 30–90 s each, on a USB stick or in Downloads. Shoot slowly
with lots of overlap. This is the test capture.

---

## Part A: setup and the pipeline, offline

### 0. The PC
- [ ] Record: Windows version, GPU, VRAM, driver (`nvidia-smi`), RAM, CPU, free disk on C:.
- [ ] Note whether the Windows account has admin rights. Step 3 needs one UAC prompt.

### 1. Parse every PowerShell script (it has never been done)
```powershell
foreach ($f in Get-ChildItem pipeline\local\*.ps1) {
  $errs = $null; [void][System.Management.Automation.Language.Parser]::ParseFile($f.FullName, [ref]$null, [ref]$errs)
  if ($errs) { "$($f.Name):"; $errs | ForEach-Object { "  line $($_.Extent.StartLineNumber): $($_.Message)" } } else { "$($f.Name): ok" }
}
```
- [ ] All four files report `ok` in **Windows PowerShell 5.1** (`powershell.exe`, which is what `start.bat` uses).
  Fix any errors before going on.
- [ ] Check that `start.bat` has CRLF line endings: `(Get-Content start.bat -Raw) -match "`r`n"` should print `True`.

### 2. Doctor on its own
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File pipeline\local\doctor.ps1
```
- [ ] It prints the GPU, RAM, disk, virtualisation and network checks, and writes `%LOCALAPPDATA%\Ply\machine.json`.
  Paste that JSON into the results file.
- [ ] The numbers match step 0. Any BLOCKER line is real, not a script mistake.

### 3. `start.bat` up to pairing
Double-click `start.bat`, or run it from a terminal so you can see the output: `cmd /c start.bat`.
- [ ] Step 2: if WSL was missing, the UAC prompt asks only for WSL, then Windows restarts. After Ricco logs back in,
  setup continues by itself (a scheduled task "Ply setup (resume after reboot)" runs once and then removes itself).
  Record whether that worked.
- [ ] Step 3: Ubuntu-24.04 installs with no questions; `wsl -d Ubuntu-24.04 -e whoami` prints `ply`.
- [ ] Step 4: `%USERPROFILE%\.wslconfig` gives Linux the RAM minus 4 GB plus 16 GB swap. An existing file is backed
  up as `.wslconfig.ply-backup`.
- [ ] Step 5: the copy lands at `~/ply` inside Ubuntu.
- [ ] Step 6: the install finishes (20–40 min) and its GPU smoke test passes. Record how long it took.
- [ ] Step 7 is **expected to fail until the API is deployed** (`/v1/pair` is a 404). Record the exact message. It
  should be a plain sentence, not a stack trace; if it isn't, make it one. Then stop here and go to step 4.

Setup log: `%LOCALAPPDATA%\Ply\start.log`. If a step fails, fix it in the Windows clone and run `start.bat`
again. Finished steps are skipped.

### 4. Inside Ubuntu: doctor and unit tests
```bash
wsl -d Ubuntu-24.04 -u ply
plylocal doctor
cd ~/ply/pipeline && $PLY_PY -m pytest tests -q -x --deselect tests/test_mushroom_seed.py --deselect tests/test_solvebench.py
```
(If pytest is missing: `$PLY_PY -m pip install pytest`.)
- [ ] `plylocal doctor` ends with `ready`. Paste its device plan: maxGaussians, maxLongSide, packed, loaderWorkers,
  sam3, and why.
- [ ] Unit tests: same result as on Linux (all pass apart from the two Modal-only tests).

### 5. First real run (quick preset)
```bash
plylocal run /mnt/c/Users/<them>/Downloads/<clip-or-folder> --name t1 --preset quick
```
- [ ] The console shows readable lines (stages, a progress bar for training with splats, VRAM, steps/s and ETA), not JSON.
- [ ] It finishes, or fails with a clear reason. Record the time of every stage from `plylocal status t1`.
- [ ] `report.json`: paste `frames` (kept vs dropped as blurry), `sfm.registeredRatio`, `train.eval` (psnr,
  ccPsnr, lpips), `train.growth` (rounds, kept or reverted, best, needsBiggerGpu), `train.density`,
  `train.peakVramGB`, `train.peakReservedGB`, `train.oomRetry`, the splat count, and the masks backend.
- [ ] Frames were not capped: the frame count follows the clip's motion, with only blurry frames dropped.
- [ ] People masks came from YOLO (`report` names the model) and `masks/` holds one mask per frame.
- [ ] `plylocal results t1` copies the twin to `%USERPROFILE%\Ply\t1`. Open `scene.spz` in a viewer, such as the
  repo's `app` or https://superspl.at/editor, and note any blurry or thin patches you see.

### 6. Restore point: Ctrl+C
Start a fresh run with a short autosave interval so the test doesn't take long:
```bash
PLY_AUTOSAVE_MIN=2 plylocal run <same clip> --name t2 --preset quick
```
- [ ] While training runs, wait for at least one `restore point saved at step N` line, then press **Ctrl+C**.
- [ ] Expected: `restore point saved at step N (... SIGINT)`, then `paused: a restore point is saved`, and the exit code is 75
  (`echo $?`).
- [ ] Run the same command again. Expected: `resumed from the restore point at step N` with the step at or just past
  where it stopped, and the loss and splat count continue rather than restarting.
- [ ] Let t2 finish. Compare its held-out PSNR and LPIPS with an uninterrupted run of the same settings. The difference
  should be within noise (about ±0.2 dB). If t1 used the same settings, compare against t1.

### 7. Restore point: hard stops
- [ ] **Close the window** while a run (t3) is training. Reopen it and run the same command. Record whether it
  resumed from a save made at the close (the user's "save and exit") or from the last autosave. If it was the
  autosave, closing didn't send the pipeline a signal: see HANDOFF.md, likely break 4. Fix it if you can, for
  example by having the launcher trap the window close.
- [ ] **`wsl --shutdown`** from PowerShell mid-training, then run again. It should resume from the last autosave.
- [ ] A restore point is never half-written: after the hard stop, resume works or cleanly starts over, and never crashes on a truncated file.

### 8. Growth, on a fuller run
```bash
plylocal run <clip> --name t4 --preset standard
```
- [ ] The console shows growth rounds with their comparison, for example `round 2: N splats, X dB (best Y) ... kept/reverted`.
- [ ] `report.json → train.growth`: every kept round improved held-out dB by ≥ 0.15 (or cut soft patches by 15 %
  and lost no dB). The final model is the best round, not the last.
- [ ] If it stopped on GPU memory, `needsBiggerGpu` is true and the report says so. No crash, no OOM loop.
- [ ] `train.density` lists under-dense cells (fixed by growth) and under-filmed cells (spots to film again), inside the room.
- [ ] Check that growth found the eval canvases (HANDOFF.md, likely break 5). If `patches` is null, find out why.

### 9. Calibrate the memory estimate
- [ ] From t1, t2 and t4: (`peakReservedGB` − about 1.2 GB) ÷ splats = bytes per splat. Compare with
  `BYTES_PER_GAUSSIAN = 2400` in `pipeline/plypipe/device.py`.
- [ ] If the real value is more than about 20 % off, update the constant and explain it in the commit message.

---

## Part B: worker mode (only after Ricco says the API is deployed)

Ricco does this from their main machine, not from this PC: deploy the API, apply migration `0003_machines.sql`, and
re-run `cloud/b2_setup.py`.

### 10. Pair
- [ ] Run `start.bat` again. Step 7 shows a code like `K7QM-3XTP`.
- [ ] Ricco approves it: `python scripts/plyctl.py machines approve K7QM-3XTP --label "<pc>"`. The PC continues on
  its own and step 8 opens the "Ply worker" window.
- [ ] `plyctl machines` lists the PC as online with its GPU, RAM and plan.
- [ ] Answer **N** to the "start at every logon" question on a friend's PC, unless the friend agrees.

### 11. Assigned job
- [ ] Ricco assigns a small job: `python scripts/plyctl.py assign <job> <pc-name>`. The worker claims it within a
  minute, shows progress lines, and the job finishes with results under `scenes/<job>/` in the bucket.
- [ ] An unassigned job in the shared queue is never taken by this PC.

### 12. Pause and resume through the API
- [ ] Close the worker window during training. The job goes back to `queued` with `pauses = 1`. Running the worker
  again (`%LOCALAPPDATA%\Ply\worker.cmd`) resumes it from the local restore point.
- [ ] Revoke: `plyctl machines revoke <pc-name>`. The worker stops within one poll (exit 3) and the token stops working.

---

## Cleanup (before handing the PC back)
- [ ] Copy anything you want to keep out first (`plylocal results`), and make sure the results file is committed.
- [ ] Inside Ubuntu: `plylocal forget`.
- [ ] From the clone, in PowerShell: `pipeline\local\ply.ps1 forget` removes the distro and the logon task, and
  restores the old `.wslconfig`.
- [ ] `plyctl machines revoke <pc-name>` (Ricco, if it was paired).
- [ ] `gh auth logout` if GitHub was signed in. Remove the clone if the friend wants it gone.

---

## Results file template (`docs/test-runs/<pc>.md`)

```markdown
# Test run: <pc>, <date>
PC: <Windows ver> | <GPU> <VRAM> GB, driver <ver> | <RAM> GB | <CPU> | <free disk> GB
Branch/commit at start: <hash>

| Step | Result | Notes / numbers |
|---|---|---|
| 0 PC | | |
| 1 PS parse | | |
| 2 doctor | | |
| 3 start.bat | | install time: |
| 4 doctor + tests | | |
| 5 t1 quick | | stage times; psnr/cc/lpips; splats; peak VRAM |
| 6 Ctrl+C resume | | |
| 7 hard stops | | close window: ; wsl --shutdown: |
| 8 growth t4 | | rounds: |
| 9 calibration | | bytes/splat: |
| 10–12 worker | | |

## Fixes made
- <commit>: <what and why>

## Still open
- 
```
