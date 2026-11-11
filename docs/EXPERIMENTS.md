# Experiments

One row per run, whether it won or not, with the cost from Modal's billing (plan, rule 4). Tables below the first
section are printed by `scripts/eval_table.py` from each run's `metrics.json`.

## Free readings from saved runs (2026-11-11)

Read from the saved High chapel run (`ply-bench:chapel/high/job`, 60,000 steps, 2M cap) and its inputs. No run was
made.

| Question | Reading | Consequence |
|---|---|---|
| Which export filter removed 47% of the splats? | The opacity filter (`opacity > 0.01`): 2,000,000 in, 1,059,736 after it, 1,058,149 after the floater filter (1,587 removed). The crop was tried and rejected by the export's own scoring | The loss is decided in training, not in export |
| Why are they transparent? | 45% of the trained splats have opacity at or below 0.005 and 34% below 0.001. They are small (median largest axis 6 mm against 31 mm for the kept ones) and 75% lie inside the kept model's bounds. While relocation runs, 70,000 to 130,000 splats (3.5 to 6.5%) are found dead and moved every 100 steps. Relocation stops at step 49,700 (83% of the run); in the 10,300 steps after it, 940,000 splats fade out and nothing brings them back | The 2M cap is in use during training and half of it is lost in the last 17%. Raising the cap would add splats that fade the same way. Q5 (a weaker opacity regulariser) tests the cause directly; a later relocation stop is a second variant if Q5 helps |
| Step time | 35 to 36 steps a second at 2M splats; 1,718 s for 60,000 steps | 15,000 steps take about 7 minutes: the size of a phase 4 variant |
| Are the chapel clips HLG or log footage? | No. HEVC Main 10, `yuv420p10le`, transfer, primaries and matrix all `bt709`, limited range (DJI Osmo Action 5 Pro; one clip probed). The extracted frames have ordinary contrast and saturation (1st to 99th percentile of luminance 12 to 239) | No colour conversion is needed for the chapel. 0.11 records the tags for every future capture |
| How much does exposure vary? | Median frame luminance ranges from 27 to 148 (of 255) over 60 frames spread through the walk; up to 8% of a frame is clipped at the windows | This is what Q1 (exposure correction folded back) is for |
| Held-out score of the saved run | PSNR 25.59 dB, colour-corrected 26.03 dB, SSIM 0.872, LPIPS 0.212 on 41 views (every 30th) | The reference the 15k-step baselines sit under |

Not read: wall crossings on the chapel grid. That needs the walking evaluation (`eval/nav.py`), which is not built.

## Phase 1: baselines and noise floor (2026-11-11)

The chapel's saved solve (1,223 views), High settings, 15,000 steps, 2M cap, on an L40S (account opservemakesalt),
code `596c02d`. Two hold-outs: every 30th view (40 views), and a block of 20 consecutive frames (`000117` to `000136`,
clip `0051`) that no training view sits between. `q0a` and `q0b` are the same run twice.

| variant | steps | every cc-PSNR | every cc-SSIM | every cc-LPIPS | block cc-PSNR | block cc-SSIM | block cc-LPIPS | viewer PSNR desktop | viewer PSNR phone | splats kept | minutes | est. cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q0a | 15000 | 24.85 | 0.8382 | 0.2655 | 22.60 | 0.8052 | 0.1637 | 25.39 | 24.20 | 61% (1.22M) | 8.7 | $0.38 |
| q0b | 15000 | 24.71 | 0.8347 | 0.2666 | 22.05 | 0.8003 | 0.1667 | 25.02 | 23.93 | 61% (1.21M) | 8.8 | $0.38 |

| metric | baseline mean | noise floor | twice the floor |
|---|---:|---:|---:|
| every cc-PSNR | 24.78 | 0.14 | 0.28 |
| every cc-SSIM | 0.8364 | 0.0035 | 0.0070 |
| every cc-LPIPS | 0.2661 | 0.0011 | 0.0022 |
| block cc-PSNR | 22.32 | 0.54 | 1.09 |
| block cc-SSIM | 0.8028 | 0.0049 | 0.0098 |
| block cc-LPIPS | 0.1652 | 0.0030 | 0.0060 |
| viewer PSNR desktop | 25.21 | 0.37 | 0.74 |
| viewer PSNR phone | 24.07 | 0.27 | 0.55 |

- **The every-30th hold-out flatters the model by about 2.5 dB**: 24.78 dB there against 22.32 dB on the block.
- **What counts as a change.** PSNR has to move by 0.28 dB on the every-30th set (the plan's 0.2 dB is inside the
  noise) and by 1.1 dB on the block. LPIPS is the sharper instrument: the plan's 0.005 is more than twice its noise
  on the every-30th set (0.0022) and just under it on the block (0.0060).
- One pair gives one difference per metric, so the floor itself is a rough number. A third baseline would firm it up.

## Phase 4: first variants (2026-11-11)

Same solve, same settings, one change each.

| variant | change | every cc-PSNR | every cc-SSIM | every cc-LPIPS | block cc-PSNR | block cc-SSIM | block cc-LPIPS | viewer desktop | viewer phone | splats kept | see-through (worst view) | desktop file | est. cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline mean | | 24.78 | 0.8364 | 0.2661 | 22.32 | 0.8028 | 0.1652 | 25.21 | 24.07 | 61% (1.21M) | 0.02 to 0.14% (0.2 to 1.5%) | 38.9 MB | |
| q5a | opacity regulariser 0.005 | 24.97 | **0.8451** | **0.2466** | 22.56 | 0.8103 | **0.1567** | 25.24 | 24.28 | 72% (1.43M) | 1.8% (14.6%) | 44.8 MB | $0.34 |
| q5b | opacity regulariser 0.002 | **25.32** | **0.8483** | **0.2336** | 22.69 | **0.8208** | **0.1475** | 25.70 | 24.50 | 80% (1.60M) | 0.4% (3.0%) | 48.8 MB | $0.38 |
| q2 | pose refinement | 24.76 | 0.8339 | 0.2688 | 22.55 | 0.8097 | **0.1584** | 25.10 | 24.19 | 60% (1.20M) | 0.6% (5.6%) | 38.3 MB | $0.41 |
| q4a | sparse depth loss | diverged | | | | | | | | | | | $0.04 |

Bold is better than the baseline by more than twice the noise floor.

- **Q5, a weaker opacity regulariser: the first real gain.** At 0.002 every hold-out metric improves and all but
  block PSNR by more than twice the noise: +0.55 dB and −0.033 LPIPS (12%) on the every-30th set, −0.018 LPIPS on the
  block, and 80% of the trained splats survive export where 61% did. This is the mechanism the free readings
  pointed at. It is not free: the desktop file grows from 38.9 to 48.8 MB, and see-through pixels in the viewer rise
  from under 0.15% to 0.4% (3.0% in the worst view). 0.005 sits between the two on quality but shows more see-through
  than either (1.8%), which one run cannot explain.
  **Not the default yet.** Next: 0.001 and 0 to find where the trend ends, a look at the see-through views, then one
  full-length run of the winner before `train_launcher.py` changes.
- **Q2, pose refinement: no gain at this length.** Every-30th PSNR and SSIM are inside the noise and LPIPS is a hair worse (+0.0027, just outside it), the
  block's LPIPS is a little better (−0.0068), and training takes 4% longer. Not worth a default on this evidence.
- **Q4a, sparse depth loss: a defect, not a result.** The loss is NaN from about step 158 and the trainer aborts in
  the MCMC relocation near step 570. `--depth-loss` cannot be used as it is wired. The cause is not found yet; the first suspect is the disparity term, which upstream scales
  for a normalised scene and ours is in metres. It has to be fixed before any depth variant (Q4a, Q4b) can run.

**Cost.** Modal's billing for the day on opservemakesalt: $3.17 ($2.88 for the eval runs, one cancelled export among
them, and $0.29 for the image and wheel checks). The runner's own estimates for the same runs add up to $2.32, so billing
runs about a quarter above them.
